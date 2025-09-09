#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import argparse
import json
import signal
import atexit
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from PIL import Image
import torch

# Keep imports for original OCR pipeline
from transformers import pipeline
from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
from PaddleOCR import PaddleOCR

import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------
# Global variables / state
# -------------------------
current_results = {}
current_output_file = None
processed_folders = set()
progress_file = None

# NOTE: corrector will be initialised in main process (not pickled to workers).
# For workers we will re-create models inside worker function to avoid pickling issues.


# -------------------------
# Utility: save / load progress
# -------------------------
def save_current_results():
    """Save current results immediately (used by main process on exit)."""
    if current_results and current_output_file:
        try:
            with open(current_output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
            print(f"Emergency save: {current_output_file}")
        except Exception as e:
            print(f"Error saving emergency results: {e}")


def save_progress_log():
    """Save progress log"""
    global progress_file
    if progress_file is None:
        print("Progress file path not set, cannot save progress")
        return

    try:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "processed_folders": sorted(list(processed_folders)),
            }, f, ensure_ascii=False, indent=2)
        print(f"Progress saved: {len(processed_folders)} folders completed")
    except Exception as e:
        print(f"Error saving progress: {e}")


def load_progress(resume_from=None, resume_file=None):
    """Load progress from file (main process only)."""
    global processed_folders, progress_file
    try:
        # priority: explicit resume_from -> progress_file -> resume_file
        if resume_from and Path(resume_from).exists():
            print(f"Loading progress from specified file: {resume_from}")
            with open(resume_from, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                processed_folders = set(progress.get("processed_folders", []))
                print(f"Loaded {len(processed_folders)} processed folders from specified resume file")
                # propagate to current progress file
                save_progress_log()
                return True

        elif progress_file and progress_file.exists():
            print(f"Loading progress from: {progress_file}")
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                processed_folders = set(progress.get("processed_folders", []))
                print(f"Loaded {len(processed_folders)} processed folders from progress file")
                return True

        elif resume_file and Path(resume_file).exists():
            print(f"Loading progress from default resume file: {resume_file}")
            with open(resume_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                processed_folders = set(progress.get("processed_folders", []))
                print(f"Loaded {len(processed_folders)} processed folders from default resume file")
                save_progress_log()
                return True

    except Exception as e:
        print(f"Error loading progress: {e}")
        processed_folders = set()
        return False

    print("No progress file found, starting fresh")
    processed_folders = set()
    return False


# -------------------------
# The "predict" logic - kept same interface but we add a batch-friendly detector call
# -------------------------
def predict_single(recognitor, detector, img_path, padding=4):
    """Original predict logic for a single image (kept as-is)."""
    img = cv2.imread(img_path)
    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    # PaddleOCR returns nested structure; normalize
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        result = result[0]
    boxes = []
    for line in result:
        boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
    boxes = boxes[::-1]
    for box in boxes:
        box[0][0] = max(0, box[0][0] - padding)
        box[0][1] = max(0, box[0][1] - padding)
        box[1][0] = min(img.shape[1], box[1][0] + padding)
        box[1][1] = min(img.shape[0], box[1][1] + padding)

    texts = []
    for i, box in enumerate(boxes):
        try:
            cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                continue
            cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            rec_result = recognitor.predict(cropped_image)
            texts.append(rec_result)
        except Exception as e:
            continue

    # Attempt correction if pipeline available (recognitor/corrector may be None in workers)
    if texts:
        return texts
    else:
        return []


def predict_batch(recognitor, detector, img_paths, padding=4, corrector=None, hf_batch_size=16):
    """
    Batch detection using PaddleOCR (detector.ocr can accept list of paths).
    For each image, do recognition using recognitor (per-crop), then optionally correct via HF pipeline in batch.
    Returns dict: {img_path: [texts...], ...}
    """
    results = {}
    if not img_paths:
        return results

    # PaddleOCR supports list of images -> returns list of results
    try:
        detections = detector.ocr(img_paths, cls=False, det=True, rec=False)
    except Exception as e:
        # fallback: per-image detection
        detections = []
        for p in img_paths:
            try:
                det = detector.ocr(p, cls=False, det=True, rec=False)
            except Exception:
                det = []
            detections.append(det)


    # Normalize paddle results: each element corresponds to an image (some versions return [ [boxes], ... ])
    # We'll iterate paired
    for idx, img_path in enumerate(img_paths):
        det = detections[idx] if idx < len(detections) else []
        if isinstance(det, list) and len(det) > 0 and isinstance(det[0], list):
            det = det[0]
        img = cv2.imread(img_path)
        boxes = []
        for line in det:
            try:
                boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
            except Exception:
                continue
        boxes = boxes[::-1]
        for box in boxes:
            box[0][0] = max(0, box[0][0] - padding)
            box[0][1] = max(0, box[0][1] - padding)
            box[1][0] = min(img.shape[1], box[1][0] + padding)
            box[1][1] = min(img.shape[0], box[1][1] + padding)

        texts = []
        for i, box in enumerate(boxes):
            try:
                cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                    continue
                cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                rec_result = recognitor.predict(cropped_image)
                texts.append(rec_result)
            except Exception:
                continue

        results[os.path.basename(img_path)] = texts

    # If provided a HuggingFace correction pipeline, batch-correct across all texts per-image
    if corrector and results:
        # collect per-image lists flattened
        all_texts = []
        keys = []
        for k, tlist in results.items():
            if tlist:
                all_texts.extend(tlist)
                keys.extend([k] * len(tlist))
        if all_texts:
            try:
                corrected = corrector(all_texts, batch_size=hf_batch_size, max_new_tokens=256)
                # corrected is list of dicts with 'generated_text'
                corrected_texts = [c.get('generated_text', '') for c in corrected]
                # map back
                mapped = {}
                for k in results.keys():
                    mapped[k] = []
                idx = 0
                for txt in corrected_texts:
                    k = keys[idx]
                    mapped[k].append(txt)
                    idx += 1
                # For images that had no detected texts, keep empty list
                for k in results.keys():
                    results[k] = mapped.get(k, [])
            except Exception:
                # if correction fails, keep raw texts
                pass

    return results


# -------------------------
# Folder processing (worker-friendly)
# -------------------------
def process_folder_worker(args_tuple):
    """
    Worker function executed in separate process.
    args_tuple: (folder_name, input_folder, output_folder, device, batch_size_images, hf_model_name_or_none)
    This function will:
      - load models locally inside the worker
      - process images in the folder in batches (detection batched)
      - write output_file {folder_name}.json
      - return (folder_name, status_string)
    """
    folder_name, input_folder, output_folder, device, batch_size_images, hf_model_name = args_tuple

    # Prepare output file path
    output_file = os.path.join(output_folder, f"{folder_name}.json")

    # If output exists and seems complete (we don't know completeness), we will resume reading it
    current_results_local = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                current_results_local = json.load(f)
        except Exception:
            current_results_local = {}

    # Gather image files (sorted)
    image_files = [f for f in sorted(os.listdir(input_folder))
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        # write empty result and return
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return (folder_name, "empty")

    # Load models inside worker (so each process has its own Predictor & PaddleOCR)
    try:
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = True
        config['predictor']['beamsearch'] = True
        config['device'] = device
        recognitor = Predictor(config)
    except Exception as e:
        print(f"[Worker {folder_name}] Error loading VietOCR: {e}")
        recognitor = None

    try:
        detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=(device == 'cuda'))
    except Exception as e:
        print(f"[Worker {folder_name}] Error loading PaddleOCR: {e}")
        detector = None

    # Optionally load HF corrector inside worker if name provided
    corrector_local = None
    if hf_model_name:
        try:
            corrector_local = pipeline("text2text-generation", model=hf_model_name)
        except Exception as e:
            corrector_local = None

    total_images = len(image_files)
    processed_count = len([f for f in image_files if f in current_results_local])
    remaining_images = total_images - processed_count

    if remaining_images == 0:
        return (folder_name, "already_complete")

    # We'll process images in batches
    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    try:
        for batch in chunked(image_files, batch_size_images):
            # skip already processed in this batch
            to_process = [os.path.join(input_folder, f) for f in batch if f not in current_results_local]
            if not to_process:
                continue

            # Run batched detection+recognition
            batch_results = predict_batch(recognitor, detector, to_process, corrector=corrector_local)

            # Merge to current_results_local
            for p in to_process:
                key = os.path.basename(p)
                current_results_local[key] = batch_results.get(key, [])

            processed_count = len([f for f in image_files if f in current_results_local])

            # Save intermediate (after each batch)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(current_results_local, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # final save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_results_local, f, ensure_ascii=False, indent=2)

        return (folder_name, "success")
    except Exception as e:
        # on error, attempt to save partial
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results_local, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return (folder_name, f"failed:{e}")


# -------------------------
# Main: orchestrates folder-level parallelism
# -------------------------
def main():
    global processed_folders, progress_file, current_output_file

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Root directory containing folders with images')
    parser.add_argument('--output_dir', default='./runs/batch_predict', help='Directory to save JSON files')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use: auto (detect automatically), cpu, or cuda')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--resume_file', type=str, help='Specific resume file to load from')
    parser.add_argument('--progress_dir', type=str, help='Directory to save progress file (default: output_dir)')
    parser.add_argument('--max_workers', type=int, default=None, help='Max parallel folder workers (default auto)')
    parser.add_argument('--batch_images', type=int, default=8, help='Number of images to batch per PaddleOCR call')
    parser.add_argument('--hf_corrector', type=str, default="bmd1905/vietnamese-correction-v2",
                        help='HF model name for correction (set empty "" to disable in workers)')
    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Prepare progress file path
    if args.progress_dir:
        progress_dir = Path(args.progress_dir)
    else:
        progress_dir = Path(args.output_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file_local = progress_dir / "processing_progress.json"
    progress_file = progress_file_local

    # default resume file (backward compatibility)
    resume_file_default = Path("/kaggle/input/l25-gen/processing_progress.json")

    print("=" * 60)
    print("🚀 Starting Vietnamese OCR Batch Processing (folder-level parallel + batched detection)")
    print("=" * 60)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Progress file: {progress_file}")
    print(f"🎮 Device: {device}")
    if args.resume_file:
        print(f"📄 Resume from provided file: {args.resume_file}")

    # Register signal handlers in main process
    def signal_handler_main(signum, frame):
        print(f"Received signal {signum}. Saving progress and exiting...")
        save_current_results()
        save_progress_log()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler_main)
    signal.signal(signal.SIGINT, signal_handler_main)
    atexit.register(save_current_results)

    # Load progress if requested
    resume_loaded = False
    if args.resume or args.resume_file or progress_file.exists():
        resume_loaded = load_progress(args.resume_file, resume_file_default)
        print(f"📋 Resume mode: {len(processed_folders)} folders already completed")
    else:
        processed_folders = set()
        print("🆕 Fresh start mode")

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Scan input directory for folders to process (supports one-level or two-level structure)
    all_folders = []
    direct_images = [f for f in os.listdir(args.input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if direct_images:
        folder_name = os.path.basename(args.input_dir.rstrip("/"))
        all_folders.append((folder_name, args.input_dir))
        print(f"📂 Found {len(direct_images)} images directly in input directory")
    else:
        for folder in sorted(os.listdir(args.input_dir)):
            folder_path = os.path.join(args.input_dir, folder)
            if os.path.isdir(folder_path):
                images_in_folder = [f for f in os.listdir(folder_path)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                if images_in_folder:
                    all_folders.append((folder, folder_path))
                else:
                    # one level deeper
                    for subfolder in sorted(os.listdir(folder_path)):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        if os.path.isdir(subfolder_path):
                            images_in_subfolder = [f for f in os.listdir(subfolder_path)
                                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                            if images_in_subfolder:
                                all_folders.append((subfolder, subfolder_path))

    total_folders = len(all_folders)
    remaining_folders = [f for f in all_folders if f[0] not in processed_folders]

    print(f"\n📊 FOLDER SUMMARY:")
    print(f"   Total folders found: {total_folders}")
    print(f"   Already processed: {len(processed_folders)}")
    print(f"   Remaining to process: {len(remaining_folders)}")

    if not remaining_folders:
        print("🎉 ALL FOLDERS HAVE BEEN PROCESSED!")
        return

    # Decide max_workers
    if device == 'cuda':
        # On Kaggle GPU (single GPU), safer to run only 1 worker to avoid multiple processes competing for GPU memory.
        if args.max_workers is None:
            max_workers = 1
        else:
            max_workers = args.max_workers
            if max_workers > 1:
                print("⚠️ Warning: You set >1 workers while device is cuda. This may cause GPU contention.")
    else:
        # CPU mode: allow multiple workers
        if args.max_workers is None:
            max_workers = min(4, (os.cpu_count() or 2))
        else:
            max_workers = args.max_workers

    print(f"\n🔁 Will process folders in parallel with max_workers={max_workers}, batch_images={args.batch_images}")

    # Build worker args
    worker_args = []
    for folder_name, folder_path in remaining_folders:
        worker_args.append((folder_name, folder_path, args.output_dir, device, args.batch_images,
                             args.hf_corrector if args.hf_corrector else None))

    # Use ProcessPoolExecutor to run folder-level parallelism (each worker loads its own models)
    newly_processed = 0
    failed_folders = 0
    empty_folders = 0
    skipped_folders = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {executor.submit(process_folder_worker, wa): wa[0] for wa in worker_args}
        for fut in as_completed(future_to_folder):
            folder = future_to_folder[fut]
            try:
                folder_name, result = fut.result()
                # update global processed_folders & save progress
                processed_folders.add(folder_name)
                save_progress_log()

                if result == "success" or result == "already_complete":
                    newly_processed += 1
                    print(f"✅ [{folder_name}] {result}")
                elif result.startswith("failed"):
                    failed_folders += 1
                    print(f"❌ [{folder_name}] {result}")
                elif result == "empty":
                    empty_folders += 1
                    print(f"📁 [{folder_name}] EMPTY")
                else:
                    print(f"ℹ️ [{folder_name}] {result}")

            except Exception as e:
                print(f"❌ Error in worker for {folder}: {e}")
                failed_folders += 1

            # print quick overall progress
            current_total_processed = len(processed_folders)
            progress_pct = (current_total_processed / total_folders) * 100 if total_folders > 0 else 100.0
            print(f"📈 Overall progress: {current_total_processed}/{total_folders} ({progress_pct:.1f}%)")

    # Final summary
    print("\n" + "=" * 60)
    print("🏁 PROCESSING SUMMARY")
    print("=" * 60)
    print(f"📊 Total folders: {total_folders}")
    print(f"📋 Previously processed: {len(processed_folders) - newly_processed}")
    print(f"🆕 Newly processed this session: {newly_processed}")
    print(f"📁 Empty folders: {empty_folders}")
    print(f"❌ Failed: {failed_folders}")
    print(f"✅ Final total processed: {len(processed_folders)}/{total_folders}")
    if len(processed_folders) == total_folders:
        print("🎉 ALL FOLDERS COMPLETED SUCCESSFULLY!")
    elif failed_folders == 0:
        print("✅ All remaining folders processed successfully!")
    else:
        remaining = total_folders - len(processed_folders)
        if remaining > 0:
            print(f"📝 {remaining} folders still need to be processed")
    print("=" * 60)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
