import os
import sys
import argparse
import json
import signal
import atexit
import shutil
from pathlib import Path
from datetime import datetime

import cv2
from PIL import Image
import torch
from transformers import pipeline
from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
from PaddleOCR import PaddleOCR

import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variables for checkpoint
current_results = {}
current_output_file = None
processed_folders = []

# Paths for progress log
resume_file = Path("/kaggle/input/l25-gen/processing_progress.json")
progress_file = Path("/kaggle/working/processing_progress.json")

# Initialize corrector globally to avoid reloading
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")


def signal_handler(signum, frame):
    print(f"Received signal {signum}. Saving progress before exit...")
    save_current_results()
    save_progress_log()
    sys.exit(0)


def save_current_results():
    """Save current results immediately"""
    if current_results and current_output_file:
        try:
            with open(current_output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
            print(f"Emergency save: {current_output_file}")
        except Exception as e:
            print(f"Error saving emergency results: {e}")


def save_progress_log():
    """Save progress log into /kaggle/working/"""
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "processed_folders": processed_folders,
                "current_file": str(current_output_file) if current_output_file else None
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")


def predict(recognitor, detector, img_path, padding=4):
    """VietOCR + PaddleOCR prediction with correction"""
    # Load image
    img = cv2.imread(img_path)
    
    # Text detection using PaddleOCR
    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    result = result[:][:][0]
    
    # Filter Boxes
    boxes = []
    for line in result:
        boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
    boxes = boxes[::-1]
    
    # Add padding to boxes
    for box in boxes:
        box[0][0] = max(0, box[0][0] - padding)
        box[0][1] = max(0, box[0][1] - padding)
        box[1][0] = min(img.shape[1], box[1][0] + padding)
        box[1][1] = min(img.shape[0], box[1][1] + padding)
    
    # Text recognition using VietOCR
    texts = []
    for i, box in enumerate(boxes):
        try:
            # Extract cropped region
            cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            
            # Check if cropped image has valid dimensions
            if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                print(f"Warning: Skipping box {i} with invalid dimensions: {cropped_image.shape}")
                continue
            
            # Convert to PIL Image
            cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            
            # Check PIL image dimensions
            if cropped_image.size[0] <= 0 or cropped_image.size[1] <= 0:
                print(f"Warning: Skipping box {i} with invalid PIL dimensions: {cropped_image.size}")
                continue

            rec_result = recognitor.predict(cropped_image)
            texts.append(rec_result)
            
        except Exception as e:
            print(f"Warning: Error processing box {i}: {e}")
            continue

    # Apply Vietnamese text correction
    if texts:
        try:
            corrections = corrector(texts, max_new_tokens=256)
            corrected_texts = [pred['generated_text'] for pred in corrections]
            return corrected_texts
        except Exception as e:
            print(f"Warning: Error in text correction: {e}")
            return texts
    else:
        return []


def process_folder(recognitor, detector, input_folder, output_folder):
    global current_results, current_output_file
    
    folder_name = os.path.basename(input_folder)
    output_file = os.path.join(output_folder, f"{folder_name}.json")
    current_output_file = output_file

    # Skip if already processed
    if folder_name in processed_folders:
        print(f"Skipping {folder_name}, already processed.")
        return "skipped"

    current_results = {}
    
    # Check if file already exists (resume from checkpoint)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                current_results = json.load(f)
            print(f"Resuming from existing file: {output_file}")
        except:
            current_results = {}
    
    image_files = [f for f in sorted(os.listdir(input_folder)) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {folder_name}")
        processed_folders.append(folder_name)
        save_progress_log()
        return "empty"

    print(f"Processing folder {folder_name} with {len(image_files)} images...")
    
    for i, file in enumerate(image_files, 1):
        # Skip if already processed in this session
        if file in current_results:
            print(f"Skipping already processed: {file}")
            continue
            
        img_path = os.path.join(input_folder, file)
        print(f"Processing [{i}/{len(image_files)}]: {file}")
        
        try:
            # Use the advanced OCR pipeline
            corrected_texts = predict(recognitor, detector, img_path)
            current_results[file] = corrected_texts
            
            # Print recognized text
            for text in corrected_texts:
                print(f"  Text: {text}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            current_results[file] = []  # Store empty result for failed images

        # Save immediately after each image
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving intermediate results: {e}")

    # Final save for this folder
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
        processed_folders.append(folder_name)
        save_progress_log()
        return "success"
    except Exception as e:
        print(f"Error saving final results: {e}")
        return "failed"


def main():
    global processed_folders
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(save_current_results)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Root directory containing folders with images')
    parser.add_argument('--output_dir', default='./runs/batch_predict', help='Directory to save JSON files')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use: auto (detect automatically), cpu, or cuda')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    # Resume logic
    initial_processed_count = 0
    if args.resume:
        if resume_file.exists() and not progress_file.exists():
            shutil.copy(resume_file, progress_file)
            print(f"Copied resume file from {resume_file} to {progress_file}")

        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    processed_folders = progress.get("processed_folders", [])
                initial_processed_count = len(processed_folders)
                print(f"Resuming: {initial_processed_count} folders already processed")
            except Exception as e:
                print(f"Error loading progress log: {e}")
                processed_folders = []

    # Configure VietOCR
    print("Loading VietOCR model...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    
    # Device configuration
    if args.device == 'auto':
        if torch.cuda.is_available():
            config['device'] = 'cuda'
            print("Auto-detected device: CUDA GPU")
        else:
            config['device'] = 'cpu'
            print("Auto-detected device: CPU")
    else:
        config['device'] = args.device
        print(f"Using specified device: {args.device}")

    recognitor = Predictor(config)

    # Configure PaddleOCR for text detection
    print("Loading PaddleOCR detector...")
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=(config['device'] == 'cuda'))
    
    print("Models loaded successfully!")

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all folders to process (flat structure: /kaggle/input/l25-keyframes/keyframes/L25_V001/)
    all_folders = []
    for folder in sorted(os.listdir(args.input_dir)):
        folder_path = os.path.join(args.input_dir, folder)
        if os.path.isdir(folder_path):
            all_folders.append((folder, folder_path))

    total_folders = len(all_folders)
    remaining_folders = [f for f in all_folders if f[0] not in processed_folders]
    
    print(f"Found {total_folders} total folders")
    print(f"Already processed: {len(processed_folders)} folders")
    print(f"Remaining to process: {len(remaining_folders)} folders")
    
    if not remaining_folders:
        print("🎉 All folders have already been processed!")
        return
    
    # Process remaining folders
    newly_processed = 0
    failed_folders = 0
    empty_folders = 0
    
    for i, (folder_name, folder_path) in enumerate(all_folders, 1):
        if folder_name in processed_folders:
            # Don't process folders that were already processed
            continue
            
        remaining_count = len([f for f in all_folders[i-1:] if f[0] not in processed_folders])
        print(f"\n=== Processing folder {i}/{total_folders}: {folder_name} ===")
        print(f"Remaining folders: {remaining_count}")
        
        result = process_folder(recognitor, detector, folder_path, args.output_dir)
        
        if result == "success":
            newly_processed += 1
            print(f"✅ Successfully processed {folder_name}")
        elif result == "failed":
            failed_folders += 1
            print(f"❌ Failed to process {folder_name}")
        elif result == "empty":
            empty_folders += 1
            print(f"📁 Empty folder: {folder_name}")
        elif result == "skipped":
            # This shouldn't happen since we filter above
            print(f"⏭️  Unexpectedly skipped: {folder_name}")
            
        current_total_processed = len(processed_folders)
        print(f"Progress: {current_total_processed}/{total_folders} folders completed")

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total folders: {total_folders}")
    print(f"Previously processed: {initial_processed_count}")
    print(f"Newly processed this session: {newly_processed}")
    print(f"Empty folders: {empty_folders}")
    print(f"Failed: {failed_folders}")
    print(f"Final total processed: {len(processed_folders)}/{total_folders}")
    
    if len(processed_folders) == total_folders:
        print("🎉 ALL FOLDERS COMPLETED!")
    elif failed_folders == 0:
        print("✅ All remaining folders processed successfully!")
    else:
        print(f"⚠️  {failed_folders} folders failed to process")
        remaining = total_folders - len(processed_folders)
        if remaining > 0:
            print(f"📝 {remaining} folders still need to be processed")


if __name__ == '__main__':
    main()
