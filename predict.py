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
processed_folders = set()

# Global variables for progress file path
progress_file = None

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
    """Save progress log"""
    global progress_file
    if progress_file is None:
        print("Progress file path not set, cannot save progress")
        return
        
    try:
        # Ensure the directory exists
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "processed_folders": list(processed_folders),
                "current_file": str(current_output_file) if current_output_file else None
            }, f, ensure_ascii=False, indent=2)
        print(f"Progress saved: {len(processed_folders)} folders completed")
    except Exception as e:
        print(f"Error saving progress: {e}")


def load_progress(resume_from=None):
    """Load progress from file"""
    global processed_folders
    
    try:
        # First try to load from specified resume file
        if resume_from and Path(resume_from).exists():
            print(f"Loading progress from specified file: {resume_from}")
            with open(resume_from, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                processed_folders = set(progress.get("processed_folders", []))
                print(f"Loaded {len(processed_folders)} processed folders from specified resume file")
                # Save to current progress file for future use
                save_progress_log()
                return True
        
        # Then try to load from current progress file
        elif progress_file and progress_file.exists():
            print(f"Loading progress from: {progress_file}")
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                processed_folders = set(progress.get("processed_folders", []))
                print(f"Loaded {len(processed_folders)} processed folders from progress file")
                return True
        
        # Finally try to load from default resume file (backward compatibility)
        elif resume_file and resume_file.exists():
            print(f"Loading progress from default resume file: {resume_file}")
            with open(resume_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                processed_folders = set(progress.get("processed_folders", []))
                print(f"Loaded {len(processed_folders)} processed folders from default resume file")
                # Copy to current progress file for future use
                save_progress_log()
                return True
                
    except Exception as e:
        print(f"Error loading progress: {e}")
        processed_folders = set()
        return False
    
    print("No progress file found, starting fresh")
    processed_folders = set()
    return False


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

    # Check if already processed
    if folder_name in processed_folders:
        print(f"✅ Skipping {folder_name} - already processed")
        return "skipped"

    # Check if output file exists (partial completion)
    file_existed = os.path.exists(output_file)
    current_results = {}
    
    if file_existed:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                current_results = json.load(f)
            print(f"📂 Resuming {folder_name} from existing file with {len(current_results)} images")
        except Exception as e:
            print(f"⚠️  Error reading existing file {output_file}: {e}")
            current_results = {}
    else:
        print(f"🆕 Starting fresh: {folder_name}")
    
    # Get image files
    image_files = [f for f in sorted(os.listdir(input_folder)) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"📁 No images found in {folder_name}")
        processed_folders.add(folder_name)
        save_progress_log()
        return "empty"

    total_images = len(image_files)
    already_processed = len([f for f in image_files if f in current_results])
    remaining_images = total_images - already_processed
    
    print(f"📊 Folder {folder_name}: {total_images} total, {already_processed} done, {remaining_images} remaining")
    
    if remaining_images == 0:
        print(f"✅ Folder {folder_name} already completed!")
        processed_folders.add(folder_name)
        save_progress_log()
        return "already_complete"
    
    # Process remaining images
    processed_count = already_processed
    for i, file in enumerate(image_files, 1):
        # Skip if already processed in this session
        if file in current_results:
            continue
            
        img_path = os.path.join(input_folder, file)
        print(f"🔄 Processing [{processed_count + 1}/{total_images}]: {file}")
        
        try:
            # Use the advanced OCR pipeline
            corrected_texts = predict(recognitor, detector, img_path)
            current_results[file] = corrected_texts
            processed_count += 1
            
            # Print recognized text (limit output)
            if corrected_texts:
                print(f"   📝 Found {len(corrected_texts)} text blocks")
                for j, text in enumerate(corrected_texts[:3], 1):  # Show max 3 texts
                    print(f"      {j}. {text[:50]}{'...' if len(text) > 50 else ''}")
                if len(corrected_texts) > 3:
                    print(f"      ... and {len(corrected_texts) - 3} more")
            else:
                print(f"   📄 No text detected")
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            current_results[file] = []  # Store empty result for failed images
            processed_count += 1

        # Save progress every 5 images
        if processed_count % 5 == 0:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(current_results, f, ensure_ascii=False, indent=2)
                print(f"💾 Intermediate save: {processed_count}/{total_images}")
            except Exception as e:
                print(f"⚠️  Error saving intermediate results: {e}")

    # Final save for this folder
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Completed {folder_name}: {output_file}")
        processed_folders.add(folder_name)
        save_progress_log()
        return "success"
    except Exception as e:
        print(f"❌ Error saving final results: {e}")
        return "failed"


def main():
    global processed_folders, progress_file, resume_file
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Root directory containing folders with images')
    parser.add_argument('--output_dir', default='./runs/batch_predict', help='Directory to save JSON files')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use: auto (detect automatically), cpu, or cuda')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--resume_file', type=str, help='Specific resume file to load from')
    parser.add_argument('--progress_dir', type=str, help='Directory to save progress file (default: output_dir)')
    args = parser.parse_args()

    # Set up progress file paths
    if args.progress_dir:
        progress_dir = Path(args.progress_dir)
    else:
        progress_dir = Path(args.output_dir)
    
    progress_file = progress_dir / "processing_progress.json"
    
    # Default resume file for backward compatibility
    resume_file = Path("/kaggle/input/l25-gen/processing_progress.json")
    
    print("="*60)
    print("🚀 Starting Vietnamese OCR Batch Processing")
    print("="*60)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Progress file: {progress_file}")
    if args.resume_file:
        print(f"📄 Resume from: {args.resume_file}")

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(save_current_results)

    # Load progress if resume is requested or if progress file exists
    if args.resume or args.resume_file or progress_file.exists():
        load_progress(args.resume_file)
        initial_processed_count = len(processed_folders)
        print(f"📋 Resume mode: {initial_processed_count} folders already completed")
    else:
        initial_processed_count = 0
        print("🆕 Fresh start mode")

    # Configure VietOCR
    print("\n🔧 Loading VietOCR model...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    
    # Device configuration
    if args.device == 'auto':
        if torch.cuda.is_available():
            config['device'] = 'cuda'
            print("🎮 Auto-detected device: CUDA GPU")
        else:
            config['device'] = 'cpu'
            print("💻 Auto-detected device: CPU")
    else:
        config['device'] = args.device
        print(f"⚙️  Using specified device: {args.device}")

    recognitor = Predictor(config)

    # Configure PaddleOCR for text detection
    print("🔧 Loading PaddleOCR detector...")
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=(config['device'] == 'cuda'))
    
    print("✅ Models loaded successfully!")

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)
    # Prepare progress dir
    progress_dir.mkdir(parents=True, exist_ok=True)

    # Get all folders to process 
    print(f"\n🔍 Scanning input directory: {args.input_dir}")
    all_folders = []
    
    # First check if there are image files directly in input_dir
    direct_images = [f for f in os.listdir(args.input_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if direct_images:
        # Images are directly in input_dir
        folder_name = os.path.basename(args.input_dir)
        all_folders.append((folder_name, args.input_dir))
        print(f"📂 Found {len(direct_images)} images directly in input directory")
    else:
        # Scan subfolders for images
        for folder in sorted(os.listdir(args.input_dir)):
            folder_path = os.path.join(args.input_dir, folder)
            if os.path.isdir(folder_path):
                # Check if this folder has images
                images_in_folder = [f for f in os.listdir(folder_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                if images_in_folder:
                    all_folders.append((folder, folder_path))
                else:
                    # Check one level deeper
                    for subfolder in os.listdir(folder_path):
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
    
    if processed_folders:
        print(f"   Previously processed: {', '.join(sorted(list(processed_folders))[:5])}{'...' if len(processed_folders) > 5 else ''}")
    
    if not remaining_folders:
        print("🎉 ALL FOLDERS HAVE BEEN PROCESSED!")
        return
    
    print(f"\n🎯 Will process {len(remaining_folders)} folders:")
    for i, (name, _) in enumerate(remaining_folders[:10], 1):
        print(f"   {i}. {name}")
    if len(remaining_folders) > 10:
        print(f"   ... and {len(remaining_folders) - 10} more")
    
    # Process remaining folders
    newly_processed = 0
    failed_folders = 0
    empty_folders = 0
    skipped_folders = 0
    
    print(f"\n🚀 STARTING PROCESSING...")
    print("="*60)
    
    for i, (folder_name, folder_path) in enumerate(all_folders, 1):
        if folder_name in processed_folders:
            continue
            
        remaining_count = len([f for f in all_folders[i-1:] if f[0] not in processed_folders])
        print(f"\n📁 [{i}/{total_folders}] Processing: {folder_name}")
        print(f"   Remaining after this: {remaining_count - 1}")
        
        result = process_folder(recognitor, detector, folder_path, args.output_dir)
        
        if result == "success":
            newly_processed += 1
            print(f"✅ SUCCESS: {folder_name}")
        elif result == "already_complete":
            newly_processed += 1
            print(f"✅ ALREADY COMPLETE: {folder_name}")
        elif result == "failed":
            failed_folders += 1
            print(f"❌ FAILED: {folder_name}")
        elif result == "empty":
            empty_folders += 1
            print(f"📁 EMPTY: {folder_name}")
        elif result == "skipped":
            skipped_folders += 1
            print(f"⏭️  SKIPPED: {folder_name}")
            
        current_total_processed = len(processed_folders)
        progress_pct = (current_total_processed / total_folders) * 100
        print(f"📈 Overall progress: {current_total_processed}/{total_folders} ({progress_pct:.1f}%)")

    print(f"\n" + "="*60)
    print(f"🏁 PROCESSING SUMMARY")
    print(f"="*60)
    print(f"📊 Total folders: {total_folders}")
    print(f"📋 Previously processed: {initial_processed_count}")
    print(f"🆕 Newly processed this session: {newly_processed}")
    print(f"📁 Empty folders: {empty_folders}")
    print(f"❌ Failed: {failed_folders}")
    print(f"⏭️  Skipped: {skipped_folders}")
    print(f"✅ Final total processed: {len(processed_folders)}/{total_folders}")
    
    if len(processed_folders) == total_folders:
        print("🎉 ALL FOLDERS COMPLETED SUCCESSFULLY!")
    elif failed_folders == 0:
        print("✅ All remaining folders processed successfully!")
    else:
        print(f"⚠️  {failed_folders} folders failed to process")
        remaining = total_folders - len(processed_folders)
        if remaining > 0:
            print(f"📝 {remaining} folders still need to be processed")
    
    print("="*60)


if __name__ == '__main__':
    main()
