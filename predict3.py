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

# Global variables for checkpoint and models
current_results = {}
current_output_file = None
processed_files = set()  # Store output file paths that are completed

# Global models - initialize once, use everywhere
corrector = None
recognitor = None
detector = None

# Paths for progress log
resume_file = Path("/kaggle/input/processing/processing_progress.json")
progress_file = Path("/kaggle/working/processing_progress.json")


def initialize_models(device, no_correction=False):
    """Initialize all models once at startup"""
    global corrector, recognitor, detector
    
    # Initialize text correction if not disabled
    if not no_correction and corrector is None:
        print("🔧 Loading Vietnamese text correction model...")
        try:
            corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")
            print("✅ Text correction model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load correction model: {e}")
            corrector = None

    # Configure VietOCR
    if recognitor is None:
        print("🔧 Loading VietOCR model...")
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = True
        config['predictor']['beamsearch'] = True
        config['device'] = device
        recognitor = Predictor(config)
        print(f"✅ VietOCR loaded on {device}")

    # Configure PaddleOCR for text detection
    if detector is None:
        print("🔧 Loading PaddleOCR detector...")
        detector = PaddleOCR(
            use_angle_cls=False, 
            lang="vi", 
            use_gpu=(device == 'cuda'),
            show_log=False  # Reduce verbose logging
        )
        print("✅ PaddleOCR detector loaded")


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}. Saving progress before exit...")
    save_current_results()
    save_progress_log()
    sys.exit(0)


def save_current_results():
    """Save current results immediately"""
    if current_results and current_output_file:
        try:
            with open(current_output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
            print(f"💾 Emergency save: {current_output_file}")
        except Exception as e:
            print(f"❌ Error saving emergency results: {e}")


def save_progress_log():
    """Save progress log into /kaggle/working/"""
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "processed_files": list(processed_files),
                "current_file": str(current_output_file) if current_output_file else None
            }, f, ensure_ascii=False, indent=2)
        print(f"📋 Progress saved: {len(processed_files)} files completed")
    except Exception as e:
        print(f"❌ Error saving progress: {e}")


def convert_old_format_to_new(old_progress, output_dir):
    """Convert old processed_folders format to new processed_files format"""
    converted_files = set()
    
    if "processed_folders" in old_progress:
        print(f"🔄 Converting {len(old_progress['processed_folders'])} old entries...")
        for folder_path in old_progress["processed_folders"]:
            # Extract folder name from full path
            folder_name = os.path.basename(folder_path)
            # Create corresponding output file path
            output_file = os.path.join(output_dir, f"{folder_name}.json")
            
            # Only add if the output file actually exists
            if os.path.exists(output_file):
                converted_files.add(output_file)
                print(f"   ✅ Found: {folder_name}.json")
            else:
                print(f"   ❌ Missing: {folder_name}.json")
                
    return converted_files


def load_progress(output_dir):
    """Load progress from file"""
    global processed_files
    
    try:
        # First try to load from working directory
        if progress_file.exists():
            print(f"📂 Loading progress from: {progress_file}")
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                
                if "processed_files" in progress:
                    # New format - direct use
                    processed_files = set(progress["processed_files"])
                    # Verify files still exist
                    existing_files = set()
                    missing_count = 0
                    for file_path in processed_files:
                        if os.path.exists(file_path):
                            existing_files.add(file_path)
                        else:
                            missing_count += 1
                    processed_files = existing_files
                    if missing_count > 0:
                        print(f"⚠️  {missing_count} previously processed files no longer exist")
                    
                elif "processed_folders" in progress:
                    # Old format - convert
                    print(f"⚠️  Converting old format from working directory")
                    processed_files = convert_old_format_to_new(progress, output_dir)
                else:
                    processed_files = set()
                    
                print(f"✅ Loaded {len(processed_files)} processed files from progress")
                return True
        
        # If not found, try resume file
        elif resume_file.exists():
            print(f"📂 Loading progress from resume file: {resume_file}")
            with open(resume_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                
                if "processed_files" in progress:
                    # New format
                    processed_files = set(progress["processed_files"])
                    # Verify files still exist
                    existing_files = set()
                    missing_count = 0
                    for file_path in processed_files:
                        if os.path.exists(file_path):
                            existing_files.add(file_path)
                        else:
                            missing_count += 1
                    processed_files = existing_files
                    if missing_count > 0:
                        print(f"⚠️  {missing_count} previously processed files no longer exist")
                    
                elif "processed_folders" in progress:
                    # Old format - convert
                    print(f"⚠️  Converting old format from resume file")
                    processed_files = convert_old_format_to_new(progress, output_dir)
                else:
                    processed_files = set()
                    
                print(f"✅ Loaded {len(processed_files)} processed files from resume file")
                # Copy to working directory for future use
                save_progress_log()
                return True
                
    except Exception as e:
        print(f"❌ Error loading progress: {e}")
        processed_files = set()
        return False
    
    print("📄 No progress file found, starting fresh")
    processed_files = set()
    return False


def predict(img_path, padding=4):
    """VietOCR + PaddleOCR prediction with correction using global models"""
    global recognitor, detector, corrector
    
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        # Text detection using PaddleOCR
        result = detector.ocr(img_path, cls=False, det=False, rec=False)
        if not result or not result[0]:
            return []
            
        result = result[0]
        
        # Filter Boxes
        boxes = []
        for line in result:
            if len(line) >= 1 and len(line[0]) >= 4:
                try:
                    boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[0][2]), int(line[0][3])]])
                except (ValueError, IndexError):
                    continue
        
        if not boxes:
            return []
            
        boxes = boxes[::-1]
        
        # Add padding to boxes
        img_height, img_width = img.shape[:2]
        for box in boxes:
            box[0][0] = max(0, box[0][0] - padding)
            box[0][1] = max(0, box[0][1] - padding)
            box[1][0] = min(img_width, box[1][0] + padding)
            box[1][1] = min(img_height, box[1][1] + padding)
        
        # Text recognition using VietOCR
        texts = []
        for box in boxes:
            try:
                # Validate box coordinates
                if (box[1][1] <= box[0][1]) or (box[1][0] <= box[0][0]):
                    continue
                    
                # Extract cropped region
                cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                
                # Check if cropped image has valid dimensions
                if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                    continue
                
                # Convert to PIL Image
                cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                
                # Check PIL image dimensions
                if cropped_image.size[0] <= 0 or cropped_image.size[1] <= 0:
                    continue

                rec_result = recognitor.predict(cropped_image)
                if rec_result and rec_result.strip():
                    texts.append(rec_result.strip())
                
            except Exception:
                continue

        # Apply Vietnamese text correction
        if texts and corrector:
            try:
                corrections = corrector(texts, max_new_tokens=256)
                corrected_texts = [pred['generated_text'].strip() for pred in corrections if pred.get('generated_text')]
                corrected_texts = [text for text in corrected_texts if text]
                return corrected_texts
            except Exception:
                return texts
        
        return texts
        
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return []


def process_folder(input_folder, output_folder):
    global current_results, current_output_file
    
    folder_name = os.path.basename(input_folder)
    output_file = os.path.join(output_folder, f"{folder_name}.json")
    current_output_file = output_file

    # Check if already processed
    if output_file in processed_files:
        print(f"✅ Skipping {folder_name} - already processed")
        return "skipped"

    # Validate input folder
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        return "failed"

    # Check if output file exists (partial completion)
    current_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                current_results = json.load(f)
            print(f"📂 Resuming {folder_name} with {len(current_results)} processed images")
        except Exception as e:
            print(f"⚠️  Error reading {output_file}: {e}")
            current_results = {}
    
    # Get image files
    try:
        all_files = os.listdir(input_folder)
        image_files = [f for f in sorted(all_files) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
    except Exception as e:
        print(f"❌ Error reading folder {input_folder}: {e}")
        return "failed"
    
    if not image_files:
        print(f"📁 Empty folder: {folder_name}")
        # Create empty JSON and mark as processed
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            processed_files.add(output_file)
            save_progress_log()
        except Exception as e:
            print(f"⚠️  Error creating empty file: {e}")
        return "empty"

    total_images = len(image_files)
    already_processed = len([f for f in image_files if f in current_results])
    remaining_images = total_images - already_processed
    
    if remaining_images == 0:
        print(f"✅ {folder_name} already complete ({total_images} images)")
        processed_files.add(output_file)
        save_progress_log()
        return "already_complete"
    
    print(f"📊 {folder_name}: {remaining_images}/{total_images} images to process")
    
    # Process remaining images
    processed_count = already_processed
    for file in image_files:
        if file in current_results:
            continue
            
        img_path = os.path.join(input_folder, file)
        processed_count += 1
        
        print(f"🔄 [{processed_count}/{total_images}] {file}", end=" ")
        
        try:
            corrected_texts = predict(img_path)
            current_results[file] = corrected_texts
            
            if corrected_texts:
                print(f"✅ {len(corrected_texts)} texts")
            else:
                print("📄 no text")
            
        except Exception as e:
            print(f"❌ error: {e}")
            current_results[file] = []

        # Save progress every 5 images
        if processed_count % 5 == 0:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(current_results, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved progress: {processed_count}/{total_images}")
            except Exception as e:
                print(f"⚠️  Save error: {e}")

    # Final save
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Completed {folder_name}")
        processed_files.add(output_file)
        save_progress_log()
        return "success"
    except Exception as e:
        print(f"❌ Final save error: {e}")
        return "failed"


def main():
    global processed_files
    
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
    parser.add_argument('--no-correction', action='store_true', help='Skip text correction step')
    args = parser.parse_args()

    print("="*60)
    print("🚀 Vietnamese OCR Batch Processing")
    print("="*60)

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🎮 Auto-detected device: {device.upper()}")
    else:
        device = args.device
        print(f"⚙️  Using device: {device.upper()}")

    # Load progress
    if args.resume or progress_file.exists() or resume_file.exists():
        load_progress(args.output_dir)
        initial_processed_count = len(processed_files)
        print(f"📋 Resume mode: {initial_processed_count} files completed")
    else:
        initial_processed_count = 0
        print("🆕 Starting fresh")

    # Initialize all models once
    print(f"\n🔧 Initializing models on {device}...")
    initialize_models(device, args.no_correction)

    # Scan for folders
    print(f"\n🔍 Scanning: {args.input_dir}")
    all_folders = []
    
    try:
        # Check for direct images
        direct_images = [f for f in os.listdir(args.input_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        
        if direct_images:
            folder_name = os.path.basename(args.input_dir)
            output_file = os.path.join(args.output_dir, f"{folder_name}.json")
            all_folders.append((folder_name, args.input_dir, output_file))
        else:
            # Scan subfolders
            for folder in sorted(os.listdir(args.input_dir)):
                folder_path = os.path.join(args.input_dir, folder)
                if os.path.isdir(folder_path):
                    try:
                        images_in_folder = [f for f in os.listdir(folder_path) 
                                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
                        if images_in_folder:
                            output_file = os.path.join(args.output_dir, f"{folder}.json")
                            all_folders.append((folder, folder_path, output_file))
                        else:
                            # Check deeper level
                            for subfolder in os.listdir(folder_path):
                                subfolder_path = os.path.join(folder_path, subfolder)
                                if os.path.isdir(subfolder_path):
                                    try:
                                        images_in_subfolder = [f for f in os.listdir(subfolder_path) 
                                                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
                                        if images_in_subfolder:
                                            output_file = os.path.join(args.output_dir, f"{subfolder}.json")
                                            all_folders.append((subfolder, subfolder_path, output_file))
                                    except (PermissionError, FileNotFoundError):
                                        continue
                    except (PermissionError, FileNotFoundError):
                        continue
                        
    except Exception as e:
        print(f"❌ Error scanning directory: {e}")
        sys.exit(1)

    total_folders = len(all_folders)
    remaining_folders = [f for f in all_folders if f[2] not in processed_files]
    
    print(f"📊 Found {total_folders} folders, {len(remaining_folders)} remaining")
    
    if not remaining_folders:
        print("🎉 All folders already processed!")
        return
    
    # Process folders
    print(f"\n🚀 Processing {len(remaining_folders)} folders...")
    print("="*60)
    
    stats = {"success": 0, "failed": 0, "empty": 0, "skipped": 0}
    
    for i, (folder_name, folder_path, output_file) in enumerate(remaining_folders, 1):
        print(f"\n📁 [{i}/{len(remaining_folders)}] {folder_name}")
        
        result = process_folder(folder_path, args.output_dir)
        stats[result] += 1
        
        progress_pct = (len(processed_files) / total_folders) * 100
        print(f"📈 Progress: {len(processed_files)}/{total_folders} ({progress_pct:.1f}%)")

    # Final summary
    print(f"\n" + "="*60)
    print(f"🏁 FINAL SUMMARY")
    print(f"="*60)
    print(f"📊 Total folders: {total_folders}")
    print(f"✅ Success: {stats['success']}")
    print(f"📁 Empty: {stats['empty']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print(f"🎯 Final completion: {len(processed_files)}/{total_folders} ({len(processed_files)/total_folders*100:.1f}%)")
    
    if len(processed_files) == total_folders:
        print("🎉 ALL PROCESSING COMPLETE!")
    
    print("="*60)


if __name__ == '__main__':
    main()
