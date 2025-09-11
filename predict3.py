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
processed_files = set()  # Store output file paths that are completed

# Paths for progress log
resume_file = Path("/kaggle/input/processing/processing_progress.json")
progress_file = Path("/kaggle/working/processing_progress.json")

# Initialize corrector globally to avoid reloading
corrector = None


def initialize_corrector():
    """Initialize corrector only when needed"""
    global corrector
    if corrector is None:
        print("🔧 Loading Vietnamese text correction model...")
        corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")
        print("✅ Text correction model loaded")


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
            print(f"💾 Emergency save: {current_output_file}")
        except Exception as e:
            print(f"❌ Error saving emergency results: {e}")


def save_progress_log():
    """Save progress log into /kaggle/working/"""
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "processed_files": list(processed_files),  # Convert set to list
                "current_file": str(current_output_file) if current_output_file else None
            }, f, ensure_ascii=False, indent=2)
        print(f"📋 Progress saved: {len(processed_files)} files completed")
    except Exception as e:
        print(f"❌ Error saving progress: {e}")


def convert_old_format_to_new(old_progress, output_dir):
    """Convert old processed_folders format to new processed_files format"""
    converted_files = set()
    
    if "processed_folders" in old_progress:
        for folder_path in old_progress["processed_folders"]:
            # Extract folder name from full path
            folder_name = os.path.basename(folder_path)
            # Create corresponding output file path
            output_file = os.path.join(output_dir, f"{folder_name}.json")
            
            # Only add if the output file actually exists
            if os.path.exists(output_file):
                converted_files.add(output_file)
                
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
                    for file_path in processed_files:
                        if os.path.exists(file_path):
                            existing_files.add(file_path)
                        else:
                            print(f"⚠️  Previously processed file no longer exists: {file_path}")
                    processed_files = existing_files
                    
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
                    for file_path in processed_files:
                        if os.path.exists(file_path):
                            existing_files.add(file_path)
                        else:
                            print(f"⚠️  Previously processed file no longer exists: {file_path}")
                    processed_files = existing_files
                    
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


def predict(recognitor, detector, img_path, padding=4):
    """VietOCR + PaddleOCR prediction with correction"""
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Could not load image: {img_path}")
            return []
        
        # Text detection using PaddleOCR
        result = detector.ocr(img_path, cls=False, det=True, rec=False)
        if not result or not result[0]:
            print(f"📄 No text detected in: {os.path.basename(img_path)}")
            return []
            
        result = result[0]
        
        # Filter Boxes
        boxes = []
        for line in result:
            if len(line) >= 1 and len(line[0]) >= 4:  # Validate box format
                try:
                    boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[0][2]), int(line[0][3])]])
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Skipping invalid box: {e}")
                    continue
        
        if not boxes:
            print(f"📄 No valid text boxes found in: {os.path.basename(img_path)}")
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
        for i, box in enumerate(boxes):
            try:
                # Validate box coordinates
                if (box[1][1] <= box[0][1]) or (box[1][0] <= box[0][0]):
                    print(f"⚠️  Skipping invalid box {i}: {box}")
                    continue
                    
                # Extract cropped region
                cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                
                # Check if cropped image has valid dimensions
                if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                    print(f"⚠️  Skipping box {i} with invalid dimensions: {cropped_image.shape}")
                    continue
                
                # Convert to PIL Image
                cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                
                # Check PIL image dimensions
                if cropped_image.size[0] <= 0 or cropped_image.size[1] <= 0:
                    print(f"⚠️  Skipping box {i} with invalid PIL dimensions: {cropped_image.size}")
                    continue

                rec_result = recognitor.predict(cropped_image)
                if rec_result and rec_result.strip():  # Only add non-empty results
                    texts.append(rec_result.strip())
                
            except Exception as e:
                print(f"⚠️  Error processing box {i}: {e}")
                continue

        # Apply Vietnamese text correction
        if texts and corrector:
            try:
                corrections = corrector(texts, max_new_tokens=256)
                corrected_texts = [pred['generated_text'].strip() for pred in corrections if pred.get('generated_text')]
                # Filter out empty results
                corrected_texts = [text for text in corrected_texts if text]
                return corrected_texts
            except Exception as e:
                print(f"⚠️  Error in text correction: {e}")
                return texts
        
        return texts
        
    except Exception as e:
        print(f"❌ Error in predict function: {e}")
        return []


def process_folder(recognitor, detector, input_folder, output_folder):
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
    try:
        all_files = os.listdir(input_folder)
        image_files = [f for f in sorted(all_files) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
    except Exception as e:
        print(f"❌ Error reading folder {input_folder}: {e}")
        return "failed"
    
    if not image_files:
        print(f"📁 No images found in {folder_name}")
        # Mark as processed even if empty
        processed_files.add(output_file)
        save_progress_log()
        # Create empty JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Error creating empty file: {e}")
        return "empty"

    total_images = len(image_files)
    already_processed = len([f for f in image_files if f in current_results])
    remaining_images = total_images - already_processed
    
    print(f"📊 Folder {folder_name}: {total_images} total, {already_processed} done, {remaining_images} remaining")
    
    if remaining_images == 0:
        print(f"✅ Folder {folder_name} already completed!")
        processed_files.add(output_file)
        save_progress_log()
        return "already_complete"
    
    # Process remaining images
    processed_count = already_processed
    for i, file in enumerate(image_files, 1):
        # Skip if already processed
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
                for j, text in enumerate(corrected_texts[:2], 1):  # Show max 2 texts
                    print(f"      {j}. {text[:60]}{'...' if len(text) > 60 else ''}")
                if len(corrected_texts) > 2:
                    print(f"      ... and {len(corrected_texts) - 2} more")
            else:
                print(f"   📄 No text detected")
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            current_results[file] = []  # Store empty result for failed images
            processed_count += 1

        # Save progress every 5 images (more frequent saves)
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
        print(f"✅ Completed {folder_name}: saved to {output_file}")
        # Mark as fully processed
        processed_files.add(output_file)
        save_progress_log()
        return "success"
    except Exception as e:
        print(f"❌ Error saving final results: {e}")
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

    # Prepare output dir first (needed for progress loading)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load progress automatically if file exists, or if --resume is specified
    if args.resume or progress_file.exists() or resume_file.exists():
        load_progress(args.output_dir)
        initial_processed_count = len(processed_files)
        print(f"📋 Resume mode: {initial_processed_count} files already completed")
    else:
        initial_processed_count = 0
        print("🆕 Fresh start mode")

    # Initialize text correction if not disabled
    if not args.no_correction:
        initialize_corrector()

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

    # Get all folders to process 
    print(f"\n🔍 Scanning input directory: {args.input_dir}")
    all_folders = []
    
    # First check if there are image files directly in input_dir
    try:
        direct_images = [f for f in os.listdir(args.input_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        
        if direct_images:
            # Images are directly in input_dir
            folder_name = os.path.basename(args.input_dir)
            output_file = os.path.join(args.output_dir, f"{folder_name}.json")
            all_folders.append((folder_name, args.input_dir, output_file))
            print(f"📂 Found {len(direct_images)} images directly in input directory")
        else:
            # Scan subfolders for images
            for folder in sorted(os.listdir(args.input_dir)):
                folder_path = os.path.join(args.input_dir, folder)
                if os.path.isdir(folder_path):
                    try:
                        # Check if this folder has images
                        images_in_folder = [f for f in os.listdir(folder_path) 
                                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
                        if images_in_folder:
                            output_file = os.path.join(args.output_dir, f"{folder}.json")
                            all_folders.append((folder, folder_path, output_file))
                        else:
                            # Check one level deeper
                            try:
                                for subfolder in os.listdir(folder_path):
                                    subfolder_path = os.path.join(folder_path, subfolder)
                                    if os.path.isdir(subfolder_path):
                                        images_in_subfolder = [f for f in os.listdir(subfolder_path) 
                                                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
                                        if images_in_subfolder:
                                            output_file = os.path.join(args.output_dir, f"{subfolder}.json")
                                            all_folders.append((subfolder, subfolder_path, output_file))
                            except PermissionError:
                                print(f"⚠️  Permission denied accessing: {folder_path}")
                    except PermissionError:
                        print(f"⚠️  Permission denied accessing: {folder_path}")
                        
    except Exception as e:
        print(f"❌ Error scanning input directory: {e}")
        sys.exit(1)

    total_folders = len(all_folders)
    remaining_folders = [f for f in all_folders if f[2] not in processed_files]
    
    print(f"\n📊 FOLDER SUMMARY:")
    print(f"   Total folders found: {total_folders}")
    print(f"   Already processed: {len(processed_files)}")
    print(f"   Remaining to process: {len(remaining_folders)}")
    
    if processed_files:
        processed_names = [os.path.basename(f).replace('.json', '') for f in processed_files]
        print(f"   Previously processed: {', '.join(sorted(processed_names)[:5])}{'...' if len(processed_names) > 5 else ''}")
    
    if not remaining_folders:
        print("🎉 ALL FOLDERS HAVE BEEN PROCESSED!")
        return
    
    print(f"\n🎯 Will process {len(remaining_folders)} folders:")
    for i, (name, _, _) in enumerate(remaining_folders[:10], 1):
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
    
    for i, (folder_name, folder_path, output_file) in enumerate(all_folders, 1):
        if output_file in processed_files:
            continue
            
        remaining_count = len([f for f in all_folders[i-1:] if f[2] not in processed_files])
        print(f"\n📁 [{i}/{total_folders}] Processing: {folder_name}")
        print(f"   Output: {output_file}")
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
            
        current_total_processed = len(processed_files)
        progress_pct = (current_total_processed / total_folders) * 100 if total_folders > 0 else 0
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
    print(f"✅ Final total processed: {len(processed_files)}/{total_folders}")
    
    if len(processed_files) == total_folders:
        print("🎉 ALL FOLDERS COMPLETED SUCCESSFULLY!")
    elif failed_folders == 0:
        print("✅ All remaining folders processed successfully!")
    else:
        print(f"⚠️  {failed_folders} folders failed to process")
        remaining = total_folders - len(processed_files)
        if remaining > 0:
            print(f"📝 {remaining} folders still need to be processed")
    
    print("="*60)


if __name__ == '__main__':
    main()
