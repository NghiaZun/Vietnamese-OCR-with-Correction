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
    
    # Add padding to boxes (with bounds checking)
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
            
            # Convert to PIL Image (BGR to RGB for proper display)
            cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            
            # Check PIL image dimensions
            if cropped_image.size[0] <= 0 or cropped_image.size[1] <= 0:
                print(f"Warning: Skipping box {i} with invalid PIL dimensions: {cropped_image.size}")
                continue

            rec_result = recognitor.predict(cropped_image)
            texts.append(rec_result)
            print(rec_result)
            
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


def find_image_folders(root_dir):
    """Find all folders containing images using recursive search"""
    root_path = Path(root_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_folders = []
    
    for folder_path in root_path.rglob('*'):
        if folder_path.is_dir():
            has_images = any(
                f.suffix.lower() in image_extensions 
                for f in folder_path.iterdir() 
                if f.is_file()
            )
            
            if has_images:
                image_folders.append(folder_path)
    
    return sorted(image_folders)


def filter_folders_by_range(folders, filter_range=None, include_pattern=None):
    """Filter folders by range or pattern"""
    if not filter_range and not include_pattern:
        return folders
    
    import re
    filtered_folders = []
    
    if filter_range:
        # Parse range like "L21-L24" or single "L25"
        if '-' in filter_range:
            start_str, end_str = filter_range.split('-')
            # Extract number from L21 format
            start_match = re.search(r'L(\d+)', start_str)
            end_match = re.search(r'L(\d+)', end_str)
            
            if start_match and end_match:
                start_num = int(start_match.group(1))
                end_num = int(end_match.group(1))
                
                for folder in folders:
                    folder_match = re.search(r'L(\d+)', folder.name)
                    if folder_match:
                        folder_num = int(folder_match.group(1))
                        if start_num <= folder_num <= end_num:
                            filtered_folders.append(folder)
        else:
            # Single value like "L25"
            target_match = re.search(r'L(\d+)', filter_range)
            if target_match:
                target_num = int(target_match.group(1))
                for folder in folders:
                    folder_match = re.search(r'L(\d+)', folder.name)
                    if folder_match and int(folder_match.group(1)) == target_num:
                        filtered_folders.append(folder)
    
    if include_pattern:
        # Parse pattern like "L21,L22,L23"
        patterns = [p.strip() for p in include_pattern.split(',')]
        for folder in folders:
            for pattern in patterns:
                if pattern in folder.name:
                    if folder not in filtered_folders:  # Avoid duplicates
                        filtered_folders.append(folder)
                    break
    
    return filtered_folders
    """Find all folders containing images using recursive search"""
    root_path = Path(root_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_folders = []
    
    for folder_path in root_path.rglob('*'):
        if folder_path.is_dir():
            has_images = any(
                f.suffix.lower() in image_extensions 
                for f in folder_path.iterdir() 
                if f.is_file()
            )
            
            if has_images:
                image_folders.append(folder_path)
    
    return sorted(image_folders)


def process_folder(recognitor, detector, folder_path, output_dir, input_root):
    """Process all images in a folder and save to JSON"""
    global current_results, current_output_file
    
    folder_path = Path(folder_path)
    folder_name = folder_path.name
    
    # Skip if already processed (using full path for comparison)
    if str(folder_path) in processed_folders:
        print(f"Skipping {folder_name}, already processed.")
        return "skipped"

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(folder_path.glob(ext))
    
    if not image_files:
        print(f"No image files found in {folder_name}")
        processed_folders.append(str(folder_path))
        save_progress_log()
        return "empty"

    image_files.sort()
    print(f"Processing folder {folder_name} with {len(image_files)} images...")
    
    # Create output path structure
    input_root = Path(input_root)
    relative_path = folder_path.relative_to(input_root)
    path_parts = list(relative_path.parts)
    
    # Handle duplicate folder names in path (like L13/L13/L13_V001)
    # Remove consecutive duplicate parts
    cleaned_parts = []
    for part in path_parts:
        if not cleaned_parts or part != cleaned_parts[-1]:
            cleaned_parts.append(part)
    
    # Create output directory structure
    if len(cleaned_parts) > 1:
        # Create nested structure: L13/L13_V001.json
        output_subdir = Path(output_dir) / cleaned_parts[0]
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"{cleaned_parts[-1]}.json"
    else:
        # Single level: L13_V001.json
        output_file = Path(output_dir) / f"{cleaned_parts[-1]}.json"
    
    # Set current file for emergency saving
    current_output_file = output_file
    current_results = {}
    
    # Check if file already exists (resume from checkpoint)
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                current_results = json.load(f)
            print(f"Resuming from existing file: {output_file}")
        except:
            current_results = {}
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        # Skip if already processed in this session
        if img_path.name in current_results:
            print(f"Skipping already processed: {img_path.name}")
            continue
            
        print(f"Processing [{i}/{len(image_files)}]: {img_path.name}")
        
        try:
            # Use the advanced OCR pipeline
            corrected_texts = predict(recognitor, detector, str(img_path))
            current_results[img_path.name] = corrected_texts
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            current_results[img_path.name] = []  # Store empty result for failed images

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
        processed_folders.append(str(folder_path))
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
    parser.add_argument('--filter_range', type=str, help='Filter folders by range, e.g., "L21-L24" or "L25"')
    parser.add_argument('--include_pattern', type=str, help='Include folders matching pattern, e.g., "L21,L22,L23"')
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

    # Find all folders containing images using recursive search
    print(f"Scanning for image folders in: {args.input_dir}")
    all_image_folders = find_image_folders(args.input_dir)
    
    if not all_image_folders:
        print(f"No image folders found in: {args.input_dir}")
        return
    
    # Apply filtering if specified
    if args.filter_range or args.include_pattern:
        image_folders = filter_folders_by_range(all_image_folders, args.filter_range, args.include_pattern)
        print(f"Found {len(all_image_folders)} total folders, filtered to {len(image_folders)} folders")
        
        if args.filter_range:
            print(f"Filter range: {args.filter_range}")
        if args.include_pattern:
            print(f"Include pattern: {args.include_pattern}")
            
        # Show which folders will be processed
        print("Folders to process:")
        for folder in image_folders:
            print(f"  - {folder.name} ({folder})")
    else:
        image_folders = all_image_folders
        print(f"Found {len(image_folders)} folders with images")
    
    total_folders = len(image_folders)
    
    # Filter out already processed folders if resuming
    if args.resume:
        remaining_folders = [f for f in image_folders if str(f) not in processed_folders]
        print(f"Found {total_folders} total folders with images")
        print(f"Already processed: {len(processed_folders)} folders")
        print(f"Remaining to process: {len(remaining_folders)} folders")
    else:
        remaining_folders = image_folders
        print(f"Found {total_folders} folders with images")
    
    if not remaining_folders:
        print("🎉 All folders have already been processed!")
        return
    
    # Process remaining folders
    newly_processed = 0
    failed_folders = 0
    empty_folders = 0
    
    for i, folder_path in enumerate(remaining_folders, 1):
        print(f"\n=== Processing folder {i}/{len(remaining_folders)}: {folder_path.name} ===")
        print(f"Full path: {folder_path}")
        
        result = process_folder(recognitor, detector, folder_path, args.output_dir, args.input_dir)
        
        if result == "success":
            newly_processed += 1
            print(f"✅ Successfully processed {folder_path.name}")
        elif result == "failed":
            failed_folders += 1
            print(f"❌ Failed to process {folder_path.name}")
        elif result == "empty":
            empty_folders += 1
            print(f"📁 Empty folder: {folder_path.name}")
        elif result == "skipped":
            print(f"⏭️  Unexpectedly skipped: {folder_path.name}")
            
        current_total_processed = len(processed_folders)
        print(f"Progress: {current_total_processed}/{total_folders} total folders completed")

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total folders found: {total_folders}")
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
