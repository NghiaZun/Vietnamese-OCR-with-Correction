import os
import sys
import argparse
import json
import signal
import atexit
import shutil
from pathlib import Path
from datetime import datetime
from paddleocr import PaddleOCR

# Global variables for checkpoint
current_results = {}
current_output_file = None
processed_folders = []

# Paths for progress log
resume_file = Path("/kaggle/input/l25-gen/processing_progress.json")
progress_file = Path("/kaggle/working/processing_progress.json")


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


def process_folder(ocr, input_folder, output_folder):
    global current_results, current_output_file
    
    folder_name = os.path.basename(input_folder)
    output_file = os.path.join(output_folder, f"{folder_name}.json")
    current_output_file = output_file

    # Skip if already processed
    if folder_name in processed_folders:
        print(f"Skipping {folder_name}, already processed.")
        return True  # Return True to indicate this folder was handled (skipped)

    current_results = {}
    image_files = [f for f in sorted(os.listdir(input_folder)) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {folder_name}")
        processed_folders.append(folder_name)
        save_progress_log()
        return True

    print(f"Processing folder {folder_name} with {len(image_files)} images...")
    
    for i, file in enumerate(image_files, 1):
        img_path = os.path.join(input_folder, file)
        print(f"Processing [{i}/{len(image_files)}]: {file}")
        try:
            result = ocr.ocr(img_path, cls=True)
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    text_lines.append(line[1][0])
            current_results[file] = text_lines
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            current_results[file] = []  # Store empty result for failed images

    # Save results for this folder
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
        processed_folders.append(folder_name)
        save_progress_log()
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


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
    if args.resume:
        if resume_file.exists() and not progress_file.exists():
            shutil.copy(resume_file, progress_file)
            print(f"Copied resume file from {resume_file} to {progress_file}")

        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    processed_folders = progress.get("processed_folders", [])
                print(f"Resuming: {len(processed_folders)} folders already processed")
            except Exception as e:
                print(f"Error loading progress log: {e}")
                processed_folders = []

    # Set device
    use_gpu = False
    if args.device == 'cuda':
        use_gpu = True
    elif args.device == 'auto':
        import paddle
        use_gpu = paddle.device.is_compiled_with_cuda()

    print(f"Using device: {'GPU' if use_gpu else 'CPU'}")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all folders to process
    all_folders = []
    for folder in sorted(os.listdir(args.input_dir)):
        folder_path = os.path.join(args.input_dir, folder)
        if os.path.isdir(folder_path):
            all_folders.append((folder, folder_path))

    total_folders = len(all_folders)
    print(f"Found {total_folders} folders to process")
    
    # Process all folders
    successful_folders = 0
    failed_folders = 0
    
    for i, (folder_name, folder_path) in enumerate(all_folders, 1):
        print(f"\n=== Processing folder {i}/{total_folders}: {folder_name} ===")
        success = process_folder(ocr, folder_path, args.output_dir)
        if success:
            successful_folders += 1
        else:
            failed_folders += 1
            
        print(f"Progress: {successful_folders + failed_folders}/{total_folders} folders completed")

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total folders: {total_folders}")
    print(f"Successfully processed: {successful_folders}")
    print(f"Failed: {failed_folders}")
    print(f"Already processed (skipped): {len([f for f in processed_folders if f in [folder[0] for folder in all_folders]])}")
    
    if failed_folders == 0:
        print("✅ All folders processed successfully!")
    else:
        print(f"⚠️  {failed_folders} folders failed to process")


if __name__ == '__main__':
    main()
