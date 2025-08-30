import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
from PIL import Image
import difflib
import re
import math
import json
import sys
import argparse
from pathlib import Path
import glob
import signal
import atexit
from datetime import datetime

import torch

import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg

from PaddleOCR import PaddleOCR, draw_ocr

# Specifying output path and font path.
FONT = './PaddleOCR/doc/fonts/latin.ttf'

from transformers import pipeline

corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")

MAX_LENGTH = 256

# Global variables for checkpoint
current_results = {}
current_output_file = None
processed_folders = []

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
    try:
        with open("processing_progress.json", 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "processed_folders": processed_folders,
                "current_file": str(current_output_file) if current_output_file else None
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")

def signal_handler(signum, frame):
    """Handle interruption signals"""
    print(f"\nReceived signal {signum}. Saving current progress...")
    save_current_results()
    save_progress_log()
    print("Progress saved. Exiting...")
    sys.exit(0)


def predict(recognitor, detector, img_path, padding=4):
    # Load image
    img = cv2.imread(img_path)

    # Text detection
    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    result = result[:][:][0]

    # Filter Boxes
    boxes = []
    for line in result:
        boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
    boxes = boxes[::-1]

    # Add padding to boxes
    for box in boxes:
        box[0][0] = box[0][0] - padding
        box[0][1] = box[0][1] - padding
        box[1][0] = box[1][0] + padding
        box[1][1] = box[1][1] + padding

    # Text recognizion
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
            cropped_image = Image.fromarray(cropped_image)
            
            # Check PIL image dimensions
            if cropped_image.size[0] <= 0 or cropped_image.size[1] <= 0:
                print(f"Warning: Skipping box {i} with invalid PIL dimensions: {cropped_image.size}")
                continue

            rec_result = recognitor.predict(cropped_image)
            text = rec_result

            texts.append(text)
            print(text)
            
        except Exception as e:
            print(f"Warning: Error processing box {i}: {e}")
            continue

    return boxes, texts


def process_folder(recognitor, detector, folder_path, output_dir):
    """Process all images in a folder and save to JSON"""
    global current_results, current_output_file
    
    folder_path = Path(folder_path)
    folder_name = folder_path.name
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(folder_path.glob(ext))
    
    if not image_files:
        print(f"No images found in: {folder_path}")
        return
    
    image_files.sort()
    print(f"Processing {len(image_files)} images in folder: {folder_name}")
    
    # Set up output file
    output_file = Path(output_dir) / f"{folder_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    current_output_file = output_file
    
    # Check if file already exists (resume from checkpoint)
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                current_results = json.load(f)
            print(f"Resuming from existing file: {output_file}")
        except:
            current_results = {}
    else:
        current_results = {}
    
    # Process each image
    for img_path in image_files:
        # Skip if already processed
        if img_path.name in current_results:
            print(f"Skipping already processed: {img_path.name}")
            continue
            
        try:
            print(f"Processing: {img_path.name}")
            boxes, texts = predict(recognitor, detector, str(img_path))
            
            # Apply correction
            if texts:
                corrections = corrector(texts, max_new_tokens=256)
                corrected_texts = [pred['generated_text'] for pred in corrections]
                current_results[img_path.name] = corrected_texts
            else:
                current_results[img_path.name] = []
            
            # Save immediately after each image
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            current_results[img_path.name] = []
            
            # Still save the error result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
    
    print(f"Completed folder: {output_file}")
    processed_folders.append(str(folder_path))
    save_progress_log()


def find_image_folders(root_dir):
    """Find all folders containing images"""
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

    # Load previous progress if resuming
    if args.resume and os.path.exists("processing_progress.json"):
        try:
            with open("processing_progress.json", 'r') as f:
                progress = json.load(f)
                processed_folders = progress.get("processed_folders", [])
            print(f"Resuming: {len(processed_folders)} folders already processed")
        except:
            processed_folders = []

    # Configure of VietOCR
    config = Cfg.load_config_from_name('vgg_transformer')
    # Custom weight
    # config = Cfg.load_config_from_file('vi00_vi01_transformer.yml')
    # config['weights'] = './pretrain_ocr/vi00_vi01_transformer.pth'

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

    # Config of PaddleOCR
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=(config['device'] == 'cuda'))
    
    # Find all folders containing images
    print(f"Scanning for image folders in: {args.input_dir}")
    image_folders = find_image_folders(args.input_dir)
    
    if not image_folders:
        print(f"No image folders found in: {args.input_dir}")
        return
    
    print(f"Found {len(image_folders)} folders with images")
    
    # Filter out already processed folders if resuming
    if args.resume:
        image_folders = [f for f in image_folders if str(f) not in processed_folders]
        print(f"Remaining folders to process: {len(image_folders)}")
    
    # Process each folder
    for i, folder_path in enumerate(image_folders, 1):
        print(f"\n=== Processing folder {i}/{len(image_folders)}: {folder_path} ===")
        try:
            process_folder(recognitor, detector, folder_path, args.output_dir)
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")
            save_current_results()  # Save what we have so far
            continue
    
    print(f"\nBatch processing completed! Results saved in: {args.output_dir}")


if __name__ == "__main__":    
    main()
