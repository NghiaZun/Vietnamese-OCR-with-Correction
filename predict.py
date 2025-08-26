# predict.py
import os
import sys
import argparse
import warnings
import cv2
from PIL import Image
import torch

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
# CHỈNH: dùng paddleocr (chữ thường)
from paddleocr import PaddleOCR, draw_ocr
from transformers import pipeline

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===== cấu hình phụ =====
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_NEW_TOKENS = 256


def is_image_file(path: str) -> bool:
    return os.path.isfile(path) and (os.path.splitext(path)[1].lower() in VALID_EXTS)


def list_images_recursively(root: str):
    """Quét đệ quy mọi cấp thư mục, trả về list đường dẫn ảnh."""
    if os.path.isfile(root) and is_image_file(root):
        return [root]
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in VALID_EXTS:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def build_vietocr_predictor(device: str) -> Predictor:
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    config['device'] = device
    return Predictor(config)


def build_paddle(device: str) -> PaddleOCR:
    """PaddleOCR dùng GPU khi device='cuda'."""
    return PaddleOCR(use_angle_cls=False, lang="vi",
                     use_gpu=(device == "cuda"), det=True, rec=False)


def predict(recognitor: Predictor, detector: PaddleOCR, img_path: str, padding=4):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Không đọc được ảnh: {img_path}")
        return [], []

    # Text detection (ổn định cấu trúc trả về)
    det_out = detector.ocr(img_path, cls=False, det=True, rec=False)
    lines = det_out[0] if det_out and len(det_out) > 0 else []

    # Lấy bbox (x1,y1,x2,y2) từ polygon
    H, W = img.shape[:2]
    boxes = []
    for item in lines:
        try:
            poly = item[0]  # 4 điểm
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            x1 = max(0, int(min(xs)) - padding)
            y1 = max(0, int(min(ys)) - padding)
            x2 = min(W - 1, int(max(xs)) + padding)
            y2 = min(H - 1, int(max(ys)) + padding)
            if x2 > x1 and y2 > y1:
                boxes.append([[x1, y1], [x2, y2]])
        except Exception:
            continue

    # Đảo thứ tự nếu muốn (giống code cũ đọc từ dưới lên)
    boxes = boxes[::-1]

    # Text recognition
    texts = []
    for i, box in enumerate(boxes):
        try:
            (x1, y1), (x2, y2) = box
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pil_img = Image.fromarray(crop)
            text = recognitor.predict(pil_img)
            texts.append(text)
        except Exception as e:
            print(f"[WARN] Lỗi box {i} ({os.path.basename(img_path)}): {e}")
            continue

    return boxes, texts


def save_txt_mirrored(root_img: str, output_root: str, image_path: str, texts, corrections):
    """
    Lưu theo cấu trúc cây:
      output_root/<relative_dir_of_image>/<image_stem>.txt
    Ví dụ:
      --img /kaggle/input/dataset-batch-3
      ảnh: /kaggle/input/dataset-batch-3/L26_a/L26_V001/000.jpg
      -> ./runs/predict/L26_a/L26_V001/000.txt
    """
    if os.path.isdir(root_img):
        rel_dir = os.path.dirname(os.path.relpath(image_path, root_img))
        out_dir = os.path.join(output_root, rel_dir)
    else:
        out_dir = output_root

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_txt = os.path.join(out_dir, f"{stem}.txt")

    with open(out_txt, "w", encoding="utf-8") as f:
        for raw, corr in zip(texts, corrections):
            f.write(f"RAW: {raw}\n")
            f.write(f"CORR: {corr['generated_text']}\n")
            f.write("-" * 20 + "\n")
    print(f"[OK] Saved: {out_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True,
                        help='Đường dẫn 1 ảnh hoặc thư mục gốc chứa nhiều cấp thư mục ảnh')
    parser.add_argument('--output', default='./runs/predict',
                        help='Thư mục gốc lưu .txt kết quả (giữ cấu trúc cây)')
    parser.add_argument('--use_gpu', required=False, help='(không dùng, giữ để tương thích)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='auto (ưu tiên GPU nếu có), cpu, hoặc cuda')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {device.upper()}")
    else:
        device = args.device
        print(f"Using specified device: {device.upper()} (CUDA available: {torch.cuda.is_available()})")

    # Build models
    recognitor = build_vietocr_predictor(device)
    detector = build_paddle(device)
    corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")

    # Gom danh sách ảnh (đệ quy)
    img_paths = list_images_recursively(args.img)
    if not img_paths:
        print(f"[ERROR] Không tìm thấy ảnh trong: {args.img}")
        sys.exit(1)

    print(f"[INFO] Số ảnh: {len(img_paths)}")
    for p in img_paths:
        print(f"\n=== {p}")
        boxes, texts = predict(recognitor, detector, p, padding=2)
        if not texts:
            print("[INFO] Bỏ qua (không có text).")
            continue

        corrections = corrector(texts, max_new_tokens=MAX_NEW_TOKENS)

        # In nhanh ra console
        for raw, pred in zip(texts, corrections):
            print("- " + pred['generated_text'])

        # Lưu theo cấu trúc cây
        save_txt_mirrored(args.img, args.output, p, texts, corrections)


if __name__ == "__main__":
    main()
