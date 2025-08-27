# predict.py - Modified to save JSON
import os
import sys
import argparse
import warnings
import cv2
import json
from PIL import Image
import torch

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
from paddleocr import PaddleOCR
from transformers import pipeline

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_NEW_TOKENS = 256


def is_image_file(path: str) -> bool:
    return os.path.isfile(path) and (os.path.splitext(path)[1].lower() in VALID_EXTS)


def list_images_recursively(root: str):
    """Quét đệ quy mọi cấp, trả về danh sách ảnh."""
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


def build_paddle(device: str,
                 angle_cls: bool = True,
                 ocr_version: str = "PP-OCRv4",
                 det_limit_side_len: int = 1536,
                 det_db_box_thresh: float = 0.30,
                 det_db_unclip_ratio: float = 2.0,
                 force_cpu: bool = False) -> PaddleOCR:
    """Khởi tạo PaddleOCR (det-only). Mặc định cho chạy 'dễ ăn' hơn."""
    use_gpu = (device == "cuda") and (not force_cpu) and torch.cuda.is_available()
    print(f"[INFO] PaddleOCR GPU: {use_gpu}")
    return PaddleOCR(
        use_angle_cls=angle_cls,
        lang="vi",
        use_gpu=use_gpu,
        det=True, rec=False,
        ocr_version=ocr_version,
        det_limit_side_len=det_limit_side_len,
        det_db_box_thresh=det_db_box_thresh,
        det_db_unclip_ratio=det_db_unclip_ratio,
        show_log=False  # Giảm log spam
    )


def preprocess_for_det(img_bgr, min_long_side: int = 800):
    """Tăng tương phản + phóng to nhẹ cho ảnh khó."""
    if img_bgr is None:
        return None
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )
    h, w = th.shape
    long_side = max(h, w)
    if long_side < min_long_side:
        scale = max(2, int(min_long_side / max(1, long_side)))
        th = cv2.resize(th, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    th3 = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return th3


def safe_get_lines(det_out):
    """Chuẩn hóa output của detector về list các dòng; rỗng nếu không có."""
    if det_out is None:
        return []
    if not isinstance(det_out, list) or len(det_out) == 0:
        return []
    first = det_out[0]
    if first is None or first == []:
        return []
    return first


def predict(recognitor: Predictor,
            detector: PaddleOCR,
            img_path: str,
            padding: int = 4,
            use_preprocess: bool = False):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Không đọc được ảnh: {img_path}")
        return [], []

    # Tiền xử lý (tùy chọn)
    det_input = preprocess_for_det(img) if use_preprocess else img

    # Detect (an toàn với None)
    det_out = detector.ocr(det_input, cls=False, det=True, rec=False) or []
    lines = safe_get_lines(det_out)
    if not lines:
        return [], []

    # Lấy bbox từ polygon
    H, W = img.shape[:2]
    boxes = []
    for item in lines:
        try:
            poly = item[0]
            if poly is None:
                continue
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

    boxes = boxes[::-1]

    # Recognize
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True,
                        help='Đường dẫn 1 ảnh hoặc thư mục gốc chứa nhiều cấp thư mục ảnh')
    parser.add_argument('--output', default='/kaggle/working/results.json',
                        help='File JSON output (mặc định: /kaggle/working/results.json)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='auto (ưu tiên GPU nếu có), cpu, hoặc cuda')
    parser.add_argument('--paddle_cpu', action='store_true',
                        help='Buộc PaddleOCR chạy CPU')
    parser.add_argument('--preprocess', action='store_true',
                        help='Bật tiền xử lý ảnh trước khi detect')
    parser.add_argument('--det_limit_side_len', type=int, default=1536)
    parser.add_argument('--det_db_box_thresh', type=float, default=0.30)
    parser.add_argument('--det_db_unclip_ratio', type=float, default=2.0)
    parser.add_argument('--ocr_version', type=str, default='PP-OCRv4',
                        choices=['PP-OCRv2', 'PP-OCRv3', 'PP-OCRv4'])
    parser.add_argument('--angle_cls', action='store_true', help='Bật phân lớp góc')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Auto-detected device: {device.upper()}")
    else:
        device = args.device
        print(f"[INFO] Using device: {device.upper()}")

    # Models
    print("[INFO] Loading models...")
    recognitor = build_vietocr_predictor(device)
    detector = build_paddle(
        device=device,
        angle_cls=args.angle_cls,
        ocr_version=args.ocr_version,
        det_limit_side_len=args.det_limit_side_len,
        det_db_box_thresh=args.det_db_box_thresh,
        det_db_unclip_ratio=args.det_db_unclip_ratio,
        force_cpu=args.paddle_cpu
    )
    corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")
    print("[INFO] Models loaded successfully!")

    # Ảnh (đệ quy)
    img_paths = list_images_recursively(args.img)
    if not img_paths:
        print(f"[ERROR] Không tìm thấy ảnh trong: {args.img}")
        sys.exit(1)

    print(f"[INFO] Found {len(img_paths)} images")
    
    # JSON result: {image_path: [list_of_texts]}
    results = {}
    
    for i, img_path in enumerate(img_paths, 1):
        print(f"\n[{i}/{len(img_paths)}] Processing: {os.path.basename(img_path)}")
        
        boxes, texts = predict(
            recognitor, detector, img_path,
            padding=4,
            use_preprocess=args.preprocess
        )
        
        if not texts:
            print("  → No text detected")
            results[img_path] = []
            continue

        # Correct texts
        print(f"  → Found {len(texts)} text regions")
        corrections = corrector(texts, max_new_tokens=MAX_NEW_TOKENS)
        
        # Store corrected texts
        corrected_texts = []
        for raw, pred in zip(texts, corrections):
            corrected_text = pred['generated_text'].strip()
            corrected_texts.append(corrected_text)
            print(f"    • {corrected_text}")
        
        results[img_path] = corrected_texts

    # Save JSON
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SUCCESS] Results saved to: {args.output}")
    print(f"[INFO] Processed {len(results)} images")
    print(f"[INFO] Total texts extracted: {sum(len(texts) for texts in results.values())}")


if __name__ == "__main__":
    main()
