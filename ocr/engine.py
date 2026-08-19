import cv2
import numpy as np
import pytesseract
import tempfile
import os


# Tesseract binary path (Homebrew on macOS)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Correct skew/tilt in scanned or photographed documents.
    Uses minimum area rectangle of all text pixels to find angle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only deskew if tilt is significant
    if abs(angle) < 0.5:
        return img

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _preprocess(img: np.ndarray, strategy: str) -> np.ndarray:
    """Apply a named preprocessing strategy to a BGR image."""

    h, w = img.shape[:2]

    # Upscale to at least 2500px tall
    if h < 2500:
        scale = 2500 / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Deskew
    img = _deskew(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if strategy == "adaptive":
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        result = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )

    elif strategy == "otsu":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif strategy == "clahe":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        result = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 8
        )

    elif strategy == "morph":
        # Morphological cleanup — good for handwriting
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    else:
        result = gray

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel)

    return result


def _run_tesseract(img: np.ndarray, psm: int = 6) -> dict:
    """Run Tesseract on a preprocessed numpy image."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    tmp.close()
    try:
        data = pytesseract.image_to_data(
            tmp.name,
            config=f"--psm {psm} --oem 3",
            output_type=pytesseract.Output.DICT
        )
    finally:
        os.remove(tmp.name)
    return data


def _extract_blocks(data: dict, min_conf: int = 25) -> list:
    """Convert raw Tesseract data dict into clean block list."""
    blocks = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        blocks.append({
            "text": text,
            "confidence": round(conf / 100, 2),
            "bounding_box": [x, y, x + w, y + h]
        })
    return blocks


def _score_blocks(blocks: list) -> float:
    """Score a block list — prefer more words with higher confidence."""
    if not blocks:
        return 0.0
    avg_conf = sum(b["confidence"] for b in blocks) / len(blocks)
    return avg_conf * len(blocks)


def extract_ocr_data(image_path: str) -> list:
    """
    Extract text blocks from image using Tesseract OCR.

    Tries multiple preprocessing strategies + PSM modes
    and picks the combination with the best confidence score.

    Returns:
        list of { text, confidence, bounding_box: [x1,y1,x2,y2] }
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    best_blocks = []
    best_score = -1
    best_label = ""

    # Try strategy × PSM combinations
    # PSM 6 = uniform block, PSM 11 = sparse text (good for handwriting)
    combos = [
        ("adaptive", 6),
        ("adaptive", 11),
        ("clahe",    6),
        ("clahe",    11),
        ("otsu",     6),
        ("morph",    6),
        ("morph",    11),
    ]

    for strategy, psm in combos:
        try:
            processed = _preprocess(img.copy(), strategy)
            data = _run_tesseract(processed, psm)
            blocks = _extract_blocks(data)
            score = _score_blocks(blocks)
            label = f"{strategy}/psm{psm}"
            print(f"  {label:<20}: {len(blocks):>3} blocks, score={score:.1f}")

            if score > best_score:
                best_score = score
                best_blocks = blocks
                best_label = label
        except Exception as e:
            print(f"  {strategy}/psm{psm}: failed — {e}")

    print("\n" + "=" * 60)
    print(f"BEST: {best_label}  |  {len(best_blocks)} blocks")
    print("=" * 60)
    for block in best_blocks:
        print(f"  [{block['confidence']:.2f}]  {block['text']}")
    print("=" * 60 + "\n")

    return best_blocks
