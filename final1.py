"""
Passport / Visa Document Screening + Database System
====================================================

4-stage pipeline:
    1. CLASSIFICATION  -> Passport or Visa (using MRZ type P/V and keywords)
    2. EXTRACTION      -> OCR + MRZ validation + Field parsing
    3. VERIFICATION    -> Match against JSON database (Passport.json / Visas.json)
    4. STORAGE         -> Update verified database record (--store)

Supporting checks:
    - Font consistency analysis
    - Copy-move / Error Level Analysis (ELA)
    - Photo region tamper & boundary density check
    - Stamp tamper & color variance check
    - Face comparison (DeepFace with OpenCV fallback)
    - Multi-factor risk calculation & verdict

This is a project/demo screening system, NOT a certified authenticity
or immigration verification system.
"""

import cv2
import numpy as np
import pytesseract
import re
import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Optional DeepFace import with graceful fallback to OpenCV
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_db_path(filename):
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(os.path.dirname(BASE_DIR), filename),
        os.path.join(os.getcwd(), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

PASSPORT_DB = resolve_db_path("Passport.json")
VISA_DB = resolve_db_path("Visas.json")

# Tesseract path configuration (Windows default)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    tessdata_dir = os.path.join(os.path.dirname(TESSERACT_PATH), "tessdata")
    if os.path.exists(tessdata_dir):
        os.environ["TESSDATA_PREFIX"] = tessdata_dir

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# =========================================================
# DATABASE LOADING
# =========================================================

def load_json_database(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Database file not found: '{path}'. "
            f"Keep {os.path.basename(PASSPORT_DB)} and {os.path.basename(VISA_DB)} "
            f"in the same folder as this script."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


passports = load_json_database(PASSPORT_DB)
visas = load_json_database(VISA_DB)


# =========================================================
# STAGE 1 - DOCUMENT CLASSIFICATION
# =========================================================

def normalize_ocr_text(text):
    return re.sub(r"\s+", " ", text.upper()).strip()


def detect_mrz_lines(image):
    """
    Look for MRZ (Machine Readable Zone) lines.
    Covers TD3 (Passport, 2x44), MRV-A (Visa, 2x44), and MRV-B (Visa, 2x36).
    """
    if image is None:
        return [], False

    h, w = image.shape[:2]

    # MRZ is normally located in the bottom 35% of the page
    mrz_region = image[int(h * 0.65):h, 0:w]
    mrz_region = cv2.resize(
        mrz_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    config = (
        "--oem 3 --psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    )

    text = pytesseract.image_to_string(thresh, config=config)
    raw_lines = text.strip().splitlines()
    lines = []

    for line in raw_lines:
        cleaned = re.sub(r"\s+", "", line.upper())
        if len(cleaned) >= 28:
            lines.append(cleaned)

    if len(lines) >= 2:
        return lines[-2:], True

    return lines, False


def classify_document(image, ocr_text):
    """
    Stage 1:
    Classifies the image as PASSPORT, VISA, or UNKNOWN.

    Signals:
        - MRZ first character: 'P' indicates Passport, 'V' indicates Visa
        - OCR keywords specific to passports vs visas
    """
    mrz_lines, mrz_detected = detect_mrz_lines(image)
    text = normalize_ocr_text(ocr_text)

    passport_keywords = [
        "PASSPORT",
        "P<",
        "BOOKLET",
        "REPUBLIC OF",
        "PASSPORT NO",
        "PASSPORT NUMBER",
        "TYPE P",
    ]

    visa_keywords = [
        "VISA",
        "VISA TYPE",
        "CONTROL NUMBER",
        "ISSUING POST",
        "NUMBER OF ENTRIES",
        "VALID FOR",
        "ENTRIES",
        "NONIMMIGRANT",
        "IMMIGRANT",
        "B1/B2",
        "B-1/B-2",
        "BEARER",
    ]

    passport_score = 0
    visa_score = 0

    if mrz_detected and len(mrz_lines) >= 1:
        first_line = mrz_lines[0]
        if first_line.startswith("P"):
            passport_score += 15
        elif first_line.startswith("V"):
            visa_score += 15
        else:
            passport_score += 4
            visa_score += 4
    elif mrz_detected:
        passport_score += 3
        visa_score += 3

    for keyword in passport_keywords:
        if keyword in text:
            passport_score += 3

    for keyword in visa_keywords:
        if keyword in text:
            visa_score += 3

    if passport_score > visa_score and passport_score > 0:
        doc_type = "PASSPORT"
    elif visa_score > passport_score and visa_score > 0:
        doc_type = "VISA"
    else:
        doc_type = "UNKNOWN"

    return {
        "document_type": doc_type,
        "passport_score": int(passport_score),
        "visa_score": int(visa_score),
        "mrz_detected": bool(mrz_detected),
        "mrz_lines": mrz_lines,
    }


# =========================================================
# COMMON IMAGE PREPROCESSING / OCR
# =========================================================

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Could not load '{image_path}'. "
            "Check the path and make sure the image is valid."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    return img, gray, thresh


def extract_text(gray_or_thresh_img):
    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(gray_or_thresh_img, config=config)

    data = pytesseract.image_to_data(
        gray_or_thresh_img,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    return text, data


# =========================================================
# STAGE 2A - PASSPORT EXTRACTION
# =========================================================

MRZ_WEIGHTS = [7, 3, 1]


def mrz_char_value(c):
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    return ord(c) - 55


def compute_check_digit(data_str):
    total = 0
    for i, c in enumerate(data_str):
        total += mrz_char_value(c) * MRZ_WEIGHTS[i % 3]
    return total % 10


def validate_td3_mrz(line1, line2):
    results = {}

    if len(line2) < 28:
        results["error"] = (
            f"MRZ line 2 too short: {len(line2)} chars (expected >= 28)"
        )
        return results

    doc_number = line2[0:9]
    doc_check = line2[9]

    dob = line2[13:19]
    dob_check = line2[19]

    expiry = line2[21:27]
    expiry_check = line2[27]

    clean_doc = doc_number.replace("<", "").replace("O", "0")
    results["passport_number"] = clean_doc

    results["doc_number_valid"] = (
        str(compute_check_digit(doc_number)) == doc_check
    )

    results["dob_valid"] = (
        str(compute_check_digit(dob)) == dob_check
    )

    results["expiry_valid"] = (
        str(compute_check_digit(expiry)) == expiry_check
    )

    results["raw_dob"] = dob
    results["raw_expiry"] = expiry

    try:
        dob_date = datetime.strptime(dob, "%y%m%d")
        results["dob_plausible"] = bool(dob_date <= datetime.now())
        results["dob_parsed"] = dob_date.strftime("%Y-%m-%d")
    except ValueError:
        results["dob_plausible"] = False

    try:
        exp_date = datetime.strptime(expiry, "%y%m%d")
        results["not_expired"] = bool(exp_date >= datetime.now())
        results["expiry_parsed"] = exp_date.strftime("%Y-%m-%d")
    except ValueError:
        results["not_expired"] = False

    return results


def extract_name_from_mrz(line1):
    if not line1 or len(line1) < 10:
        return None
    name_part = line1[5:] if len(line1) > 5 else line1
    name_part = re.sub(r"[^A-Z<]", "", name_part)
    if "<<" in name_part:
        parts = name_part.split("<<")
        surname = parts[0].replace("<", " ").strip()
        given = parts[1].split("<")[0].strip() if len(parts) > 1 else ""
        return f"{given} {surname}".strip() if given else surname
    elif "<" in name_part:
        return " ".join([p for p in name_part.split("<") if p])
    return name_part


def extract_passport_data(image, ocr_text):
    mrz_lines, detected = detect_mrz_lines(image)

    result = {
        "mrz_lines": mrz_lines,
        "mrz_detected": bool(detected),
        "passport_number": None,
        "name": None,
        "nationality": None,
        "date_of_birth": None,
        "date_of_expiry": None,
        "mrz_validation": {},
        "ocr_text": ocr_text,
    }

    if len(mrz_lines) >= 2:
        validation = validate_td3_mrz(mrz_lines[0], mrz_lines[1])
        result["mrz_validation"] = validation
        result["passport_number"] = validation.get("passport_number")
        result["name"] = extract_name_from_mrz(mrz_lines[0])
        if len(mrz_lines[0]) >= 5:
            result["nationality"] = mrz_lines[0][2:5].replace("<", "")
        if validation.get("dob_parsed"):
            result["date_of_birth"] = validation["dob_parsed"]
        if validation.get("expiry_parsed"):
            result["date_of_expiry"] = validation["expiry_parsed"]

    # Fallback: look for a passport-number-like token in OCR.
    if not result["passport_number"]:
        candidates = re.findall(
            r"\b[A-Z][A-Z0-9<]{7,8}\b",
            normalize_ocr_text(ocr_text),
        )
        if candidates:
            result["passport_number"] = candidates[0].replace("<", "")

    return result


# =========================================================
# STAGE 2B - VISA EXTRACTION
# =========================================================

def normalize_passport_number(value):
    if not value:
        return None

    value = re.sub(r"[^A-Z0-9]", "", value.upper())

    if len(value) < 6:
        return None

    # Handle standard OCR errors (e.g. letter O -> 0, letter I -> 1 in digit parts)
    if value.startswith("X") and len(value) >= 7:
        value = "X" + value[1:].replace("O", "0").replace("I", "1")
    elif len(value) >= 7:
        # Common format: 2 letters followed by 6-7 digits/chars e.g. CZ6311T47
        prefix = value[:2]
        rest = value[2:].replace("I", "1").replace("O", "0")
        value = prefix + rest

    return value


def extract_date_candidates(text):
    patterns = [
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
        r"\b\d{2}[-/]\d{2}[-/]\d{4}\b",
        r"\b\d{2}[-/]\d{2}[-/]\d{2}\b",
        r"\b\d{2}[A-Z]{3}\d{4}\b",
        r"\b\d{2}\s+[A-Z]{3}\s+\d{4}\b",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, text))

    return candidates


def parse_date(value):
    if not value:
        return None

    value = value.strip().replace(" ", "")

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d-%m-%y",
        "%d/%m/%y",
        "%d%b%Y",
        "%d%b%y",
        "%d%B%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Try normalizing OCR typos in alphanumeric dates e.g. 1OJUN1981 -> 10JUN1981
    m = re.match(r"^(\d{1,2}|[OI]\d|\d[OI])([A-Z]{3})(\d{4}|\d{2})$", value)
    if m:
        day = m.group(1).replace("O", "0").replace("I", "1").zfill(2)
        month = m.group(2)
        year = m.group(3).replace("O", "0").replace("I", "1")
        clean_val = f"{day}{month}{year}"
        for fmt in ["%d%b%Y", "%d%b%y"]:
            try:
                return datetime.strptime(clean_val, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


def extract_visa_data(image, ocr_text):
    text = normalize_ocr_text(ocr_text)
    mrz_lines, mrz_detected = detect_mrz_lines(image) if image is not None else ([], False)

    result = {
        "passport_number": None,
        "name": None,
        "nationality": None,
        "visa_type": None,
        "date_of_expiry": None,
        "date_of_birth": None,
        "mrz_lines": mrz_lines,
        "mrz_detected": bool(mrz_detected),
        "mrz_validation": {},
        "ocr_text": ocr_text,
    }

    # 1. Parse MRZ if present (MRV-A or MRV-B format)
    if len(mrz_lines) >= 2:
        validation = validate_td3_mrz(mrz_lines[0], mrz_lines[1])
        result["mrz_validation"] = validation
        result["name"] = extract_name_from_mrz(mrz_lines[0])
        if len(mrz_lines) >= 2 and len(mrz_lines[1]) >= 13:
            result["nationality"] = mrz_lines[1][10:13].replace("<", "")
        if validation.get("passport_number"):
            result["passport_number"] = normalize_passport_number(validation["passport_number"])
        if validation.get("expiry_parsed"):
            result["date_of_expiry"] = validation["expiry_parsed"]
        if validation.get("dob_parsed"):
            result["date_of_birth"] = validation["dob_parsed"]

    # 2. Extract Passport Number from text if MRZ didn't provide one or for cross-check
    if not result["passport_number"]:
        passport_patterns = [
            r"(?:PASSPORT\s*(?:NO|NUMBER|#)?|PASSPORTNO)\s*[:\-]?\s*([A-Z0-9<]{6,12})",
            r"(?:DOCUMENT\s*(?:NO|NUMBER|#))\s*[:\-]?\s*([A-Z0-9<]{6,12})",
        ]
        for pattern in passport_patterns:
            match = re.search(pattern, text)
            if match:
                result["passport_number"] = normalize_passport_number(match.group(1))
                break

    if not result["passport_number"]:
        match = re.search(r"\bX[0-9O]{6,8}\b", text)
        if match:
            result["passport_number"] = normalize_passport_number(match.group(0))

    if not result["passport_number"]:
        candidates = re.findall(r"\b[A-Z]{1,2}\d{6,8}[A-Z0-9]?\b", text)
        if candidates:
            result["passport_number"] = normalize_passport_number(candidates[0])

    # 3. Extract Visa Type
    if re.search(r"\b[B8][\s\-_/]*1\s*/\s*[B8][\s\-_/]*2\b|\b[B8]1/[B8]2\b", text):
        result["visa_type"] = "B1/B2"
    elif re.search(r"\b[B8][\s\-_/]*1\b", text):
        result["visa_type"] = "Business (B1)"
    elif re.search(r"\b[B8][\s\-_/]*2\b", text):
        result["visa_type"] = "Tourist (B2)"
    elif re.search(r"\bF[\s\-_/]*1\b", text):
        result["visa_type"] = "Student (F1)"
    elif re.search(r"\bH[\s\-_/]*1B\b", text):
        result["visa_type"] = "Work (H1B)"
    else:
        for vt in ["TOURIST", "STUDENT", "BUSINESS", "TRANSIT", "DIPLOMATIC"]:
            if vt in text:
                result["visa_type"] = vt.title()
                break

    # 4. Extract Date of Expiry from text
    # Collect all parsed dates in the text
    dates_found = []
    candidates = extract_date_candidates(text)
    for cand in candidates:
        parsed = parse_date(cand)
        if parsed and parsed not in dates_found:
            dates_found.append(parsed)

    if dates_found:
        dates_found.sort()
        # The expiry date is typically in the future, or the latest date mentioned
        future_dates = [d for d in dates_found if d >= datetime.now().strftime("%Y-%m-%d")]
        if future_dates:
            result["date_of_expiry"] = future_dates[-1]
        elif not result["date_of_expiry"]:
            result["date_of_expiry"] = dates_found[-1]

    # Also capture birth date if present
    past_dates = [d for d in dates_found if d < "2015-01-01"]
    if past_dates and not result["date_of_birth"]:
        result["date_of_birth"] = past_dates[0]

    return result


# =========================================================
# STAGE 2C - FORENSIC / SUPPORTING CHECKS
# =========================================================

def check_font_consistency(data):
    heights = []

    for i in range(len(data.get("text", []))):
        text = data["text"][i].strip()
        h = data["height"][i]
        conf = int(data.get("conf", [0])[i])

        if len(text) > 1 and conf > 40:
            heights.append(h)

    if len(heights) < 5:
        return {
            "height_cv": None,
            "suspicious": False,
            "note": "Not enough high-confidence text lines to evaluate",
        }

    mean_h = float(np.mean(heights))
    std_h = float(np.std(heights))
    cv = float(std_h / mean_h) if mean_h > 0 else 0.0

    return {
        "height_cv": round(cv, 2),
        "suspicious": bool(cv > 0.8),
    }


def detect_copy_move_artifacts(gray_img):
    is_success, buffer = cv2.imencode(
        ".jpg",
        gray_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )

    if not is_success:
        return {
            "mean_error_level": 0.0,
            "max_error_level": 0,
            "suspicious_region": False,
        }

    recompressed = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    diff = cv2.absdiff(gray_img, recompressed)

    mean_diff = float(np.mean(diff))
    max_diff = int(np.max(diff))

    return {
        "mean_error_level": round(mean_diff, 2),
        "max_error_level": max_diff,
        "suspicious_region": bool(max_diff > 60),
    }


def locate_photo_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    faces = sorted(
        faces,
        key=lambda f: f[2] * f[3],
        reverse=True,
    )

    x, y, w, h = faces[0]

    pad_x = int(w * 0.6)
    pad_y = int(h * 0.8)

    x0 = int(max(0, x - pad_x))
    y0 = int(max(0, y - pad_y))
    x1 = int(min(img.shape[1], x + w + pad_x))
    y1 = int(min(img.shape[0], y + h + pad_y))

    return (x0, y0, x1, y1)


def analyze_photo_region(img, gray_full):
    box = locate_photo_region(img)

    if box is None:
        return {
            "photo_detected": False,
            "note": "No face/photo region located",
        }

    x0, y0, x1, y1 = box
    photo_crop = img[y0:y1, x0:x1]

    ela_result = detect_copy_move_artifacts(
        cv2.cvtColor(photo_crop, cv2.COLOR_BGR2GRAY)
    )

    pad = 12
    bx0 = max(0, x0 - pad)
    by0 = max(0, y0 - pad)
    bx1 = min(img.shape[1], x1 + pad)
    by1 = min(img.shape[0], y1 + pad)

    border_strip = gray_full[by0:by1, bx0:bx1]

    edges = cv2.Canny(border_strip, 50, 150)
    edge_density = float(np.mean(edges > 0))

    return {
        "photo_detected": True,
        "photo_bbox": (int(x0), int(y0), int(x1), int(y1)),
        "photo_ela": ela_result,
        "border_edge_density": round(edge_density, 3),
        "suspicious_boundary": bool(edge_density > 0.28),
    }


def analyze_stamp_regions(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_ink = np.array([100, 50, 50])
    upper_ink = np.array([160, 255, 255])

    mask = cv2.inRange(hsv, lower_ink, upper_ink)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        opened,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    stamps = []
    any_suspicious = False

    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(c)
        roi_hsv = hsv[y:y + h, x:x + w]
        hue_channel = roi_hsv[:, :, 0]
        hue_std = float(np.std(hue_channel))

        aspect = float(w) / max(h, 1)
        straightness = float(max(aspect, 1.0 / max(aspect, 1e-6)))

        is_suspicious = bool(hue_std > 20 or straightness > 2.5)

        if is_suspicious:
            any_suspicious = True

        stamps.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "hue_std": round(hue_std, 2),
            "straightness_ratio": round(straightness, 2),
            "suspicious": is_suspicious,
        })

    return {
        "stamps_detected": len(stamps),
        "stamp_details": stamps,
        "any_suspicious": bool(any_suspicious),
    }


# =========================================================
# MODULE 4 - FACE VERIFICATION (SUPPORTING CHECK)
# =========================================================

def _opencv_compare_faces(img_a, img_b):
    """
    Fallback face comparison using OpenCV Haar Cascade detection,
    histogram equalization, and normalized template matching.
    """
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    def get_face_crop(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        crop = gray[y:y + h, x:x + w]
        return cv2.resize(crop, (200, 200))

    face_a = get_face_crop(img_a)
    face_b = get_face_crop(img_b)

    if face_a is None or face_b is None:
        return {
            "comparable": False,
            "verified": False,
            "distance": None,
            "similarity_score": None,
            "likely_same_person": False,
            "method": "opencv_cascade",
            "status": "NO_FACE_DETECTED",
            "message": "Could not detect a clear face in one or both images",
        }

    face_a = cv2.equalizeHist(face_a)
    face_b = cv2.equalizeHist(face_b)

    match_res = cv2.matchTemplate(face_a, face_b, cv2.TM_CCOEFF_NORMED)
    raw_sim = float(match_res[0][0])
    similarity = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))
    distance = max(0.0, round(1.0 - similarity, 3))
    verified = bool(similarity > 0.60)

    return {
        "comparable": True,
        "verified": verified,
        "similarity_score": round(similarity, 3),
        "distance": distance,
        "likely_same_person": verified,
        "method": "opencv_cascade",
        "status": "MATCH" if verified else "MISMATCH",
        "message": "Face Match (OpenCV template comparison)" if verified else "Face Mismatch",
    }


def compare_faces(img_a, img_b):
    """
    Face Verification using DeepFace if installed,
    with an automatic, resilient fallback to OpenCV.
    """
    if DEEPFACE_AVAILABLE and DeepFace is not None:
        try:
            result = DeepFace.verify(
                img1_path=img_a,
                img2_path=img_b,
                enforce_detection=False,
            )

            verified = bool(result.get("verified", False))
            distance = result.get("distance")
            if distance is not None:
                distance = round(float(distance), 3)

            return {
                "comparable": True,
                "verified": verified,
                "distance": distance,
                "likely_same_person": verified,
                "method": "deepface",
                "status": "MATCH" if verified else "MISMATCH",
                "message": "Face Match (DeepFace)" if verified else "Face Mismatch (DeepFace)",
            }
        except Exception:
            # Fallback to OpenCV if DeepFace model execution fails
            pass

    return _opencv_compare_faces(img_a, img_b)


# =========================================================
# STAGE 3 - DATABASE VERIFICATION
# =========================================================

def find_passport(passport_no):
    if not passport_no:
        return None
    return passports.get(passport_no)


def find_visa(passport_no):
    if not passport_no:
        return None, None

    for visa_id, visa_data in visas.items():
        if visa_data.get("passport_no") == passport_no:
            return visa_id, visa_data

    return None, None


def verify_passport(passport_data):
    passport_no = passport_data.get("passport_number")
    db_record = find_passport(passport_no)

    result = {
        "verified": False,
        "passport_found": False,
        "passport_number": passport_no,
        "database_record": None,
        "field_matches": {},
        "issues": [],
    }

    if not db_record:
        result["issues"].append(
            f"Passport number '{passport_no}' not found in Passport.json"
        )
        return result

    result["passport_found"] = True
    result["database_record"] = db_record

    mrz = passport_data.get("mrz_validation", {})

    if "dob_parsed" in mrz:
        result["field_matches"]["date_of_birth"] = bool(
            mrz["dob_parsed"] == db_record.get("date_of_birth")
        )

    if "expiry_parsed" in mrz:
        result["field_matches"]["date_of_expiry"] = bool(
            mrz["expiry_parsed"] == db_record.get("date_of_expiry")
        )

    if mrz.get("doc_number_valid") is False:
        result["issues"].append(
            "Passport number MRZ check digit failed"
        )

    if mrz.get("dob_valid") is False:
        result["issues"].append(
            "Date of birth MRZ check digit failed"
        )

    if mrz.get("expiry_valid") is False:
        result["issues"].append(
            "Expiry date MRZ check digit failed"
        )

    if result["field_matches"].get("date_of_birth") is False:
        result["issues"].append(
            "DOB does not match Passport.json"
        )

    if result["field_matches"].get("date_of_expiry") is False:
        result["issues"].append(
            "Passport expiry does not match Passport.json"
        )

    result["verified"] = bool(
        result["passport_found"] and len(result["issues"]) == 0
    )

    return result


def verify_visa(visa_data):
    passport_no = visa_data.get("passport_number")
    visa_id, db_record = find_visa(passport_no)

    result = {
        "verified": False,
        "visa_found": False,
        "visa_id": visa_id,
        "passport_number": passport_no,
        "database_record": None,
        "field_matches": {},
        "issues": [],
    }

    if not db_record:
        result["issues"].append(
            f"Visa/passport number '{passport_no}' not found in Visas.json"
        )
        return result

    result["visa_found"] = True
    result["database_record"] = db_record

    # Visa type matching
    if visa_data.get("visa_type"):
        extracted_vt = visa_data["visa_type"].upper()
        db_vt = str(db_record.get("visa_type", "")).upper()
        vt_match = (
            extracted_vt == db_vt
            or (extracted_vt in ["B1", "B2", "B1/B2"] and db_vt in ["TOURIST", "BUSINESS", "B1/B2", "TOURIST/BUSINESS"])
            or (db_vt in ["B1", "B2", "B1/B2"] and extracted_vt in ["TOURIST", "BUSINESS", "B1/B2", "TOURIST/BUSINESS"])
        )
        result["field_matches"]["visa_type"] = bool(vt_match)

    # Expiry matching
    if visa_data.get("date_of_expiry"):
        exp_match = bool(
            visa_data["date_of_expiry"] == db_record.get("date_of_expiry")
        )
        result["field_matches"]["date_of_expiry"] = exp_match

    # Issues check
    if result["field_matches"].get("visa_type") is False:
        result["issues"].append(
            f"Visa type mismatch: extracted '{visa_data['visa_type']}' vs database '{db_record.get('visa_type')}'"
        )

    if result["field_matches"].get("date_of_expiry") is False:
        result["issues"].append(
            f"Visa expiry mismatch: extracted '{visa_data['date_of_expiry']}' vs database '{db_record.get('date_of_expiry')}'"
        )

    result["verified"] = bool(
        result["visa_found"] and len(result["issues"]) == 0
    )

    return result


# =========================================================
# STAGE 4 - STORAGE
# =========================================================

def store_verified_passport(passport_data, verification):
    """
    Updates or inserts the verified passport record in Passport.json.
    """
    if not verification.get("verified"):
        return False, "Passport was not verified; storage skipped."

    passport_no = passport_data.get("passport_number")

    if not passport_no:
        return False, "Passport number missing; storage skipped."

    mrz = passport_data.get("mrz_validation", {})

    if passport_no not in passports:
        passports[passport_no] = {
            "name": passport_data.get("name") or "Verified Holder",
            "nationality": passport_data.get("nationality") or "Indian",
            "date_of_birth": mrz.get("dob_parsed") or "",
            "date_of_expiry": mrz.get("expiry_parsed") or "",
        }
        msg = f"New passport registered in {os.path.basename(PASSPORT_DB)}: {passport_no}"
    else:
        record = passports[passport_no]
        if mrz.get("dob_parsed"):
            record["date_of_birth"] = mrz["dob_parsed"]
        if mrz.get("expiry_parsed"):
            record["date_of_expiry"] = mrz["expiry_parsed"]
        msg = f"Passport record updated in {os.path.basename(PASSPORT_DB)}: {passport_no}"

    with open(PASSPORT_DB, "w", encoding="utf-8") as f:
        json.dump(passports, f, indent=2, ensure_ascii=False)

    return True, msg


def store_verified_visa(visa_data, verification):
    """
    Updates or inserts the verified visa record in Visas.json.
    """
    if not verification.get("verified"):
        return False, "Visa was not verified; storage skipped."

    visa_id = verification.get("visa_id")
    passport_no = visa_data.get("passport_number")

    if not visa_id or visa_id not in visas:
        visa_id = f"MOCKVISA{len(visas)+1:03d}"
        visas[visa_id] = {
            "passport_no": passport_no or "UNKNOWN",
            "visa_type": visa_data.get("visa_type") or "Tourist",
            "date_of_expiry": visa_data.get("date_of_expiry") or "",
        }
        msg = f"New visa record created in {os.path.basename(VISA_DB)}: {visa_id} ({passport_no})"
    else:
        record = visas[visa_id]
        if visa_data.get("passport_number"):
            record["passport_no"] = visa_data["passport_number"]
        if visa_data.get("visa_type"):
            record["visa_type"] = visa_data["visa_type"]
        if visa_data.get("date_of_expiry"):
            record["date_of_expiry"] = visa_data["date_of_expiry"]
        msg = f"Visa record updated in {os.path.basename(VISA_DB)}: {visa_id}"

    with open(VISA_DB, "w", encoding="utf-8") as f:
        json.dump(visas, f, indent=2, ensure_ascii=False)

    return True, msg


# =========================================================
# RISK SCORE
# =========================================================

def calculate_risk(
    document_type,
    mrz_validation=None,
    font_check=None,
    forensics=None,
    photo_check=None,
    stamp_check=None,
    verification=None,
    face_match=None,
):
    breakdown = {}

    mrz_validation = mrz_validation or {}
    font_check = font_check or {}
    forensics = forensics or {}
    photo_check = photo_check or {}
    stamp_check = stamp_check or {}
    verification = verification or {}
    face_match = face_match or {}

    if document_type == "PASSPORT":
        if mrz_validation.get("error"):
            breakdown["mrz"] = 30
        elif mrz_validation:
            mrz_score = 0
            for key in [
                "doc_number_valid",
                "dob_valid",
                "expiry_valid",
                "dob_plausible",
            ]:
                if not mrz_validation.get(key, False):
                    mrz_score += 15
            breakdown["mrz"] = mrz_score

        if not verification.get("passport_found", False):
            breakdown["passport_database"] = 20
        elif not verification.get("verified", False):
            breakdown["passport_database_mismatch"] = 20

    elif document_type == "VISA":
        if mrz_validation.get("error"):
            breakdown["mrz"] = 30
        elif mrz_validation:
            mrz_score = 0
            for key in [
                "doc_number_valid",
                "dob_valid",
                "expiry_valid",
                "dob_plausible",
            ]:
                if not mrz_validation.get(key, False):
                    mrz_score += 15
            breakdown["mrz"] = mrz_score

        if not verification.get("visa_found", False):
            breakdown["visa_database"] = 20
        elif not verification.get("verified", False):
            breakdown["visa_database_mismatch"] = 20

    breakdown["font"] = (
        15 if font_check.get("suspicious") else 0
    )

    breakdown["general_forensics"] = (
        15 if forensics.get("suspicious_region") else 0
    )

    breakdown["photo_tamper"] = (
        20 if photo_check.get("suspicious_boundary") else 0
    )

    breakdown["stamp_tamper"] = (
        15 if stamp_check.get("any_suspicious") else 0
    )

    if face_match.get("comparable"):
        if not face_match.get("likely_same_person", False):
            breakdown["face_mismatch"] = 25
        else:
            breakdown["face_mismatch"] = 0

    total = min(sum(breakdown.values()), 100)

    if total >= 60:
        verdict = "HIGH RISK - needs manual review"
    elif total >= 30:
        verdict = "MEDIUM RISK - verify manually"
    else:
        verdict = "LOW RISK - no obvious issues"

    return total, breakdown, verdict


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass
class DocumentAnalysisResult:
    document_type: str = "UNKNOWN"
    ocr_text: str = ""
    classification: dict = field(default_factory=dict)
    passport_data: dict = field(default_factory=dict)
    visa_data: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    form_verification: dict = field(default_factory=dict)
    font_check: dict = field(default_factory=dict)
    forensics: dict = field(default_factory=dict)
    photo_check: dict = field(default_factory=dict)
    stamp_check: dict = field(default_factory=dict)
    face_match: dict = field(default_factory=dict)
    risk_score: int = 0
    risk_breakdown: dict = field(default_factory=dict)
    verdict: str = "UNKNOWN"

    def to_dict(self):
        return asdict(self)


def cross_verify_form(analysis_result, form_data):
    """
    Cross-checks user-submitted form details against OCR/MRZ data.
    """
    if not form_data:
        return {
            "form_checked": False,
            "form_matches": {},
            "form_issues": [],
        }

    form_matches = {}
    form_issues = []

    doc_data = (
        analysis_result.passport_data
        if analysis_result.document_type == "PASSPORT"
        else analysis_result.visa_data
    )

    form_pass = form_data.get("passportNo", "").strip().upper()
    extracted_pass = (doc_data.get("passport_number") or "").strip().upper()
    if form_pass and extracted_pass:
        matches = bool(form_pass == extracted_pass)
        form_matches["passport_number"] = matches
        if not matches:
            form_issues.append(
                f"Form Passport No '{form_pass}' != Document '{extracted_pass}'"
            )

    form_dob = form_data.get("dob", "").strip()
    extracted_dob = (doc_data.get("date_of_birth") or "").strip()
    if form_dob and extracted_dob:
        matches = bool(form_dob == extracted_dob)
        form_matches["date_of_birth"] = matches
        if not matches:
            form_issues.append(
                f"Form DOB '{form_dob}' != Document '{extracted_dob}'"
            )

    form_exp = form_data.get("expiry", "").strip()
    extracted_exp = (doc_data.get("date_of_expiry") or "").strip()
    if form_exp and extracted_exp:
        matches = bool(form_exp == extracted_exp)
        form_matches["date_of_expiry"] = matches
        if not matches:
            form_issues.append(
                f"Form Expiry '{form_exp}' != Document '{extracted_exp}'"
            )

    form_name = form_data.get("fullName", "").strip().upper()
    extracted_name = (doc_data.get("name") or "").strip().upper()
    if form_name and extracted_name:
        matches = bool(form_name in extracted_name or extracted_name in form_name)
        form_matches["full_name"] = matches
        if not matches:
            form_issues.append(
                f"Form Name '{form_name}' != Document '{extracted_name}'"
            )

    return {
        "form_checked": bool(len(form_matches) > 0),
        "form_matches": form_matches,
        "form_issues": form_issues,
    }


# =========================================================
# COMPLETE 4-STAGE PIPELINE
# =========================================================

def analyze_document(
    image_path,
    reference_face_path=None,
    store=False,
    form_data=None,
):
    # -----------------------------------------------------
    # STAGE 1 - CLASSIFICATION
    # -----------------------------------------------------

    img, gray, thresh = preprocess_image(image_path)
    ocr_text, ocr_data = extract_text(gray)

    classification = classify_document(
        img,
        ocr_text,
    )

    document_type = classification["document_type"]

    result = DocumentAnalysisResult(
        document_type=document_type,
        ocr_text=ocr_text,
        classification=classification,
    )

    # -----------------------------------------------------
    # STAGE 2 - EXTRACTION
    # -----------------------------------------------------

    if document_type == "PASSPORT":
        result.passport_data = extract_passport_data(
            img,
            ocr_text,
        )
    elif document_type == "VISA":
        result.visa_data = extract_visa_data(
            img,
            ocr_text,
        )
    else:
        result.verdict = (
            "UNKNOWN DOCUMENT - could not confidently classify "
            "as passport or visa"
        )
        return result

    # Supporting checks for both document types
    result.font_check = check_font_consistency(ocr_data)
    result.forensics = detect_copy_move_artifacts(gray)
    result.photo_check = analyze_photo_region(img, gray)
    result.stamp_check = analyze_stamp_regions(img)

    # Optional facial identity check
    if reference_face_path:
        ref_img = cv2.imread(reference_face_path)
        if ref_img is None:
            result.face_match = {
                "comparable": False,
                "note": f"Could not load reference image: '{reference_face_path}'",
            }
        else:
            result.face_match = compare_faces(img, ref_img)
    else:
        result.face_match = {
            "comparable": False,
            "note": "No reference face image provided",
        }

    # -----------------------------------------------------
    # STAGE 3 - VERIFICATION
    # -----------------------------------------------------

    if document_type == "PASSPORT":
        result.verification = verify_passport(result.passport_data)
        mrz_validation = result.passport_data.get("mrz_validation", {})
    else:
        result.verification = verify_visa(result.visa_data)
        mrz_validation = result.visa_data.get("mrz_validation", {})

    if form_data:
        result.form_verification = cross_verify_form(result, form_data)

    # -----------------------------------------------------
    # RISK ASSESSMENT
    # -----------------------------------------------------

    (
        result.risk_score,
        result.risk_breakdown,
        result.verdict,
    ) = calculate_risk(
        document_type=document_type,
        mrz_validation=mrz_validation,
        font_check=result.font_check,
        forensics=result.forensics,
        photo_check=result.photo_check,
        stamp_check=result.stamp_check,
        verification=result.verification,
        face_match=result.face_match,
    )

    # -----------------------------------------------------
    # STAGE 4 - STORAGE
    # -----------------------------------------------------

    if store:
        if document_type == "PASSPORT":
            stored, message = store_verified_passport(
                result.passport_data,
                result.verification,
            )
        else:
            stored, message = store_verified_visa(
                result.visa_data,
                result.verification,
            )

        result.verification["storage"] = {
            "requested": True,
            "stored": stored,
            "message": message,
        }
    else:
        result.verification["storage"] = {
            "requested": False,
            "stored": False,
            "message": "Storage not requested. Use --store flag to save updates.",
        }

    return result


# =========================================================
# OUTPUT FORMATTING
# =========================================================

def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(result):
    section("STAGE 1 - DOCUMENT CLASSIFICATION")
    print("Document Type:  ", result.document_type)
    print("Passport Score: ", result.classification.get("passport_score"))
    print("Visa Score:     ", result.classification.get("visa_score"))
    print("MRZ Detected:   ", result.classification.get("mrz_detected"))

    section("STAGE 2 - EXTRACTION")
    if result.document_type == "PASSPORT":
        print("Passport Number: ", result.passport_data.get("passport_number"))
        print("MRZ Lines:       ", result.passport_data.get("mrz_lines"))
        print("MRZ Validation:  ", result.passport_data.get("mrz_validation"))
    elif result.document_type == "VISA":
        print("Passport Number: ", result.visa_data.get("passport_number"))
        print("Visa Type:       ", result.visa_data.get("visa_type"))
        print("Visa Expiry:     ", result.visa_data.get("date_of_expiry"))
        if result.visa_data.get("mrz_lines"):
            print("MRZ Lines:       ", result.visa_data.get("mrz_lines"))
        if result.visa_data.get("mrz_validation"):
            print("MRZ Validation:  ", result.visa_data.get("mrz_validation"))

    section("STAGE 3 - DATABASE VERIFICATION")
    print("Verified:            ", result.verification.get("verified"))
    if result.document_type == "PASSPORT":
        print("Passport Found:      ", result.verification.get("passport_found"))
    else:
        print("Visa Found:          ", result.verification.get("visa_found"))
        print("Visa ID:             ", result.verification.get("visa_id"))

    print("Field Matches:       ", result.verification.get("field_matches"))
    print("Verification Issues: ", result.verification.get("issues"))

    if result.verification.get("database_record"):
        print("Database Record:")
        print(" ", json.dumps(result.verification["database_record"], indent=2))

    section("SUPPORTING CHECKS")
    print("Font Consistency Check: ", result.font_check)
    print("Forensics (ELA):        ", result.forensics)
    print("Photo Region Check:     ", result.photo_check)
    print("Stamp Region Check:     ", result.stamp_check)
    print("Face Match Verification:", result.face_match)

    section("STAGE 4 - STORAGE")
    print("Storage Status: ", result.verification.get("storage"))

    section("RISK BREAKDOWN")
    for key, value in result.risk_breakdown.items():
        print(f"  {key:<26}: {value}")

    section("FINAL RESULT")
    print(f"Risk Score: {result.risk_score}/100")
    print(f"Verdict:    {result.verdict}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0])

    if len(sys.argv) < 2:
        print(
            f"Usage:\n"
            f"  python {script_name} <document_image>\n"
            f"  python {script_name} <document_image> <reference_face>\n"
            f"  python {script_name} <document_image> --store\n"
            f"  python {script_name} <document_image> <reference_face> --store\n"
            f"  python {script_name} <document_image> --json"
        )
        sys.exit(1)

    image_path = sys.argv[1]
    reference_face_path = None
    store = "--store" in sys.argv
    output_json = "--json" in sys.argv

    # Detect optional reference face path
    for arg in sys.argv[2:]:
        if arg not in ["--store", "--json"]:
            reference_face_path = arg
            break

    if not output_json:
        print(f"Analyzing document: {image_path}")
        if reference_face_path:
            print(f"Reference face:    {reference_face_path}")
        if store:
            print("Storage mode:      ENABLED")

    try:
        res = analyze_document(
            image_path=image_path,
            reference_face_path=reference_face_path,
            store=store,
        )

        if output_json:
            print(json.dumps(res.to_dict(), indent=2, default=str))
        else:
            print_result(res)

    except pytesseract.pytesseract.TesseractNotFoundError:
        print("\nERROR: Tesseract OCR was not found.")
        print(
            "Install Tesseract or ensure TESSERACT_PATH points to tesseract.exe."
        )
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
