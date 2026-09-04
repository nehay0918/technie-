"""
Passport / Visa Document Screening + Database System
====================================================

4-stage pipeline:
    1. CLASSIFICATION  -> Passport or Visa
    2. EXTRACTION      -> OCR + MRZ / Visa fields
    3. VERIFICATION    -> Match against the correct JSON database
    4. STORAGE         -> Update the matching verified record

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
from dataclasses import dataclass, field


# =========================================================
# CONFIGURATION
# =========================================================

PASSPORT_DB = "Passport.json"
VISA_DB = "Visas.json"

# If Tesseract is installed at the normal Windows location, this works.
# If tesseract is already in PATH, leave this as None.
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# =========================================================
# DATABASE LOADING
# =========================================================

def load_json_database(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Database file not found: '{path}'. "
            f"Keep {PASSPORT_DB} and {VISA_DB} in the same folder as this script."
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
    Look for passport TD3-style MRZ lines.
    We do not rely only on the word 'PASSPORT'.
    """
    h, w = image.shape[:2]

    # MRZ is normally near the bottom of a passport page.
    mrz_region = image[int(h * 0.72):h, 0:w]
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
        if len(cleaned) >= 30:
            lines.append(cleaned)

    # A TD3 passport normally has two MRZ lines.
    if len(lines) >= 2:
        return lines[-2:], True

    return lines, False


def classify_document(image, ocr_text):
    """
    Stage 1:
    Classifies the image as PASSPORT, VISA, or UNKNOWN.

    Strongest signal:
        - Passport MRZ detected

    Secondary signals:
        - OCR keywords
    """
    mrz_lines, mrz_detected = detect_mrz_lines(image)

    text = normalize_ocr_text(ocr_text)

    passport_keywords = [
        "PASSPORT",
        "P<",
        "NATIONALITY",
        "DATE OF BIRTH",
        "DATE OF EXPIRY",
        "PERSONAL NUMBER",
    ]

    visa_keywords = [
        "VISA",
        "VISA TYPE",
        "DATE OF ISSUE",
        "NUMBER OF ENTRIES",
        "VALID FOR",
        "ENTRY",
    ]

    passport_score = 0
    visa_score = 0

    if mrz_detected:
        passport_score += 10

    for keyword in passport_keywords:
        if keyword in text:
            passport_score += 2

    for keyword in visa_keywords:
        if keyword in text:
            visa_score += 2

    if passport_score > visa_score and passport_score > 0:
        doc_type = "PASSPORT"
    elif visa_score > passport_score and visa_score > 0:
        doc_type = "VISA"
    else:
        doc_type = "UNKNOWN"

    return {
        "document_type": doc_type,
        "passport_score": passport_score,
        "visa_score": visa_score,
        "mrz_detected": mrz_detected,
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


def extract_text(thresh_img):
    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(thresh_img, config=config)

    data = pytesseract.image_to_data(
        thresh_img,
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

    if len(line2) < 44:
        results["error"] = (
            f"MRZ line 2 too short: {len(line2)} chars (expected 44)"
        )
        return results

    doc_number = line2[0:9]
    doc_check = line2[9]

    dob = line2[13:19]
    dob_check = line2[19]

    expiry = line2[21:27]
    expiry_check = line2[27]

    results["passport_number"] = doc_number.replace("<", "")

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
        results["dob_plausible"] = dob_date <= datetime.now()
        results["dob_parsed"] = dob_date.strftime("%Y-%m-%d")
    except ValueError:
        results["dob_plausible"] = False

    try:
        exp_date = datetime.strptime(expiry, "%y%m%d")
        results["not_expired"] = exp_date >= datetime.now()
        results["expiry_parsed"] = exp_date.strftime("%Y-%m-%d")
    except ValueError:
        results["not_expired"] = False

    return results


def extract_passport_data(image, ocr_text):
    mrz_lines, detected = detect_mrz_lines(image)

    result = {
        "mrz_lines": mrz_lines,
        "mrz_detected": detected,
        "passport_number": None,
        "mrz_validation": {},
        "ocr_text": ocr_text,
    }

    if len(mrz_lines) >= 2:
        validation = validate_td3_mrz(mrz_lines[0], mrz_lines[1])
        result["mrz_validation"] = validation
        result["passport_number"] = validation.get("passport_number")

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
    """
    OCR often confuses O and 0. Only normalize this for a
    passport-number-shaped value.
    """
    if not value:
        return None

    value = re.sub(r"[^A-Z0-9]", "", value.upper())

    if len(value) < 6:
        return None

    # For the mock database, passport numbers are X + digits.
    if value.startswith("X"):
        value = "X" + value[1:].replace("O", "0")

    return value


def extract_date_candidates(text):
    """
    Extract common visa date formats and normalize them to YYYY-MM-DD
    where possible.
    """
    candidates = []

    patterns = [
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
        r"\b\d{2}[-/]\d{2}[-/]\d{4}\b",
        r"\b\d{2}[-/]\d{2}[-/]\d{2}\b",
    ]

    for pattern in patterns:
        candidates.extend(re.findall(pattern, text))

    return candidates


def parse_date(value):
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d-%m-%y",
        "%d/%m/%y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


def extract_visa_data(ocr_text):
    text = normalize_ocr_text(ocr_text)

    result = {
        "passport_number": None,
        "visa_type": None,
        "date_of_expiry": None,
        "ocr_text": ocr_text,
    }

    # Passport number: first try explicit labels.
    passport_patterns = [
        r"(?:PASSPORT\s*(?:NO|NUMBER|#)?|PASSPORTNO)\s*[:\-]?\s*([A-Z][A-Z0-9<]{5,11})",
        r"(?:DOCUMENT\s*(?:NO|NUMBER|#))\s*[:\-]?\s*([A-Z][A-Z0-9<]{5,11})",
    ]

    for pattern in passport_patterns:
        match = re.search(pattern, text)

        if match:
            result["passport_number"] = normalize_passport_number(
                match.group(1)
            )
            break

    # Fallback: mock passport-number pattern such as X0001001.
    if not result["passport_number"]:
        match = re.search(r"\bX[0-9O]{6,8}\b", text)

        if match:
            result["passport_number"] = normalize_passport_number(
                match.group(0)
            )

    # Visa type.
    visa_types = ["TOURIST", "STUDENT", "BUSINESS"]

    for visa_type in visa_types:
        if visa_type in text:
            result["visa_type"] = visa_type.title()
            break

    # Expiry date.
    expiry_patterns = [
        r"(?:DATE\s*OF\s*EXPIRY|EXPIRY|EXPIRATION|VALID\s*UNTIL)"
        r"\s*[:\-]?\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{2}[-/]\d{2}[-/]\d{2})"
    ]

    for pattern in expiry_patterns:
        match = re.search(pattern, text)

        if match:
            result["date_of_expiry"] = parse_date(match.group(1))
            if result["date_of_expiry"]:
                break

    return result


# =========================================================
# STAGE 2C - FORENSIC / SUPPORTING CHECKS
# =========================================================

def check_font_consistency(data):
    heights = []

    for h, conf in zip(data["height"], data["conf"]):
        try:
            confidence = float(conf)
        except (ValueError, TypeError):
            continue

        if confidence > 30 and h > 0:
            heights.append(h)

    if len(heights) < 5:
        return {"insufficient_data": True}

    std_dev = np.std(heights)
    mean_h = np.mean(heights)

    cv_ratio = std_dev / mean_h if mean_h else 0

    return {
        "height_cv": round(float(cv_ratio), 3),
        "suspicious": cv_ratio > 0.35,
    }


def detect_copy_move_artifacts(gray_img):
    _, encoded = cv2.imencode(
        ".jpg",
        gray_img,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    recompressed = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

    diff = cv2.absdiff(gray_img, recompressed)

    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    return {
        "mean_error_level": round(float(mean_diff), 2),
        "max_error_level": int(max_diff),
        "suspicious_region": max_diff > 60,
    }


FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


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

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)

    x1 = min(img.shape[1], x + w + pad_x)
    y1 = min(img.shape[0], y + h + pad_y)

    return (x0, y0, x1, y1)


def analyze_photo_region(img, gray_full):
    box = locate_photo_region(img)

    if box is None:
        return {
            "photo_detected": False,
            "note": "No face/photo region located",
        }

    x0, y0, x1, y1 = box

    photo_gray = gray_full[y0:y1, x0:x1]

    if photo_gray.size == 0:
        return {
            "photo_detected": False,
            "note": "Photo region empty after crop",
        }

    ela = detect_copy_move_artifacts(photo_gray)

    edges = cv2.Canny(photo_gray, 50, 150)

    border_thickness = 3

    h, w = photo_gray.shape

    border_mask = np.zeros_like(edges)

    border_mask[:border_thickness, :] = 1
    border_mask[-border_thickness:, :] = 1
    border_mask[:, :border_thickness] = 1
    border_mask[:, -border_thickness:] = 1

    border_edge_density = (
        np.sum(edges * border_mask)
        / (np.sum(border_mask) * 255 + 1e-6)
    )

    return {
        "photo_detected": True,
        "photo_bbox": box,
        "photo_ela": ela,
        "border_edge_density": round(float(border_edge_density), 3),
        "suspicious_boundary": (
            border_edge_density > 0.25
            or ela.get("suspicious_region", False)
        ),
    }


def analyze_stamp_regions(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_ranges = [
        ((100, 50, 50), (130, 255, 255)),
        ((0, 50, 50), (10, 255, 255)),
        ((170, 50, 50), (180, 255, 255)),
        ((125, 40, 40), (160, 255, 255)),
    ]

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in color_ranges:
        mask |= cv2.inRange(
            hsv,
            np.array(lower),
            np.array(upper),
        )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    stamp_regions = [
        c for c in contours
        if cv2.contourArea(c) > 500
    ]

    if not stamp_regions:
        return {
            "stamps_detected": 0,
            "note": "No stamp-colored ink regions found",
        }

    results = []

    for c in stamp_regions:
        x, y, w, h = cv2.boundingRect(c)

        region_hsv = hsv[y:y + h, x:x + w]
        region_mask = mask[y:y + h, x:x + w]

        hues = region_hsv[:, :, 0][region_mask > 0]

        hue_std = float(np.std(hues)) if len(hues) > 0 else 0

        rect_perimeter = 2 * (w + h)
        contour_perimeter = cv2.arcLength(c, True)

        straightness_ratio = (
            contour_perimeter / rect_perimeter
            if rect_perimeter
            else 0
        )

        results.append({
            "bbox": (x, y, w, h),
            "hue_std": round(hue_std, 2),
            "straightness_ratio": round(straightness_ratio, 2),
            "suspicious": (
                hue_std > 25
                or straightness_ratio < 1.05
            ),
        })

    return {
        "stamps_detected": len(results),
        "stamp_details": results,
        "any_suspicious": any(
            r["suspicious"] for r in results
        ),
    }


# =========================================================
# OPTIONAL FACE MATCH
# =========================================================

def compare_faces(img_a, img_b):
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    def get_face_crop(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            1.1,
            5,
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

        crop = gray[y:y + h, x:x + w]

        return cv2.resize(crop, (200, 200))

    face_a = get_face_crop(img_a)
    face_b = get_face_crop(img_b)

    if face_a is None or face_b is None:
        return {
            "comparable": False,
            "note": "Could not detect a face in one or both images",
        }

    face_a = cv2.equalizeHist(face_a)
    face_b = cv2.equalizeHist(face_b)

    result = cv2.matchTemplate(
        face_a,
        face_b,
        cv2.TM_CCOEFF_NORMED,
    )

    similarity = float(result[0][0])

    return {
        "comparable": True,
        "similarity_score": round(similarity, 3),
        "likely_same_person": similarity > 0.5,
        "warning": "Coarse heuristic only",
    }


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
            "Passport number not found in Passport.json"
        )
        return result

    result["passport_found"] = True
    result["database_record"] = db_record

    mrz = passport_data.get("mrz_validation", {})

    if "dob_parsed" in mrz:
        result["field_matches"]["date_of_birth"] = (
            mrz["dob_parsed"] == db_record.get("date_of_birth")
        )

    if "expiry_parsed" in mrz:
        result["field_matches"]["date_of_expiry"] = (
            mrz["expiry_parsed"]
            == db_record.get("date_of_expiry")
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

    if (
        result["field_matches"].get("date_of_birth") is False
    ):
        result["issues"].append(
            "DOB does not match Passport.json"
        )

    if (
        result["field_matches"].get("date_of_expiry") is False
    ):
        result["issues"].append(
            "Passport expiry does not match Passport.json"
        )

    # For this project, a database record plus no detected mismatch
    # is considered a successful database verification.
    result["verified"] = (
        result["passport_found"]
        and len(result["issues"]) == 0
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
            "Visa/passport number not found in Visas.json"
        )
        return result

    result["visa_found"] = True
    result["database_record"] = db_record

    if visa_data.get("visa_type"):
        result["field_matches"]["visa_type"] = (
            visa_data["visa_type"].upper()
            == str(db_record.get("visa_type", "")).upper()
        )

    if visa_data.get("date_of_expiry"):
        result["field_matches"]["date_of_expiry"] = (
            visa_data["date_of_expiry"]
            == db_record.get("date_of_expiry")
        )

    if (
        result["field_matches"].get("visa_type") is False
    ):
        result["issues"].append(
            "Visa type does not match Visas.json"
        )

    if (
        result["field_matches"].get("date_of_expiry") is False
    ):
        result["issues"].append(
            "Visa expiry does not match Visas.json"
        )

    result["verified"] = (
        result["visa_found"]
        and len(result["issues"]) == 0
    )

    return result


# =========================================================
# STAGE 4 - STORAGE
# =========================================================

def store_verified_passport(passport_data, verification):
    """
    Updates the existing passport record only.
    It does NOT create an unknown passport record.
    """
    if not verification.get("verified"):
        return False, "Passport was not verified; storage skipped."

    passport_no = passport_data.get("passport_number")

    if not passport_no or passport_no not in passports:
        return False, "Passport record does not exist."

    record = passports[passport_no]

    mrz = passport_data.get("mrz_validation", {})

    if mrz.get("dob_parsed"):
        record["date_of_birth"] = mrz["dob_parsed"]

    if mrz.get("expiry_parsed"):
        record["date_of_expiry"] = mrz["expiry_parsed"]

    with open(PASSPORT_DB, "w", encoding="utf-8") as f:
        json.dump(passports, f, indent=2, ensure_ascii=False)

    return True, f"Passport record updated: {passport_no}"


def store_verified_visa(visa_data, verification):
    """
    Updates the existing visa record only.
    It does NOT create an unknown visa record.
    """
    if not verification.get("verified"):
        return False, "Visa was not verified; storage skipped."

    visa_id = verification.get("visa_id")

    if not visa_id or visa_id not in visas:
        return False, "Visa record does not exist."

    record = visas[visa_id]

    if visa_data.get("passport_number"):
        record["passport_no"] = visa_data["passport_number"]

    if visa_data.get("visa_type"):
        record["visa_type"] = visa_data["visa_type"]

    if visa_data.get("date_of_expiry"):
        record["date_of_expiry"] = visa_data["date_of_expiry"]

    with open(VISA_DB, "w", encoding="utf-8") as f:
        json.dump(visas, f, indent=2, ensure_ascii=False)

    return True, f"Visa record updated: {visa_id}"


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
):
    breakdown = {}

    mrz_validation = mrz_validation or {}
    font_check = font_check or {}
    forensics = forensics or {}
    photo_check = photo_check or {}
    stamp_check = stamp_check or {}
    verification = verification or {}

    if document_type == "PASSPORT":
        if mrz_validation.get("error"):
            breakdown["mrz"] = 30
        else:
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

    total = min(sum(breakdown.values()), 100)

    if total >= 60:
        verdict = "HIGH RISK — needs manual review"
    elif total >= 30:
        verdict = "MEDIUM RISK — verify manually"
    else:
        verdict = "LOW RISK — no obvious issues"

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

    font_check: dict = field(default_factory=dict)

    forensics: dict = field(default_factory=dict)

    photo_check: dict = field(default_factory=dict)

    stamp_check: dict = field(default_factory=dict)

    face_match: dict = field(default_factory=dict)

    risk_score: int = 0

    risk_breakdown: dict = field(default_factory=dict)

    verdict: str = "UNKNOWN"


# =========================================================
# COMPLETE 4-STAGE PIPELINE
# =========================================================

def analyze_document(
    image_path,
    reference_face_path=None,
    store=False,
):
    # -----------------------------------------------------
    # STAGE 1 - CLASSIFICATION
    # -----------------------------------------------------

    img, gray, thresh = preprocess_image(image_path)

    ocr_text, ocr_data = extract_text(thresh)

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
            ocr_text,
        )

    else:
        result.verdict = (
            "UNKNOWN DOCUMENT — could not confidently classify "
            "as passport or visa"
        )
        return result

    # Supporting checks for both document types.
    result.font_check = check_font_consistency(ocr_data)

    result.forensics = detect_copy_move_artifacts(gray)

    result.photo_check = analyze_photo_region(
        img,
        gray,
    )

    result.stamp_check = analyze_stamp_regions(img)

    # Optional identity check.
    if reference_face_path:
        ref_img = cv2.imread(reference_face_path)

        if ref_img is None:
            result.face_match = {
                "comparable": False,
                "note": (
                    f"Could not load '{reference_face_path}'"
                ),
            }
        else:
            result.face_match = compare_faces(
                img,
                ref_img,
            )
    else:
        result.face_match = {
            "comparable": False,
            "note": "No reference image provided",
        }

    # -----------------------------------------------------
    # STAGE 3 - VERIFICATION
    # -----------------------------------------------------

    if document_type == "PASSPORT":
        result.verification = verify_passport(
            result.passport_data
        )

        mrz_validation = result.passport_data.get(
            "mrz_validation",
            {},
        )

    else:
        result.verification = verify_visa(
            result.visa_data
        )

        mrz_validation = {}

    # -----------------------------------------------------
    # RISK
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
            "message": (
                "Storage not requested. Use --store after verification."
            ),
        }

    return result


# =========================================================
# OUTPUT
# =========================================================

def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(result):
    section("STAGE 1 - DOCUMENT CLASSIFICATION")

    print("Document Type:", result.document_type)

    print(
        "Passport Score:",
        result.classification.get("passport_score"),
    )

    print(
        "Visa Score:",
        result.classification.get("visa_score"),
    )

    print(
        "MRZ Detected:",
        result.classification.get("mrz_detected"),
    )

    section("STAGE 2 - EXTRACTION")

    if result.document_type == "PASSPORT":
        print(
            "Passport Number:",
            result.passport_data.get("passport_number"),
        )

        print(
            "MRZ Lines:",
            result.passport_data.get("mrz_lines"),
        )

        print(
            "MRZ Validation:",
            result.passport_data.get("mrz_validation"),
        )

    elif result.document_type == "VISA":
        print(
            "Passport Number:",
            result.visa_data.get("passport_number"),
        )

        print(
            "Visa Type:",
            result.visa_data.get("visa_type"),
        )

        print(
            "Visa Expiry:",
            result.visa_data.get("date_of_expiry"),
        )

    section("STAGE 3 - DATABASE VERIFICATION")

    print(
        "Verified:",
        result.verification.get("verified"),
    )

    if result.document_type == "PASSPORT":
        print(
            "Passport Found:",
            result.verification.get("passport_found"),
        )
    else:
        print(
            "Visa Found:",
            result.verification.get("visa_found"),
        )

        print(
            "Visa ID:",
            result.verification.get("visa_id"),
        )

    print(
        "Field Matches:",
        result.verification.get("field_matches"),
    )

    print(
        "Verification Issues:",
        result.verification.get("issues"),
    )

    if result.verification.get("database_record"):
        print("Database Record:")
        print(result.verification["database_record"])

    section("SUPPORTING CHECKS")

    print("Font Check:", result.font_check)
    print("Forensics:", result.forensics)
    print("Photo Check:", result.photo_check)
    print("Stamp Check:", result.stamp_check)
    print("Face Match:", result.face_match)

    section("STAGE 4 - STORAGE")

    print(
        "Storage:",
        result.verification.get("storage"),
    )

    section("RISK BREAKDOWN")

    for key, value in result.risk_breakdown.items():
        print(f"  {key}: {value}")

    section("FINAL RESULT")

    print(
        f"Risk Score: {result.risk_score}/100"
    )

    print(
        f"Verdict:    {result.verdict}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python ocr_4_stage.py <document_image>\n"
            "  python ocr_4_stage.py <document_image> <reference_face>\n"
            "  python ocr_4_stage.py <document_image> --store\n"
            "  python ocr_4_stage.py <document_image> <reference_face> --store"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    reference_face_path = None
    store = "--store" in sys.argv

    # Find optional reference image.
    for arg in sys.argv[2:]:
        if arg != "--store":
            reference_face_path = arg
            break

    print(f"Analyzing document: {image_path}")

    if reference_face_path:
        print(
            f"Reference face: {reference_face_path}"
        )

    if store:
        print("Storage mode: ENABLED")

    try:
        result = analyze_document(
            image_path=image_path,
            reference_face_path=reference_face_path,
            store=store,
        )

        print_result(result)

    except pytesseract.pytesseract.TesseractNotFoundError:
        print("\nERROR: Tesseract OCR was not found.")
        print(
            "Install Tesseract or set TESSERACT_PATH at the top "
            "of this file."
        )
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
