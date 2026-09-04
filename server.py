"""
Document Bifurcation, Verification & Storage Server
===================================================

Flask server providing:
  - Web UI for frontend check-in (Frontend/front.html)
  - /api/verify: Document classification (Passport vs Visa), OCR & MRZ parsing,
                 database verification (Passport.json & Visas.json),
                 facial identity check, tamper forensics, and storage.
  - /api/database: Live records in Passport.json and Visas.json.
  - /api/samples: Pre-configured sample passports and visas for instant testing.
"""

import os
import sys
import json
import tempfile
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "Backhend"))

import final1

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "Frontend"))


# =========================================================
# STATIC & FRONTEND ROUTES
# =========================================================

@app.route("/")
@app.route("/front.html")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "Frontend"), "front.html")


@app.route("/Images/<path:filename>")
def serve_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, "Images"), filename)


@app.route("/Frontend/<path:filename>")
def serve_frontend(filename):
    return send_from_directory(os.path.join(BASE_DIR, "Frontend"), filename)


# =========================================================
# SAMPLE DOCUMENTS LIST
# =========================================================

@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples = [
        {
            "id": "sample_pass_1",
            "category": "passport",
            "label": "Passport: Rahul Kumar",
            "path": "Images/Passport/WhatsApp Image 2026-09-04 at 13.21.06 (1).jpeg",
            "url": "/Images/Passport/WhatsApp Image 2026-09-04 at 13.21.06 (1).jpeg",
            "preset": {
                "name": "Rahul Kumar",
                "passport_no": "X0001006",
                "dob": "2002-05-15",
                "expiry": "2032-05-15",
            },
        },
        {
            "id": "sample_pass_2",
            "category": "passport",
            "label": "Passport: Vihaan Shah",
            "path": "Images/Passport/WhatsApp Image 2026-09-04 at 13.21.05.jpeg",
            "url": "/Images/Passport/WhatsApp Image 2026-09-04 at 13.21.05.jpeg",
            "preset": {
                "name": "Vihaan Shah",
                "passport_no": "X0001009",
                "dob": "1998-08-26",
                "expiry": "2030-10-14",
            },
        },
        {
            "id": "sample_visa_1",
            "category": "visa",
            "label": "Visa: Rahul Kumar (Student)",
            "path": "Images/Visa/WhatsApp Image 2026-09-04 at 13.21.07.jpeg",
            "url": "/Images/Visa/WhatsApp Image 2026-09-04 at 13.21.07.jpeg",
            "preset": {
                "name": "Rahul Kumar",
                "passport_no": "X0001006",
                "expiry": "2029-01-19",
            },
        },
        {
            "id": "sample_visa_2",
            "category": "visa",
            "label": "Visa: US B1/B2 (Smith)",
            "path": "Images/Passport/pass.jpg",
            "url": "/Images/Passport/pass.jpg",
            "preset": {
                "name": "John Smith",
                "passport_no": "CZ6511T47",
                "expiry": "2034-01-23",
            },
        },
        {
            "id": "sample_visa_3",
            "category": "visa",
            "label": "Visa: Tourist (Vihaan)",
            "path": "Images/Visa/WhatsApp Image 2026-09-04 at 13.21.06.jpeg",
            "url": "/Images/Visa/WhatsApp Image 2026-09-04 at 13.21.06.jpeg",
            "preset": {
                "passport_no": "X0001009",
                "expiry": "2027-10-09",
            },
        },
    ]
    return jsonify({"success": True, "samples": samples})


# =========================================================
# DATABASE VIEWER ROUTE
# =========================================================

@app.route("/api/database", methods=["GET"])
def get_database():
    try:
        passports = final1.load_json_database(final1.PASSPORT_DB)
        visas = final1.load_json_database(final1.VISA_DB)
        return jsonify({
            "success": True,
            "passports": passports,
            "visas": visas,
            "passport_db_path": final1.PASSPORT_DB,
            "visa_db_path": final1.VISA_DB,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================================================
# MAIN VERIFICATION & STORAGE API
# =========================================================

@app.route("/api/verify", methods=["POST"])
def verify_document_endpoint():
    temp_dir = tempfile.mkdtemp(prefix="doc_check_")
    doc_path = None
    live_path = None

    try:
        # 1. Resolve Document Image (Upload or Sample Path)
        if "docPhoto" in request.files and request.files["docPhoto"].filename:
            doc_file = request.files["docPhoto"]
            doc_path = os.path.join(temp_dir, doc_file.filename)
            doc_file.save(doc_path)
        elif "sampleDocPath" in request.form and request.form["sampleDocPath"]:
            sample_rel = request.form["sampleDocPath"].replace("\\", "/")
            doc_path = os.path.join(BASE_DIR, sample_rel)
            if not os.path.exists(doc_path):
                return jsonify({"success": False, "error": f"Sample image not found: {sample_rel}"}), 400
        else:
            return jsonify({"success": False, "error": "No document image provided"}), 400

        # 2. Resolve Live Selfie Image (Optional)
        if "livePhoto" in request.files and request.files["livePhoto"].filename:
            live_file = request.files["livePhoto"]
            live_path = os.path.join(temp_dir, live_file.filename)
            live_file.save(live_path)

        # 3. Gather Applicant Form Data
        form_data = {
            "fullName": request.form.get("fullName", "").strip(),
            "passportNo": request.form.get("passportNo", "").strip(),
            "nationality": request.form.get("nationality", "").strip(),
            "dob": request.form.get("dob", "").strip(),
            "expiry": request.form.get("expiry", "").strip(),
            "phone": request.form.get("phone", "").strip(),
            "email": request.form.get("email", "").strip(),
        }

        # 4. Storage mode
        store_mode = request.form.get("store", "true").lower() in ["true", "1", "yes"]

        # 5. Run 4-Stage Pipeline + Supporting Forensics
        result = final1.analyze_document(
            image_path=doc_path,
            reference_face_path=live_path,
            store=store_mode,
            form_data=form_data,
        )

        # 6. Prepare JSON Response
        extracted_info = (
            result.passport_data
            if result.document_type == "PASSPORT"
            else result.visa_data
        )

        db_found = bool(
            result.verification.get("passport_found")
            or result.verification.get("visa_found")
        )

        response_payload = {
            "success": True,
            "document_type": result.document_type,
            "bifurcation": result.classification,
            "extracted_data": extracted_info,
            "verification": {
                "verified": bool(result.verification.get("verified", False)),
                "database_found": db_found,
                "record": result.verification.get("database_record"),
                "field_matches": result.verification.get("field_matches", {}),
                "issues": result.verification.get("issues", []),
            },
            "form_verification": result.form_verification,
            "forensics": {
                "font_check": result.font_check,
                "copy_move": result.forensics,
                "photo_check": result.photo_check,
                "stamp_check": result.stamp_check,
            },
            "face_match": result.face_match,
            "risk": {
                "score": result.risk_score,
                "breakdown": result.risk_breakdown,
                "verdict": result.verdict,
            },
            "storage": result.verification.get("storage", {}),
        }

        return jsonify(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        # Clean up temporary upload files
        try:
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
        except Exception:
            pass


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================")
    print(f" Document Bifurcation & Verification Server")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f" Open http://127.0.0.1:{port} in your browser")
    print(f"==================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
