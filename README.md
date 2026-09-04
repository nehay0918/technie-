# Automated Document Bifurcation & Verification System

An intelligent, multi-stage document screening and verification system for **Passports** and **Visas**. Automatically classifies document types, extracts fields using OCR & MRZ (Machine Readable Zone) checksums, verifies records against local JSON databases (`Passport.json` and `Visas.json`), executes forensic tamper checks, performs facial identity verification, and stores verified records.

---

## Architecture & Features

```
Document Detection/
├── Frontend/
│   └── front.html          # Responsive 3-step web application UI
├── DataBase/
│   ├── Passport.json       # Passport database records
│   └── Visas.json          # Visa database records
├── Backhend/
│   ├── final1.py           # Core classification, extraction & verification engine
│   └── server.py           # Flask REST API server
├── Images/
│   ├── Passport/           # Sample passport documents
│   └── Visa/               # Sample visa documents
├── requirements.txt        # Project dependencies
└── server.py               # Root application entry point
```

### Key Modules:
1. **Document Bifurcation (Classification)**: Distinguishes Passports from Visas using ICAO Doc 9303 MRZ headers (`P<...` vs `V...`) and weighted contextual keywords.
2. **Field Extraction & Checksum Validation**: Validates standard 7-3-1 ICAO check digits for document number, date of birth, and expiry date.
3. **Database Verification**: Routes Passports to `DataBase/Passport.json` and Visas to `DataBase/Visas.json`.
4. **Forensics & Tamper Checks**: Font consistency analysis, JPEG Error Level Analysis (ELA) for copy-move splicing, photo boundary edge density, and stamp ink variance.
5. **Facial Identity Verification**: 1-to-1 comparison of live selfie against document photo (DeepFace with OpenCV Haar-cascade fallback).
6. **Database Storage**: Updates existing records or registers newly verified documents into the database.

---

## How to Run in GitHub Codespaces (Directly in Cloud Browser)

You can run this entire project in your browser without installing anything locally:

1. Open your repository on GitHub: `https://github.com/nehay0918/technie-`
2. Click the green **"<> Code"** button, select the **"Codespaces"** tab, and click **"Create codespace on main"**.
3. Once the cloud VS Code terminal opens, install dependencies:
   ```bash
   sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   python server.py
   ```
5. A popup will appear in the bottom right corner: **"Open in Browser"**. Click it to use the web application!

---

## How to Run Locally (Cloned from GitHub)

### 1. Prerequisites
- **Python 3.10+** installed
- **Tesseract OCR** installed:
  - **Windows**: Download from [UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) (default install to `C:\Program Files\Tesseract-OCR`)
  - **Linux (Ubuntu/Debian)**: `sudo apt install tesseract-ocr`
  - **macOS**: `brew install tesseract`

### 2. Clone the Repository
```bash
git clone https://github.com/nehay0918/technie-.git
cd technie-
```

### 3. Create & Activate Virtual Environment
```bash
# Windows:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Web Application
```bash
python server.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## Command-Line Usage (CLI)

```bash
# Basic check on a document
python final1.py "Images/Passport/pass.jpg"

# Check with facial identity match (Selfie vs Document)
python final1.py "Images/Passport/pass.jpg" "Images/Passport/pass.jpg"

# Verify and save record into DataBase/ folder
python final1.py "Images/Visa/WhatsApp Image 2026-09-04 at 13.21.07.jpeg" --store

# Export analysis as JSON
python final1.py "Images/Passport/pass.jpg" --json
```