# Contactless Multimodal Wellness Monitor

A hackathon project aimed at contactless wellness monitoring using computer vision and signal processing.

## Project Structure

```
project/
├── backend/
├── frontend/
├── ml/
│   ├── ml1/            # Facial landmark detection & rPPG physiological signal extraction
│   └── ml2/            # Triage model
├── .env
├── README.md
└── requirements.txt
```

## Setup Instructions

1. Create a Python Virtual Environment:
   `python -m venv venv`
2. Activate Virtual Environment:
   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
   - Windows (CMD): `.\venv\Scripts\activate.bat`
   - macOS/Linux: `source venv/bin/activate`
3. Install Dependencies:
   `pip install -r requirements.txt`
4. Run Camera Test:
   `python ml/ml1/camera_test.py`
