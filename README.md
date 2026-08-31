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

## Backend API

Install the backend dependencies, then start the API from the project root:

```text
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --reload --reload-exclude "venv/*"
```

The API provides `GET /health`, session creation/deletion, video analysis,
clarification, chat, calibration, timeline, and report endpoints. Create a
session with `POST /sessions`, then upload a short video to
`POST /sessions/{session_id}/analyze` as multipart field `file`.

For a basic camera test, open `http://127.0.0.1:8000/test` after starting
Uvicorn. Click **Allow Camera**, **Start Recording**, and then wait for the
ML1/ML2 JSON response. The page records for 10 seconds automatically.

Analysis requires at least 30 decoded video frames. The first backend version
uses one active in-memory session because ML-2 currently owns global timeline
and calibration state. This system provides informational wellness insights,
not medical diagnosis or treatment.
