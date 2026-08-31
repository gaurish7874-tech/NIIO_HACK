import sys
import logging
import json
import os

# add to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(".env")
logging.basicConfig(level=logging.DEBUG)

from ml.ml2.triage import _run_physio_agent

physio_data = {
    "hr": 105.0,
    "hrv": 18.0,
    "respiration": 22.0,
    "confidence": 0.85
}

try:
    print(_run_physio_agent(physio_data))
except Exception as e:
    print("Error:", e)
