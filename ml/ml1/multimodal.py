"""
multimodal.py
-------------
Top-level ML-1 Multimodal Wellness Extraction pipeline combining both Physiological (rPPG, HR, HRV, Respiration)
and Behavioral (Blink, Gaze, Head Pose) features into a unified FastAPI-ready JSON response.
"""

import os
import sys
import time

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.ml1.physio import extract_physio
from ml.ml1.behavioral import extract_behavioral


def analyze_multimodal_wellness(frame_buffer: list, fps: float = 30.0) -> dict:
    """
    Combines ML-1 Physiological and Behavioral pipelines into a single unified output.

    Parameters:
        frame_buffer (list): List of OpenCV BGR video frames (e.g. 150-300 frames captured at 30 FPS).
        fps (float): Video sampling rate (Frames Per Second).

    Returns:
        dict matching the exact project output contract:
            {
                "timestamp": "2026-08-19T16:59:13Z",
                "physio": {
                    "hr": float or None,
                    "hrv": float or None,
                    "respiration": float or None,
                    "confidence": float
                },
                "behavioral": {
                    "blink_rate": float,
                    "eye_closure": float,
                    "gaze_stability": float,
                    "head_pose": {
                        "pitch": float,
                        "yaw": float,
                        "roll": float
                    },
                    "confidence": float
                }
            }
    """
    # Generate ISO 8601 UTC timestamp
    iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Execute Physiological Pipeline
    physio_result = extract_physio(frame_buffer, fps=fps)

    # 2. Execute Behavioral Pipeline
    behavioral_result = extract_behavioral(frame_buffer, fps=fps)

    # 3. Combine into final target JSON structure
    return {
        "timestamp": iso_timestamp,
        "physio": {
            "hr": physio_result.get("hr"),
            "hrv": physio_result.get("hrv"),
            "respiration": physio_result.get("respiration"),
            "confidence": physio_result.get("confidence", 0.0)
        },
        "behavioral": {
            "blink_rate": behavioral_result.get("blink_rate", 0.0),
            "eye_closure": behavioral_result.get("eye_closure", 0.0),
            "gaze_stability": behavioral_result.get("gaze_stability", 0.0),
            "head_pose": {
                "pitch": behavioral_result.get("head_pose", {}).get("pitch", 0.0),
                "yaw": behavioral_result.get("head_pose", {}).get("yaw", 0.0),
                "roll": behavioral_result.get("head_pose", {}).get("roll", 0.0)
            },
            "confidence": behavioral_result.get("confidence", 0.0)
        }
    }


# -----------------------------------------------------------------------
# Offline Module Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    print("Testing analyze_multimodal_wellness(frame_buffer)...")
    dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(30)]

    output = analyze_multimodal_wellness(dummy_frames, fps=30.0)
    import json

    print("Generated Output Schema:\n", json.dumps(output, indent=2))
    assert "timestamp" in output and "physio" in output and "behavioral" in output
    print("[SUCCESS] Multimodal pipeline contract verified!")
