"""
physio.py
---------
Main entry point for ML-1 physiological extraction pipeline.
Provides `extract_physio(frame_buffer, fps=30.0)` for FastAPI backend integration.
"""

import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp

# Ensure project root is in Python path for standalone script execution & backend imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ml.ml1.roi import extract_forehead_roi
from ml.ml1.signal import RGBSignalBuffer
from ml.ml1.pos import extract_pos_pulse
from ml.ml1.filtering import apply_bandpass_filter
from ml.ml1.heart_rate import calculate_heart_rate
from ml.ml1.hrv import calculate_hrv_rmssd
from ml.ml1.confidence import calculate_confidence_score

# Locate face_landmarker.task model file
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_landmarker.task")

# Global lazy-loaded MediaPipe FaceLandmarker instance for performance
_LANDMARKER_INSTANCE = None

def _get_landmarker():
    """
    Singleton lazy-initializer for MediaPipe FaceLandmarker.
    """
    global _LANDMARKER_INSTANCE
    if _LANDMARKER_INSTANCE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model asset not found at {MODEL_PATH}")

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        _LANDMARKER_INSTANCE = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER_INSTANCE


def extract_respiration_rate(raw_pulse: np.ndarray, fps: float = 30.0):
    """
    Extracts respiration rate (in Breaths Per Minute - RPM) from low-frequency modulation of the rPPG signal.

    Respiration frequency range: 0.12 Hz (7.2 RPM) to 0.45 Hz (27 RPM).
    """
    if len(raw_pulse) < int(fps * 5.0):
        return None

    num_samples = len(raw_pulse)
    fft_spectrum = np.abs(np.fft.rfft(raw_pulse))
    fft_freqs = np.fft.rfftfreq(num_samples, d=1.0/fps)

    resp_idx = np.where((fft_freqs >= 0.12) & (fft_freqs <= 0.45))[0]

    if len(resp_idx) == 0:
        return None

    resp_spectrum = fft_spectrum[resp_idx]
    max_idx = resp_idx[np.argmax(resp_spectrum)]
    best_freq = fft_freqs[max_idx]

    respiration_rpm = round(float(best_freq * 60.0), 1)

    if 8.0 <= respiration_rpm <= 25.0:
        return respiration_rpm

    return None


def extract_physio(frame_buffer: list, fps: float = 30.0) -> dict:
    """
    FastAPI-ready main physiological extraction function.

    Parameters:
        frame_buffer (list): List of OpenCV BGR video frames (e.g. 150-300 frames captured at 30 FPS).
        fps (float): Video sampling rate (Frames Per Second).

    Returns:
        dict containing:
            {
                "hr": float or None,          # Heart Rate in BPM
                "hrv": float or None,         # Experimental RMSSD in ms
                "respiration": float or None, # Respiration Rate in RPM
                "confidence": float           # Quality/Confidence Score (0.0 to 1.0)
            }
    """
    # -------------------------------------------------------------------
    # 1. Graceful Input & Buffer Validation
    # -------------------------------------------------------------------
    if not isinstance(frame_buffer, (list, tuple, np.ndarray)) or len(frame_buffer) == 0:
        return {
            "hr": None,
            "hrv": None,
            "respiration": None,
            "confidence": 0.0
        }

    num_frames = len(frame_buffer)

    # Minimum frames required for meaningful signal processing (~1.5 seconds at 30 FPS)
    if num_frames < 30:
        return {
            "hr": None,
            "hrv": None,
            "respiration": None,
            "confidence": 0.0
        }

    # -------------------------------------------------------------------
    # 2. Extract Forehead ROI & RGB Time-Series
    # -------------------------------------------------------------------
    landmarker = _get_landmarker()
    signal_buffer = RGBSignalBuffer(max_samples=num_frames)

    face_detected_count = 0
    roi_valid_count = 0

    for idx, frame in enumerate(frame_buffer):
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = landmarker.detect(mp_image)

        if detection_result.face_landmarks:
            face_detected_count += 1
            face_landmarks = detection_result.face_landmarks[0]

            roi_crop, _, _ = extract_forehead_roi(frame, face_landmarks)

            if roi_crop is not None and roi_crop.size > 0:
                roi_valid_count += 1
                signal_buffer.add_frame_sample(roi_crop, timestamp=idx / fps)

    # -------------------------------------------------------------------
    # 3. Handle No-Face & Low Face-Detection Scenarios
    # -------------------------------------------------------------------
    face_ratio = face_detected_count / float(num_frames)
    roi_ratio = roi_valid_count / float(num_frames)

    # Never return fake values if face is missing in > 50% of frames
    if face_ratio < 0.5 or len(signal_buffer) < 30:
        return {
            "hr": None,
            "hrv": None,
            "respiration": None,
            "confidence": 0.0
        }

    r, g, b, t = signal_buffer.get_signal_arrays()
    rgb_matrix = np.column_stack((r, g, b))

    # -------------------------------------------------------------------
    # 4. Execute Signal Processing Pipeline Stages
    # Stage A: POS Algorithm
    # Stage B: Bandpass Filter (0.7 Hz to 4.0 Hz)
    # Stage C: Heart Rate Extraction
    # Stage D: Experimental HRV (RMSSD)
    # Stage E: Respiration Rate
    # -------------------------------------------------------------------
    raw_pulse = extract_pos_pulse(rgb_matrix, fps=fps)
    filtered_pulse = apply_bandpass_filter(raw_pulse, fps=fps, lowcut=0.7, highcut=4.0)

    hr_info = calculate_heart_rate(filtered_pulse, fps=fps)
    hrv_info = calculate_hrv_rmssd(hr_info['peak_positions'], fps=fps)
    respiration_rpm = extract_respiration_rate(raw_pulse, fps=fps)

    # -------------------------------------------------------------------
    # 5. Compute Dynamic Confidence Score
    # -------------------------------------------------------------------
    confidence = calculate_confidence_score(
        face_detected=(face_ratio >= 0.5),
        roi_valid=(roi_ratio >= 0.5),
        buffer_length=len(signal_buffer),
        signal_quality=hr_info['signal_quality'],
        peak_intervals=hr_info.get('peak_intervals', []),
        max_buffer_size=300
    )

    # Validate output values: set to None if confidence is too low (< 0.15) or HR is 0
    final_hr = hr_info['heart_rate'] if (hr_info['heart_rate'] > 0 and confidence >= 0.15) else None
    final_hrv = hrv_info['rmssd_ms'] if (hrv_info['rmssd_ms'] is not None and confidence >= 0.15) else None
    final_resp = respiration_rpm if (respiration_rpm is not None and confidence >= 0.15) else None

    return {
        "hr": final_hr,
        "hrv": final_hrv,
        "respiration": final_resp,
        "confidence": confidence
    }


# -----------------------------------------------------------------------
# Offline Pipeline Verification Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing extract_physio(frame_buffer)...")

    dummy_frames = []
    for _ in range(60):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:300, 200:400] = [120, 150, 200]
        dummy_frames.append(img)

    result = extract_physio(dummy_frames, fps=30.0)
    print("Test Result Output Structure:", result)
    assert "hr" in result and "hrv" in result and "respiration" in result and "confidence" in result
    print("[SUCCESS] extract_physio pipeline interface verified!")
