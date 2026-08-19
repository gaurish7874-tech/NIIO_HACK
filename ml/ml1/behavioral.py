"""
behavioral.py
-------------
Main entry point for ML Behavioral Extraction Pipeline.
Provides `extract_behavioral(frame_buffer, fps=30.0)` for FastAPI backend integration.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ml.ml1.blink import BlinkTracker
from ml.ml1.gaze import GazeTracker
from ml.ml1.head_pose import estimate_head_pose

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_landmarker.task")
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


def extract_behavioral(frame_buffer: list, fps: float = 30.0) -> dict:
    """
    FastAPI-ready main behavioral extraction function.

    Parameters:
        frame_buffer (list): List of OpenCV BGR frames (e.g. 150-300 frames captured at 30 FPS).
        fps (float): Video sampling rate (Frames Per Second).

    Returns:
        dict containing:
            {
                "blink_rate": float,          # Blinks Per Minute (BPM)
                "eye_closure": float,         # Eye closure ratio (0.0 to 1.0)
                "gaze_stability": float,      # Gaze stability index (0.0 to 1.0)
                "head_pose": {
                    "pitch": float,           # Head Pitch (degrees)
                    "yaw": float,             # Head Yaw (degrees)
                    "roll": float             # Head Roll (degrees)
                },
                "confidence": float,          # Quality/Confidence Score (0.0 to 1.0)
                "signal_available": bool      # True if face signal was detected
            }
    """
    # -------------------------------------------------------------------
    # 1. Graceful Input & Buffer Validation
    # -------------------------------------------------------------------
    default_response = {
        "blink_rate": 0.0,
        "eye_closure": 0.0,
        "gaze_stability": 0.0,
        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "confidence": 0.0,
        "signal_available": False
    }

    if not isinstance(frame_buffer, (list, tuple, np.ndarray)) or len(frame_buffer) < 10:
        return default_response

    num_frames = len(frame_buffer)
    landmarker = _get_landmarker()

    blink_tracker = BlinkTracker()
    gaze_tracker = GazeTracker(history_size=num_frames)

    face_detected_count = 0

    pitches = []
    yaws = []
    rolls = []

    last_gaze_info = {'gaze_direction': 'UNKNOWN', 'gaze_x': 0.5, 'gaze_y': 0.5, 'gaze_stability': 0.0}

    # -------------------------------------------------------------------
    # 2. Process Video Frame Buffer
    # -------------------------------------------------------------------
    for frame in frame_buffer:
        if frame is None:
            continue

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = landmarker.detect(mp_image)

        if detection_result.face_landmarks:
            face_detected_count += 1
            face_landmarks = detection_result.face_landmarks[0]

            # A. Process Blink
            blink_tracker.process_frame(face_landmarks, width, height)

            # B. Process Gaze
            last_gaze_info = gaze_tracker.process_frame(face_landmarks, width, height)

            # C. Process Head Pose
            pose_info = estimate_head_pose(face_landmarks, width, height)
            pitches.append(pose_info['pitch'])
            yaws.append(pose_info['yaw'])
            rolls.append(pose_info['roll'])

    # -------------------------------------------------------------------
    # 3. Handle No-Face & Low Face-Detection Scenarios
    # -------------------------------------------------------------------
    face_ratio = face_detected_count / float(num_frames)

    if face_ratio < 0.3:
        return default_response

    duration_sec = num_frames / float(fps)

    # Calculate metrics
    blink_rate = blink_tracker.get_blink_rate(duration_sec)
    eye_closure = blink_tracker.get_eye_closure_ratio()
    gaze_stability = last_gaze_info['gaze_stability']

    mean_pitch = round(float(np.mean(pitches)), 1) if len(pitches) > 0 else 0.0
    mean_yaw = round(float(np.mean(yaws)), 1) if len(yaws) > 0 else 0.0
    mean_roll = round(float(np.mean(rolls)), 1) if len(rolls) > 0 else 0.0

    # Calculate Confidence Score
    confidence = round(float(np.clip(face_ratio * (0.5 + 0.5 * gaze_stability), 0.0, 1.0)), 2)

    return {
        "blink_rate": blink_rate,
        "eye_closure": eye_closure,
        "gaze_stability": gaze_stability,
        "head_pose": {
            "pitch": mean_pitch,
            "yaw": mean_yaw,
            "roll": mean_roll
        },
        "confidence": confidence,
        "signal_available": True
    }


# -----------------------------------------------------------------------
# Offline Pipeline Verification Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing extract_behavioral(frame_buffer)...")

    dummy_frames = []
    for _ in range(30):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frames.append(img)

    result = extract_behavioral(dummy_frames, fps=30.0)
    print("Behavioral Output Structure:", result)
    assert "blink_rate" in result and "head_pose" in result and "confidence" in result
    print("[SUCCESS] extract_behavioral pipeline interface verified!")
