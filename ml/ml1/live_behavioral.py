"""
live_behavioral.py
------------------
Full real-time ML Behavioral pipeline demo running on Camera 1.
Renders on-screen HUD for:
1. EAR (Eye Aspect Ratio) & Blink count
2. Eye state (OPEN / CLOSED)
3. Gaze direction (LEFT / RIGHT / CENTER / UP / DOWN) & Gaze Stability
4. Head Pose (Pitch, Yaw, Roll angles)
5. 3D nose direction vector overlay on video feed
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ml.ml1.blink import BlinkTracker
from ml.ml1.gaze import GazeTracker
from ml.ml1.head_pose import estimate_head_pose, draw_head_pose_axis

CAMERA_INDEX = 1
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_landmarker.task")


def run_live_behavioral():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    blink_tracker = BlinkTracker()
    gaze_tracker = GazeTracker(history_size=30)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        return

    print("================================================================")
    print("      ML BEHAVIORAL PIPELINE (BLINK + GAZE + HEAD POSE)         ")
    print("================================================================")
    print(f"[SUCCESS] Camera {CAMERA_INDEX} & Behavioral Pipeline initialized!")
    print("Press 'q' in the window to exit.")

    prev_time = time.time()
    start_time = time.time()
    fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        current_time = time.time()
        frame_timestamp_ms = int((current_time - start_time) * 1000)

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        face_detected = False
        blink_info = {'ear': 0.0, 'eye_state': 'N/A', 'total_blinks': 0}
        gaze_info = {'gaze_direction': 'N/A', 'gaze_stability': 0.0}
        pose_info = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

        if detection_result.face_landmarks:
            face_detected = True
            face_landmarks = detection_result.face_landmarks[0]

            # 1. Blink
            blink_info = blink_tracker.process_frame(face_landmarks, width, height)

            # 2. Gaze
            gaze_info = gaze_tracker.process_frame(face_landmarks, width, height)

            # 3. Head Pose & 3D vector drawing
            pose_info = estimate_head_pose(face_landmarks, width, height)
            frame = draw_head_pose_axis(frame, pose_info)

        # FPS
        time_diff = current_time - prev_time
        if time_diff > 0:
            fps = float(np.clip(1.0 / time_diff, 10.0, 60.0))
        prev_time = current_time

        duration_sec = current_time - start_time
        blink_rate = blink_tracker.get_blink_rate(duration_sec)
        eye_closure = blink_tracker.get_eye_closure_ratio()

        # -------------------------------------------------------------
        # Live HUD Telemetry Overlay
        # -------------------------------------------------------------
        cv2.rectangle(frame, (10, 10), (360, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (360, 220), (255, 255, 0), 1)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"Face: {'YES' if face_detected else 'NO'}", (160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if face_detected else (0, 0, 255), 1)

        # Blink Telemetry
        ear_str = f"EAR: {blink_info['ear']:.2f} ({blink_info['eye_state']})"
        cv2.putText(frame, ear_str, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(frame, f"Blinks: {blink_info['total_blinks']} | Rate: {blink_rate} BPM", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(frame, f"Eye Closure Ratio: {eye_closure:.2f}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        # Gaze Telemetry
        gaze_str = f"Gaze: {gaze_info['gaze_direction']} (Stab: {gaze_info['gaze_stability']:.2f})"
        cv2.putText(frame, gaze_str, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1)

        # Head Pose Telemetry
        pose_str = f"Pose: P:{pose_info['pitch']} Y:{pose_info['yaw']} R:{pose_info['roll']}"
        cv2.putText(frame, pose_str, (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 1)

        cv2.imshow("ML Behavioral Pipeline Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting live behavioral demo...")
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Behavioral pipeline closed cleanly.")


if __name__ == "__main__":
    run_live_behavioral()
