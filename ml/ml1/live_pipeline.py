"""
live_pipeline.py
----------------
Full real-time ML-1 physiological pipeline demo running on live webcam feed.
Simultaneously executes:
1. MediaPipe FaceLandmarker detection
2. Forehead ROI extraction
3. Spatial RGB signal buffering
4. POS algorithm
5. SciPy Bandpass filtering
6. Heart Rate extraction (BPM)
7. Experimental HRV calculation (RMSSD in ms)
8. Respiration Rate extraction (RPM)
9. Dynamic Confidence scoring
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

from ml.ml1.roi import extract_forehead_roi
from ml.ml1.signal import RGBSignalBuffer
from ml.ml1.pos import extract_pos_pulse
from ml.ml1.filtering import apply_bandpass_filter
from ml.ml1.heart_rate import calculate_heart_rate
from ml.ml1.hrv import calculate_hrv_rmssd
from ml.ml1.confidence import calculate_confidence_score
from ml.ml1.physio import extract_respiration_rate

CAMERA_INDEX = 1
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_landmarker.task")


def run_live_pipeline():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    # Initialize FaceLandmarker in VIDEO mode
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    signal_buffer = RGBSignalBuffer(max_samples=300)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        return

    print("================================================================")
    print("      ML-1 PHYSIOLOGICAL PIPELINE (rPPG + HR + HRV + CONF)       ")
    print("================================================================")
    print(f"[SUCCESS] Camera {CAMERA_INDEX} & ML-1 Pipeline initialized!")
    print("Press 'q' in the window to exit.")

    prev_time = time.time()
    start_time = time.time()
    fps = 30.0
    prev_hr = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        current_time = time.time()
        frame_timestamp_ms = int((current_time - start_time) * 1000)

        # Convert to RGB & process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        face_detected = False
        roi_valid = False
        mean_r, mean_g, mean_b = None, None, None

        if detection_result.face_landmarks:
            face_detected = True
            face_landmarks = detection_result.face_landmarks[0]

            roi_crop, polygon_pts, bbox = extract_forehead_roi(frame, face_landmarks)

            if roi_crop is not None and bbox is not None:
                roi_valid = True
                x, y, w, h = bbox
                mean_r, mean_g, mean_b = signal_buffer.add_frame_sample(roi_crop, timestamp=current_time)

                # Draw Visual ROI Overlays
                cv2.polylines(frame, [polygon_pts], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(frame, "Forehead ROI", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Compute stable FPS from buffer timestamps
        time_diff = current_time - prev_time
        if time_diff > 0:
            inst_fps = 1.0 / time_diff
            fps = float(np.clip(inst_fps, 10.0, 60.0))
        prev_time = current_time

        # Retrieve buffer contents
        r, g, b, t = signal_buffer.get_signal_arrays()
        buffer_length = len(t)

        hr_display = "Calculating..."
        hrv_display = "Calculating..."
        resp_display = "Calculating..."
        confidence_val = 0.0

        if buffer_length >= 30:
            rgb_matrix = np.column_stack((r, g, b))

            # POS + Bandpass
            raw_pulse = extract_pos_pulse(rgb_matrix, fps=fps)
            filtered_pulse = apply_bandpass_filter(raw_pulse, fps=fps, lowcut=0.7, highcut=4.0)

            # HR Extraction
            hr_info = calculate_heart_rate(filtered_pulse, fps=fps, prev_hr=prev_hr, smoothing_factor=0.3)
            if hr_info['heart_rate'] > 0:
                prev_hr = hr_info['heart_rate']

            # Experimental HRV
            hrv_info = calculate_hrv_rmssd(hr_info['peak_positions'], fps=fps)

            # Respiration Rate
            resp_rpm = extract_respiration_rate(raw_pulse, fps=fps)

            # Confidence Score
            confidence_val = calculate_confidence_score(
                face_detected=face_detected,
                roi_valid=roi_valid,
                buffer_length=buffer_length,
                signal_quality=hr_info['signal_quality'],
                peak_intervals=hr_info.get('peak_intervals', []),
                max_buffer_size=300
            )

            if hr_info['heart_rate'] > 0 and confidence_val >= 0.15:
                hr_display = f"{hr_info['heart_rate']} BPM"
            else:
                hr_display = "Low Confidence"

            if hrv_info['rmssd_ms'] is not None and confidence_val >= 0.15:
                hrv_display = f"{hrv_info['rmssd_ms']} ms (Exp)"
            else:
                hrv_display = "N/A"

            if resp_rpm is not None and confidence_val >= 0.15:
                resp_display = f"{resp_rpm} RPM"
            else:
                resp_display = "N/A"
        else:
            confidence_val = calculate_confidence_score(
                face_detected=face_detected,
                roi_valid=roi_valid,
                buffer_length=buffer_length,
                signal_quality=0.0
            )

        # -------------------------------------------------------------
        # Live HUD Telemetry Overlay
        # -------------------------------------------------------------
        # Black background panel for HUD text
        cv2.rectangle(frame, (10, 10), (320, 210), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 210), (0, 255, 0), 1)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Face: {'YES' if face_detected else 'NO'}", (150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if face_detected else (0, 0, 255), 1)
        cv2.putText(frame, f"Buffer: {buffer_length}/300", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.putText(frame, f"Heart Rate: {hr_display}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"HRV (RMSSD): {hrv_display}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1)
        cv2.putText(frame, f"Respiration: {resp_display}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        # Color-coded confidence bar
        conf_color = (0, 255, 0) if confidence_val >= 0.7 else ((0, 255, 255) if confidence_val >= 0.4 else (0, 0, 255))
        cv2.putText(frame, f"Confidence: {confidence_val:.2f}", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

        cv2.imshow("ML-1 Physiological Pipeline Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting live pipeline...")
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Pipeline closed cleanly.")


if __name__ == "__main__":
    run_live_pipeline()
