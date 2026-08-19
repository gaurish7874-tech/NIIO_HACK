"""
roi_test.py
------------
Webcam test script using MediaPipe Tasks API (FaceLandmarker),
Forehead ROI extraction, spatial RGB signal averaging, and live telemetry overlays.
"""

import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ml.ml1.roi import extract_forehead_roi
from ml.ml1.signal import RGBSignalBuffer

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
# Set camera index (0 for default built-in camera, 1 for external/secondary camera)
CAMERA_INDEX = 1

# Model asset path for MediaPipe Tasks API
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_landmarker.task")


def run_roi_test():
    # ---------------------------------------------------------
    # 1. Initialize MediaPipe FaceLandmarker Task
    # ---------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Face Landmarker model asset not found at: {MODEL_PATH}")
        print("Please ensure face_landmarker.task is located in F:\\NIO_HACK\\models\\")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # ---------------------------------------------------------
    # 2. Initialize Signal Buffer & Video Capture
    # ---------------------------------------------------------
    signal_buffer = RGBSignalBuffer(max_samples=300)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX from 0 to 1 in roi_test.py.")
        return

    print(f"[SUCCESS] Webcam (Index {CAMERA_INDEX}) and FaceLandmarker initialized!")
    print("Press 'q' in the camera window to exit.")

    prev_time = time.time()
    fps = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[ERROR] Failed to capture frame.")
            break

        current_timestamp = time.time()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Frame timestamp in milliseconds required by VIDEO mode
        frame_timestamp_ms = int((current_timestamp - start_time) * 1000)

        # Detect facial landmarks
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        mean_r, mean_g, mean_b = None, None, None
        face_detected = False

        # ---------------------------------------------------------
        # Process Detected Face Landmarks & ROI
        # ---------------------------------------------------------
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            face_detected = True

            # Extract forehead ROI crop, polygon points, and bounding box
            roi_crop, polygon_pts, bbox = extract_forehead_roi(frame, face_landmarks)

            if roi_crop is not None and bbox is not None:
                x, y, w, h = bbox

                # Extract spatial mean RGB and add sample to signal buffer
                mean_r, mean_g, mean_b = signal_buffer.add_frame_sample(roi_crop, timestamp=current_timestamp)

                # Draw yellow polygon boundary on forehead ROI
                cv2.polylines(frame, [polygon_pts], isClosed=True, color=(0, 255, 255), thickness=2)

                # Draw blue bounding rectangle around forehead ROI
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

                # Label ROI box
                cv2.putText(frame, "Forehead ROI", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ---------------------------------------------------------
        # Compute FPS
        # ---------------------------------------------------------
        time_diff = current_timestamp - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = current_timestamp

        # ---------------------------------------------------------
        # Telemetry Overlays
        # ---------------------------------------------------------
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Face: {'YES' if face_detected else 'NO'}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if face_detected else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Buffer: {len(signal_buffer)}/300",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ---------------------------------------------------------
        # RGB values
        # ---------------------------------------------------------
        if mean_r is not None:
            rgb_text = (
                f"R:{mean_r:.1f} "
                f"G:{mean_g:.1f} "
                f"B:{mean_b:.1f}"
            )

            cv2.putText(
                frame,
                rgb_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # ---------------------------------------------------------
        # Display Window
        # ---------------------------------------------------------
        cv2.imshow(
            "Forehead ROI & Signal Extraction Test",
            frame
        )

        # ---------------------------------------------------------
        # Quit Key Handling
        # ---------------------------------------------------------
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting ROI test...")
            break

    # ---------------------------------------------------------
    # Cleanup Resources
    # ---------------------------------------------------------
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("ROI test closed cleanly.")


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    run_roi_test()
