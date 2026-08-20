"""
verify_multimodal.py
--------------------
Captures real live frames from Camera 1 (~10 seconds = 300 frames),
runs the combined Multimodal Wellness pipeline, and prints the exact target JSON structure.
"""

import json
import os
import sys
import time
import cv2

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.ml1.multimodal import analyze_multimodal_wellness

CAMERA_INDEX = 1
NUM_FRAMES_TO_CAPTURE = 500  # ~10 seconds of video at 60 FPS


def run_multimodal_verification():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        return

    print("================================================================")
    print("   MULTIMODAL WELLNESS MONITOR - FINAL OUTPUT VERIFICATION      ")
    print("================================================================")
    print("Position your face in front of the camera...")
    print(f"Capturing {NUM_FRAMES_TO_CAPTURE} live frames (~10 seconds)...\n")

    frame_buffer = []
    start_time = time.time()

    while len(frame_buffer) < NUM_FRAMES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_buffer.append(frame)
        display_frame = frame.copy()

        cv2.rectangle(display_frame, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(
            display_frame,
            f"Capturing Multimodal Frames: {len(frame_buffer)}/{NUM_FRAMES_TO_CAPTURE}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_frame,
            "Look at the camera, blink naturally...",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow("Capturing Real Frames for Multimodal Test", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elapsed_time = time.time() - start_time
    actual_fps = len(frame_buffer) / elapsed_time if elapsed_time > 0 else 30.0

    cap.release()
    cv2.destroyAllWindows()

    if len(frame_buffer) < 10:
        print("[ERROR] Not enough frames captured.")
        return

    print(f"[SUCCESS] Captured {len(frame_buffer)} real frames in {elapsed_time:.1f}s (Actual FPS: {actual_fps:.1f}).")
    print("Executing full Multimodal Wellness Pipeline...\n")

    final_output = analyze_multimodal_wellness(frame_buffer, fps=actual_fps)

    print("================================================================")
    print("              FINAL MULTIMODAL WELLNESS JSON OUTPUT             ")
    print("================================================================")
    print(json.dumps(final_output, indent=2))
    print("================================================================\n")


if __name__ == "__main__":
    run_multimodal_verification()
