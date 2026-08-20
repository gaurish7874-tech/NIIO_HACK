"""
verify_behavioral.py
--------------------
Captures real live video frames from Camera 1 (~10 seconds = 300 frames),
passes them to `extract_behavioral(frame_buffer)`, and prints the real behavioral metrics.
"""

import os
import sys
import time
import cv2

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.ml1.behavioral import extract_behavioral

CAMERA_INDEX = 1
NUM_FRAMES_TO_CAPTURE = 300  # ~10 seconds of video at 30 FPS


def run_real_behavioral_test():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        print("Please verify your webcam connection.")
        return

    print("================================================================")
    print("      REAL WEBCAM VERIFICATION FOR extract_behavioral()        ")
    print("================================================================")
    print("Position your face in front of the camera...")
    print(f"Capturing {NUM_FRAMES_TO_CAPTURE} live frames (~10 seconds)... Look at camera and blink naturally.\n")

    frame_buffer = []
    start_time = time.time()

    while len(frame_buffer) < NUM_FRAMES_TO_CAPTURE:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[ERROR] Failed to grab frame from camera.")
            break

        frame_buffer.append(frame)

        display_frame = frame.copy()
        current_count = len(frame_buffer)

        cv2.rectangle(display_frame, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(
            display_frame,
            f"Capturing Behavioral Frames: {current_count}/{NUM_FRAMES_TO_CAPTURE}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )
        cv2.putText(
            display_frame,
            "Blink naturally and look at the screen...",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow("Capturing Real Frames for Behavioral Test", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture aborted by user.")
            break

    elapsed_time = time.time() - start_time
    actual_fps = len(frame_buffer) / elapsed_time if elapsed_time > 0 else 30.0

    cap.release()
    cv2.destroyAllWindows()

    if len(frame_buffer) < 10:
        print("[ERROR] Not enough frames captured.")
        return

    print(f"\n[SUCCESS] Captured {len(frame_buffer)} real frames in {elapsed_time:.1f} seconds (Actual FPS: {actual_fps:.1f}).")
    print("Running extract_behavioral(frame_buffer)... Processing behavioral pipeline...\n")

    result = extract_behavioral(frame_buffer, fps=actual_fps)

    print("================================================================")
    print("               REAL EXTRACTED BEHAVIORAL RESULTS                ")
    print("================================================================")
    print(f"  Blink Rate           : {result['blink_rate']} BPM")
    print(f"  Eye Closure Ratio    : {result['eye_closure']}")
    print(f"  Gaze Stability       : {result['gaze_stability']}")
    print(f"  Head Pose (Pitch)    : {result['head_pose']['pitch']}°")
    print(f"  Head Pose (Yaw)      : {result['head_pose']['yaw']}°")
    print(f"  Head Pose (Roll)     : {result['head_pose']['roll']}°")
    print(f"  Confidence Score     : {result['confidence']:.2f}")
    print(f"  Signal Available     : {result['signal_available']}")
    print("================================================================")
    print(f"Raw Output Dictionary  : {result}\n")


if __name__ == "__main__":
    run_real_behavioral_test()
