"""
camera_test.py
--------------
Simple OpenCV webcam test script to check:
1. Camera initialization
2. Live frame capture & display
3. Real-time FPS (Frames Per Second) calculation
4. Clean exit on pressing 'q'
"""

import time
import cv2

def run_camera_test():
    # Step 1: Open the default webcam (index 0 is usually the built-in camera)
    cap = cv2.VideoCapture(0)

    # Graceful error check: verify if the camera opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("Please check if another app is using the camera, or try changing camera index (e.g. 0 to 1).")
        return

    print("[SUCCESS] Webcam opened successfully!")
    print("Press 'q' in the video window to exit.")

    # Initialize variables for FPS (Frames Per Second) calculation
    prev_time = time.time()
    fps = 0.0

    while True:
        # Step 2: Read a single frame from the camera
        # 'ret' is True if frame read was successful
        # 'frame' is the image array (NumPy pixel matrix)
        ret, frame = cap.read()

        # Handle failure gracefully if camera disconnects or frame fails to capture
        if not ret or frame is None:
            print("[ERROR] Failed to capture frame from camera.")
            break

        # Step 3: Calculate real-time FPS
        current_time = time.time()
        time_diff = current_time - prev_time

        if time_diff > 0:
            fps = 1.0 / time_diff

        prev_time = current_time

        # Step 4: Draw FPS text on the video frame
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            img=frame,
            text=fps_text,
            org=(20, 40),                      # Position (X=20, Y=40)
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, # Simple font
            fontScale=1.0,                      # Font size
            color=(0, 255, 0),                  # Color (B=0, G=255, R=0 -> Green)
            thickness=2,                        # Text thickness
            lineType=cv2.LINE_AA                # Smooth anti-aliased lines
        )

        # Step 5: Display the frame in a GUI window
        cv2.imshow("Webcam Test", frame)

        # Step 6: Wait 1ms for key press; exit loop if user presses 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting camera test...")
            break

    # Step 7: Release camera hardware and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test closed cleanly.")

if __name__ == "__main__":
    run_camera_test()
