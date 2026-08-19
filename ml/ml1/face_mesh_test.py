"""
face_mesh_test.py
-----------------
Webcam program integrating OpenCV and MediaPipe Face Mesh.

Requirements implemented:
1. Capture webcam frames with OpenCV.
2. Convert frame color from BGR to RGB (MediaPipe expects RGB).
3. Run MediaPipe Face Mesh detection.
4. Draw facial landmarks on detected face.
5. Calculate & overlay real-time FPS.
6. Press 'q' to exit cleanly.
"""

import time
import cv2
import mediapipe as mp
from ml.ml1.landmarks import get_landmark_pixel_coords

def run_face_mesh_test():
    # -------------------------------------------------------------------
    # Step 1: Initialize MediaPipe Face Mesh & Drawing Utilities
    # -------------------------------------------------------------------
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize FaceMesh object:
    # - max_num_faces=1: Detect only 1 face for optimal performance
    # - refine_landmarks=True: Adds detailed landmarks for eyes, irises, lips
    # - min_detection_confidence=0.5: Minimum confidence to consider detection valid
    # - min_tracking_confidence=0.5: Tracking confidence between frames
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # -------------------------------------------------------------------
    # Step 2: Open Default Webcam with OpenCV
    # -------------------------------------------------------------------
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("Please check if another app is using the camera.")
        return

    print("[SUCCESS] Webcam opened and MediaPipe Face Mesh initialized!")
    print("Press 'q' in the window to exit.")

    prev_time = time.time()
    fps = 0.0

    while True:
        # Step 3: Read a single frame from webcam (OpenCV returns BGR format)
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[ERROR] Failed to capture frame.")
            break

        image_height, image_width, _ = frame.shape

        # Step 4: Convert BGR to RGB
        # OpenCV uses BGR (Blue-Green-Red) order by default.
        # MediaPipe expects RGB (Red-Green-Blue) order.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Performance tip: Mark frame as non-writeable to speed up MediaPipe processing
        rgb_frame.flags.writeable = False

        # Step 5: Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        # Mark frame back as writeable for OpenCV drawing
        rgb_frame.flags.writeable = True

        # Step 6: Draw Face Mesh landmarks if a face is detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw full face mesh tessellation (network of landmarks)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELLATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tessellation_style()
                )

                # Draw face contours (eyes, lips, face oval)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                # Example: Extract nose tip landmark (Index 1) using landmarks.py helper
                nose_tip = get_landmark_pixel_coords(face_landmarks, image_width, image_height, 1)
                if nose_tip:
                    # Draw a distinct green circle on the nose tip (Index 1)
                    cv2.circle(frame, (nose_tip[0], nose_tip[1]), 4, (0, 255, 0), -1)

        # Step 7: Calculate real-time FPS
        current_time = time.time()
        time_diff = current_time - prev_time

        if time_diff > 0:
            fps = 1.0 / time_diff

        prev_time = current_time

        # Overlay FPS on the video feed
        cv2.putText(
            img=frame,
            text=f"FPS: {fps:.1f}",
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Step 8: Display the frame in an OpenCV window
        cv2.imshow("MediaPipe Face Mesh Test", frame)

        # Step 9: Press 'q' to exit loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting Face Mesh test...")
            break

    # Step 10: Clean up resources
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("Face Mesh test closed cleanly.")

if __name__ == "__main__":
    run_face_mesh_test()
