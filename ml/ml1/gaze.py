"""
gaze.py
-------
Gaze direction & Gaze Stability estimation module using MediaPipe Iris landmarks.

How Iris Gaze Tracking Works:
1. MediaPipe FaceLandmarker refined landmarks include 3D iris centers:
   - Left Iris Center: Index 468
   - Right Iris Center: Index 473
2. We calculate the relative horizontal and vertical position of the iris center
   within the eye bounding box (between eye corner landmarks 33 & 133).
3. Normalization (0.0 to 1.0) makes gaze measurement independent of video resolution or head distance:
   - ratio_x ~ 0.50: CENTER
   - ratio_x < 0.38: RIGHT
   - ratio_x > 0.62: LEFT
4. Gaze Stability is derived from the standard deviation of gaze positions over a rolling window.
   Low position variance yields high gaze stability (~1.0).
"""

import math
from collections import deque
import numpy as np
from ml.ml1.landmarks import get_landmark_pixel_coords

# Iris & Eye Landmarks
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

LEFT_EYE_CORNER_LEFT = 33
LEFT_EYE_CORNER_RIGHT = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOT = 145


class GazeTracker:
    """
    Gaze Direction & Stability Tracker.
    """
    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.gaze_history_x = deque(maxlen=history_size)
        self.gaze_history_y = deque(maxlen=history_size)

    def process_frame(self, face_landmarks, image_width: int, image_height: int):
        """
        Estimates gaze position, direction, and rolling stability from a single frame.

        Returns:
            dict containing:
                - 'gaze_direction': str ("CENTER", "LEFT", "RIGHT", "UP", "DOWN", "UNKNOWN")
                - 'gaze_x': float (0.0 to 1.0 normalized position)
                - 'gaze_y': float (0.0 to 1.0 normalized position)
                - 'gaze_stability': float (0.0 to 1.0)
        """
        if face_landmarks is None:
            return {
                'gaze_direction': 'UNKNOWN',
                'gaze_x': 0.5,
                'gaze_y': 0.5,
                'gaze_stability': 0.0
            }

        landmarks_list = getattr(face_landmarks, 'landmark', face_landmarks)

        # Handle missing iris landmarks (require at least 469 landmarks)
        if len(landmarks_list) <= LEFT_IRIS_CENTER:
            return {
                'gaze_direction': 'UNKNOWN',
                'gaze_x': 0.5,
                'gaze_y': 0.5,
                'gaze_stability': 0.0
            }

        iris_pt = get_landmark_pixel_coords(face_landmarks, image_width, image_height, LEFT_IRIS_CENTER)
        corner_left = get_landmark_pixel_coords(face_landmarks, image_width, image_height, LEFT_EYE_CORNER_LEFT)
        corner_right = get_landmark_pixel_coords(face_landmarks, image_width, image_height, LEFT_EYE_CORNER_RIGHT)
        eye_top = get_landmark_pixel_coords(face_landmarks, image_width, image_height, LEFT_EYE_TOP)
        eye_bot = get_landmark_pixel_coords(face_landmarks, image_width, image_height, LEFT_EYE_BOT)

        if None in (iris_pt, corner_left, corner_right, eye_top, eye_bot):
            return {
                'gaze_direction': 'UNKNOWN',
                'gaze_x': 0.5,
                'gaze_y': 0.5,
                'gaze_stability': 0.0
            }

        # Calculate horizontal eye width and iris position ratio
        eye_width = float(corner_right[0] - corner_left[0])
        eye_height = float(eye_bot[1] - eye_top[1])

        if eye_width <= 0 or eye_height <= 0:
            return {
                'gaze_direction': 'UNKNOWN',
                'gaze_x': 0.5,
                'gaze_y': 0.5,
                'gaze_stability': 0.0
            }

        gaze_x = float((iris_pt[0] - corner_left[0]) / eye_width)
        gaze_y = float((iris_pt[1] - eye_top[1]) / eye_height)

        # Append to rolling history
        self.gaze_history_x.append(gaze_x)
        self.gaze_history_y.append(gaze_y)

        # Estimate Gaze Direction
        if gaze_x < 0.38:
            h_dir = "RIGHT"
        elif gaze_x > 0.62:
            h_dir = "LEFT"
        else:
            h_dir = "CENTER"

        if gaze_y < 0.30:
            v_dir = "UP"
        elif gaze_y > 0.70:
            v_dir = "DOWN"
        else:
            v_dir = ""

        gaze_direction = f"{v_dir} {h_dir}".strip() if v_dir else h_dir

        # Calculate Gaze Stability (1.0 - normalized standard deviation)
        gaze_stability = 0.0
        if len(self.gaze_history_x) >= 5:
            std_x = float(np.std(self.gaze_history_x))
            std_y = float(np.std(self.gaze_history_y))
            total_std = math.sqrt(std_x * std_x + std_y * std_y)

            # High stability = low std (< 0.05)
            gaze_stability = float(np.clip(1.0 - (total_std / 0.15), 0.0, 1.0))

        return {
            'gaze_direction': gaze_direction,
            'gaze_x': round(gaze_x, 3),
            'gaze_y': round(gaze_y, 3),
            'gaze_stability': round(gaze_stability, 2)
        }

    def reset(self):
        """
        Clears gaze history.
        """
        self.gaze_history_x.clear()
        self.gaze_history_y.clear()


# -----------------------------------------------------------------------
# Offline Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Gaze Module...")
    tracker = GazeTracker()
    print("Gaze tracker initialized successfully.")
