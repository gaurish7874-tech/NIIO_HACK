"""
blink.py
--------
Eye Blink Detection & Eye Aspect Ratio (EAR) module using MediaPipe facial landmarks.

How EAR Works:
The Eye Aspect Ratio (EAR) is an elegant geometric ratio measuring the open/closed state of the eye.
As the eye closes, the vertical distance between the eyelids drops towards 0, while the horizontal
width between eye corners remains constant. Thus, EAR drops significantly during a blink.

EAR Formula:
          || p2 - p6 || + || p3 - p5 ||
  EAR = ---------------------------------
                2 * || p1 - p4 ||

Where:
  p1, p4: Left and Right horizontal eye corners
  p2, p6: Top and Bottom vertical eyelid points (outer pair)
  p3, p5: Top and Bottom vertical eyelid points (inner pair)
"""

import math
import numpy as np
from ml.ml1.landmarks import get_landmark_pixel_coords, get_landmark_distance

# MediaPipe 468/478 Landmark Indices for Left and Right Eyes:
# Left Eye
LEFT_EYE_CORNER_LEFT = 33
LEFT_EYE_CORNER_RIGHT = 133
LEFT_EYE_TOP1 = 159
LEFT_EYE_BOT1 = 145
LEFT_EYE_TOP2 = 158
LEFT_EYE_BOT2 = 144

# Right Eye
RIGHT_EYE_CORNER_LEFT = 362
RIGHT_EYE_CORNER_RIGHT = 263
RIGHT_EYE_TOP1 = 386
RIGHT_EYE_BOT1 = 374
RIGHT_EYE_TOP2 = 387
RIGHT_EYE_BOT2 = 373

# Default EAR threshold below which eye is considered closed
EAR_THRESHOLD = 0.21
MIN_CONSECUTIVE_CLOSED_FRAMES = 2


def calculate_single_eye_ear(face_landmarks, image_width: int, image_height: int, eye_indices: dict) -> float:
    """
    Calculates EAR for a single eye given landmark indices.
    """
    p1 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, eye_indices['corner_left'])
    p4 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, eye_indices['corner_right'])
    p2 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, eye_indices['top1'])
    p6 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, eye_indices['bot1'])
    p3 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, eye_indices['top2'])
    p5 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, eye_indices['bot2'])

    if None in (p1, p2, p3, p4, p5, p6):
        return 0.0

    # Vertical distances
    v1 = get_landmark_distance(p2, p6)
    v2 = get_landmark_distance(p3, p5)

    # Horizontal distance
    h = get_landmark_distance(p1, p4)

    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


def calculate_face_ear(face_landmarks, image_width: int, image_height: int) -> float:
    """
    Calculates average EAR across both eyes.
    """
    if face_landmarks is None:
        return 0.0

    left_indices = {
        'corner_left': LEFT_EYE_CORNER_LEFT, 'corner_right': LEFT_EYE_CORNER_RIGHT,
        'top1': LEFT_EYE_TOP1, 'bot1': LEFT_EYE_BOT1,
        'top2': LEFT_EYE_TOP2, 'bot2': LEFT_EYE_BOT2
    }
    right_indices = {
        'corner_left': RIGHT_EYE_CORNER_LEFT, 'corner_right': RIGHT_EYE_CORNER_RIGHT,
        'top1': RIGHT_EYE_TOP1, 'bot1': RIGHT_EYE_BOT1,
        'top2': RIGHT_EYE_TOP2, 'bot2': RIGHT_EYE_BOT2
    }

    left_ear = calculate_single_eye_ear(face_landmarks, image_width, image_height, left_indices)
    right_ear = calculate_single_eye_ear(face_landmarks, image_width, image_height, right_indices)

    if left_ear > 0 and right_ear > 0:
        return (left_ear + right_ear) / 2.0
    return max(left_ear, right_ear)


class BlinkTracker:
    """
    Stateful Blink Tracker maintaining rolling history, blink count, and eye closure duration.
    """
    def __init__(self, ear_threshold: float = EAR_THRESHOLD, min_closed_frames: int = MIN_CONSECUTIVE_CLOSED_FRAMES):
        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames

        self.consecutive_closed_frames = 0
        self.total_blinks = 0
        self.total_closed_frames = 0
        self.total_frames = 0
        self.eye_state = "OPEN"  # "OPEN" or "CLOSED"

    def process_frame(self, face_landmarks, image_width: int, image_height: int):
        """
        Processes a single frame and updates blink telemetry.

        Returns:
            dict containing:
                - 'ear': float (current average EAR)
                - 'eye_state': str ("OPEN" or "CLOSED")
                - 'total_blinks': int
                - 'consecutive_closed_frames': int
        """
        ear = calculate_face_ear(face_landmarks, image_width, image_height)
        self.total_frames += 1

        if ear > 0 and ear < self.ear_threshold:
            self.consecutive_closed_frames += 1
            self.total_closed_frames += 1
            self.eye_state = "CLOSED"
        else:
            # Check if transitioning from CLOSED to OPEN after minimum consecutive frames
            if self.consecutive_closed_frames >= self.min_closed_frames:
                self.total_blinks += 1

            self.consecutive_closed_frames = 0
            self.eye_state = "OPEN"

        return {
            'ear': round(float(ear), 3),
            'eye_state': self.eye_state,
            'total_blinks': self.total_blinks,
            'consecutive_closed_frames': self.consecutive_closed_frames
        }

    def get_blink_rate(self, duration_seconds: float) -> float:
        """
        Calculates rolling blink rate in Blinks Per Minute (BPM).
        """
        if duration_seconds <= 0:
            return 0.0
        return round(float((self.total_blinks / duration_seconds) * 60.0), 1)

    def get_eye_closure_ratio(self) -> float:
        """
        Calculates ratio of time eyes were closed (0.0 to 1.0).
        """
        if self.total_frames == 0:
            return 0.0
        return round(float(self.total_closed_frames / self.total_frames), 3)

    def reset(self):
        """
        Resets tracking stats.
        """
        self.consecutive_closed_frames = 0
        self.total_blinks = 0
        self.total_closed_frames = 0
        self.total_frames = 0
        self.eye_state = "OPEN"


# -----------------------------------------------------------------------
# Offline Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Blink & EAR Module...")
    tracker = BlinkTracker()
    print("Blink tracker initialized successfully.")
