"""
roi.py
------
Module responsibility:
Extract stable forehead Region of Interest (ROI) using MediaPipe facial landmarks.

Why Forehead ROI?
- Forehead skin has high capillary density and minimal facial muscle movement.
- Hair and facial hair (beards/mustaches) do not obstruct the forehead, providing clean rPPG signals.
"""

import cv2
import numpy as np
from ml.ml1.landmarks import get_landmark_pixel_coords

# Key MediaPipe landmark indices defining the forehead bounds:
# 10  : Upper forehead center
# 67  : Upper left forehead
# 297 : Upper right forehead
# 107 : Left eyebrow top
# 336 : Right eyebrow top
FOREHEAD_LANDMARK_INDICES = [10, 67, 107, 336, 297]

def extract_forehead_roi(frame, face_landmarks):
    """
    Extracts forehead Region of Interest (ROI) polygon and bounding box.

    Parameters:
        frame: OpenCV image frame (BGR format)
        face_landmarks: MediaPipe Face Mesh landmark object

    Returns:
        roi_crop (ndarray): Image cropped to forehead ROI (or None if invalid)
        polygon_pts (ndarray): (N, 1, 2) array of pixel coordinates for polygon drawing
        bbox (tuple): (x, y, w, h) bounding rectangle coordinates
    """
    if frame is None or face_landmarks is None:
        return None, None, None

    image_height, image_width, _ = frame.shape

    # Extract pixel coordinates for forehead landmarks
    points = []
    for idx in FOREHEAD_LANDMARK_INDICES:
        pt = get_landmark_pixel_coords(face_landmarks, image_width, image_height, idx)
        if pt is None:
            return None, None, None
        points.append((pt[0], pt[1]))

    # Convert to NumPy array format required by OpenCV polygon functions
    polygon_pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # Calculate bounding box (x, y, width, height) around forehead points
    x, y, w, h = cv2.boundingRect(polygon_pts)

    # Ensure bounding box is within frame boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(image_width - x, w)
    h = min(image_height - y, h)

    if w <= 0 or h <= 0:
        return None, None, None

    # Crop rectangular forehead region from frame
    roi_crop = frame[y:y+h, x:x+w]
    bbox = (x, y, w, h)

    return roi_crop, polygon_pts, bbox
