"""
landmarks.py
------------
Reusable helper module for processing MediaPipe face landmarks.

Key concepts for beginners:
1. MediaPipe returns landmark coordinates (x, y, z) normalized between 0.0 and 1.0.
   - landmark.x = 0.5 means 50% across image width (from left).
   - landmark.y = 0.5 means 50% down image height (from top).
   - landmark.z = depth relative to face center (smaller/negative values mean closer to camera).
2. To convert normalized coordinates to pixel coordinates on your screen:
   - pixel_x = int(landmark.x * image_width)
   - pixel_y = int(landmark.y * image_height)
"""

import math

def get_landmark_pixel_coords(face_landmarks, image_width: int, image_height: int, landmark_index: int):
    """
    Converts a MediaPipe normalized landmark into pixel (x, y, z) coordinates.

    Parameters:
        face_landmarks: MediaPipe NormalizedLandmarkList (e.g. face_landmarks.landmark)
        image_width (int): Width of the frame in pixels
        image_height (int): Height of the frame in pixels
        landmark_index (int): Index of landmark (0 to 467 for MediaPipe Face Mesh)

    Returns:
        tuple (int, int, float) -> (pixel_x, pixel_y, landmark_z) or None if index is invalid
    """
    if face_landmarks is None:
        return None

    # Handle MediaPipe LandmarkList or standard Python list
    landmarks_list = getattr(face_landmarks, 'landmark', face_landmarks)

    # Safe index check
    if not (0 <= landmark_index < len(landmarks_list)):
        print(f"[WARNING] Invalid landmark_index: {landmark_index}. Must be between 0 and {len(landmarks_list) - 1}.")
        return None

    landmark = landmarks_list[landmark_index]

    # Convert normalized (0.0 to 1.0) coordinates to screen pixels
    pixel_x = int(landmark.x * image_width)
    pixel_y = int(landmark.y * image_height)
    landmark_z = landmark.z  # Relative depth value

    return (pixel_x, pixel_y, landmark_z)


def get_landmark_distance(pt1, pt2, use_3d: bool = False) -> float:
    """
    Calculates Euclidean distance between two landmark coordinate tuples.

    Parameters:
        pt1: Tuple (x1, y1) or (x1, y1, z1)
        pt2: Tuple (x2, y2) or (x2, y2, z2)
        use_3d (bool): If True, includes z-coordinate in distance calculation

    Returns:
        float: Distance in pixels (or 3D space)
    """
    if pt1 is None or pt2 is None:
        return 0.0

    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]

    if use_3d and len(pt1) > 2 and len(pt2) > 2:
        dz = pt2[2] - pt1[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    return math.sqrt(dx * dx + dy * dy)


def get_landmark_midpoint(pt1, pt2):
    """
    Calculates the midpoint between two landmark coordinate tuples.

    Parameters:
        pt1: Tuple (x1, y1) or (x1, y1, z1)
        pt2: Tuple (x2, y2) or (x2, y2, z2)

    Returns:
        tuple (int, int) -> (mid_x, mid_y)
    """
    if pt1 is None or pt2 is None:
        return None

    mid_x = int((pt1[0] + pt2[0]) / 2)
    mid_y = int((pt1[1] + pt2[1]) / 2)

    return (mid_x, mid_y)


def get_distance_between_indices(face_landmarks, image_width: int, image_height: int, idx1: int, idx2: int, use_3d: bool = False) -> float:
    """
    Helper function to directly compute distance between two landmark indices.
    """
    pt1 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, idx1)
    pt2 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, idx2)
    return get_landmark_distance(pt1, pt2, use_3d=use_3d)


def get_midpoint_between_indices(face_landmarks, image_width: int, image_height: int, idx1: int, idx2: int):
    """
    Helper function to directly compute midpoint between two landmark indices.
    """
    pt1 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, idx1)
    pt2 = get_landmark_pixel_coords(face_landmarks, image_width, image_height, idx2)
    return get_landmark_midpoint(pt1, pt2)
