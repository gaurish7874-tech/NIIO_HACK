"""
head_pose.py
------------
Head Pose Estimation (Pitch, Yaw, Roll) using OpenCV solvePnP.

Concept:
OpenCV solvePnP (Perspective-n-Point) takes:
1. 3D reference facial model points (in 3D world space, e.g. millimeters)
2. 2D pixel coordinates of those points extracted from MediaPipe Face Mesh
3. Camera intrinsic parameters (focal length, optical center)

It calculates the 3D orientation of the user's head relative to the camera sensor.

Euler Angles Breakdown:
- Pitch : Head Up / Down tilt (Rotation around X-axis)
- Yaw   : Head Left / Right turn (Rotation around Y-axis)
- Roll  : Head Sideways tilt (Rotation around Z-axis)
"""

import math
import cv2
import numpy as np
from ml.ml1.landmarks import get_landmark_pixel_coords

# Standard 3D facial model coordinates (Canonical 3D Face Model)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose Tip (Index 1)
    (0.0, -330.0, -65.0),        # Chin (Index 152)
    (-225.0, 170.0, -135.0),     # Left Eye Outer Corner (Index 33)
    (225.0, 170.0, -135.0),      # Right Eye Outer Corner (Index 263)
    (-150.0, -150.0, -125.0),    # Left Mouth Corner (Index 61)
    (150.0, -150.0, -125.0)      # Right Mouth Corner (Index 291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(face_landmarks, image_width: int, image_height: int):
    """
    Estimates 3D Head Pose (Pitch, Yaw, Roll) using cv2.solvePnP.

    Parameters:
        face_landmarks: MediaPipe FaceLandmarker result
        image_width (int): Frame width in pixels
        image_height (int): Frame height in pixels

    Returns:
        dict containing:
            - 'pitch': float (degrees, + = looking down, - = looking up)
            - 'yaw': float (degrees, + = turned right, - = turned left)
            - 'roll': float (degrees, + = tilted right, - = tilted left)
            - 'nose_end_point2d': tuple (x, y) 2D projected point for 3D pose axis drawing
            - 'rvec': rotation vector
            - 'tvec': translation vector
    """
    if face_landmarks is None or image_width <= 0 or image_height <= 0:
        return {
            'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
            'nose_end_point2d': None, 'rvec': None, 'tvec': None
        }

    # Extract 2D image points
    image_points_2d = []
    for idx in LANDMARK_INDICES:
        pt = get_landmark_pixel_coords(face_landmarks, image_width, image_height, idx)
        if pt is None:
            return {
                'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
                'nose_end_point2d': None, 'rvec': None, 'tvec': None
            }
        image_points_2d.append((pt[0], pt[1]))

    image_points_2d = np.array(image_points_2d, dtype=np.float64)

    # -------------------------------------------------------------------
    # Construct Camera Matrix & Distortion Coefficients
    # -------------------------------------------------------------------
    focal_length = image_width
    center = (image_width / 2.0, image_height / 2.0)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assuming zero lens distortion

    # -------------------------------------------------------------------
    # Solve PnP
    # -------------------------------------------------------------------
    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return {
            'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
            'nose_end_point2d': None, 'rvec': None, 'tvec': None
        }

    # -------------------------------------------------------------------
    # Convert Rotation Vector to Euler Angles (Pitch, Yaw, Roll)
    # -------------------------------------------------------------------
    rmat, _ = cv2.Rodrigues(rvec)

    # Combine rotation matrix and translation vector into 3x4 projection matrix
    proj_matrix = np.hstack((rmat, tvec))

    # Decompose projection matrix to extract Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = float(euler_angles[0, 0])
    yaw = float(euler_angles[1, 0])
    roll = float(euler_angles[2, 0])

    # -------------------------------------------------------------------
    # Project 3D Nose Axis Point onto 2D image for Visual Indicator
    # -------------------------------------------------------------------
    nose_axis_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
    nose_end_point2d, _ = cv2.projectPoints(nose_axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

    p1 = (int(image_points_2d[0][0]), int(image_points_2d[0][1]))
    p2 = (int(nose_end_point2d[0][0][0]), int(nose_end_point2d[0][0][1]))

    return {
        'pitch': round(pitch, 1),
        'yaw': round(yaw, 1),
        'roll': round(roll, 1),
        'nose_start': p1,
        'nose_end': p2,
        'rvec': rvec,
        'tvec': tvec
    }


def draw_head_pose_axis(frame, pose_result):
    """
    Draws a 3D nose direction vector line on the camera preview.
    """
    if pose_result is None or pose_result.get('nose_start') is None or pose_result.get('nose_end') is None:
        return frame

    p1 = pose_result['nose_start']
    p2 = pose_result['nose_end']

    # Draw cyan line indicating head orientation vector
    cv2.line(frame, p1, p2, (255, 255, 0), 2)
    cv2.circle(frame, p1, 3, (0, 0, 255), -1)

    return frame


# -----------------------------------------------------------------------
# Offline Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Head Pose Module...")
    res = estimate_head_pose(None, 640, 480)
    print("Default response on empty input:", res)
    assert res['pitch'] == 0.0 and res['yaw'] == 0.0 and res['roll'] == 0.0
    print("[SUCCESS] Head Pose module test passed cleanly!")
