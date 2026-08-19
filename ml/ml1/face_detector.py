"""
face_detector.py
----------------
Module responsibility:
Detect face and facial landmarks using MediaPipe Face Mesh and return landmarks + annotated frames.
"""

import cv2
import mediapipe as mp

class FaceMeshDetector:
    """
    Modular Face Mesh Detector wrapper class.
    """
    def __init__(self, max_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame_bgr):
        """
        Converts BGR frame to RGB and runs MediaPipe Face Mesh.

        Returns:
            results: MediaPipe FaceMesh processing results object
            rgb_frame: Converted RGB image frame
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results, rgb_frame

    def draw_landmarks(self, frame_bgr, results):
        """
        Draws face mesh tessellation and contours on the BGR frame.
        """
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELLATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tessellation_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        return frame_bgr

    def close(self):
        """
        Releases FaceMesh resources.
        """
        self.face_mesh.close()
