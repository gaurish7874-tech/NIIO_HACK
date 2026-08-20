"""
signal.py
---------
Module responsibility:
1. Extract spatial average (mean) R, G, B values from the ROI.
2. Maintain a rolling buffer of RGB signal values over time with timestamps.
"""

import time
from collections import deque
import numpy as np
import cv2

class RGBSignalBuffer:
    """
    Rolling buffer for storing physiological RGB signal samples over time.
    """
    def __init__(self, max_samples: int = 300):
        """
        Parameters:
            max_samples (int): Buffer capacity (e.g. 300 frames = ~10 seconds at 30 FPS)
        """
        self.max_samples = max_samples
        self.timestamps = deque(maxlen=max_samples)
        self.r_values = deque(maxlen=max_samples)
        self.g_values = deque(maxlen=max_samples)
        self.b_values = deque(maxlen=max_samples)

    def extract_mean_rgb(self, roi_crop):
        """
        Calculates spatial average (mean) R, G, B pixel values from an ROI image.

        Note: OpenCV stores pixels in BGR order (Blue, Green, Red).
        Returns:
            mean_r, mean_g, mean_b (floats)
        """
        if roi_crop is None or roi_crop.size == 0:
            return None, None, None

        # cv2.mean returns average for each channel in BGR order: (mean_b, mean_g, mean_r, _)
        mean_b, mean_g, mean_r, _ = cv2.mean(roi_crop)
        return float(mean_r), float(mean_g), float(mean_b)

    def add_frame_sample(self, roi_crop, timestamp: float = None):
        """
        Extracts mean RGB from ROI crop and appends sample to the rolling buffer.

        Returns:
            tuple (mean_r, mean_g, mean_b) if sample added, else (None, None, None)
        """
        if timestamp is None:
            timestamp = time.time()

        mean_r, mean_g, mean_b = self.extract_mean_rgb(roi_crop)

        if mean_r is not None:
            self.timestamps.append(timestamp)
            self.r_values.append(mean_r)
            self.g_values.append(mean_g)
            self.b_values.append(mean_b)
            return mean_r, mean_g, mean_b

        return None, None, None

    def get_signal_arrays(self):
        """
        Returns buffered signals as 1D NumPy arrays (R, G, B, Timestamps).
        """
        t = np.array(self.timestamps)
        r = np.array(self.r_values)
        g = np.array(self.g_values)
        b = np.array(self.b_values)
        return r, g, b, t

    def clear(self):
        """
        Clears all stored signals.
        """
        self.timestamps.clear()
        self.r_values.clear()
        self.g_values.clear()
        self.b_values.clear()

    def __len__(self):
        return len(self.timestamps)
