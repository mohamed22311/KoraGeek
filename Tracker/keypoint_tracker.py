from .base_tracker import BaseTracker

import numpy as np
import cv2
import supervision as sv
from ultralytics.engine.results import Results
from utils import file_loader, file_saver
from typing import List

class KeypointTracker(BaseTracker):
    """Detection and Tracking of football field keypoints"""
    def __init__(self, model_path: str, conf: float=0.1, kp_conf: float = 0.8) -> None:
        """
        Initialize KeypointsTracker for tracking keypoints.
        
        Args:
            model_path (str): Model path.
            conf (float): Confidence threshold for field detection.
            kp_conf (float): Confidence threshold for keypoints.
        """
        super().__init__(model_path, conf)
        
        self.kp_conf = kp_conf
        self.tracks = {}

    def get_detections(self, frames: List[np.ndarray], batch_size: int = 16, read_from_stub: bool=False, stub_name: str=None) -> List[Results]:
        """
        Perform KeyPoint detection on the input frames.
        Args:
            frames (List[ndarray]): List of frames to perform object detection on.
            batch_size (int): Number of frames in a batch. Default is 16.
            read_from_stub (bool): Whether to read from stub file. Default is False.
            stub_name (str): Name of the stub file. Default is None.

        Returns:
            List[Results]: Keypoint Detection results for each frame.
        """
        if read_from_stub:
            detections= file_loader(dir='keypoint_detections',file_name=stub_name)
            if detections:
                return detections
            
        frames = [self._adjust_contrast(frame) for frame in frames]

        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=self.conf)
            detections+=detections_batch

        if stub_name:
            file_saver(detections,'keypoint_detections',stub_name)            
        return detections

    def get_tracks(self, detections: List[Results]) -> List[sv.KeyPoints]:
        """ Get the keypoints tracks of a video frame.
        Args:
            detections (Results): Detections from the frame.
        Returns:
            List[sv.KeyPoints]: List of keypoints tracks.
        """
        tracks = []
        filters = []
        for detection in detections:
            detection_sv = sv.KeyPoints.from_ultralytics(detection)
            filter = detection_sv.confidence[0] > self.kp_conf
            filtered_keypoints = detection_sv.xy[0][filter]
            filtered_keypoints = sv.KeyPoints(xy=filtered_keypoints[np.newaxis, ...])

            tracks.append(filtered_keypoints)
            filters.append(filter)
        
        # return detection_sv
        return tracks, filters
    
    def _adjust_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Adjust the contrast of the frame using Histogram Equalization.
        
        Args:
            frame (ndarray): The input image frame.
        
        Returns:
            ndarray: The frame with adjusted contrast.
        """
        # Check if the frame is colored (3 channels). If so, convert to grayscale for histogram equalization.
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert to YUV color space
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            
            # Apply histogram equalization to the Y channel (luminance)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to BGR format
            frame_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # If the frame is already grayscale, apply histogram equalization directly
            frame_equalized = cv2.equalizeHist(frame)

        return frame_equalized