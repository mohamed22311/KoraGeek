from typing import Any, Dict, List, Tuple

import cv2
import pandas as pd
from .base import BaseTransformer
from .homography import Homography
from utils import get_anchors_coordinates
from Enums import Position
from supervision import KeyPoints

import numpy as np

class ViewTransformer(BaseTransformer):
    """
    A class to map object positions from detected keypoints to a top-down view.

    This class implements the mapping of detected objects to their corresponding
    positions in a top-down representation based on the homography obtained from 
    detected keypoints.
    """

    def __init__(self, top_down_keypoints: np.ndarray, alpha: float = 0.9) -> None:
        """
        Initializes the ObjectPositionMapper.

        Args:
            top_down_keypoints (np.ndarray): An array of shape (n, 2) containing the top-down keypoints.
            alpha (float): Smoothing factor for homography smoothing.
        """
        super().__init__()
        self.top_down_keypoints = top_down_keypoints
        self.homography : Homography = Homography(alpha) 
    
    def transform(self, object_tracks: Dict[str, Dict[int, Dict[str, Any]]], keypoints_tracks: KeyPoints, filter: np.ndarray[bool]) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Maps the detection data to their positions in the top-down view.

        This method retrieves keypoints and object information from the detection data,
        computes the homography matrix, smooths it over frames, and projects the foot positions
        of detected objects.

        Args:
            object_tracks (Dict[str, Dict[int, Dict[str, Any]]]): Object tracks containing the object information.
            keypoints_tracks (KeyPoints): Detected keypoints.
            filter (np.ndarray[bool]): A boolean array to filter the keypoints.
        
        Returns:
            Dict[str, Dict[int, Dict[str, Any]]]: The object tracks with projected positions.
        """
        
        object_data = object_tracks
        keypoints = keypoints_tracks.xy[0]
        top_down_keypoints = self.top_down_keypoints[filter]

        homography_matrix = self.homography.find_homography(keypoints, top_down_keypoints)
        
        for class_name, track_data in object_data.items():
            for track_id, track_info in track_data.items():
                bbox = track_info['bbox']
                feet_pos = get_anchors_coordinates(bbox,anchor=Position.BOTTOM_CENTER)
                projected_pos = self.homography.perspective_transform(posistion=feet_pos, H_mat=homography_matrix)
                object_data[class_name][track_id]['projection'] = projected_pos
                object_data[class_name][track_id]['position'] = feet_pos
        
        return object_data
    
    def adjust_transforms(self,object_tracks: Dict[str, Dict[int, Dict[str, Any]]], camera_movement: Tuple[float, float]):
        """
        Adjust the projected positions of the objects based on the camera movement in each frame

        Args:
            tracks (Dict[str, Dict[int, Dict[str, Any]]]): Object tracks containing the object information.
            camera_movement (Tuple[float, float]): The camera movement in frame

        Returns:
            Dict[str, Dict[int, Dict[str, Any]]]: The object tracks with adjusted projected positions
        """
        for class_name, track_data in object_tracks.items():
            for track_id, track_info in track_data.items():
                position = track_info['projection']
                position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                object_tracks[class_name][track_id]['projection'] = position_adjusted

        return object_tracks