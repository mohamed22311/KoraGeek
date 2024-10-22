import torch
import numpy as np
from abc import ABC, abstractmethod
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List
from numpy import ndarray
from supervision import Detections
from utils import get_settings

class BaseTracker(ABC):

    def __init__(self,
                 model_path: str,
                 conf: float = 0.3) -> None:
        """
        Load the model from the given path and set the confidence threshold.

        Args:
            model_path (str): Path to the model.
            conf (float): Confidence threshold for detections.
        """
        device = torch.device(get_settings().DEVICE)
        self.model = YOLO(model_path).to(device)
        self.conf = conf  # Set confidence threshold
        

    @abstractmethod
    def get_detections(self,frames: List[ndarray], batch_size: int=16) -> List[Results]:
        """
        Abstract method for YOLO detection.

        Args:
            frames (List[np.ndarray]): List of frames for detection.

        Returns:
            List[Results]: List of YOLO detection result objects.
        """
        pass
        
    @abstractmethod
    def get_tracks(self, detections: List[Detections]) -> dict:
        """
        Abstract method for tracking detections.

        Args:
            detection (Results): YOLO detection results for a single frame.

        Returns:
            dict: Tracking data.
        """
        pass

