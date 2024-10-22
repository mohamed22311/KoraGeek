from .base_tracker import BaseTracker

import numpy as np
import supervision as sv
from ultralytics.engine.results import Results
from utils import file_loader, file_saver
from typing import Dict, List


class ObjectTracker(BaseTracker):
    """Object Tracker class to track the object in the video"""

    def __init__(self, model_path: str, conf: float=0.3, ball_conf: float = 0.3) -> None:
        """
        Initialize ObjectTracker with detection and tracking.

        Args:
            model_path (str): Model Path.
            conf (float): Confidence threshold for detection.
        """
        super().__init__(model_path, conf)

        self.classes = ['ball', 'goalkeeper', 'player', 'referee']
        self.ball_conf = ball_conf
        self.tracker = sv.ByteTrack()
        self.tracker.reset()

    def get_detections(self, frames: List[np.ndarray], batch_size: int = 16, read_from_stub: bool=False, stub_name: str=None) -> List[Results]:
        """
        Get the object detections from the video frames.

        Args:
            frames (List[ndarray]): List of frames to perform object detection on.
            batch_size (int): Number of frames in a batch. Default is 16.
            read_from_stub (bool): Whether to read from stub file. Default is False.
            stub_name (str): Name of the stub file. Default is None.

        Returns:
            List[Results]: Detection results for each frame.
        """
        if read_from_stub:
            detections= file_loader(dir='detections',file_name=stub_name)
            if detections:
                return detections
            
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=self.conf)
            detections+=detections_batch

        if stub_name:
            file_saver(detections,'detections',stub_name)
            
        return detections
    
    def get_tracks(self, detections: List[Results]) -> List[Dict[str, Dict[int, Dict[str, np.ndarray]]]]:
        """
        Get the tracks from the detections. 

        Args:
            detections (List[Results]): Detections from the frames.

        Returns:
            List[Dict[str, Dict[int, Dict[str, np.ndarray]]]]: List of tracks for each frame.
        
        Example:
            tracks = [ {
                'players': {
                    1: {
                        'bbox': np.ndarray([x, y, w, h])
                    }, 
                    2: {
                        'bbox': np.ndarray([x, y, w, h])
                    }
                },
                'referees': {
                    1: {
                        'bbox': np.ndarray([x, y, w, h])
                    }
                },
                'ball': {
                    1: {
                        'bbox': np.ndarray([x, y, w, h])
                    }
                },
                'goalkeepers': {
                    1: {
                        'bbox': np.ndarray([x, y, w, h])
                    }
                }
            } ] 
        """
        tracks = []

        for frame_detections in detections:
            detections_sv = sv.Detections.from_ultralytics(frame_detections)
            # apply nms to all detections except ball
            all_detections = detections_sv[detections_sv.class_id != 0]
            ball_detections = detections_sv[detections_sv.class_id == 0]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True) 
            frame_tracks = sv.Detections.merge([all_detections, ball_detections])

            frame_tracks = self.tracker.update_with_detections(frame_tracks)
            

            current_frame_tracks = self._tracks_mapper(frame_tracks)
            tracks.append(current_frame_tracks)

        return tracks

    def _tracks_mapper(self, tracks: sv.Detections) -> dict:
        """
        Maps tracks to a dictionary by class and tracker ID. Also, adjusts bounding boxes to 1920x1080 resolution.

        Args:
            tracks (sv.Detections): Tracks from the frame.
        Returns:
            dict: Mapped detections for the frame.
        """
        results={
            'players':{},
            'referees':{},
            'ball':{},
            'goalkeepers':{}
        }

        bboxes = tracks.xyxy
        class_ids = tracks.class_id 
        tracker_ids = tracks.tracker_id  
        confs = tracks.confidence 

        # Iterate over all tracks
        for bbox, class_id, track_id, conf in zip(bboxes, class_ids, tracker_ids, confs):

            # if self.classes[class_id] not in results:
            #     results[self.classes[class_id]] = {}

            # if track_id not in results[self.classes[class_id]]:
            if self.classes[class_id] == "player":
                results["players"][track_id] = {"bbox":bbox}
            
            if self.classes[class_id] == "referee":
                results["referees"][track_id] = {"bbox":bbox}

            if self.classes[class_id] == "ball":
                results["ball"][1] = {"bbox":bbox}
            
            if self.classes[class_id] == "goalkeeper":
                results["goalkeepers"][track_id] = {"bbox":bbox}

        return results
