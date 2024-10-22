from typing import List
import cv2
import numpy as np
from utils import point_distance, xy_distance

class CameraMovementEstimator():
    
    def __init__(self, frame: np.ndarray):
        """
        Initialize the CameraMovementEstimator object with the first frame of the video

        Args:
            frame (np.ndarray): The first frame of the video
        """
        self.minimum_distance = 5 # Minimum distance between two points to consider them as the same point in pixels

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        ) # Parameters for the Lucas-Kanade optical flow algorithm 

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Convert the first frame to grayscale

        mask_features = np.zeros_like(first_frame_grayscale) # Create a mask to limit the features to the field area 
        mask_features[:,0:20] = 1 # Left border
        mask_features[:,900:1050] = 1 # Right border

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        ) # Parameters for the goodFeaturesToTrack function to detect features in the first frame of the video 

    def get_camera_movement(self, frames: List[np.ndarray]) -> List[List[int]]:
        """
        Estimate the camera movement in each frame of the video using the Lucas-Kanade optical flow algorithm

        Args:
            frames (List[np.ndarray]): A list of frames of the video

        Returns:
            List[List[int]]: A list of the camera movement in each frame of the video
        """
        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)
            
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = point_distance(new_features_point,old_features_point)
                
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()

        return camera_movement
    

    