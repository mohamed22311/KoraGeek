from typing import Dict, List
from Tracker import ObjectTracker, KeypointTracker
from TeamAssigner import TeamAssigner, Team
from supervision import Color
from utils import Settings
from ViewTransformer import ViewTransformer, CameraMovementEstimator
from BallAssigner import BallPlayerAssigner
from SpeedEstimator import SpeedEstimator
from Annotator import ObjectAnnotator, KeypointAnnotator, ProjectionAnnotator

from numpy import ndarray, zeros, uint8
import cv2
from supervision import Detections, crop_image

class FootballVideoProcessor():
    """
    A video processor for football footage that tracks objects and keypoints,
    estimates speed, assigns the ball to player, calculates the ball possession 
    and adds various annotations.
    """

    def __init__(self,
                 object_tracker: ObjectTracker,
                 object_annotator: ObjectAnnotator,
                 keypoints_tracker: KeypointTracker, 
                 keypoints_annotator: KeypointAnnotator,
                 team_assigner: TeamAssigner,
                 ball_player_assigner: BallPlayerAssigner,
                 projection_annotator: ProjectionAnnotator,
                 speed_estimator: SpeedEstimator,
                 transformer: ViewTransformer,
                 camera_movement_estimator: CameraMovementEstimator,
                 settings: Settings,
                 display_map: bool=True,
                 display_keypoints: bool=True,
                 read_from_stub: bool=False,
                 stub_name: str=None,
                 ) -> None:
        """
        Initializes the FootballVideoProcessor with the necessary trackers, assigners, estimator, transformer, settings, and field image.
        
        Args:
            object_tracker (ObjectTracker): The object tracker used to track objects in the video.
            keypoints_tracker (KeypointTracker): The keypoint tracker used to track keypoints in the video.
            team_assigner (TeamAssigner): The team assigner used to assign players to teams.
            ball_player_assigner (BallPlayerAssigner): The ball player assigner used to assign the ball to a player.
            speed_estimator (SpeedEstimator): The speed estimator used to estimate the speed of players.
            transformer (ViewTransformer): The view transformer used to transform the video to a top-down view.
            settings (Settings): The settings used to configure the video processor.
            read_from_stub (bool): Whether to read the detections from a stub file.
            stub_name (str): The name of the stub file to read the detections from.
        """
        self.object_tracker : ObjectTracker = object_tracker
        self.object_annotator : ObjectAnnotator = object_annotator
        self.keypoints_tracker : KeypointTracker = keypoints_tracker
        self.keypoints_annotator : KeypointAnnotator = keypoints_annotator
        self.team_assigner : TeamAssigner = team_assigner
        self.ball_player_assigner : BallPlayerAssigner = ball_player_assigner
        self.projection_annotator : ProjectionAnnotator = projection_annotator
        self.speed_estimator : SpeedEstimator = speed_estimator
        self.transformer : ViewTransformer = transformer
        self.camera_movement_estimator : CameraMovementEstimator = camera_movement_estimator
        self.settings : Settings = settings
        self.display_map : bool = display_map
        self.read_from_stub : bool = read_from_stub
        self.stub_name : str = stub_name
        self.display_keypoints : bool = display_keypoints

    
    def process_video(self, video_frames: List[ndarray], fps:int) -> List[ndarray]:
        """
        Processes the video frames by tracking objects and keypoints, estimating speed, assigning the ball to a player,
        calculating ball possession, and adding annotations.
        
        Args:
            video_frames (List[ndarray]): The list of video frames to process.
            fps (int): The frames per second of the video.

        Returns:
            List[ndarray]: The processed video frames.
        """
        processed_frames = []
        annotated_frames = []
        voronoi_frames = []
        blended_frames = []

        # Get Detections
        object_detections = self.object_tracker.get_detections(video_frames, read_from_stub=True,stub_name=self.stub_name)
        keypoints_detections = self.keypoints_tracker.get_detections(video_frames, read_from_stub=True,stub_name=self.stub_name)

        # Get Tracks
        object_tracks = self.object_tracker.get_tracks(detections=object_detections)
        keypoints_tracks, filters = self.keypoints_tracker.get_tracks(detections=keypoints_detections)

        # Fit Team Assigner
        self.team_assigner.fit(frames=video_frames, all_tracks=object_tracks)

        # calculate camera movement
        camera_movement = self.camera_movement_estimator.get_camera_movement(video_frames)

        for frame_num, frame in enumerate(video_frames):
            
            frame_object_tracks = object_tracks[frame_num]
            frame_keypoints_tracks = keypoints_tracks[frame_num]

            frame_tracks = {
                'object_tracks': frame_object_tracks,
                'keypoints_tracks': frame_keypoints_tracks
            }
            
            # Assign Player Teams
            frame_tracks['object_tracks'] = self.team_assigner.assign_teams(frame, frame_object_tracks)

            # Transform to top-down view
            frame_tracks['object_tracks'] = self.transformer.transform(object_tracks=frame_tracks['object_tracks'], keypoints_tracks=frame_tracks['keypoints_tracks'], filter=filters[frame_num])

            frame_tracks['object_tracks'] = self.transformer.adjust_transforms(object_tracks=frame_tracks['object_tracks'], camera_movement=camera_movement[frame_num])
            
            point1, point2 = None, None
            for key, val in enumerate(frame_tracks['keypoints_tracks'].xy[0]):
                if key == 8:
                    point1 = (val[0], val[1])
                elif key == 24:
                    point2 = (val[0], val[1])
                else:
                    continue
            # Assign Ball to closest Player
            frame_tracks['object_tracks'], _ = self.ball_player_assigner.assign(
                tracks=frame_tracks['object_tracks'],
                current_frame=frame_num,
                penalty_point_1_pos=point1, # keypoint for player 1
                penalty_point_2_pos=point2  # keypoint for player 2
            )
            

            # Speed Estimation for Players
            frame_tracks['object_tracks'] = self.speed_estimator.calculate_speed(
                tracks=frame_tracks['object_tracks'],
                frame_number=frame_num,
            )

            # Annotate the current frame with the tracking information
            frame, annoteted_frame, voronoi_frame, blended_frame = self.annotate(frame, frame_tracks)

            # Append the annotated frame to the processed frames list
            processed_frames.append(frame)
            annotated_frames.append(annoteted_frame)
            voronoi_frames.append(voronoi_frame)
            blended_frames.append(blended_frame)
        
        return processed_frames, annotated_frames, voronoi_frames, blended_frames
    
    def annotate(self, frame: ndarray, tracks: Dict) -> ndarray:
        """
        Annotates the given frame with analised data

        Args:
            frame (ndarray): The current video frame to be annotated.
            tracks (Dict[str, Dict[int, ndarray]]): A dictionary containing tracking data for objects and keypoints.

        Returns:
            ndarray: The annotated video frame.
        """
         
        frame = self.object_annotator.annotate(frame, tracks['object_tracks'])

        if self.display_keypoints:
            frame = self.keypoints_annotator.annotate(frame, tracks['keypoints_tracks'])

        annoteted_frame = None
        voronoi_frame = None
        blended_frame = None
        if self.display_map:
            #projection = self.projection_annotator.annotate(frame, tracks=tracks['object_tracks'])
            #frame = self._combine_frame_projection(frame, projection)
            annoteted_frame, voronoi_frame, blended_frame, frame = self.projection_annotator.annotate(frame, tracks['object_tracks'])
                    
        return frame, annoteted_frame, voronoi_frame, blended_frame


    def _combine_frame_projection(self, frame: ndarray, projection_frame: ndarray) -> ndarray:
        """
        Combines the original video frame with the projection of player positions on the field image.

        Args:
            frame (ndarray): The original video frame.
            projection_frame (ndarray): The projected field image with annotations.

        Returns:
            ndarray: The combined frame.
        """
        # Target canvas size
        canvas_width, canvas_height = 1920, 1080
        
        # Get dimensions of the original frame and projection frame
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape

        # Scale the projection to 70% of its original size
        scale_proj = 0.2

        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)

        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))

        # Create a blank canvas of 1920x1080
        combined_frame = zeros((canvas_height, canvas_width, 3), dtype=uint8)

        # Copy the main frame onto the canvas (top-left corner)
        combined_frame[:h_frame, :w_frame] = frame

        # Set the position for the projection frame at the bottom-middle
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25  # 25px margin from bottom

        # Blend the projection with 75% visibility (alpha transparency)
        alpha = 0.60
        overlay = combined_frame[y_offset:y_offset + new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha, overlay, 1 - alpha, 0, overlay)

        return combined_frame

            