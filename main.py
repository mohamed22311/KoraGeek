import cv2
import numpy as np
from Tracker import ObjectTracker, KeypointTracker
from TeamAssigner import TeamAssigner, Team
from supervision import Color
from utils import get_settings, read_video, save_video
from ViewTransformer import ViewTransformer, CameraMovementEstimator
from BallAssigner import BallPlayerAssigner
from SpeedEstimator import SpeedEstimator
from VideoProcessor import FootballVideoProcessor
from Annotator import ObjectAnnotator, KeypointAnnotator, ProjectionAnnotator

def main():
    # Settings 
    settings = get_settings()

    # Read Video
    input_video_folder = settings.INPUT_VIDEO_PATH
    input_video_name = settings.INPUT_VIDEO_NAME
    input_video_path = input_video_folder + input_video_name

    video_frames, fps = read_video(input_video_path)

    # Initialize Trackers
    object_tracker = ObjectTracker('models\weights\object-detection.pt')
    keypoints_tracker = KeypointTracker('models\weights\keypoints-detection.pt')

    # Intialize Team Assigner
    team1 = Team('team1', Color.from_hex(settings.OBJECT_ANNOTATION_COLOR[0]), Color.from_hex(settings.OBJECT_ANNOTATION_COLOR[0]))
    team2 = Team('team2', Color.from_hex(settings.OBJECT_ANNOTATION_COLOR[1]), Color.from_hex(settings.OBJECT_ANNOTATION_COLOR[1]))
    team_assigner = TeamAssigner(team1=team1, team2=team2, settings=settings)

    # Initialize View Transformer
    transformer = ViewTransformer(top_down_keypoints=np.array(settings.vertices()), alpha=0.9)

    camera_movement_estimator = CameraMovementEstimator(frame=video_frames[0])

    # Initialize Ball Player Assigner
    ball_player_assigner = BallPlayerAssigner(team1=team1, team2=team2, fps=fps)
    
    # Initialize Speed Estimator
    speed_estimator = SpeedEstimator(
        field_image_width=780,
        field_image_height=1150,
        real_field_length=105,
        real_field_width=68,
        smoothing_window=5,
        fps=fps
    )

    # Initialize Annotators
    object_annotator = ObjectAnnotator(settings)
    keypoints_annotator = KeypointAnnotator(settings)
    projection_annotator = ProjectionAnnotator(settings, team1, team2)

    # Initialize Video Processor
    video_processor = FootballVideoProcessor(
        object_tracker=object_tracker,
        object_annotator=object_annotator,
        keypoints_tracker=keypoints_tracker,
        keypoints_annotator=keypoints_annotator,
        team_assigner=team_assigner,
        ball_player_assigner=ball_player_assigner,
        projection_annotator=projection_annotator,
        speed_estimator=speed_estimator,
        transformer=transformer,
        camera_movement_estimator=camera_movement_estimator,
        settings=settings,
        display_map=True,
        display_keypoints=False,
        read_from_stub=True,
        stub_name=input_video_name
    )

    output_video_frames, annoteted_frames, voronoi_frames, blended_frames = video_processor.process_video(video_frames, fps)
    
    # Save video
    if output_video_frames is None:
        print('Output video is empty')
    else:
        save_video(output_video_frames, './videos/output/new/new_output_video_4.mp4')
        print('Output video saved at output_videos/output_video.mp4')
    
    if annoteted_frames is None:
        print('Annotated video is empty')
    else:
        save_video(annoteted_frames, './videos/output/new/annotated_video_4.mp4')
        print('Annotated video saved at output_videos/annotated_video.mp4')
    
    if voronoi_frames is None:
        print('Voronoi video is empty')
    else:
        save_video(voronoi_frames, './videos/output/new/voronoi_video_4.mp4')
        print('Voronoi video saved at output_videos/voronoi_video.mp4')
    
    if blended_frames is None:
        print('Blended video is empty')
    else:
        save_video(blended_frames, './videos/output/new/blended_video_4.mp4')
        print('Blended video saved at output_videos/blended_video.mp4')

if __name__ == '__main__':
    main()
