from dataclasses import field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Tuple, Union
import torch

class Settings(BaseSettings):
    """
    Class to store all the settings used in the project and handle pitch-specific geometry.
    """

    # Annotation Colors
    PLAYER_BALL_ANNOTATION_COLOR: Union[str, List[str]]
    BALL_ANNOTATION_COLOR: Union[str, List[str]]
    OBJECT_ANNOTATION_COLOR: Union[str, List[str]]
    LABEL_ANNOTATION_COLOR: Union[str, List[str]]
    TEXT_ANNOTATION_COLOR: Union[str, List[str]]
    KEYPOINT_ANNOTATION_COLOR: str
    KEYPOINT_TEXT_ANNOTATION_COLOR: str
    OUTLINE_COLOR: str

    # Device
    DEVICE: Union[str, torch.device] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_VIDEO_PATH: str = "videos/input/"
    INPUT_VIDEO_NAME: str = "0bfacc_0.mp4"

    # Pitch Dimensions
    PITCH_WIDTH: int = 7000  
    PITCH_LENGTH: int = 12000  
    PITCH_PENALTY_BOX_WIDTH: int = 4100  
    PITCH_PENALTY_BOX_LENGTH: int = 2015  
    PITCH_GOAL_BOX_WIDTH: int = 1832  
    PITCH_GOAL_BOX_LENGTH: int = 550  
    PITCH_CENTER_CIRCLE_RADIUS: int = 915  
    PITCH_PENALTY_SPOT_DISTANCE: int = 1100  

    PITCH_BACKGROUND_COLOR: str = "#228b22"
    PITCH_LINE_COLOR: str = "#FFFFFF"

    # PITCH EDGES
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])

    # PITCH LABELS
    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "14", "19"
    ])

    # PITCH COLORS
    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#00BFFF", "#00BFFF"
    ])

    # Method to compute pitch vertices
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (0, 0),  # 1
            (0, (self.PITCH_WIDTH - self.PITCH_PENALTY_BOX_WIDTH) / 2),  # 2
            (0, (self.PITCH_WIDTH - self.PITCH_GOAL_BOX_WIDTH) / 2),  # 3
            (0, (self.PITCH_WIDTH + self.PITCH_GOAL_BOX_WIDTH) / 2),  # 4
            (0, (self.PITCH_WIDTH + self.PITCH_PENALTY_BOX_WIDTH) / 2),  # 5
            (0, self.PITCH_WIDTH),  # 6
            (self.PITCH_GOAL_BOX_LENGTH, (self.PITCH_WIDTH - self.PITCH_GOAL_BOX_WIDTH) / 2),  # 7
            (self.PITCH_GOAL_BOX_LENGTH, (self.PITCH_WIDTH + self.PITCH_GOAL_BOX_WIDTH) / 2),  # 8
            (self.PITCH_PENALTY_SPOT_DISTANCE, self.PITCH_WIDTH / 2),  # 9
            (self.PITCH_PENALTY_BOX_LENGTH, (self.PITCH_WIDTH - self.PITCH_PENALTY_BOX_WIDTH) / 2),  # 10
            (self.PITCH_PENALTY_BOX_LENGTH, (self.PITCH_WIDTH - self.PITCH_GOAL_BOX_WIDTH) / 2),  # 11
            (self.PITCH_PENALTY_BOX_LENGTH, (self.PITCH_WIDTH + self.PITCH_GOAL_BOX_WIDTH) / 2),  # 12
            (self.PITCH_PENALTY_BOX_LENGTH, (self.PITCH_WIDTH + self.PITCH_PENALTY_BOX_WIDTH) / 2),  # 13
            (self.PITCH_LENGTH / 2, 0),  # 14
            (self.PITCH_LENGTH / 2, self.PITCH_WIDTH / 2 - self.PITCH_CENTER_CIRCLE_RADIUS),  # 15
            (self.PITCH_LENGTH / 2, self.PITCH_WIDTH / 2 + self.PITCH_CENTER_CIRCLE_RADIUS),  # 16
            (self.PITCH_LENGTH / 2, self.PITCH_WIDTH),  # 17
            (
                self.PITCH_LENGTH - self.PITCH_PENALTY_BOX_LENGTH,
                (self.PITCH_WIDTH - self.PITCH_PENALTY_BOX_WIDTH) / 2
            ),  # 18
            (
                self.PITCH_LENGTH - self.PITCH_PENALTY_BOX_LENGTH,
                (self.PITCH_WIDTH - self.PITCH_GOAL_BOX_WIDTH) / 2
            ),  # 19
            (
                self.PITCH_LENGTH - self.PITCH_PENALTY_BOX_LENGTH,
                (self.PITCH_WIDTH + self.PITCH_GOAL_BOX_WIDTH) / 2
            ),  # 20
            (
                self.PITCH_LENGTH - self.PITCH_PENALTY_BOX_LENGTH,
                (self.PITCH_WIDTH + self.PITCH_PENALTY_BOX_WIDTH) / 2
            ),  # 21
            (self.PITCH_LENGTH - self.PITCH_PENALTY_SPOT_DISTANCE, self.PITCH_WIDTH / 2),  # 22
            (
                self.PITCH_LENGTH - self.PITCH_GOAL_BOX_LENGTH,
                (self.PITCH_WIDTH - self.PITCH_GOAL_BOX_WIDTH) / 2
            ),  # 23
            (
                self.PITCH_LENGTH - self.PITCH_GOAL_BOX_LENGTH,
                (self.PITCH_WIDTH + self.PITCH_GOAL_BOX_WIDTH) / 2
            ),  # 24
            (self.PITCH_LENGTH, 0),  # 25
            (self.PITCH_LENGTH, (self.PITCH_WIDTH - self.PITCH_PENALTY_BOX_WIDTH) / 2),  # 26
            (self.PITCH_LENGTH, (self.PITCH_WIDTH - self.PITCH_GOAL_BOX_WIDTH) / 2),  # 27
            (self.PITCH_LENGTH, (self.PITCH_WIDTH + self.PITCH_GOAL_BOX_WIDTH) / 2),  # 28
            (self.PITCH_LENGTH, (self.PITCH_WIDTH + self.PITCH_PENALTY_BOX_WIDTH) / 2),  # 29
            (self.PITCH_LENGTH, self.PITCH_WIDTH),  # 30
            (self.PITCH_LENGTH / 2 - self.PITCH_CENTER_CIRCLE_RADIUS, self.PITCH_WIDTH / 2),  # 31
            (self.PITCH_LENGTH / 2 + self.PITCH_CENTER_CIRCLE_RADIUS, self.PITCH_WIDTH / 2),  # 32
        ]

    class Config:
        env_file = ".env"

# Function to retrieve settings
def get_settings():
    return Settings()
