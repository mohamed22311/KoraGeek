from .team import Team
import numpy as np
from typing import Tuple

class TeamAssignerModel:
    
    def __init__(self, team1: Team, team2: Team) -> None:
        """
        Initializes the TeamAssignerModel with jersey colors for the teams.

        Args:
            team1 (team): The first team object.
            team2 (team): The second team object.
        """
        self.player_centroids = np.array([team1.player_jersey_color.as_rgb(), team2.player_jersey_color.as_rgb()])
        self.goalkeeper_centroids = np.array([team1.goalkeeper_jersey_color.as_rgb(), team2.goalkeeper_jersey_color.as_rgb()])

    def predict(self, extracted_color: Tuple[int, int, int]) -> int:
        """
        Predict the team for a given jersey color based on the centroids.

        Args:
            extracted_color (Tuple[int, int, int]): The extracted jersey color in BGR format.
            is_goalkeeper (bool): Flag to indicate if the color is for a goalkeeper.

        Returns:
            int: The index of the predicted team (0 or 1).
        """
        centroids = self.player_centroids

        # Calculate distances
        distances = np.linalg.norm(extracted_color - centroids, axis=1)
        
        return np.argmin(distances)