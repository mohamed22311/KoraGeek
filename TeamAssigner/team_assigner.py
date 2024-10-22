from .team import Team
from .team_assigner_model import TeamAssignerModel

import numpy as np
import cv2
import supervision as sv
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
from utils import Settings

class TeamAssigner:
    """A class to assign players to teams based on their jersey colors."""
    
    def __init__(self, team1: Team, team2: Team, settings: Settings):

        self.team1 = team1
        self.team2 = team2

        self.player_team_dict = {} # Dictionary to store player to team mapping

        self.team_colors: Dict[str, sv.Color] = {
            team1.name: team1.player_jersey_color,
            team2.name: team2.player_jersey_color
        }

        self.goalkeeper_colors: Dict[str, sv.Color] = {
            team1.name: team1.goalkeeper_jersey_color,
            team2.name: team2.goalkeeper_jersey_color
        }

        self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)


    def cluster(self, frame: np.ndarray):
        """
        Cluster the frame to extract the jersey colors.

        Args:
            frame (np.ndarray): The frame to extract the jersey colors from.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The cluster centers and labels.
        """
        frame = frame.reshape((-1, 3))
        self.kmeans.fit(frame)
        return self.kmeans.cluster_centers_, self.kmeans.labels_
    
    def _apply_mask(self, frame: np.ndarray, green_threshold: float = 0.08) -> np.ndarray:
        """
        Apply a mask to an frame based on green color in HSV space. 
        If the mask covers more than green_threshold of the frame, apply the inverse of the mask.

        Args:
            frame (np.ndarray): An frame to apply the mask to.
            green_threshold (float): Threshold for green color coverage.

        Returns:
            np.ndarray: The masked frame.
        """
        # Convert the frame to HSV
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the green color range in HSV
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])

        # Create the mask
        mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Count the number of masked pixels
        total_pixels = frame.shape[0] * frame.shape[1]
        masked_pixels = cv2.countNonZero(cv2.bitwise_not(mask))
        mask_percentage = masked_pixels / total_pixels
        
        if mask_percentage > green_threshold:
            # Apply inverse mask
            return cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        
        # Apply normal mask
        return frame
    
    def get_player_jersey_color(self, frame: np.ndarray, bbox: List[int]) -> sv.Color:
        """
        Get the jersey color of a player in the frame.

        Args:
            frame (np.ndarray): The frame to extract the jersey color from.
            bbox (List[int]): The bounding box of the player.
        
        Returns:
            sv.Color: The jersey color of the player.
        """
         # Extract the player image from the whole frame
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        # Get the top half of the image (to avoid the grass and the player shadow)
        top_half_image = image[0:int(image.shape[0]/2),:]
        
        # Apply the mask to the top half of the image 
        masked_img_top = self._apply_mask(frame=top_half_image, green_threshold=0.08)

        # Get Clustering model
        clusters_centers, labels = self.cluster(frame=masked_img_top)
        
        # Reshape the labels to the image shape
        clustered_image = labels.reshape(masked_img_top.shape[0],masked_img_top.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        background_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - background_cluster

        player_jersey_color_bgr = clusters_centers[player_cluster]

        #player_jersey_color = sv.Color.from_bgr_tuple(player_jersey_color_bgr)
        player_jersey_color_rgb = player_jersey_color_bgr[::-1]
        return player_jersey_color_rgb
    
    def get_player_team(self, frame: np.ndarray, bbox: List[int], player_id: int) -> str:
        """
        Get the team of a player based on their jersey color.

        Args:
            frame (np.ndarray): The frame to extract the jersey color from.
            bbox (List[int]): The bounding box of the player.
            player_id (int): The ID of the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.
        
        Returns:
            str: The team name of the player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_jersey_color(frame,bbox=bbox)

        team_id = self.model.predict(extracted_color=player_color)
        team = self.team1.name if team_id == 0 else self.team2.name
        self.player_team_dict[player_id] = team

        return team

    def assign_teams(self, frame: np.ndarray, tracks: Dict[int, Dict[str, List[int]]]) -> Dict[str, List[List[int]]]:
        """
        Assign teams to the players based on their jersey colors.

        Args:
            frame (np.ndarray): The frame to extract the jersey colors from.
            tracks (Dict[int, Dict[str, List[int]]]): The tracks of the players.

        Returns:
            Dict[str, List[List[int]]]: The tracks of the players with the assigned teams.
        """

        for track_type in ['goalkeepers', 'players']:
            for id, track in tracks[track_type].items():
                bbox = track["bbox"]
                team = self.get_player_team(frame=frame, bbox=bbox, player_id=id)
                tracks[track_type][id]['team'] = team
                tracks[track_type][id]['team_color'] = self.team_colors[team]
        return tracks
    
    def fit(self, frames: List[np.ndarray], all_tracks: Dict[int, Dict[str, List[int]]]):
        """
        Fit the team assigner to the frames and tracks.

        Args:
            frames (List[np.ndarray]): The frames to fit the team assigner to.
            all_tracks (Dict[int, Dict[str, List[int]]]): The tracks of the players.
        """
        
        player_colors = []
        for idx in range(len(all_tracks)):
            tracks = all_tracks[idx]
            frame = frames[idx]
            for track_type in ['goalkeepers', 'players']:
                for id, track in tracks[track_type].items():
                    bbox = track["bbox"]
                    player_color =  self.get_player_jersey_color(frame,bbox)
                    player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.team1.player_jersey_color = sv.Color.from_bgr_tuple(kmeans.cluster_centers_[0])
        self.team2.player_jersey_color = sv.Color.from_bgr_tuple(kmeans.cluster_centers_[1])
        self.team1.goalkeeper_jersey_color = sv.Color.from_bgr_tuple(kmeans.cluster_centers_[0])
        self.team2.goalkeeper_jersey_color = sv.Color.from_bgr_tuple(kmeans.cluster_centers_[1])
        
        self.model = TeamAssignerModel(self.team1, self.team2)

        # self.team_colors[self.team1.name] = self.team1.player_jersey_color
        # self.team_colors[self.team2.name] = self.team2.player_jersey_color