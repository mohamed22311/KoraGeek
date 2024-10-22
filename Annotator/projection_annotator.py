from typing import Any, Dict, List, Optional, Tuple
import supervision as sv
import numpy as np
import cv2
from TeamAssigner import Team

from utils import Settings
from .base_annotator import BaseAnnotator

class ProjectionAnnotator(BaseAnnotator):
    """
    Class to annotate projections on a projection image, including Voronoi regions for players (and goalkeepers), 
    and different markers for ball, players, referees, and goalkeepers.
    """
    def __init__(self, settings: Settings, team1: Team, team2: Team) -> None:
        """
        Initializes the ProjectionAnnotator with the settings.

        Args:

            settings (Settings): The settings used to configure the annotator.
        """
        self.settings = settings
        self.team1 = team1
        self.team2 = team2


    def annotate(self, frame: np.ndarray, tracks: Dict[str, Dict[int, Dict[str, Any]]], draw_voronoi: bool = True) -> np.ndarray:
        """
        Annotates an image with projected player, goalkeeper, referee, and ball positions, along with Voronoi regions.
        
        Parameters:
            frame (np.ndarray): The image on which to draw the annotations.
            tracks (Dict): A dictionary containing tracking information for 'player', 'goalkeeper', 'referee', and 'ball'.

        Returns:
            np.ndarray: The annotated frame.
        """
        pitch_ball_xy = None
        if 1 in tracks['ball']:
            pitch_ball_xy = np.array( [tracks['ball'][1]['projection']])

        pitch_referee_xy = np.array([track['projection'] for track in tracks['referees'].values()])
        
        pitch_player_team1_xy = np.array([track['projection'] for track in tracks['players'].values() if track['team'] == self.team1.name])
        pitch_player_team2_xy = np.array([track['projection'] for track in tracks['players'].values() if track['team'] == self.team2.name])
        
        goalkeepers_team1 = np.array([track['projection'] for track in tracks['goalkeepers'].values() if track['team'] == self.team1.name])
        goalkeepers_team2 = np.array([track['projection'] for track in tracks['goalkeepers'].values() if track['team'] == self.team2.name])
        
        arrays_to_concat = [pitch_player_team1_xy]
        if goalkeepers_team1.size > 0:
            arrays_to_concat.append(goalkeepers_team1)
        
        pitch_player_team1_xy = np.concatenate(arrays_to_concat)

        arrays_to_concat = [pitch_player_team2_xy]
        if goalkeepers_team2.size > 0:
            arrays_to_concat.append(goalkeepers_team2)

        pitch_player_team2_xy = np.concatenate(arrays_to_concat)

        annotated_frame = self._draw_pitch()
        voronoi_frame = self._draw_pitch()
        blended_frame = self._draw_pitch()

        # Draw Ball
        annotated_frame = self._draw_points_on_pitch(xy=pitch_ball_xy,
                                                    face_color=sv.Color.WHITE,
                                                    edge_color=sv.Color.BLACK,
                                                    radius=10,
                                                    pitch=annotated_frame)
        
        # Draw Players Team 1
        annotated_frame = self._draw_points_on_pitch(xy=pitch_player_team1_xy,
                                                    face_color=sv.Color.from_hex('00BFFF'),
                                                    edge_color=sv.Color.BLACK,
                                                    radius=16,
                                                    pitch=annotated_frame)
        
        # Draw Players Team 2
        annotated_frame = self._draw_points_on_pitch(xy=pitch_player_team2_xy,
                                                    face_color=sv.Color.from_hex('FF1493'),
                                                    edge_color=sv.Color.BLACK,
                                                    radius=16,
                                                    pitch=annotated_frame)
        
        # Draw Referees
        annotated_frame = self._draw_points_on_pitch(xy=pitch_referee_xy,
                                                    face_color=sv.Color.from_hex('FFD700'),
                                                    edge_color=sv.Color.BLACK,
                                                    radius=16,
                                                    pitch=annotated_frame)

        voronoi_frame = self._draw_pitch_voronoi_diagram(team_1_xy=pitch_player_team1_xy,
                                                        team_2_xy=pitch_player_team2_xy,
                                                        team_1_color=sv.Color.from_hex('00BFFF'),
                                                        team_2_color=sv.Color.from_hex('FF1493'),
                                                        pitch=voronoi_frame)
        
    
        
        blended_frame = self._draw_pitch_voronoi_diagram_2(team_1_xy=pitch_player_team1_xy,
                                                                team_2_xy=pitch_player_team2_xy,
                                                                team_1_color=sv.Color.from_hex('00BFFF'),
                                                                team_2_color=sv.Color.from_hex('FF1493'),
                                                                pitch=blended_frame)
        
        blended_frame = self._draw_points_on_pitch(xy=pitch_ball_xy,
                                                face_color=sv.Color.WHITE,
                                                edge_color=sv.Color.WHITE,
                                                radius=8,
                                                thickness=1,
                                                pitch=blended_frame)
    
        blended_frame = self._draw_points_on_pitch(xy=pitch_player_team1_xy,
                                                face_color=sv.Color.from_hex('00BFFF'),
                                                edge_color=sv.Color.WHITE,
                                                radius=16,
                                                thickness=1,
                                                pitch=blended_frame)
        
        blended_frame = self._draw_points_on_pitch(xy=pitch_player_team2_xy,
                                                        face_color=sv.Color.from_hex('FF1493'),
                                                        edge_color=sv.Color.WHITE,
                                                        radius=16,
                                                        thickness=1,
                                                        pitch=blended_frame)
        
        # Combine orignal frame with annotated frame to make the annotated frame like small map in the bottom middle center of the orignal frame with low opacity 
        original_frame = self._overlay_minimap_on_frame(frame, annotated_frame, opacity=0.5, scale=0.2)

        return annotated_frame, voronoi_frame, blended_frame, original_frame
        

    def _draw_pitch(self,
                   background_color: sv.Color = sv.Color(34, 139, 34),
                   line_color: sv.Color = sv.Color.WHITE,
                   padding: int = 50,
                   line_thickness: int = 4,
                   point_radius: int = 8,
                   scale: float = 0.1) -> np.ndarray:
        """
        Draws a soccer pitch with specified dimensions, colors, and scale.

        Args:
            self.settings (SoccerPitchConfiguration): Configuration object containing the
                dimensions and layout of the pitch.
            background_color (sv.Color, optional): Color of the pitch background.
                Defaults to sv.Color(34, 139, 34).
            line_color (sv.Color, optional): Color of the pitch lines.
                Defaults to sv.Color.WHITE.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            line_thickness (int, optional): Thickness of the pitch lines in pixels.
                Defaults to 4.
            point_radius (int, optional): Radius of the penalty spot points in pixels.
                Defaults to 8.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.

        Returns:
            np.ndarray: Image of the soccer pitch.
        """
        scaled_width = int(self.settings.PITCH_WIDTH * scale)
        scaled_length = int(self.settings.PITCH_LENGTH * scale)
        scaled_circle_radius = int(self.settings.PITCH_CENTER_CIRCLE_RADIUS * scale)
        scaled_penalty_spot_distance = int(self.settings.PITCH_PENALTY_SPOT_DISTANCE * scale)

        pitch_width = scaled_width + 2 * padding
        pitch_length = scaled_length + 2 * padding
        pitch_shape = (pitch_width, pitch_length, 3)
        pitch_image = np.ones(pitch_shape,dtype=np.uint8) * np.array(background_color.as_bgr(), dtype=np.uint8)

        for start, end in self.settings.edges:

            point1 = (int(self.settings.vertices()[start - 1][0] * scale) + padding,
                    int(self.settings.vertices()[start - 1][1] * scale) + padding)
            
            point2 = (int(self.settings.vertices()[end - 1][0] * scale) + padding,
                    int(self.settings.vertices()[end - 1][1] * scale) + padding)
            
            cv2.line(
                img=pitch_image,
                pt1=point1,
                pt2=point2,
                color=line_color.as_bgr(),
                thickness=line_thickness
            )

        centre_circle_center = ( scaled_length // 2 + padding, scaled_width // 2 + padding)
        
        cv2.circle(
            img=pitch_image,
            center=centre_circle_center,
            radius=scaled_circle_radius,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

        penalty_spots = [
            (scaled_penalty_spot_distance + padding,scaled_width // 2 + padding),
            
            (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
        ]

        for spot in penalty_spots:

            cv2.circle(
                img=pitch_image,
                center=spot,
                radius=point_radius,
                color=line_color.as_bgr(),
                thickness=-1
            )

        return pitch_image
    

    def _draw_points_on_pitch(self,
                             xy: np.ndarray,
                             face_color: sv.Color = sv.Color.RED,
                             edge_color: sv.Color = sv.Color.BLACK,
                             radius: int = 10,
                             thickness: int = 2,
                             padding: int = 50,
                             scale: float = 0.1,
                             pitch: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draws points on a soccer pitch.

        Args:
            xy (np.ndarray): Array of points to be drawn, with each point represented by
                its (x, y) coordinates.
            face_color (sv.Color, optional): Color of the point faces.
                Defaults to sv.Color.RED.
            edge_color (sv.Color, optional): Color of the point edges.
                Defaults to sv.Color.BLACK.
            radius (int, optional): Radius of the points in pixels.
                Defaults to 10.
            thickness (int, optional): Thickness of the point edges in pixels.
                Defaults to 2.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.
            pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
                If None, a new pitch will be created. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer pitch with points drawn on it.
        """
        if xy is None or len(xy) == 0:
            return pitch
        
        if pitch is None:
            pitch = self._draw_pitch(padding=padding,scale=scale)

        for point in xy:

            scaled_point = ( int(point[0] * scale) + padding, int(point[1] * scale) + padding)
            
            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=radius,
                color=face_color.as_bgr(),
                thickness=-1
            )

            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=radius,
                color=edge_color.as_bgr(),
                thickness=thickness
            )

        return pitch
    

    def _draw_paths_on_pitch(self,
                             paths: List[np.ndarray],
                             color: sv.Color = sv.Color.WHITE,
                             thickness: int = 2,
                             padding: int = 50,
                             scale: float = 0.1,
                             pitch: Optional[np.ndarray] = None
                             ) -> np.ndarray:
        """
        Draws paths on a soccer pitch.

        Args:
            paths (List[np.ndarray]): List of paths, where each path is an array of (x, y)
                coordinates.
            color (sv.Color, optional): Color of the paths.
                Defaults to sv.Color.WHITE.
            thickness (int, optional): Thickness of the paths in pixels.
                Defaults to 2.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.
            pitch (Optional[np.ndarray], optional): Existing pitch image to draw paths on.
                If None, a new pitch will be created. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer pitch with paths drawn on it.
        """
        if pitch is None:
            pitch = self._draw_pitch(padding=padding, scale=scale)

        for path in paths:

            scaled_path = [
                (int(point[0] * scale) + padding, int(point[1] * scale) + padding)
                for point in path if point.size > 0
            ]

            if len(scaled_path) < 2:
                continue

            for i in range(len(scaled_path) - 1):

                cv2.line(
                    img=pitch,
                    pt1=scaled_path[i],
                    pt2=scaled_path[i + 1],
                    color=color.as_bgr(),
                    thickness=thickness
                )

            return pitch
        

    def _draw_pitch_voronoi_diagram(self,
                                    team_1_xy: np.ndarray,
                                    team_2_xy: np.ndarray,
                                    team_1_color: sv.Color = sv.Color.RED,
                                    team_2_color: sv.Color = sv.Color.WHITE,
                                    opacity: float = 0.5,
                                    padding: int = 50,
                                    scale: float = 0.1,
                                    pitch: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draws a Voronoi diagram on a soccer pitch representing the control areas of two
        teams.

        Args:
            team_1_xy (np.ndarray): Array of (x, y) coordinates representing the positions
                of players in team 1.
            team_2_xy (np.ndarray): Array of (x, y) coordinates representing the positions
                of players in team 2.
            team_1_color (sv.Color, optional): Color representing the control area of
                team 1. Defaults to sv.Color.RED.
            team_2_color (sv.Color, optional): Color representing the control area of
                team 2. Defaults to sv.Color.WHITE.
            opacity (float, optional): Opacity of the Voronoi diagram overlay.
                Defaults to 0.5.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.
            pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
                Voronoi diagram on. If None, a new pitch will be created. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer pitch with the Voronoi diagram overlay.
        """
        if pitch is None:
            pitch = self._draw_pitch(padding=padding,scale=scale)

        scaled_width = int(self.settings.PITCH_WIDTH * scale)
        scaled_length = int(self.settings.PITCH_LENGTH * scale)

        voronoi = np.zeros_like(pitch, dtype=np.uint8)

        team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
        team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

        y_coordinates, x_coordinates = np.indices(( scaled_width + 2 * padding, scaled_length + 2 * padding))

        y_coordinates -= padding
        x_coordinates -= padding

        def calculate_distances(xy, x_coordinates, y_coordinates):
            return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                        (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)


           

        distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
        distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

        min_distances_team_1 = np.min(distances_team_1, axis=0)
        min_distances_team_2 = np.min(distances_team_2, axis=0)

        control_mask = min_distances_team_1 < min_distances_team_2

        voronoi[control_mask] = team_1_color_bgr
        voronoi[~control_mask] = team_2_color_bgr

        overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

        return overlay

            
    def _draw_pitch_voronoi_diagram_2(self,
                                     team_1_xy: np.ndarray,
                                     team_2_xy: np.ndarray,
                                     team_1_color: sv.Color = sv.Color.RED,
                                     team_2_color: sv.Color = sv.Color.WHITE,
                                     opacity: float = 0.5,
                                     padding: int = 50,
                                     scale: float = 0.1,
                                     pitch: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draws a Voronoi diagram on a soccer pitch representing the control areas of two
        teams with smooth color transitions.

        Args:
            config (SoccerPitchConfiguration): Configuration object containing the
                dimensions and layout of the pitch.
            team_1_xy (np.ndarray): Array of (x, y) coordinates representing the positions
                of players in team 1.
            team_2_xy (np.ndarray): Array of (x, y) coordinates representing the positions
                of players in team 2.
            team_1_color (sv.Color, optional): Color representing the control area of
                team 1. Defaults to sv.Color.RED.
            team_2_color (sv.Color, optional): Color representing the control area of
                team 2. Defaults to sv.Color.WHITE.
            opacity (float, optional): Opacity of the Voronoi diagram overlay.
                Defaults to 0.5.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.
            pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
                Voronoi diagram on. If None, a new pitch will be created. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer pitch with the Voronoi diagram overlay.
        """
        if pitch is None:
            pitch = self._draw_pitch(padding=padding,scale=scale)

        scaled_width = int(self.settings.PITCH_WIDTH * scale)
        scaled_length = int(self.settings.PITCH_LENGTH * scale)

        voronoi = np.zeros_like(pitch, dtype=np.uint8)

        team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
        team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

        y_coordinates, x_coordinates = np.indices((scaled_width + 2*padding, scaled_length + 2*padding))

        y_coordinates -= padding
        x_coordinates -= padding

        def calculate_distances(xy, x_coordinates, y_coordinates):
            return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                        (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)
        
        distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
        distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

        min_distances_team_1 = np.min(distances_team_1, axis=0)
        min_distances_team_2 = np.min(distances_team_2, axis=0)

        # Increase steepness of the blend effect
        steepness = 15  # Increased steepness for sharper transition
        distance_ratio = min_distances_team_2 / np.clip(min_distances_team_1 + min_distances_team_2, a_min=1e-5, a_max=None)
        blend_factor = np.tanh((distance_ratio - 0.5) * steepness) * 0.5 + 0.5

        # Create the smooth color transition
        for c in range(3):  # Iterate over the B, G, R channels
            voronoi[:, :, c] = (blend_factor * team_1_color_bgr[c] +
                                (1 - blend_factor) * team_2_color_bgr[c]).astype(np.uint8)

        overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

        return overlay
    

    def _overlay_minimap_on_frame(self, frame: np.ndarray, minimap: np.ndarray, opacity: float=0.5, scale: float=0.2):
        """
        Overlay the minimap image on the frame at the bottom center with a given opacity and scale.

        :param frame: The original frame (soccer game image), expected as a 3-channel RGB image.
        :param minimap: The mini-map image, either RGB or grayscale.
        :param opacity: The transparency level of the mini-map (0.0 to 1.0).
        :param scale: The scale factor to resize the mini-map (0.2 means 20% of the frame width).
        :return: The frame with the mini-map overlay.
        """
        # Get the dimensions of the frame
        frame_height, frame_width = frame.shape[:2]

        # Resize the mini-map to be smaller (bottom center) based on the scale of the frame width
        minimap_width = int(frame_width * scale)
        minimap_height = int(minimap.shape[0] * minimap_width / minimap.shape[1])
        minimap_resized = cv2.resize(minimap, (minimap_width, minimap_height))

        # Calculate the position (bottom center)
        x_offset = (frame_width - minimap_width) // 2  # Horizontally center the minimap
        y_offset = frame_height - minimap_height - 10  # Place it 10 pixels from the bottom

        # Extract the region of interest (ROI) from the frame where the mini-map will be placed
        roi = frame[y_offset:y_offset + minimap_height, x_offset:x_offset + minimap_width]

        # Convert minimap to have 3 channels if it's grayscale
        if len(minimap_resized.shape) == 2:
            minimap_resized = cv2.cvtColor(minimap_resized, cv2.COLOR_GRAY2BGR)

        # Blend the mini-map with the frame (overlay with opacity)
        blended = cv2.addWeighted(roi, 1 - opacity, minimap_resized, opacity, 0)

        # Replace the ROI on the original frame with the blended image
        frame[y_offset:y_offset + minimap_height, x_offset:x_offset + minimap_width] = blended

        return frame