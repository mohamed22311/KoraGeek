import math
from collections import deque
from typing import Dict, Any, Tuple

class SpeedEstimator:
    """Estimates the speed of objects (km/h)."""

    def __init__(self,
                 field_image_width: int = 780,
                 field_image_height: int = 1150,
                 real_field_length: float = 105,  # in meters (FIFA standard length)
                 real_field_width: float = 68,    # in meters (FIFA standard width)
                 smoothing_window: int = 5,
                 fps: int = 24) -> None:
        """
        Initialize the SpeedEstimator with the field dimensions and real-world measurements.

        Args:
            field_image_width (int): Width of the field in pixels.
            field_image_height (int): Height of the field in pixels.
            real_field_length (float): Real-world length of the field in meters.
            real_field_width (float): Real-world width of the field in meters.
            smoothing_window (int): Number of frames to consider for speed smoothing.
            fps (int): Frames per second of the video.
        """
        self.field_image_width = field_image_width
        self.field_image_height = field_image_height

        self.real_field_length = real_field_length  # in meters
        self.real_field_width = real_field_width    # in meters

        self.previous_positions: Dict[int, Tuple[Tuple[float, float], int]] = {}  # player_id -> (position, frame_number)

        self.speed_history: Dict[int, deque] = {}  # player_id -> deque of last window_size speeds for smoothing

        self.smoothing_window = smoothing_window

        self.fps = fps

        # Calculate scaling factors (pixels to meters conversion)
        self.scale_x = real_field_length / field_image_width
        self.scale_y = real_field_width / field_image_height

        # Maximum realistic speed (km/h)
        self.max_speed = 40.0

    def calculate_speed(self, tracks: Dict[str, Dict[int, Dict[str, Any]]], frame_number: int) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Calculate the speed of players based on their projections and update the track information.

        Args:
            tracks (Dict[str, Dict[int, Dict[str, Any]]]): A dictionary containing tracking information for players.
            frame_number (int): The current frame number of the video.

        Returns:
            Dict[str, Dict[int, Dict[str, Any]]]: Updated tracks with calculated speeds.
        """
        for track_type in tracks:

            if track_type == 'ball' or track_type == 'referees':
                continue  # Skip ball and referee speed calculations

            for player_id, track in tracks[track_type].items():
                if 'projection' in track:
                    current_position = track['projection']  # The player's current position (x, y)

                    if player_id in self.previous_positions:
                        prev_position, prev_frame = self.previous_positions[player_id]

                        # Calculate the distance in meters
                        distance = self._calculate_distance(prev_position, current_position)/80

                        # Calculate the time difference in seconds
                        time_diff = (frame_number - prev_frame) / self.fps

                        # Prevent division by zero
                        if time_diff <= 0:
                            time_diff = 1 / self.fps

                        # Calculate speed in km/h
                        speed = (distance / time_diff) * 3.6  # Convert m/s to km/h

                        # Apply maximum speed check
                        speed = min(speed, self.max_speed)

                        # Apply smoothing to speed
                        smoothed_speed = self._smooth_speed(player_id, speed)

                        # Update the player's speed and distance in the track
                        tracks[track_type][player_id]['speed'] = smoothed_speed
                        tracks[track_type][player_id]['distance'] = distance

                        # Debugging: Print values to understand what's happening
                        print(f"Player {player_id} - Distance: {distance:.2f} meters, "
                              f"Time: {time_diff:.2f} seconds, Raw Speed: {speed:.2f} km/h")
                        print(f"Player {player_id} - Smoothed Speed: {smoothed_speed:.2f} km/h")

                    else:
                        # If it's the first time we're seeing this player, set speed to 0
                        tracks[track_type][player_id]['speed'] = 0.0
                        tracks[track_type][player_id]['distance'] = 0.0
                        self.speed_history[player_id] = deque([0.0] * self.smoothing_window, maxlen=self.smoothing_window)

                    # Update the previous position with the current position and frame number
                    self.previous_positions[player_id] = (current_position, frame_number)
                else:
                    # If there's no projection, set speed to 0
                    tracks[track_type][player_id]['speed'] = 0.0
                    tracks[track_type][player_id]['distance'] = 0.0

        return tracks

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate the Euclidean distance between two positions in meters.

        Args:
            pos1 (Tuple[float, float]): The first position (x, y) in pixels.
            pos2 (Tuple[float, float]): The second position (x, y) in pixels.

        Returns:
            float: The distance in meters.
        """
        dx = (pos2[0] - pos1[0]) * self.scale_x
        dy = (pos2[1] - pos1[1]) * self.scale_y
        return math.sqrt(dx**2 + dy**2)

    def _smooth_speed(self, player_id: Any, speed: float) -> float:
        """
        Smooth the speed measurement using a moving average.

        Args:
            player_id (Any): The identifier for the player.
            speed (float): The calculated speed to be smoothed.

        Returns:
            float: The smoothed speed value.
        """
        if player_id not in self.speed_history:
            self.speed_history[player_id] = deque([0.0] * self.smoothing_window, maxlen=self.smoothing_window)

        self.speed_history[player_id].append(speed)
        return sum(self.speed_history[player_id]) / len(self.speed_history[player_id])

    def reset(self) -> None:
        """
        Reset the previous positions and speed history. 
        Call this at the start of a new video or when needed.
        """
        self.previous_positions = {}
        self.speed_history = {}
