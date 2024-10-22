from utils import point_distance, get_bbox_center
from PossessionTracker import PossessionTracker
from TeamAssigner import Team

from typing import Dict, Tuple, Any

class BallPlayerAssigner:
    """Assigns the ball to a player if it fits the criteria"""

    def __init__(self, 
                 team1: Team, 
                 team2: Team, 
                 max_ball_player_distance: float = 10.0, 
                 grace_period: float = 4.0, 
                 ball_grace_period: float = 2.0, 
                 fps: int = 24, 
                 max_ball_speed: float = 250.0, 
                 speed_check_frames: int = 15, 
                 penalty_point_distance: float = 15.0) -> None:
        """
        Initializes the BallToPlayerAssigner with necessary parameters.

        Args:
            team1 (Team): The Team object of the first team.
            team2 (Team): The Team object of the second team.
            max_ball_player_distance (float): The maximum distance to consider a player as being able to possess the ball.
            grace_period (float): The time in seconds a player retains possession after losing the ball.
            ball_grace_period (float): The time in seconds to allow a player to retain possession after the ball detection is lost.
            fps (int): Frames per second for the video feed.
            max_ball_speed (float): The maximum allowed ball movement in pixels between frames.
            speed_check_frames (int): The number of frames to check for ball movement.
            penalty_point_distance (float): The distance from the penalty point within which the ball is considered invalid.
        """
        self.max_ball_player_distance = max_ball_player_distance
        self.grace_period_frames = int(grace_period * fps)
        self.ball_grace_period_frames = int(ball_grace_period * fps)
        self.max_ball_speed = max_ball_speed
        self.speed_check_frames = speed_check_frames
        self.possession_tracker = PossessionTracker(team1, team2)
        self.last_possession_frame = 0
        self.last_player_w_ball = 1
        self.last_possessing_team = -1
        self.ball_lost_frame = 0
        self.ball_history = []
        self.penalty_point_distance = penalty_point_distance

    def _is_ball_movement_valid(self, ball_pos: Tuple[float, float], current_frame: int) -> bool:
        """
        Checks if the ball's movement is valid based on its previous position.

        Args:
            ball_pos (Tuple[float, float]): The current position of the ball (x, y).
            current_frame (int): The current frame number.

        Returns:
            bool: True if the ball movement is valid, False otherwise.
        """
        if not self.ball_history:
            return True  # No history, so movement is valid
        
        last_ball_pos, last_frame = self.ball_history[-1]

        if current_frame - last_frame <= self.speed_check_frames:
            distance_moved = point_distance(ball_pos, last_ball_pos)

            if distance_moved > self.max_ball_speed:
                return False  # Movement is too large, likely invalid

        return True

    def _calc_ball_possession(self, tracks: Dict[str, Any], current_frame: int, penalty_point_1_pos: Tuple[float, float], penalty_point_2_pos: Tuple[float, float]) -> Tuple[Dict[str, Any], int]:
        """
        Checks if a player has possession of the ball based on various criteria.

        Args:
            tracks (Dict[str, Any]): A dictionary containing tracked objects.
            current_frame (int): The current frame number.

        Returns:
            Tuple[Dict[str, Any], int]: Updated tracks and the ID of the player with the ball.
        """
        ball_pos = (None,None)
        # Check if there's a ball track in the field 
        if 'ball' in tracks and tracks['ball']:
            if 'projection' in tracks['ball'][1]:
                
                test_ball_pos = tracks['ball'][1]['projection']  
                ball_bbox_center = get_bbox_center(tracks['ball'][1]['bbox'])

                is_near_penalty_point = False
                if penalty_point_1_pos is not None:
                    if point_distance(ball_bbox_center, penalty_point_1_pos) < self.penalty_point_distance:
                        is_near_penalty_point = True
                if penalty_point_2_pos is not None:
                    if point_distance(ball_bbox_center, penalty_point_2_pos) < self.penalty_point_distance:
                        is_near_penalty_point = True
                if not is_near_penalty_point and self._is_ball_movement_valid(test_ball_pos, current_frame):
                    ball_pos = test_ball_pos
        return ball_pos

    def assign(self, tracks: Dict[str, Any], current_frame: int, penalty_point_1_pos: Tuple[float, float], penalty_point_2_pos: Tuple[float, float]) -> Tuple[Dict[str, Any], int]:
        """
        Assigns the ball to the nearest player based on various criteria.

        Args:
            tracks (Dict[str, Any]): A dictionary containing tracked objects.
            current_frame (int): The current frame number.
            penalty_point_1_pos (Tuple[float, float]): The position of the first penalty point (x, y).
            penalty_point_2_pos (Tuple[float, float]): The position of the second penalty point (x, y).

        Returns:
            Tuple[Dict[str, Any], int]: Updated tracks and the ID of the player with the ball.
        """
        # Copy the tracks to avoid mutating the original data
        tracks = tracks.copy()
        min_dis = self.max_ball_player_distance
        players = {**tracks.get('players', {}), **tracks.get('goalkeepers', {})}
        ball_pos = self._calc_ball_possession(tracks, current_frame, penalty_point_1_pos, penalty_point_2_pos)
        elapsed_frames = 0

        if ball_pos != (None, None):
            self.ball_lost_frame = current_frame
            self.ball_history.append((ball_pos, current_frame))
            if len(self.ball_history) > self.speed_check_frames:
                self.ball_history.pop(0)
            
            elapsed_frames = current_frame - self.last_possession_frame if self.last_possession_frame else float('inf')

            for player_id, player in players.items():
                player_pos = player['projection']
                dis = point_distance(ball_pos, player_pos)
                if dis <= min_dis:
                    min_dis = dis
                    self.last_player_w_ball = player_id
                    self.last_possessing_team = players[self.last_player_w_ball]['team']
                    self.last_possession_frame = current_frame
        else:
            elapsed_frames = current_frame - self.ball_lost_frame if self.ball_lost_frame else float('inf')
                
        if self.last_player_w_ball is not None:

            if elapsed_frames <= self.grace_period_frames:

                player_w_ball = self.last_player_w_ball
                self.possession_tracker.add_possession(self.last_possessing_team)                    

                if player_w_ball in tracks['players']:
                    tracks['players'][player_w_ball]['has_ball'] = True
                    self.possession_tracker.add_possession(tracks['players'][player_w_ball]['team'])
                elif player_w_ball in tracks['goalkeepers']:
                    tracks['goalkeepers'][player_w_ball]['has_ball'] = True
                    self.possession_tracker.add_possession(tracks['goalkeepers'][player_w_ball]['team'])
                else:
                    self.possession_tracker.add_possession(self.last_possessing_team)
            else:
                self.possession_tracker.add_possession(-1)
                self.last_player_w_ball = None  
        else:
            self.possession_tracker.add_possession(-1)
        
        return tracks, self.last_player_w_ball

    def get_ball_possessions(self) -> Any:
        """
        Returns the current ball possessions tracked by the possession tracker.

        Returns:
            Any: The current ball possessions.
        """
        return self.possession_tracker.get_ball_possessions()