from TeamAssigner import Team

from typing import Dict, List

class PossessionTracker:
    """Tracking the ball possession of each team"""

    def __init__(self,
                 team1: Team, team2: Team) -> None:
        """
        Initializes the PossessionTracker with team names and possession statistics.

        Args:
            team1 (Team): The first team object
            team2 (Team): The second team object
        """
        self.possession_dict: Dict[str, int] = {-1: 0, team1.name: 0, team2.name: 0}
        self.team1_name: str = team1.name
        self.team2_name: str = team2.name
        self.possession: List[Dict[int, float]] = []  # List to track possession percentages over time
        self.sum: int = 0  # Total number of possession instances

    def add_possession(self, team_name: str) -> None:
        """
        Records possession for a specific team and updates possession statistics.

        Args:
            team_name (str): The name of the team that currently has possession.
        """
        self.possession_dict[team_name] += 1
        self.sum += 1
        self.possession.append({
            -1: self.possession_dict[-1] / self.sum, 
            0: self.possession_dict[self.team1_name] / self.sum, 
            1: self.possession_dict[self.team2_name] / self.sum
        })
    
    def get_ball_possessions(self) -> List[float]:
        """
        Returns the possession percentages for each team.

        Returns:
            List[float]: The possession percentages for each team.
        """
        return self.possession[-1].values()