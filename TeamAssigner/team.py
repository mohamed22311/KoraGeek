from dataclasses import dataclass
from typing import Tuple
from supervision import Color

@dataclass(init=True)
class Team:
    """
    A class to represent a football team.

    Attributes:
        name (str): The name of the team.
        player_jersey_color (sv.Color): The jersey color of the players.
        goalkeeper_jersey_color (sv.Color): The jersey color of the goalkeeper.
    """
    name: str
    player_jersey_color: Color
    goalkeeper_jersey_color: Color