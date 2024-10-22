from enum import Enum

class Position(Enum):
    """
    Enum representing the position of an anchor point.
    """

    CENTER = "CENTER"
    CENTER_LEFT = "CENTER_LEFT"
    CENTER_RIGHT = "CENTER_RIGHT"
    TOP_CENTER = "TOP_CENTER"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    CENTER_OF_MASS = "CENTER_OF_MASS"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))