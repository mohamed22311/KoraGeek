from .base_annotator import BaseAnnotator
from .AnnotationDrawer import EllipseAnnotator, TriangleAnnotator, LabelAnnotator
from Enums import Position
from utils import Settings

from supervision import Color, ColorPalette
from numpy import ndarray
from typing import Dict, List

class ObjectAnnotator(BaseAnnotator):
    """Annotates objects in a frame, such as the ball, players, referees, and goalkeepers."""

    def __init__(self, settings: Settings) -> None:
        """
        Initializes the ObjectAnnotator with predefined ball and referee annotation colors.
        """
        super().__init__()
        self.settings = settings

        if isinstance(settings.BALL_ANNOTATION_COLOR, List):
            triangle_color = ColorPalette.from_hex(settings.BALL_ANNOTATION_COLOR)
        else:
            triangle_color = Color.from_hex(settings.BALL_ANNOTATION_COLOR)
        
        self.triangle_annotator = TriangleAnnotator(
            color=triangle_color,
            base=25,
            height=21,
            outline_thickness=1,
            outline_color=Color.BLACK,
            classes=['ball', 'players']
        )   
        
        if isinstance(settings.OBJECT_ANNOTATION_COLOR, List):
            ellipse_color = ColorPalette.from_hex(settings.OBJECT_ANNOTATION_COLOR)
        else:
            ellipse_color = Color.from_hex(settings.OBJECT_ANNOTATION_COLOR)

        self.ellipse_annotator = EllipseAnnotator(
            color=ellipse_color,
            thickness=2
        )
        
        if isinstance(settings.LABEL_ANNOTATION_COLOR, List):
            label_color = ColorPalette.from_hex(settings.LABEL_ANNOTATION_COLOR)
        else:
            label_color = Color.from_hex(settings.LABEL_ANNOTATION_COLOR)
        
        if isinstance(settings.TEXT_ANNOTATION_COLOR, List):
            text_color = ColorPalette.from_hex(settings.TEXT_ANNOTATION_COLOR)
        else:
            text_color = Color.from_hex(settings.TEXT_ANNOTATION_COLOR)

        self.label_annotator = LabelAnnotator(
            color=label_color,
            text_color=text_color,
            text_position=Position.BOTTOM_CENTER,
        )
        
    def annotate(self, frame: ndarray, tracks: Dict) -> ndarray:
        """
        Annotates the frame with objects like players, referees, and the ball.

        Args:
            frame (ndarray): The current frame to be annotated.
            tracks (Dict): A dictionary containing object tracking data, categorized by object types.

        Returns:
            ndarray: The annotated frame.
        """
        frame = frame.copy()

        # Iterate over the tracked objects
        for track in tracks:
            for track_id, item in tracks[track].items():

                if track == 'ball':
                    frame = self.triangle_annotator.draw(frame, item['bbox'],obj_cls='ball')
                elif track == 'referees':
                    frame = self.ellipse_annotator.draw(frame, item['bbox'], type='Dashed')
                else:
                    speed = item.get('speed', 0)
                    team_color = item.get('team_color', None)

                    if track == 'players':
                        frame = self.ellipse_annotator.draw(frame, item['bbox'], type='Normal', color=team_color)
                    else:
                        frame = self.ellipse_annotator.draw(frame, item['bbox'], type='Double', color=team_color)   
                    
                    # If the player has the ball, draw a triangle to indicate it
                    if 'has_ball' in item and item['has_ball']:
                        frame = self.triangle_annotator.draw(frame, item['bbox'],obj_cls='players', color=Color.from_hex(self.settings.PLAYER_BALL_ANNOTATION_COLOR))

                    labels = [str(track_id)]
                    
                    frame = self.label_annotator.draw(frame, item['bbox'], labels, color=team_color, posistion=Position.BOTTOM_CENTER, background=True)

                    if speed >= 0:
                        labels = [f"{speed:.2f} km/h"]

                        frame = self.label_annotator.draw(frame, item['bbox'], labels, posistion=Position.TOP_RIGHT, background=False)

                   
        return frame

    