from .base_annotator import BaseAnnotator

import numpy as np
import cv2
from supervision import Color, ColorPalette
from typing import List, Optional, Tuple, Union


class EllipseAnnotator(BaseAnnotator):
    def __init__(self,
                 color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
                 thickness: int=2,
                 start_angle: int=-45,
                 end_angle: int=235):
        """
        Initialize the EllipseAnnotator with the given color, thickness, and start and end angles.

        Args:
            color (Color | ColorPalette): The color of the ellipse.
            thickness (int): The thickness of the ellipse outline.
            start_angle (int): The starting angle of the ellipse arc.
            end_angle (int): The ending angle of the ellipse arc.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.start_angle: int = start_angle
        self.end_angle: int = end_angle
    
    def draw(self, frame: np.ndarray, bbox: List[int], type: str = 'Normal', color: Optional[Color] = None) -> np.ndarray:
        """
        Draw an ellipse on the frame with the given bounding box coordinates.
        Args:
            frame (np.ndarray): The frame to draw the ellipse on.
            bbox (List[int]): The bounding box coordinates [x_min, y_min, x_max, y_max] of the ellipse.
            type (str): The type of the ellipse to draw. it takes one of the following values:
                - 'Normal': Draw a normal ellipse.
                - 'Dashed': Draw a dashed ellipse.
                - 'Double': Draw two concentric ellipses.
        Returns:
            np.ndarray: The frame with the ellipse drawn on it.
        Examples:
            ellipse_annotator = EllipseAnnotator()
            frame = ellipse_annotator.draw(frame, [10, 20, 30, 40])
        """
        x1, y1, x2, y2 = bbox
        center = (int((x1 + x2) / 2), int(y2))
        width = x2 - x1
        
        if color is None:
            color = self.color

        color = self._reslove_color(color,type)

        # Determine the ellipse style based on the object class
        if type == 'Dashed':
            self._draw_dashed_ellipse(frame, center, width, color)
        elif type == 'Double':
            self._draw_double_ellipse(frame, center, width, color)
        else:
            self._draw_ellipse(frame, center, width, color)

        
        return frame
    
    def _draw_ellipse(self, frame: np.ndarray, center: Tuple[int, int], width: int, color: Color) -> np.ndarray:
        """
        Draws a normal ellipse on the frame.
        Args:
            frame (np.ndarray): The frame to draw the ellipse on.
            center (Tuple[int, int]): The center of the ellipse.
            width (int): The width of the ellipse.
            color (Color): The color of the ellipse.
        Returns:         
            np.ndarray: The frame with the ellipse drawn on it.
        """
        cv2.ellipse(
                frame,
                center=center,
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=self.start_angle,
                endAngle=self.end_angle,
                color=color.as_bgr(),
                thickness=self.thickness,
                lineType=cv2.LINE_4,
            )
        return frame

    def _draw_double_ellipse(self,
                             frame: np.ndarray,
                             center: Tuple[int, int],
                             width: int,
                             color: Color) -> np.ndarray:
        """
        Draws two concentric ellipses for special objects like goalkeepers.

        Args:
            frame (np.ndarray): The frame where the ellipses will be drawn.
            center (Tuple[int, int]): The center of the ellipses.
            width (int): The width of the first ellipse.
            color (Color): The color of the ellipses.
        Returns:
            np.ndarray: The frame with the ellipses drawn on it.
        """
        size_decrement = 5  # Reduce the size of the second ellipse
        # Draw two concentric ellipses
        for i in range(2):
            cv2.ellipse(
                frame,
                center=center,
                axes=(int(width - i * size_decrement), int(20 - i * size_decrement)),
                angle=0.0,
                startAngle=self.start_angle,
                endAngle=self.end_angle,
                color=color.as_bgr(),
                thickness=self.thickness,
                lineType=cv2.LINE_4,
            )
        return frame
        
    def _draw_dashed_ellipse(self,
                             frame: np.ndarray,
                             center: Tuple[int, int],
                             width: int,
                             color: Color) -> np.ndarray:
        """
        Draws a dashed ellipse for special objects

        Args:
            frame (np.ndarray): The frame where the ellipse will be drawn.
            center (Tuple[int, int]): The center of the ellipse.
            width (int): The width of the ellipse.
            color (Color): The color of the ellipse.
        Returns:
            np.ndarray: The frame with the ellipse drawn on it.
        """
        dash_length = 15  # Length of each dash
        total_angle = self.end_angle - self.start_angle  # Total angle to cover

        # Draw dashed lines by alternating between dashes and gaps
        for angle in range(self.start_angle, total_angle, dash_length * 2):
            cv2.ellipse(
                frame,
                center=center,
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=angle,
                endAngle=angle + dash_length,
                color=color.as_bgr(),
                thickness=self.thickness,
                lineType=cv2.LINE_4,
            )
        return frame
    
    def _reslove_color(self, color: Union[Color, ColorPalette], type: str) -> Color:
        """
        Resolve the color of the ellipse based on the object class and the ellipse type.
        Args:
            color (Union[Color, ColorPalette]): The color of the ellipse.
            type (str): The type of the ellipse.
        Returns:
            Color: The resolved color of the ellipse.
        """
        if isinstance(color, ColorPalette):
            if type == 'Dashed':
                return self.color.by_idx(2)
            elif type == 'Double':
                return self.color.by_idx(1)
            else:
                return self.color.by_idx(0)
        return color


