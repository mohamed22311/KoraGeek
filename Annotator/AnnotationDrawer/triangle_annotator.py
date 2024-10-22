from .base_annotator import BaseAnnotator

import cv2
from supervision import Color, ColorPalette
from numpy import ndarray, array, int32

from Enums import Position
from utils import get_anchors_coordinates
from typing import List, Optional, Union

class TriangleAnnotator(BaseAnnotator):
    def __init__(self,
                 color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
                 base: int = 10,
                 height: int = 10,
                 position: Position = Position.TOP_CENTER,
                 outline_thickness: int = 0,
                 outline_color: Union[Color, ColorPalette] = Color.BLACK,
                 classes: List[str] = None):
        
        """
        Initialize the TriangleAnnotator with the given color, base, height, and outline thickness.
        
        Args:
            color (Union[Color, ColorPalette]): The color of the triangle.
            base (int): The base width of the triangle.
            height (int): The height of the triangle.
            position (Position): The anchor position for placing the triangle.
            outline_thickness (int): The thickness of the triangle outline.
            outline_color (Union[Color, ColorPalette]): The color of the triangle outline.
            classes (List[str]): The list of classes to draw the triangle on.
        """
        self.color: Union[Color, ColorPalette] = color
        self.base: int = base
        self.height: int = height
        self.position: Position = position
        self.outline_thickness: int = outline_thickness
        self.outline_color: Union[Color, ColorPalette] = outline_color
        self.classes: List[str] = classes

    def draw(self, frame: ndarray, bbox: list[int], color: Optional[Color] = None, outline_color: Union[Color, ColorPalette]= None, obj_cls: str = None) -> ndarray:
        """
        Draw a triangle on the frame with the given bounding box coordinates.

        Args:
            frame (ndarray): The frame to draw the triangle on.
            bbox (List[int]): The bounding box coordinates [x_min, y_min, x_max, y_max] of the triangle.
            color (Optional[Color]): The color of the triangle.
            outline_color (Union[Color, ColorPalette]): The color of the triangle outline.
            obj_cls (str): The class of the object to draw the triangle on.
        Returns:
            ndarray: The frame with the triangle drawn on it.
        Examples:
            triangle_annotator = TriangleAnnotator()
            frame = triangle_annotator.draw(frame, [10, 20, 30, 40])
        """
        xy = get_anchors_coordinates(anchor=self.position, bbox=bbox)

        if color is None:
            color = self.color

        color = self._reslove_color(color,obj_cls)

        tip_x, tip_y = int(xy[0]), int(xy[1])
        vertices = array(
            [
                [tip_x - self.base // 2, tip_y - self.height],
                [tip_x + self.base // 2, tip_y - self.height],
                [tip_x, tip_y],
            ],
            int32,
        )
        
        cv2.fillPoly(frame, [vertices], color.as_bgr())

        if self.outline_thickness:
            
            if outline_color is None:
                outline_color = self.outline_color
        
            outline_color = self._reslove_color(outline_color,type)
            
            cv2.polylines(
                frame,
                [vertices],
                True,
                outline_color.as_bgr(),
                thickness=self.outline_thickness,
            )

        return frame
    
    def _reslove_color(self, color: Union[Color, ColorPalette], type: str) -> Color:
        """
        Resolves the color of the triangle based on its type.
        
        Args:
            color (Union[Color, ColorPalette]): The color of the triangle.
            type (str): The type of the triangle to draw. It takes one of the following values:
                - 'Normal': Draw a normal triangle.
                - 'Dashed': Draw a dashed triangle.
                - 'Double': Draw two concentric triangles.
        
        Returns:
            Color: The resolved color of the triangle.
        """
        if isinstance(color, ColorPalette):
            if self.classes is not None:
                for idx, cls in enumerate(self.classes):
                    if cls == type:
                        if idx < len(color):
                            color = color.by_idx(idx)
                        else:
                            color = color.by_idx(100000)
                        break
            else:
                color = color.by_idx(100000)
        return color

    