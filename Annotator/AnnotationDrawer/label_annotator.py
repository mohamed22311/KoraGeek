from .base_annotator import BaseAnnotator

import cv2
from typing import List, Tuple, Union
import numpy as np
from supervision import Color, ColorPalette
from Enums import Position
from utils import get_anchors_coordinates

class LabelAnnotator(BaseAnnotator):
    def __init__(self,
                 color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
                 text_color: Union[Color, ColorPalette] = Color.WHITE,
                 text_position: Position = Position.TOP_LEFT,
                 text_scale: float = 0.5,
                 text_thickness: int = 1,
                 text_padding: int = 3,
                 border_radius: int = 0,):
        """
        Initialize the LabelAnnotator with the given color, text color, text position, font scale, font thickness, font face, and label offset.

        Args:
            color (sv.Color | sv.ColorPalette): The color of the label background.
            text_color (sv.Color): The color of the label text.
            text_position (sv.Position): The position of the label text.
            font_scale (float): The scale of the font.
            font_thickness (int): The thickness of the font.
            font_face (int): The font face.
            label_offset (int): The offset of the label from the object.
        """
        self.border_radius: int = border_radius
        self.color: Union[Color, ColorPalette] = color
        self.text_color: Union[Color, ColorPalette] = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.text_anchor: Position = text_position

    def draw(self, frame: np.ndarray, bbox: List[int], labels: List[str], color: Union[Color, ColorPalette]=None, text_color: Union[Color, ColorPalette]=None, posistion: Position = None, background: bool=True) -> np.ndarray:
        """
        Draw a label on the frame with the given bounding box coordinates.

        Args:
            frame (np.ndarray): The frame to draw the label on.
            bbox (List[int]): The bounding box coordinates [x_min, y_min, x_max, y_max] of the label.
            labels (List[str]): The labels to draw on the frame.
            color (Union[Color, ColorPalette]): The color of the label background.
            text_color (Union[Color, ColorPalette]): The color of the label text.

        Returns:
            np.ndarray: The frame with the label drawn on it.
        
        Examples:
            label_annotator = LabelAnnotator()
            frame = label_annotator.draw(frame, [10, 20, 30, 40], ['person', 'car'])
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        if posistion is None:
            posistion = self.text_anchor
        center_coordinates = get_anchors_coordinates(anchor=posistion, bbox=bbox).astype(int)

        if color is None:
            color = self.color

        if text_color is None:
            text_color = self.text_color

        color = self._reslove_color(color)
        text_color = self._reslove_color(text_color)
        text = self._reslove_labels(labels)

        text_w, text_h = cv2.getTextSize(
            text=text,
            fontFace=font,
            fontScale=self.text_scale,
            thickness=self.text_thickness,
        )[0]

        text_w_padded = text_w + 2 * self.text_padding
        text_h_padded = text_h + 2 * self.text_padding

        text_background_xyxy = self._resolve_text_background_xyxy(
            center_coordinates=tuple(center_coordinates),
            text_wh=(text_w_padded, text_h_padded),
            position=self.text_anchor,
        )

        text_x = text_background_xyxy[0] + self.text_padding
        text_y = text_background_xyxy[1] + self.text_padding + text_h

        if background:
            self._draw_rounded_rectangle(
                frame=frame,
                bbox=text_background_xyxy,
                color=color.as_bgr(),
                border_radius=self.border_radius,
            )
            
        cv2.putText(
            img=frame,
            text=text,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=self.text_scale,
            color=text_color.as_bgr(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )

        return frame
    


    def _resolve_text_background_xyxy(self, center_coordinates: Tuple[int, int], text_wh: Tuple[int, int], position: Position) -> Tuple[int, int, int, int]:
        """
        Resolve the text background coordinates based on the text position.
        
        Args:
            center_coordinates (Tuple[int, int]): The center coordinates of the text.
            text_wh (Tuple[int, int]): The width and height of the text.
            position (Position): The position of the text.
        
        Returns:
            Tuple[int, int, int, int]: The text background coordinates [x1, y1, x2, y2].
        """
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh

        if position == Position.TOP_LEFT:
            return center_x, center_y - text_h, center_x + text_w, center_y
        
        elif position == Position.TOP_RIGHT:
            return center_x - text_w, center_y - text_h, center_x, center_y
        
        elif position == Position.TOP_CENTER:
            return (
                center_x - text_w // 2,
                center_y - text_h,
                center_x + text_w // 2,
                center_y,
            )
        
        elif position == Position.CENTER:
            return (
                center_x - text_w // 2,
                center_y - text_h // 2,
                center_x + text_w // 2,
                center_y + text_h // 2,
            )
        
        elif position == Position.BOTTOM_LEFT:
            return center_x, center_y, center_x + text_w, center_y + text_h
        
        elif position == Position.BOTTOM_RIGHT:
            return center_x - text_w, center_y, center_x, center_y + text_h
        
        elif position == Position.BOTTOM_CENTER:
            return (
                center_x - text_w // 2,
                center_y,
                center_x + text_w // 2,
                center_y + text_h,
            )
        
        elif position == Position.CENTER_LEFT:
            return (
                center_x - text_w,
                center_y - text_h // 2,
                center_x,
                center_y + text_h // 2,
            )
        
        elif position == Position.CENTER_RIGHT:
            return (
                center_x,
                center_y - text_h // 2,
                center_x + text_w,
                center_y + text_h // 2,
            )
        
    def _draw_rounded_rectangle(self,frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], border_radius: int) -> np.ndarray:
        """
        Draw a rounded rectangle on the frame with the given bounding box coordinates.

        Args:
            frame (np.ndarray): The frame to draw the rounded rectangle on.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates [x1, y1, x2, y2] of the rounded rectangle.
            color (Tuple[int, int, int]): The color of the rounded rectangle.
            border_radius (int): The radius of the rounded rectangle corners.

        Returns:
            np.ndarray: The frame with the rounded rectangle drawn on it.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        border_radius = min(border_radius, min(width, height) // 2)

        rectangle_coordinates = [
            ((x1 + border_radius, y1), (x2 - border_radius, y2)),
            ((x1, y1 + border_radius), (x2, y2 - border_radius)),
        ]

        circle_centers = [
            (x1 + border_radius, y1 + border_radius),
            (x2 - border_radius, y1 + border_radius),
            (x1 + border_radius, y2 - border_radius),
            (x2 - border_radius, y2 - border_radius),
        ]

        for coordinates in rectangle_coordinates:
            cv2.rectangle(
                img=frame,
                pt1=coordinates[0],
                pt2=coordinates[1],
                color=color,
                thickness=-1,
            )

        for center in circle_centers:
            cv2.circle(
                img=frame,
                center=center,
                radius=border_radius,
                color=color,
                thickness=-1,
            )

        return frame
    

    def _reslove_color(self, color: Union[Color, ColorPalette]) -> Color:
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
            color = color.by_idx(100000)
        return color
    

    def _reslove_labels(self, labels: List[str]) -> str:
        """
        Resolves the labels to a single string.
        
        Args:
            labels (List[str]): The labels to resolve.
            
        Returns:    
            str: The resolved labels.
        """
        return '\n'.join(labels)
