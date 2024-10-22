from .base_annotator import BaseAnnotator
from utils import Settings

import numpy as np
import cv2
from supervision import Color, KeyPoints

class KeypointAnnotator(BaseAnnotator):
    """Annotates keypoints in a frame, such as the head, shoulders, and feet."""

    def __init__(self, settings: Settings) -> None:
        """
        Initializes the KeypointAnnotator with predefined keypoint annotation colors.
        """
        super().__init__()

        self.settings = settings

        self.keypoint_color = Color.from_hex(settings.KEYPOINT_ANNOTATION_COLOR)
        self.keypoint_text_color = Color.from_hex(settings.KEYPOINT_TEXT_ANNOTATION_COLOR)


    def annotate(self, frame: np.ndarray, keypoints: KeyPoints) -> np.ndarray:
        """
        Annotates keypoints in a frame. Draws a circle for each keypoint (dot) with a radius of 5 and color (0, 255, 0) (green).
        Annotates the keypoint ID next to the dot.

        Args:
            frame (np.ndarray): The frame to annotate.
            keypoints (KeyPoints): The keypoints to annotate.

        Returns:
            np.ndarray: The annotated frame.
        """
        frame = frame.copy()
        xys = keypoints.xy[0]

        for keypoint_id, (x, y) in enumerate(xys):
            
            # Draw a circle for each keypoint (dot) with a radius of 5 and color (0, 255, 0) (green)
            cv2.circle(img=frame,
                        center=(int(x), int(y)),
                        radius=5,
                        color=self.keypoint_color.as_bgr(),
                        thickness=-1)
            
            # Annotate the keypoint ID next to the dot
            cv2.putText(img=frame,
                        text=str(keypoint_id),
                        org=(int(x) + 10, int(y) - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=self.keypoint_text_color.as_bgr(),
                        thickness=1)
        return frame
    
