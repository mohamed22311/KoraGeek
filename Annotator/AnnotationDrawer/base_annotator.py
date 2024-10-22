from abc import ABC, abstractmethod
from typing import List
import numpy as np

"""
An image of type `np.ndarray` or `PIL.Image.Image`.

Unlike a `Union`, ensures the type remains consistent. If a function
takes an `ImageType` argument and returns an `ImageType`, when you
pass an `np.ndarray`, you will get an `np.ndarray` back.
"""


class BaseAnnotator(ABC):
    @abstractmethod
    def draw(self, frame: np.ndarray, bbox: List[int], **kwargs) -> np.ndarray:
        pass
