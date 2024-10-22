import cv2
from typing import List, Generator, Iterable, List, TypeVar
V = TypeVar("V")

def read_video(video_path: str) -> List:
    """
    Read a video file and calculate the fps. 
    Args:
        video_path: str, path to the video file
    Returns:
        frames: List, list of frames
    Examples:
        video_path = 'data/video.mp4'
        frames = read_video(video_path)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    
    return frames, fps


def save_video(video_frames: List,
               video_path: str) -> None:
    """
    Save a list of frames as a video file.
    Args:
        video_frames: List, list of frames
        video_path: str, path to save the video file
    Returns:
        None
    Examples:
        video_path = 'data/video.mp4'
        save_video(video_frames, video_path)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 24, (video_frames[0].shape[1], video_frames[0].shape[0]))
    
    for frame in video_frames:
        out.write(frame)

    out.release()


def create_batches( sequence: Iterable[V], batch_size: int ) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch