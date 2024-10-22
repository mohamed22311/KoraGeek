import os
import pickle

from utils import get_settings

settings = get_settings()

def file_loader(
                dir: str,
                file_name: str) -> any:
    """
    Load the object file.
    Args:
        dir: str, directory name
        file_name: str, name to the file
    Returns:
        file: Any, object loaded from the file
        
    """

    file_path = './stub/'+dir+'/'+file_name+'.pkl'

    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path,'rb') as f:
            file = pickle.load(f)
    except:
        return None

    return file

def file_saver(
            content: any,
            dir: str,
            file_name: str) -> None:
    """
    Save the object file.
    Args:
        content: Any, object to be saved
        dir: str, directory name
        file_name: str, name to the file
    Returns:
        None
    """
    file_path = './stub/'+dir+'/'+file_name+'.pkl'

    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

    try:
        with open(file_path,'wb') as f:
            pickle.dump(content,f)
    except:
        return None