import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from PIL import Image

# List of 16 evaluation objects from Grosse et al.
EVAL_OBJECTS = [
    'box', 'cup1', 'cup2', 'dinosaur', 'panther', 'squirrel', 'sun', 'teabag2',
    'deer', 'frog1', 'frog2', 'paper1', 'paper2', 'raccoon', 'teabag1', 'turtle'
]

# 4 extra objects with slight issues
EXTRA_OBJECTS = ['apple', 'pear', 'phone', 'potato']

ALL_OBJECTS = EVAL_OBJECTS + EXTRA_OBJECTS

class MITIntrinsicDataset:
    """
    Loader for the MIT Intrinsic Images dataset (Grosse et al. 2009).
    """
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Args:
            data_dir (Path, optional): Path to the project data directory. 
                                       If None, attempts to find it relative to project root.
        """
        if data_dir is None:
            # Assume we are in src/data/mit.py, project root is 2 levels up
            project_root = Path(__file__).resolve().parent.parent.parent
            self.data_dir = project_root / "data" / "mit" / "MIT-intrinsic" / "data"
        else:
            self.data_dir = Path(data_dir) / "MIT-intrinsic" / "data"
            
        if not self.data_dir.exists():
            # Try fallback to just data/mit/data or similar if needed
            fallback_dir = Path(__file__).resolve().parent.parent.parent / "data" / "mit" / "data"
            if fallback_dir.exists():
                self.data_dir = fallback_dir
                
    def get_object_path(self, tag: str) -> Path:
        """Returns the path to an object's folder."""
        path = self.data_dir / tag
        if not path.exists():
            raise FileNotFoundError(f"Object folder {tag} not found at {path}")
        return path

    def load_image(self, path: Path) -> np.ndarray:
        """Loads a PNG image and normalizes it to [0, 1] float."""
        with Image.open(path) as img:
            arr = np.array(img, dtype=np.float32)
            # Normalization depending on bit depth
            if img.mode == 'I;16' or img.mode == 'I' or np.max(arr) > 255.0:
                arr = arr / 65535.0
            else:
                arr = arr / 255.0
        return arr

    def load_object(self, tag: str) -> Dict[str, np.ndarray]:
        """
        Loads all default components for a given object tag.
        
        Args:
            tag (str): The object identifier (e.g. 'box').
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'diffuse': The color diffuse input image [H, W, 3]
                - 'reflectance': The color reflectance GT [H, W, 3]
                - 'shading': The grayscale shading GT [H, W] (already 2D in dataset)
                - 'mask': The boolean mask [H, W]
        """
        obj_dir = self.get_object_path(tag)
        
        # Load mask
        mask_path = obj_dir / "mask.png"
        with Image.open(mask_path) as img:
            # Grayscale check
            mask_arr = np.array(img.convert('L'))
            mask = mask_arr > 0
            
        # Load diffuse (original input to algorithm)
        diffuse = self.load_image(obj_dir / "diffuse.png")
        
        # Load reflectance
        reflectance = self.load_image(obj_dir / "reflectance.png")
        
        # Load shading
        shading_img = self.load_image(obj_dir / "shading.png")
        # Shading in the dataset is grayscale, but PIL might load it with 3 channels (or L)
        # Let's ensure it is 2D
        if shading_img.ndim == 3:
            shading = np.mean(shading_img, axis=2)
        else:
            shading = shading_img
            
        return {
            'diffuse': diffuse,
            'reflectance': reflectance,
            'shading': shading,
            'mask': mask
        }

    def load_multiple_lights(self, tag: str) -> np.ndarray:
        """
        Loads the 10 light-varying images of the object.
        
        Args:
            tag (str): The object identifier.
            
        Returns:
            np.ndarray: [H, W, 3, 10] array containing the 10 light conditions.
        """
        obj_dir = self.get_object_path(tag)
        # Load light01.png to get dimensions
        img0 = self.load_image(obj_dir / "light01.png")
        h, w, c = img0.shape
        
        result = np.zeros((h, w, c, 10), dtype=np.float32)
        for i in range(10):
            light_path = obj_dir / f"light{i+1:02d}.png"
            result[..., i] = self.load_image(light_path)
            
        return result
