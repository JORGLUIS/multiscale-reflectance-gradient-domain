import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from PIL import Image

class MPISintelDataset:
    """
    Loader for the MPI-Sintel dataset training images (Butler et al. 2012).
    It loads the 'albedo' (reflectance) and 'clean' (input/shading) passes.
    """
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Args:
            data_dir (Path, optional): Path to the project data directory.
                                       If None, attempts to find it relative to project root.
        """
        if data_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.data_dir = project_root / "data" / "sintel"
        else:
            self.data_dir = Path(data_dir)
            
        # Sintel training zip extracts as 'training/' inside data_dir
        # Let's handle both direct 'training/' and 'sintel/training/' cases
        self.training_dir = self.data_dir / "training"
        if not self.training_dir.exists():
            # Try checking if there is a subfolder 'sintel' or similar
            sub_sintel = self.data_dir / "MPI-Sintel" / "training"
            if sub_sintel.exists():
                self.training_dir = sub_sintel
            else:
                sub_training = self.data_dir / "MPI-Sintel-training_images" / "training"
                if sub_training.exists():
                    self.training_dir = sub_training

    def get_scenes(self) -> List[str]:
        """Returns the list of scene names in the dataset."""
        albedo_dir = self.training_dir / "albedo"
        if not albedo_dir.exists():
            raise FileNotFoundError(f"Sintel albedo directory not found at {albedo_dir}")
        return sorted([d.name for d in albedo_dir.iterdir() if d.is_dir()])

    def get_frames_in_scene(self, scene: str) -> List[str]:
        """Returns the list of frame filenames for a given scene."""
        scene_dir = self.training_dir / "albedo" / scene
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory {scene} not found.")
        return sorted([f.name for f in scene_dir.glob("*.png")])

    def load_image(self, path: Path) -> np.ndarray:
        """Loads a PNG image and normalizes it to [0, 1] float."""
        with Image.open(path) as img:
            arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def load_frame(self, scene: str, frame_file: str) -> Dict[str, np.ndarray]:
        """
        Loads the albedo, clean and calculates the shading for a single frame.
        
        Args:
            scene (str): Scene name (e.g. 'alley_1').
            frame_file (str): Frame filename (e.g. 'frame_0001.png').
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'diffuse': The color 'clean' image [H, W, 3] (acts as input I)
                - 'reflectance': The color 'albedo' GT [H, W, 3] (acts as R)
                - 'shading': Grayscale shading GT [H, W] (S = I_gray / (R_gray + eps))
                - 'mask': The boolean mask [H, W] (all True, as Sintel has no invalid region)
        """
        albedo_path = self.training_dir / "albedo" / scene / frame_file
        clean_path = self.training_dir / "clean" / scene / frame_file
        
        reflectance = self.load_image(albedo_path)
        diffuse = self.load_image(clean_path)
        
        # Calculate shading: shading = diffuse / reflectance
        # Sintel files are color. To get a 2D shading, we can convert diffuse and reflectance
        # to log-luminance or grayscale first.
        # Let's compute grayscale luminance: 0.2989 * R + 0.5870 * G + 0.1140 * B
        diffuse_gray = 0.2989 * diffuse[..., 0] + 0.5870 * diffuse[..., 1] + 0.1140 * diffuse[..., 2]
        refl_gray = 0.2989 * reflectance[..., 0] + 0.5870 * reflectance[..., 1] + 0.1140 * reflectance[..., 2]
        
        # Use epsilon to avoid division by zero
        eps = 1e-4
        shading = diffuse_gray / (refl_gray + eps)
        shading = np.clip(shading, 0.0, 1.0)
        
        h, w = shading.shape
        mask = np.ones((h, w), dtype=bool)
        
        return {
            'diffuse': diffuse,
            'reflectance': reflectance,
            'shading': shading,
            'mask': mask
        }
