import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root and Grosse code to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / "data" / "mit_code" / "MIT-intrinsic"))

from src.data.mit import MITIntrinsicDataset
from src.metrics.lmse import score_image, ssq_error, local_error

def test_lmse_identity():
    """Test that LMSE of an image against itself is exactly 0."""
    # Create a dummy image and mask
    np.random.seed(0)
    img = np.random.rand(100, 100)
    mask = np.ones((100, 100), dtype=bool)
    
    # Using our local_error directly
    err = local_error(img, img, mask, window_size=20, window_shift=10)
    assert np.isclose(err, 0.0), f"LMSE of identical images should be 0, got {err}"

def test_lmse_vs_grosse():
    """Validate our LMSE implementation against Grosse et al. (2009) original python code."""
    # Check if the Grosse code is available
    grosse_path = project_root / "data" / "mit_code" / "MIT-intrinsic" / "intrinsic.py"
    if not grosse_path.exists():
        pytest.skip("Grosse's original code not found. Skip cross-validation.")
        
    import intrinsic as grosse_intrinsic
    
    # Load a real object from MIT dataset to test
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not fully loaded yet (e.g. 'box' not found). Skip test. Error: {e}")
        
    diffuse = data['diffuse']
    reflectance = data['reflectance']
    shading = data['shading']
    mask = data['mask']
    
    # Create some dummy estimates by adding perturbations
    np.random.seed(42)
    # Convert reflectance to grayscale for Grosse's metric
    refl_gray = np.mean(reflectance, axis=2)
    
    # Perturb
    est_refl = refl_gray * 0.9 + 0.05 * np.random.rand(*refl_gray.shape)
    est_shading = shading * 1.1 + 0.05 * np.random.rand(*shading.shape)
    
    # 1. Compute using Grosse's implementation
    grosse_score = grosse_intrinsic.score_image(
        shading, refl_gray, est_shading, est_refl, mask, window_size=20
    )
    
    # 2. Compute using our implementation
    our_score = score_image(
        shading, refl_gray, est_shading, est_refl, mask, window_size=20
    )
    
    print(f"Grosse score: {grosse_score}")
    print(f"Our score: {our_score}")
    
    assert np.isclose(our_score, grosse_score, rtol=1e-5), \
        f"Scores do not match. Grosse: {grosse_score}, Ours: {our_score}"

def test_mit_loader():
    """Verify that MIT dataset loader reads objects correctly."""
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not fully loaded yet (e.g. 'box' not found). Skip test. Error: {e}")
        
    assert 'diffuse' in data
    assert 'reflectance' in data
    assert 'shading' in data
    assert 'mask' in data
    
    # Check dimensions
    assert data['diffuse'].ndim == 3
    assert data['reflectance'].ndim == 3
    assert data['shading'].ndim == 2
    assert data['mask'].ndim == 2
    
    # Check shape compatibility
    h, w, c = data['diffuse'].shape
    assert data['reflectance'].shape == (h, w, c)
    assert data['shading'].shape == (h, w)
    assert data['mask'].shape == (h, w)
    
    # Check value ranges
    assert np.all(data['diffuse'] >= 0.0) and np.all(data['diffuse'] <= 1.0)
    assert np.all(data['reflectance'] >= 0.0) and np.all(data['reflectance'] <= 1.0)
    assert np.all(data['shading'] >= 0.0) and np.all(data['shading'] <= 1.0)
    assert data['mask'].dtype == bool

from src.data.sintel import MPISintelDataset

def test_sintel_loader():
    """Verify that Sintel dataset loader reads objects correctly."""
    dataset = MPISintelDataset()
    try:
        scenes = dataset.get_scenes()
        if len(scenes) == 0:
            pytest.skip("MPI-Sintel dataset scenes empty. Skip test.")
        scene = scenes[0]
        frames = dataset.get_frames_in_scene(scene)
        if len(frames) == 0:
            pytest.skip("MPI-Sintel dataset frames empty. Skip test.")
        frame = frames[0]
        data = dataset.load_frame(scene, frame)
    except Exception as e:
        pytest.skip(f"MPI-Sintel dataset not fully loaded yet. Skip test. Error: {e}")
        
    assert 'diffuse' in data
    assert 'reflectance' in data
    assert 'shading' in data
    assert 'mask' in data
    
    # Check dimensions
    assert data['diffuse'].ndim == 3
    assert data['reflectance'].ndim == 3
    assert data['shading'].ndim == 2
    assert data['mask'].ndim == 2
    
    # Check shape compatibility
    h, w, c = data['diffuse'].shape
    assert data['reflectance'].shape == (h, w, c)
    assert data['shading'].shape == (h, w)
    assert data['mask'].shape == (h, w)
    
    # Check value ranges
    assert np.all(data['diffuse'] >= 0.0) and np.all(data['diffuse'] <= 1.0)
    assert np.all(data['reflectance'] >= 0.0) and np.all(data['reflectance'] <= 1.0)
    assert np.all(data['shading'] >= 0.0) and np.all(data['shading'] <= 1.0)
    assert data['mask'].dtype == bool

