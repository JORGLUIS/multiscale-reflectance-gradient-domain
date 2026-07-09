import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.ssr import ssr_decomposition
from src.baselines.msr import msr_decomposition
from src.baselines.horn import horn_decomposition
from src.data.mit import MITIntrinsicDataset

def test_ssr_energy_conservation():
    """Verify that SSR estimated reflectance and shading multiply back to original image (R * S = I)."""
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not fully loaded. Skip test. Error: {e}")
        
    diffuse = data['diffuse']
    mask = data['mask']
    
    S, R = ssr_decomposition(diffuse, mask, sigma=15.0)
    
    reconstructed = R * S[..., np.newaxis]
    
    diffuse_masked = diffuse * mask[..., np.newaxis]
    reconstructed_masked = reconstructed * mask[..., np.newaxis]
    
    assert np.allclose(diffuse_masked, reconstructed_masked, rtol=1e-3, atol=1e-3), \
        "SSR R * S does not reconstruct the original color image."

def test_msr_energy_conservation():
    """Verify that MSR estimated reflectance and shading multiply back to original image (R * S = I)."""
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not fully loaded. Skip test. Error: {e}")
        
    diffuse = data['diffuse']
    mask = data['mask']
    
    S, R = msr_decomposition(diffuse, mask, sigmas=[15.0, 80.0, 250.0])
    
    reconstructed = R * S[..., np.newaxis]
    
    diffuse_masked = diffuse * mask[..., np.newaxis]
    reconstructed_masked = reconstructed * mask[..., np.newaxis]
    
    assert np.allclose(diffuse_masked, reconstructed_masked, rtol=1e-3, atol=1e-3), \
        "MSR R * S does not reconstruct the original color image."

def test_horn_execution():
    """Verify that Retinex-Horn decomposition executes and returns valid shapes and ranges."""
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not fully loaded. Skip test. Error: {e}")
        
    diffuse = data['diffuse']
    mask = data['mask']
    
    # Horn is slow, we can test on a small crop or run it directly since box is small (400x434)
    # Box is fast to solve with pyamg (ruge-stuben solver)
    S, R = horn_decomposition(diffuse, mask, threshold=0.1, L1=False)
    
    assert S.shape == mask.shape
    assert R.shape == diffuse.shape
    
    # Check that they are inside valid bounds
    assert np.all(S >= 0.0) and np.all(S <= 1.0)
    assert np.all(R >= 0.0)
