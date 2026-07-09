import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.transforms.starlet import starlet_decompose, starlet_reconstruct
from src.baselines.homomorphic import homomorphic_decomposition
from src.decompose.multiscale import multiscale_decomposition
from src.data.mit import MITIntrinsicDataset

def test_starlet_reconstruction_grayscale():
    """Verify that Starlet decomposition and reconstruction is a perfect identity on 2D images."""
    np.random.seed(0)
    image = np.random.rand(128, 128)
    
    coeffs, residual = starlet_decompose(image, levels=4)
    reconstructed = starlet_reconstruct(coeffs, residual)
    
    assert np.allclose(image, reconstructed, rtol=1e-10, atol=1e-10), \
        "Starlet reconstruction on grayscale image does not match original."

def test_starlet_reconstruction_color():
    """Verify that Starlet decomposition and reconstruction is a perfect identity on 3D images."""
    np.random.seed(1)
    image = np.random.rand(128, 128, 3)
    
    coeffs, residual = starlet_decompose(image, levels=3)
    reconstructed = starlet_reconstruct(coeffs, residual)
    
    assert np.allclose(image, reconstructed, rtol=1e-10, atol=1e-10), \
        "Starlet reconstruction on color image does not match original."

def test_homomorphic_energy_conservation():
    """Verify that estimated reflectance and shading multiply back to original color image (R * S = I)."""
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not fully loaded. Skip test. Error: {e}")
        
    diffuse = data['diffuse']
    mask = data['mask']
    
    S, R = homomorphic_decomposition(diffuse, mask, sigma=15.0)
    
    # R * S should match diffuse inside the mask (with small error due to eps and masking)
    reconstructed = R * S[..., np.newaxis]
    
    # Only test inside mask
    diffuse_masked = diffuse * mask[..., np.newaxis]
    reconstructed_masked = reconstructed * mask[..., np.newaxis]
    
    # Because of eps=1e-5 inside log, values close to 0 might have tiny mismatches, but generally they match
    assert np.allclose(diffuse_masked, reconstructed_masked, rtol=1e-3, atol=1e-3), \
        "Homomorphic R * S does not reconstruct the original color image."

def test_multiscale_energy_conservation():
    """Verify that multiscale Starlet R * S multiplies back to original color image (R * S = I)."""
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not fully loaded. Skip test. Error: {e}")
        
    diffuse = data['diffuse']
    mask = data['mask']
    
    S, R = multiscale_decomposition(diffuse, mask, levels=3, threshold_factor=2.0)
    
    # R * S should match diffuse inside the mask
    reconstructed = R * S[..., np.newaxis]
    
    diffuse_masked = diffuse * mask[..., np.newaxis]
    reconstructed_masked = reconstructed * mask[..., np.newaxis]
    
    assert np.allclose(diffuse_masked, reconstructed_masked, rtol=1e-3, atol=1e-3), \
        "Multiscale Starlet R * S does not reconstruct the original color image."
