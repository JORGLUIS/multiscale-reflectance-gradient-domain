import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.transforms.mmt import mmt_decompose, mmt_reconstruct
from src.decompose.multiscale import multiscale_decomposition
from src.data.mit import MITIntrinsicDataset

def test_mmt_reconstruction_grayscale():
    """Verify that MMT decomposition and reconstruction is a perfect identity on 2D images."""
    np.random.seed(0)
    image = np.random.rand(128, 128)
    
    coeffs, residual = mmt_decompose(image, levels=4)
    reconstructed = mmt_reconstruct(coeffs, residual)
    
    assert np.allclose(image, reconstructed, rtol=1e-10, atol=1e-10), \
        "MMT reconstruction on grayscale image does not match original."

def test_mmt_reconstruction_color():
    """Verify that MMT decomposition and reconstruction is a perfect identity on 3D images."""
    np.random.seed(1)
    image = np.random.rand(128, 128, 3)
    
    coeffs, residual = mmt_decompose(image, levels=3)
    reconstructed = mmt_reconstruct(coeffs, residual)
    
    assert np.allclose(image, reconstructed, rtol=1e-10, atol=1e-10), \
        "MMT reconstruction on color image does not match original."

def test_f3_energy_conservation():
    """Verify that multiscale MMT R * S multiplies back to original color image (R * S = I)
    when using all advanced rules (scale coherence and color coherence)."""
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not fully loaded. Skip test. Error: {e}")
        
    diffuse = data['diffuse']
    mask = data['mask']
    
    S, R = multiscale_decomposition(
        diffuse, mask, 
        levels=3, 
        threshold_factor=2.0, 
        transform_type="mmt",
        scale_coherence=True,
        color_coherence=True
    )
    
    # R * S should match diffuse inside the mask
    reconstructed = R * S[..., np.newaxis]
    
    diffuse_masked = diffuse * mask[..., np.newaxis]
    reconstructed_masked = reconstructed * mask[..., np.newaxis]
    
    assert np.allclose(diffuse_masked, reconstructed_masked, rtol=1e-3, atol=1e-3), \
        "F3 Multiscale MMT R * S does not reconstruct the original color image."
