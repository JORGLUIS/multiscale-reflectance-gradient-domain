import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.transforms.wavelet import wavelet_decompose, wavelet_reconstruct

def test_wavelet_reconstruction_powers_of_two():
    """Verify that SWT decomposition and reconstruction is a perfect identity on standard 128x128 images."""
    np.random.seed(0)
    image = np.random.rand(128, 128)
    
    coeffs, orig_shape = wavelet_decompose(image, levels=3, wavelet_name="db2")
    reconstructed = wavelet_reconstruct(coeffs, wavelet_name="db2", original_shape=orig_shape)
    
    assert np.allclose(image, reconstructed, rtol=1e-10, atol=1e-10), \
        "Wavelet reconstruction on standard power-of-two image shape failed."

def test_wavelet_reconstruction_arbitrary_shape():
    """Verify that SWT decomposition and reconstruction is a perfect identity on arbitrary shapes (e.g. 123x137)."""
    np.random.seed(1)
    image = np.random.rand(123, 137)
    
    coeffs, orig_shape = wavelet_decompose(image, levels=3, wavelet_name="db2")
    reconstructed = wavelet_reconstruct(coeffs, wavelet_name="db2", original_shape=orig_shape)
    
    assert np.allclose(image, reconstructed, rtol=1e-10, atol=1e-10), \
        "Wavelet reconstruction on padded arbitrary image shape failed."

def test_wavelet_decomposition_energy_conservation():
    """Verify that multiscale Wavelet R * S multiplies back to original color image (R * S = I)
    when using all advanced rules (scale coherence and color coherence)."""
    from src.decompose.multiscale import multiscale_decomposition
    from src.data.mit import MITIntrinsicDataset
    
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        pytest.skip(f"MIT dataset not loaded: {e}")
        
    diffuse = data['diffuse']
    mask = data['mask']
    
    S, R = multiscale_decomposition(
        diffuse, mask, 
        levels=3, 
        threshold_factor=2.0, 
        transform_type="wavelet",
        scale_coherence=True,
        color_coherence=True
    )
    
    # R * S should match diffuse inside the mask
    reconstructed = R * S[..., np.newaxis]
    diffuse_masked = diffuse * mask[..., np.newaxis]
    reconstructed_masked = reconstructed * mask[..., np.newaxis]
    
    assert np.allclose(diffuse_masked, reconstructed_masked, rtol=1e-3, atol=1e-3), \
        "Wavelet R * S does not multiply back to original image."

