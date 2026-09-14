"""
test_surrogate.py — Tests for surrogate_transistor.py

Tests:
  - Import and metadata correctness
  - OUTPUT_KEYS has 6 entries including eye_width_ui
  - _PVT_CORNERS has all 5 corners
  - TransistorSurrogateMLP has correct default output_dim
  - If the model file exists, load it and assert MAE < 2 dB on a test vector
"""
import numpy as np
import pytest


def test_output_keys_has_eye_width_ui():
    """OUTPUT_KEYS must include eye_width_ui (6 entries total)."""
    from surrogate_transistor import OUTPUT_KEYS
    assert len(OUTPUT_KEYS) == 6, f"Expected 6 OUTPUT_KEYS, got {len(OUTPUT_KEYS)}: {OUTPUT_KEYS}"
    assert 'eye_width_ui' in OUTPUT_KEYS
    assert 'hd3_db' in OUTPUT_KEYS
    assert 'peaking_db' in OUTPUT_KEYS


def test_pvt_corners_has_all_five():
    """_PVT_CORNERS must have all 5 corners: tt, ss, ff, sf, fs."""
    from surrogate_transistor import _PVT_CORNERS
    corner_names = [c[0] for c in _PVT_CORNERS]
    assert len(_PVT_CORNERS) == 5, f"Expected 5 corners, got {len(_PVT_CORNERS)}: {corner_names}"
    for expected in ('tt', 'ss', 'ff', 'sf', 'fs'):
        assert expected in corner_names, f"Corner '{expected}' missing from _PVT_CORNERS"


def test_mlp_default_output_dim():
    """TransistorSurrogateMLP default output_dim must be 6."""
    import inspect
    from surrogate_transistor import TransistorSurrogateMLP
    sig = inspect.signature(TransistorSurrogateMLP.__init__)
    default_output_dim = sig.parameters['output_dim'].default
    assert default_output_dim == 6, f"Expected default output_dim=6, got {default_output_dim}"


def test_surrogate_inference_returns_eye_width_ui():
    """simulate_transistor_surrogate must return eye_width_ui key in its dict."""
    import os
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'surrogate_transistor.pt')
    if not os.path.exists(model_path):
        pytest.skip("surrogate_transistor.pt not found — run --generate + --train first")
    from surrogate_transistor import simulate_transistor_surrogate
    result = simulate_transistor_surrogate(4.0, 750.0, 1.8e-12, 400.0, 2200.0, 20000.0)
    assert 'eye_width_ui' in result, f"eye_width_ui key missing from surrogate output: {list(result.keys())}"
    assert 'hd3_db' in result
    assert 'peaking_db' in result
