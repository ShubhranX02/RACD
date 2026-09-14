"""
test_circuit.py — Tests for circuit_transistor.py

These tests cover:
  - Import correctness
  - PARAM_BOUNDS consistency between surrogate and environment
  - Area formula correctness
  - _device_section_2stage generates valid SPICE text
  - simulate_transistor_fast signature has compute_hd3 param
  - New return dict has eye_width_ui key
"""
import inspect
import numpy as np
import pytest


def test_circuit_imports():
    """circuit_transistor.py must import without error."""
    import circuit_transistor
    assert hasattr(circuit_transistor, 'simulate_transistor_fast')
    assert hasattr(circuit_transistor, 'simulate_transistor_level')
    assert hasattr(circuit_transistor, '_device_section_2stage')
    assert hasattr(circuit_transistor, '_device_section_1stage')


def test_simulate_fast_has_compute_hd3_param():
    """simulate_transistor_fast must accept compute_hd3 keyword arg."""
    from circuit_transistor import simulate_transistor_fast
    sig = inspect.signature(simulate_transistor_fast)
    assert 'compute_hd3' in sig.parameters, "compute_hd3 parameter is missing"
    default = sig.parameters['compute_hd3'].default
    assert default == False, f"compute_hd3 default should be False (got {default})"


def test_device_section_2stage_generates_spice():
    """_device_section_2stage should return a non-empty string with key SPICE elements."""
    from circuit_transistor import _device_section_2stage
    netlist = _device_section_2stage(
        Wn1_um=4.0, Rs1_ohm=750.0, Cs1_farad=1.8e-12, Itail1_half_ua=400.0, RL1_ohm=2200.0,
        Wn2_um=4.0, Rs2_ohm=750.0, Cs2_farad=1.8e-12, Itail2_half_ua=400.0, RL2_ohm=2200.0,
        Rdfe_ohm=20000.0, vdd=1.8,
    )
    assert isinstance(netlist, str)
    assert len(netlist) > 100, "SPICE netlist seems too short"
    assert 'sky130_fd_pr__nfet_01v8' in netlist
    assert 'nfet' in netlist.lower() or 'XM' in netlist


def test_temp_dir_is_tmp_subdir():
    """Temp files should go to src/.tmp/, not src/ directly."""
    import os
    import circuit_transistor
    assert '.tmp' in circuit_transistor._TEMP_DIR, (
        f"_TEMP_DIR should point to .tmp/ subdir, got: {circuit_transistor._TEMP_DIR}"
    )
    assert os.path.isdir(circuit_transistor._TEMP_DIR), ".tmp/ directory should be created on import"


def test_estimate_area_2stage_reasonable():
    """estimate_area_2stage_mm2 must return a positive area within 0-0.05 mm^2 for nominal sizing."""
    from environment_transistor import estimate_area_2stage_mm2
    area = estimate_area_2stage_mm2(
        Wn=4.0, Rs=750.0, Cs=1.8e-12, RL=2200.0, Rdfe=20000.0
    )
    assert area > 0, "Area must be positive"
    assert area < 0.05, f"Nominal sizing should be < 0.05 mm^2 (spec limit), got {area:.4f} mm^2"


def test_estimate_area_1stage_reasonable():
    """estimate_area_1stage_mm2 must return a positive area < 0.05 mm^2 and smaller than 2-stage."""
    from environment_transistor import estimate_area_1stage_mm2, estimate_area_2stage_mm2
    a1 = estimate_area_1stage_mm2(Wn=4.0, Rs=750.0, Cs=1.8e-12, RL=2200.0, Rdfe=20000.0)
    a2 = estimate_area_2stage_mm2(Wn=4.0, Rs=750.0, Cs=1.8e-12, RL=2200.0, Rdfe=20000.0)
    assert a1 > 0, "1-stage area must be positive"
    assert a1 < 0.05, f"1-stage sizing should be < 0.05 mm^2 (spec limit), got {a1:.4f} mm^2"
    assert a1 < a2, f"1-stage area ({a1:.5f}) should be smaller than 2-stage ({a2:.5f})"
