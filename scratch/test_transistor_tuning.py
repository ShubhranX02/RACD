"""
scratch/test_transistor_tuning.py — Tune transistor CTLE parameters to hit ~6-8 dB peaking,
measure all 7 specs, and run across worst-case PVT corners.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from circuit_transistor import simulate_transistor_level

def estimate_area_mm2(Wn_um, Rs_ohm, Cs_farad, RL_ohm, Rdfe_ohm):
    # Transistor active area: 2 * (W * L) with L=0.15um
    a_diff_um2 = 2 * (Wn_um * 0.15)
    # Poly resistors (R_sheet ~ 100-200 ohm/sq, W_res ~ 1um)
    a_res_um2 = (2 * RL_ohm + Rs_ohm + 2 * Rdfe_ohm) / 150.0 * (1.0**2)
    # MIM/MOM capacitor (~ 2 fF / um^2)
    a_cap_um2 = (Cs_farad * 1e15) / 2.0
    total_um2 = (a_diff_um2 + a_res_um2 + a_cap_um2) * 2.5 # 2.5x layout overhead/spacing
    return total_um2 / 1e6 # convert to mm^2

def test_tuning():
    print("Testing transistor sizing for CTLE...")
    # Sizing for strong HF boost:
    # High Rs (600-1000 ohm), small Cs (1-3 pF), reasonable tail current
    candidates = [
        (4.0, 600.0, 1.5e-12, 400.0, 2500.0, 15000.0),
        (6.0, 800.0, 2.0e-12, 500.0, 3000.0, 20000.0),
        (8.0, 1000.0, 2.5e-12, 600.0, 3500.0, 25000.0),
    ]
    for Wn, Rs, Cs, Itail, RL, Rdfe in candidates:
        try:
            res = simulate_transistor_level(Wn, Rs, Cs, Itail, RL, Rdfe, corner='tt', temp=27, vdd=1.8)
            area = estimate_area_mm2(Wn, Rs, Cs, RL, Rdfe)
            print(f"W={Wn}u Rs={Rs} Cs={Cs*1e12:.1f}p Itail={Itail}u RL={RL} -> "
                  f"Peak={res['peaking_db']:.2f}dB LF={res['low_freq_gain_db']:.2f}dB HF={res['high_freq_gain_db']:.2f}dB "
                  f"Noise={res['noise_mvrms']:.3f}mV HD3={res['hd3_db']:.1f}dB Pwr={res['power_mw']:.2f}mW "
                  f"Eye={res['eye_height_proxy_mv']:.1f}mV Area={area:.5f}mm2")
        except Exception as e:
            print(f"Failed candidate: {e}")

if __name__ == '__main__':
    test_tuning()
