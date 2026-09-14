"""
validate_pvt.py — Master PVT Validation Sign-Off Script for Transistor-Level Equalizer.

Evaluates synthesized designs across:
- 5 Process Corners: TT, SS, FF, SF, FS
- 3 Supply Voltages: 1.71V, 1.80V, 1.89V (+/- 5%)
- 3 Operating Temps: 0°C, 27°C, 125°C
Total: 5 representative extreme corners (default) or full 45 PVT conditions with all 7 specs.
Supports Hybrid RACD (FAISS + RL) and Pure RL sizing modes.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from circuit_transistor import simulate_transistor_level
from environment_transistor import TransistorEqualizerEnv, estimate_area_2stage_mm2


def run_pvt_validation(target_peaking_db=6.0, mode='racd', full_matrix=False):
    """
    Validates the transistor equalizer across PVT corners.
    mode: 'racd' (Hybrid FAISS + RL) or 'rl' (Pure PPO)
    If full_matrix is False, evaluates the 5 worst-case representative corners.
    If full_matrix is True, evaluates all 45 grid points.
    """
    print("=" * 115)
    print(f"  PVT Multi-Corner Validation Sign-Off (Target: {target_peaking_db:.1f} dB, Mode: {mode.upper()})")
    print("=" * 115)

    source_info = "UNKNOWN"
    if mode == 'racd':
        from retrieval_faiss_transistor import synthesize_transistor_racd
        synth = synthesize_transistor_racd(target_peaking_db, corner='tt')
        sz = synth['sizing']
        Wn, Rs, Cs, Itail, RL, Rdfe = sz['Wn_um'], sz['Rs_ohm'], sz['Cs_farad'], sz['Itail_half_ua'], sz['RL_ohm'], sz['Rdfe_ohm']
        source_info = f"{synth['source']} (Error: {synth['error_db']:.2f} dB, Latency: {synth['elapsed_ms']:.2f} ms)"
    else:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor')
        if os.path.exists(model_path + '.zip'):
            from stable_baselines3 import PPO
            model = PPO.load(model_path, device='cpu')
            env = TransistorEqualizerEnv(peaking_target_range=(target_peaking_db, target_peaking_db))
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            Wn, Rs, Cs, Itail, RL, Rdfe = env._rescale_action(action)
            source_info = "PURE_PPO_POLICY"
        else:
            Wn, Rs, Cs, Itail, RL, Rdfe = 4.0, 750.0, 1.8e-12, 400.0, 2200.0, 20000.0
            source_info = "ANALYTICAL_FALLBACK"

    area = estimate_area_2stage_mm2(Wn, Rs, Cs, RL, Rdfe)
    print(f"Synthesis Origin: {source_info}")
    print(f"Sizing: Wn={Wn:.2f} um, Rs={Rs:.1f} Ohm, Cs={Cs*1e12:.2f} pF, Itail={Itail:.1f} uA, RL={RL:.1f} Ohm, Rdfe={Rdfe:.1f} Ohm")
    print(f"Estimated Die Area: {area:.5f} mm² (Spec: < 0.05 mm²)\n")

    corners = ['tt', 'ss', 'ff', 'sf', 'fs']
    volts = [1.80] if not full_matrix else [1.71, 1.80, 1.89]
    temps = [27] if not full_matrix else [0, 27, 125]

    if not full_matrix:
        test_points = [
            ('tt', 1.80, 27),
            ('ss', 1.71, 125),
            ('ff', 1.89, 0),
            ('sf', 1.80, 27),
            ('fs', 1.80, 27)
        ]
    else:
        test_points = [(c, v, t) for c in corners for v in volts for t in temps]

    print("=" * 115)
    print(f"{'Corner':<8} {'VDD':<6} {'Temp':<6} {'Peaking (dB)':<15} {'HD3 (<-30dB)':<14} {'Noise (<1.5mV)':<16} {'Power (<15mW)':<14} {'Eye (>100mV)':<14} {'Area (<0.05mm²)':<16} {'Compliance'}")
    print("=" * 115)

    results = []
    passes = 0

    for c, v, t in test_points:
        res = simulate_transistor_level(
            Wn, Rs, Cs, Itail, RL, Rdfe,
            corner=c, temp=t, vdd=v,
            topology='2stage'
        )
        p = res['peaking_db']
        h = res['hd3_db']
        n = res['noise_mvrms']
        pwr = res['power_mw']
        eye = res['eye_height_proxy_mv']

        peaking_ok = abs(p - target_peaking_db) <= 1.5  # within 1.5 dB of target
        ok = peaking_ok and (h <= -30.0) and (n <= 1.5) and (pwr <= 15.0) and (eye >= 100.0) and (area <= 0.05)
        if ok:
            passes += 1
        # MARGINAL: only if HD3 is the single failing metric AND it's close
        only_hd3_fail = (not ok) and peaking_ok and (n <= 1.5) and (pwr <= 15.0) and (eye >= 100.0) and (area <= 0.05)
        status = "PASS" if ok else ("MARGINAL" if (only_hd3_fail and h <= -28.0) else "FAIL")

        print(f"{c.upper():<8} {v:<6.2f} {t:<6d} {p:<15.2f} {h:<14.1f} {n:<16.3f} {pwr:<14.2f} {eye:<14.1f} {area:<16.5f} {status}")
        results.append({
            'corner': c.upper(), 'vdd': v, 'temp': t,
            'peaking_db': p, 'hd3_db': h, 'noise_mvrms': n,
            'power_mw': pwr, 'eye_height_mv': eye, 'area_mm2': area,
            'status': status
        })

    print("=" * 115)
    pass_rate = (passes / len(test_points)) * 100
    print(f"PVT Multi-Corner Audit: {passes}/{len(test_points)} Passed ({pass_rate:.1f}% Compliance Rate)")

    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pvt_validation_log.json')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete multi-corner audit results to: {log_file}\n")
    return pass_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=float, default=8.0, help='Target peaking dB (default: 8.0)')
    parser.add_argument('--mode', type=str, default='racd', choices=['racd', 'rl'], help='Sizing mode: racd (hybrid) or rl (pure PPO)')
    parser.add_argument('--full', action='store_true', help='Run full 45-point PVT matrix instead of 5 representative corners')
    args = parser.parse_args()

    run_pvt_validation(target_peaking_db=args.target, mode=args.mode, full_matrix=args.full)
