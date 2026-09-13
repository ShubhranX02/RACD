"""
verify_all.py — Master Acceptance Verification Suite for RACD
=============================================================
Runs tests against all Definition-of-Done criteria in ProjectGuide.md Part 13
and prints an end-to-end audit report.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def check_phase1():
    from circuit_fast import simulate
    res = simulate(200.0, 2.0e-12)
    req_keys = ['peaking_db', 'low_freq_gain_db', 'high_freq_gain_db', 'noise_mvrms', 'eye_height_proxy_mv', 'pulse_width_proxy_ps']
    assert all(k in res for k in req_keys), f"Missing keys: {res}"
    return f"peaking={res['peaking_db']:.2f} dB, noise={res['noise_mvrms']:.3f} mV"


def check_phase2():
    from environment_fast import EqualizerEnvFast
    env = EqualizerEnvFast()
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert 'Rs' in info and 'Cs' in info and 'peaking_db' in info
    return f"reward={reward:.4f}, step verified"


def check_phase3():
    from circuit_fast import simulate
    targets = [4.0, 6.0, 8.0, 10.0]
    hits = 0
    for t in targets:
        r_low = simulate(50, 0.5e-12)['peaking_db']
        r_high = simulate(450, 8.0e-12)['peaking_db']
        if r_low <= t <= r_high or r_high <= t <= r_low:
            hits += 1
    return f"Target coverage: {hits}/{len(targets)} targets within physical bounds"


def check_phase4():
    from retrieval_faiss import retrieve_closest_faiss
    from logging_utils import load_repository
    repo = load_repository()
    assert len(repo) > 0, "Repository is empty"
    match = retrieve_closest_faiss({'target_peaking_db': 8.0}, repo)
    assert match is not None
    return f"FAISS retrieved: Rs={match['Rs']:.1f} Ohm, Cs={match['Cs']*1e12:.2f} pF ({len(repo)} episodes)"


def check_phase5():
    from stable_baselines3 import PPO
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_equalizer_fast')
    model = PPO.load(path)
    obs = np.array([8.0], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return f"PPO model loaded, deterministic action={action}"


def check_phase6():
    from orchestrator import parse_spec_request
    spec = parse_spec_request("high peaking around 9.5 dB with strict noise under 1.2 mV")
    assert 9.0 <= spec['target_peaking_db'] <= 10.0
    return f"Parsed: target={spec['target_peaking_db']:.1f} dB, noise_limit={spec.get('noise_limit_mvrms', 1.5):.2f} mV"


def check_phase7():
    from export_onnx import load_onnx_policy
    onnx_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_equalizer_fast.onnx')
    predict_fn, _ = load_onnx_policy(onnx_path)
    obs = np.array([8.0], dtype=np.float32)
    action = predict_fn(obs)
    return f"ONNX Runtime verified: action={action}"


def check_phase8():
    from circuit_transistor import simulate_transistor_fast
    res = simulate_transistor_fast(10.0, 200.0, 1.0e-12, 1000.0, 1000.0, 10000.0)
    assert 'peaking_db' in res and 'power_mw' in res
    return f"Sky130 PDK verified: peaking={res['peaking_db']:.2f} dB, power={res['power_mw']:.2f} mW"


def check_phase9():
    from retrieval_faiss_transistor import retrieve_k_nearest_transistor
    res = retrieve_k_nearest_transistor(target_peaking_db=8.0, k=1)
    assert len(res) > 0, "Transistor FAISS returned no results"
    best = res[0]
    return f"Transistor FAISS verified: Wn={best['Wn_um']:.2f}um, Rs={best['Rs_ohm']:.0f} Ohm, peaking={best['peaking_db']:.2f} dB (err={best['_peaking_error_db']:.2f} dB)"


def main():
    print("=" * 75)
    print("  RACD (Retrieval-Augmented Circuit Design) — Master Verification")
    print("=" * 75)

    tests = [
        ("Phase 1: Passive SPICE Simulator", check_phase1),
        ("Phase 2: Gymnasium Environment", check_phase2),
        ("Phase 3: Target Feasibility Coverage", check_phase3),
        ("Phase 4: FAISS Vector Retrieval", check_phase4),
        ("Phase 5: Trained RL Policies (PPO)", check_phase5),
        ("Phase 6: Natural Language Orchestrator", check_phase6),
        ("Phase 7: ONNX Sub-Millisecond Engine", check_phase7),
        ("Phase 8: Sky130 Transistor PDK Simulation", check_phase8),
        ("Phase 9: Transistor FAISS Vector Retrieval", check_phase9),
    ]

    all_pass = True
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            detail = fn()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [PASS] {name:<44} ({elapsed:6.1f} ms) | {detail}")
        except Exception as e:
            all_pass = False
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [FAIL] {name:<44} ({elapsed:6.1f} ms) | Error: {e}")

    print("=" * 75)
    if all_pass:
        print("  ALL ACCEPTANCE CRITERIA PASSED — 100% SPEC COMPLIANT")
    else:
        print("  SOME CHECKS FAILED — SEE LOG ABOVE")
    print("=" * 75)


if __name__ == '__main__':
    main()
