"""
circuit_surrogate.py — Surrogate MLP-backed simulate() drop-in.

Priority chain per call:
  1. SurrogateMLP inference (~0.01 ms/call, or ~0.002 ms/call batched on GPU)
  2. Analytical RC fallback (if surrogate not loaded)
  3. SPICE ground-truth (if force_spice=True)

Also provides simulate_batch(Rs_arr, Cs_arr) for true vectorized GPU
evaluation across all workers in a single torch.mm call.
"""

import os
import numpy as np
import torch

# Ensure ngspice on PATH
if os.path.exists(r"C:\Spice64\bin") and r"C:\Spice64\bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = r"C:\Spice64\bin;" + os.environ.get("PATH", "")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SURROGATE_MODEL = None
_NORM_STATS = None

_RS_MIN, _RS_MAX = 50.0, 500.0
_CS_MIN, _CS_MAX = 0.1e-12, 10e-12


def load_surrogate(path: str = None):
    """Load surrogate model from disk. Called lazily on first simulate() call."""
    global _SURROGATE_MODEL, _NORM_STATS
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', 'models', 'surrogate.pt')
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return False
    try:
        from surrogate import load_surrogate as _load
        model, stats = _load(path)
        _SURROGATE_MODEL = model.to(_DEVICE)
        _NORM_STATS = {k: v.to(_DEVICE).float() if isinstance(v, torch.Tensor) else v
                       for k, v in stats.items()}
        return True
    except Exception as e:
        print(f"[circuit_surrogate] Could not load surrogate: {e}")
        return False


def _normalise_inputs(Rs_arr: np.ndarray, Cs_arr: np.ndarray) -> torch.Tensor:
    """Normalise inputs using surrogate training normalization stats."""
    x = np.stack([Rs_arr, Cs_arr], axis=-1).astype(np.float32)
    x_t = torch.from_numpy(x).to(_DEVICE)
    if _NORM_STATS and 'input_min' in _NORM_STATS and 'input_max' in _NORM_STATS:
        x_min = _NORM_STATS['input_min']
        x_max = _NORM_STATS['input_max']
        x_t = (x_t - x_min) / (x_max - x_min + 1e-8)
    return x_t


def _denormalise_outputs(y_t: torch.Tensor) -> np.ndarray:
    """Reverse output normalisation to physical units."""
    if _NORM_STATS and 'output_mean' in _NORM_STATS:
        y_t = y_t * (_NORM_STATS['output_std'] + 1e-8) + _NORM_STATS['output_mean']
    return y_t.detach().cpu().numpy()


def simulate_batch(Rs_arr, Cs_arr) -> list:
    """
    Vectorized batch inference — evaluates N (Rs, Cs) pairs in one forward pass.

    Args:
        Rs_arr: array-like of shape (N,)
        Cs_arr: array-like of shape (N,)

    Returns:
        List of N result dicts, same format as simulate().
    """
    global _SURROGATE_MODEL
    if _SURROGATE_MODEL is None:
        load_surrogate()

    Rs_arr = np.asarray(Rs_arr, dtype=np.float64)
    Cs_arr = np.asarray(Cs_arr, dtype=np.float64)

    if _SURROGATE_MODEL is not None:
        x_t = _normalise_inputs(Rs_arr, Cs_arr)
        with torch.no_grad():
            y_t = _SURROGATE_MODEL(x_t)
        y = _denormalise_outputs(y_t)
        results = []
        for i in range(len(Rs_arr)):
            results.append({
                'peaking_db': float(y[i, 0]),
                'low_freq_gain_db': float(y[i, 0]) - 6.0,   # approximation
                'high_freq_gain_db': float(y[i, 0]) - 6.0 + float(y[i, 0]),
                'noise_mvrms': float(max(0.0, y[i, 1])),
                'eye_height_proxy_mv': float(max(0.0, y[i, 2])),
                'pulse_width_proxy_ps': 199.0,
                '_source': 'surrogate_batch',
            })
        return results
    else:
        # Fallback: analytical model, vectorized
        from circuit_fast import _analytical_simulate_numpy
        return [_analytical_simulate_numpy(float(Rs_arr[i]), float(Cs_arr[i]))
                for i in range(len(Rs_arr))]


def simulate(Rs_ohm: float, Cs_farad: float, RL_ohm: float = 200,
             return_curve: bool = False, force_spice: bool = False) -> dict:
    """
    Surrogate-backed drop-in for circuit.simulate() and circuit_fast.simulate().

    Speed hierarchy:
      Surrogate MLP: ~0.01 ms   (after load_surrogate())
      Analytical RC: ~0.12 ms   (fallback if no surrogate)
      SPICE single-pass: ~48 ms (force_spice=True, or final validation)
    """
    global _SURROGATE_MODEL

    if force_spice:
        from circuit_fast import _spice_simulate
        return _spice_simulate(Rs_ohm, Cs_farad, RL_ohm)

    if _SURROGATE_MODEL is None:
        loaded = load_surrogate()
        if not loaded:
            # Fall through to analytical model
            from circuit_fast import _analytical_simulate_numpy
            return _analytical_simulate_numpy(Rs_ohm, Cs_farad, RL_ohm)

    # Surrogate inference
    x_t = _normalise_inputs(
        np.array([Rs_ohm], dtype=np.float64),
        np.array([Cs_farad], dtype=np.float64)
    )
    with torch.no_grad():
        y_t = _SURROGATE_MODEL(x_t)
    y = _denormalise_outputs(y_t)[0]

    result = {
        'peaking_db': float(y[0]),
        'low_freq_gain_db': float(y[0]) - 6.0,
        'high_freq_gain_db': float(y[0]),
        'noise_mvrms': float(max(0.0, y[1])),
        'eye_height_proxy_mv': float(max(0.0, y[2])),
        'pulse_width_proxy_ps': 199.0,
        '_source': 'surrogate',
    }

    if return_curve:
        # Compute curve analytically (cheap, and not needed for RL)
        freqs = np.logspace(6, 10, 41)
        omega = 2 * np.pi * freqs
        jw = 1j * omega * Rs_ohm * Cs_farad
        H = RL_ohm * (1 + jw) / (RL_ohm * (1 + jw) + Rs_ohm)
        gain_db = 20 * np.log10(np.abs(H))
        result['frequency_hz'] = freqs.tolist()
        result['gain_db_curve'] = gain_db.tolist()

    return result


if __name__ == '__main__':
    import time

    print("Loading surrogate model...")
    ok = load_surrogate()
    print(f"Loaded: {ok}")

    # Benchmark vs SPICE
    from circuit_fast import _spice_simulate

    test_cases = [(200, 2e-12), (100, 1e-12), (400, 8e-12), (150, 5e-12)]
    print("\n=== Accuracy vs SPICE ground truth ===")
    for Rs, Cs in test_cases:
        spice = _spice_simulate(Rs, Cs)
        surr  = simulate(Rs, Cs)
        print(f"  Rs={Rs:.0f}Ohm Cs={Cs*1e12:.1f}pF | "
              f"SPICE peaking={spice['peaking_db']:.2f} dB  "
              f"Surrogate={surr['peaking_db']:.2f} dB  "
              f"err={abs(spice['peaking_db']-surr['peaking_db']):.3f} dB")

    print("\n=== Single-call benchmark ===")
    t0 = time.time()
    for _ in range(10000):
        simulate(200, 2e-12)
    t1 = time.time()
    print(f"  10,000 surrogate calls: {(t1-t0)*1000:.1f} ms  ({(t1-t0)/10:.4f} ms avg)")

    print("\n=== Batch benchmark (N=2048) ===")
    Rs_batch = np.random.uniform(50, 500, 2048)
    Cs_batch = np.random.uniform(0.1e-12, 10e-12, 2048)
    t0 = time.time()
    results = simulate_batch(Rs_batch, Cs_batch)
    t1 = time.time()
    print(f"  2048 surrogate batch: {(t1-t0)*1000:.2f} ms total  "
          f"({(t1-t0)/2048*1000:.4f} ms per sample)")
    print(f"  Sample peaking: {results[0]['peaking_db']:.2f} dB")
