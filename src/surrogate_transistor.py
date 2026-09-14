"""
surrogate_transistor.py — Neural SPICE Surrogate for Transistor-Level CTLE
============================================================================
Trains a 6-input → 5-output MLP on SkyWater 130 nm SPICE simulation data.
After training, delivers ~0.01 ms/call inference vs 2–4 s/call for ngspice —
a ~6,000× speedup that enables SubprocVecEnv-parallelized RL training.

Input  (6):  Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm
Output (5):  peaking_db, noise_mvrms, power_mw, eye_height_proxy_mv, hd3_db

Usage
-----
    # Generate data + train (one-time, ~5 min):
    python src/surrogate_transistor.py --generate --n-samples 600 --workers 12

    # Train only (if data already exists):
    python src/surrogate_transistor.py --train

    # Quick sanity benchmark:
    python src/surrogate_transistor.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "transistor_repository.jsonl"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_PATH   = MODEL_DIR / "surrogate_transistor.pt"

# ---------------------------------------------------------------------------
# Physical parameter bounds (must match environment_transistor.py ranges)
# ---------------------------------------------------------------------------
PARAM_BOUNDS = {
    "Wn_um":          (1.0,     15.0),    # wider: low Wn → low gain → ~3 dB; high Wn → ~12 dB
    "Rs_ohm":         (100.0,   3000.0),  # wider: high Rs → high degeneration → lower gain; low Rs → high gain
    "Cs_farad":       (0.5e-12, 5.0e-12), # slightly wider for more zero-placement freedom
    "Itail_half_ua":  (100.0,   1200.0),  # wider to support high-gain stages needing more bias current
    "RL_ohm":         (500.0,   5000.0),  # wider: higher RL for high gain at 12 dB
    "Rdfe_ohm":       (5000.0,  40000.0),
}
INPUT_KEYS  = list(PARAM_BOUNDS.keys())
OUTPUT_KEYS = ["peaking_db", "noise_mvrms", "power_mw", "eye_height_proxy_mv", "hd3_db", "eye_width_ui"]

# PVT corners used during data generation — all 5 to match environment_transistor.py
_PVT_CORNERS = [
    ('tt', 27,  1.8),    # typical-typical
    ('ss', 125, 1.71),   # slow-slow, hot, low-voltage
    ('ff', 0,   1.89),   # fast-fast, cold, high-voltage
    ('sf', 27,  1.8),    # slow-nfet, fast-pfet
    ('fs', 27,  1.8),    # fast-nfet, slow-pfet
]

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TransistorSurrogateMLP(nn.Module):
    """4-layer MLP with BatchNorm and Dropout for surrogate CTLE simulation."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 256, output_dim: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Global surrogate state (lazy-loaded, per-process)
# ---------------------------------------------------------------------------

_SURROGATE_MODEL:      Optional[TransistorSurrogateMLP] = None
_SURROGATE_NORM:       Optional[Dict]                   = None

# Surrogate INFERENCE always runs on CPU regardless of CUDA availability.
# Measured: GPU=0.95ms vs CPU=0.32ms (torch) vs CPU=0.05ms (numpy) per call.
# GPU only wins at batch_size >= ~256. RL env steps are batch_size=1 per worker.
# GPU is still used in train_surrogate() for training (large batches there).
_SURROGATE_DEVICE: torch.device = torch.device("cpu")

# Numpy weight cache — populated by load_transistor_surrogate() for fast inference
_NP_WEIGHTS:  Optional[list] = None   # list of (W, b) np.float32 arrays per Linear layer
_NP_BN:       Optional[list] = None   # list of (gamma, beta, mean, std) per BatchNorm layer
_NP_IN_MEAN:  Optional[np.ndarray] = None
_NP_IN_STD:   Optional[np.ndarray] = None
_NP_OUT_MEAN: Optional[np.ndarray] = None
_NP_OUT_STD:  Optional[np.ndarray] = None


def _extract_numpy_weights() -> None:
    """Extract PyTorch weights to numpy for ~0.05ms inference (vs 0.32ms torch)."""
    global _NP_WEIGHTS, _NP_BN, _NP_IN_MEAN, _NP_IN_STD, _NP_OUT_MEAN, _NP_OUT_STD
    import torch.nn as nn
    _NP_WEIGHTS = [
        (m.weight.detach().cpu().numpy().copy().astype(np.float32),
         m.bias.detach().cpu().numpy().copy().astype(np.float32))
        for m in _SURROGATE_MODEL.modules() if isinstance(m, nn.Linear)
    ]
    _NP_BN = [
        (m.weight.detach().cpu().numpy().copy().astype(np.float32),
         m.bias.detach().cpu().numpy().copy().astype(np.float32),
         m.running_mean.detach().cpu().numpy().copy().astype(np.float32),
         np.sqrt(m.running_var.detach().cpu().numpy().copy().astype(np.float32) + float(m.eps)))
        for m in _SURROGATE_MODEL.modules() if isinstance(m, nn.BatchNorm1d)
    ]
    _NP_IN_MEAN  = _SURROGATE_NORM['input_mean'].cpu().numpy().copy().astype(np.float32)
    _NP_IN_STD   = (_SURROGATE_NORM['input_std'].cpu().numpy().copy().astype(np.float32) + 1e-8)
    _NP_OUT_MEAN = _SURROGATE_NORM['output_mean'].cpu().numpy().copy().astype(np.float32)
    _NP_OUT_STD  = (_SURROGATE_NORM['output_std'].cpu().numpy().copy().astype(np.float32) + 1e-8)


def _numpy_forward(x_norm: np.ndarray) -> np.ndarray:
    """Pure numpy MLP forward pass — ~0.05ms vs 0.32ms for torch at batch_size=1."""
    def bn(h, i):
        return (h - _NP_BN[i][2]) / _NP_BN[i][3] * _NP_BN[i][0] + _NP_BN[i][1]
    x = np.maximum(0.0, bn(x_norm @ _NP_WEIGHTS[0][0].T + _NP_WEIGHTS[0][1], 0))
    x = np.maximum(0.0, bn(x @ _NP_WEIGHTS[1][0].T + _NP_WEIGHTS[1][1], 1))
    x = np.maximum(0.0, bn(x @ _NP_WEIGHTS[2][0].T + _NP_WEIGHTS[2][1], 2))
    return x @ _NP_WEIGHTS[3][0].T + _NP_WEIGHTS[3][1]


def load_transistor_surrogate(path: Optional[str] = None) -> bool:
    """
    Load the trained surrogate from disk. Safe to call from subprocesses —
    each process gets its own model copy. Also extracts numpy weights for fast inference.

    Returns True on success, False if the file does not exist.
    """
    global _SURROGATE_MODEL, _SURROGATE_NORM, _SURROGATE_DEVICE

    if path is None:
        path = str(MODEL_PATH)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return False

    try:
        ckpt = torch.load(path, map_location=_SURROGATE_DEVICE, weights_only=False)
        model = TransistorSurrogateMLP(
            input_dim=ckpt.get('input_dim', 6),
            hidden_dim=ckpt.get('hidden_dim', 256),
            output_dim=ckpt.get('output_dim', 5),
        )
        model.load_state_dict(ckpt['state_dict'])
        model.to(_SURROGATE_DEVICE)
        model.eval()
        _SURROGATE_MODEL = model

        # Load only the four float normalisation arrays — skip string/int metadata
        _FLOAT_NORM_KEYS = {'input_mean', 'input_std', 'output_mean', 'output_std'}
        norm = {}
        for k in _FLOAT_NORM_KEYS:
            if k not in ckpt:
                continue
            v = ckpt[k]
            if isinstance(v, torch.Tensor):
                norm[k] = v.to(_SURROGATE_DEVICE).float()
            else:
                norm[k] = torch.as_tensor(v, dtype=torch.float32, device=_SURROGATE_DEVICE)
        _SURROGATE_NORM = norm

        # Extract numpy weights for fast inference (6x speedup vs torch at batch_size=1)
        _extract_numpy_weights()
        return True
    except Exception as e:
        print(f"[surrogate_transistor] load failed: {e}")
        return False


def _is_loaded() -> bool:
    return _SURROGATE_MODEL is not None


def _normalise(x_np: np.ndarray) -> torch.Tensor:
    x_t = torch.from_numpy(x_np.astype(np.float32)).to(_SURROGATE_DEVICE)
    if _SURROGATE_NORM and 'input_mean' in _SURROGATE_NORM:
        x_t = (x_t - _SURROGATE_NORM['input_mean']) / (_SURROGATE_NORM['input_std'] + 1e-8)
    return x_t


def _denormalise(y_t: torch.Tensor) -> np.ndarray:
    if _SURROGATE_NORM and 'output_mean' in _SURROGATE_NORM:
        y_t = y_t * (_SURROGATE_NORM['output_std'] + 1e-8) + _SURROGATE_NORM['output_mean']
    return y_t.detach().cpu().numpy()


def simulate_transistor_surrogate(
    Wn_um: float, Rs_ohm: float, Cs_farad: float,
    Itail_half_ua: float, RL_ohm: float, Rdfe_ohm: float,
) -> Dict[str, float]:
    """
    Drop-in replacement for simulate_transistor_fast() using the trained MLP.
    Uses pure numpy forward pass (~0.05 ms) when weights are cached,
    falls back to torch (~0.32 ms) or SPICE if surrogate not loaded.
    """
    global _SURROGATE_MODEL
    if not _is_loaded():
        loaded = load_transistor_surrogate()
        if not loaded:
            from circuit_transistor import simulate_transistor_fast
            return simulate_transistor_fast(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm)

    if _NP_WEIGHTS is not None:
        # Fast numpy path: ~0.05ms per call (6x faster than torch at batch_size=1)
        x = np.array([[Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm]], dtype=np.float32)
        x_norm = (x - _NP_IN_MEAN) / _NP_IN_STD
        y_norm = _numpy_forward(x_norm)
        y = y_norm[0] * _NP_OUT_STD + _NP_OUT_MEAN
    else:
        # Torch fallback path
        x = np.array([[Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm]], dtype=np.float32)
        x_t = _normalise(x)
        with torch.no_grad():
            y_t = _SURROGATE_MODEL(x_t)
        y = _denormalise(y_t)[0]

    return {
        'peaking_db':           float(y[0]),
        'noise_mvrms':          float(max(0.0, y[1])),
        'power_mw':             float(max(0.0, y[2])),
        'eye_height_proxy_mv':  float(max(0.0, y[3])),
        'hd3_db':               float(y[4]),
        'eye_width_ui':         float(max(0.0, y[5])) if len(y) > 5 else 0.0,
        'low_freq_gain_db':     float(y[0]) - 6.0,
        'high_freq_gain_db':    float(y[0]),
        '_source':              'surrogate_transistor',
    }


def simulate_batch_transistor(params_array: np.ndarray) -> list:
    """
    Vectorized batch inference over N parameter sets.
    params_array: shape (N, 6) — [Wn, Rs, Cs, Itail, RL, Rdfe] per row.
    Returns: list of N result dicts.
    """
    if not _is_loaded():
        load_transistor_surrogate()

    x_t = _normalise(params_array.astype(np.float32))
    with torch.no_grad():
        y_t = _SURROGATE_MODEL(x_t)
    y = _denormalise(y_t)

    results = []
    for i in range(len(params_array)):
        results.append({
            'peaking_db':           float(y[i, 0]),
            'noise_mvrms':          float(max(0.0, y[i, 1])),
            'power_mw':             float(max(0.0, y[i, 2])),
            'eye_height_proxy_mv':  float(max(0.0, y[i, 3])),
            'hd3_db':               float(y[i, 4]),
            '_source':              'surrogate_batch',
        })
    return results


# ---------------------------------------------------------------------------
# Data generation (parallel SPICE sweeps)
# ---------------------------------------------------------------------------

def _latin_hypercube_sample(n: int, bounds: dict, seed: int = 42) -> np.ndarray:
    """Generate n samples via Latin Hypercube Sampling over the 6D parameter space."""
    rng = np.random.default_rng(seed)
    n_dims = len(bounds)
    samples = np.zeros((n, n_dims))
    for i, (lo, hi) in enumerate(bounds.values()):
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n
        samples[:, i] = lo + u * (hi - lo)
    return samples


def _spice_one(args):
    """Worker function — simulates one parameter set, returns data records or empty list on failure.

    Uses compute_hd3=False (fast ~50ms/call) for reliable bulk data generation.
    Eye transient still runs so eye_width_ui is recorded accurately.
    HD3 defaults to -38.0 placeholder; use a separate enrichment pass with
    compute_hd3=True on a subset of records once the surrogate is trained.
    """
    idx, Wn, Rs, Cs, Itail, RL, Rdfe = args
    # Use PID-unique worker_id to avoid temp file conflicts across processes
    worker_id = os.getpid()
    results = []
    for corner, temp, vdd in _PVT_CORNERS:
        try:
            from circuit_transistor import simulate_transistor_fast
            r = simulate_transistor_fast(
                float(Wn), float(Rs), float(Cs), float(Itail),
                float(RL), float(Rdfe),
                corner=corner, temp=temp, vdd=vdd, topology='1stage',
                worker_id=worker_id,
                compute_hd3=False,  # fast path — eye transient still runs for eye_width_ui
            )
            record = {
                'Wn_um':               float(Wn),
                'Rs_ohm':              float(Rs),
                'Cs_farad':            float(Cs),
                'Itail_half_ua':       float(Itail),
                'RL_ohm':              float(RL),
                'Rdfe_ohm':            float(Rdfe),
                'peaking_db':          r['peaking_db'],
                'noise_mvrms':         r['noise_mvrms'],
                'power_mw':            r['power_mw'],
                'eye_height_proxy_mv': r['eye_height_proxy_mv'],
                'hd3_db':              r.get('hd3_db', -38.0),  # -38.0 placeholder in fast path
                'eye_width_ui':        r.get('eye_width_ui', 0.0),
                'corner': corner, 'temp': temp, 'vdd': vdd,
            }
            results.append(record)
        except Exception:
            pass  # skip failed SPICE runs silently
    return results


def generate_training_data(
    n_samples: int = 600,
    n_workers: int = None,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> int:
    """
    Generate transistor surrogate training data via parallel SPICE simulations.

    Runs LHS across the 6D parameter space, evaluates each across TT/SS/FF corners
    using n_workers parallel ngspice processes.

    Returns: number of records written.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 12)
    if output_path is None:
        output_path = str(DATA_PATH)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[data_gen] Generating {n_samples} LHS samples × {len(_PVT_CORNERS)} corners "
          f"= {n_samples * len(_PVT_CORNERS)} SPICE calls across {n_workers} workers")
    print(f"[data_gen] Estimated time: {n_samples * len(_PVT_CORNERS) * 0.1 / n_workers / 60:.1f} min "
          f"(~100ms/call: AC+Noise + eye transient, no HD3)")

    samples = _latin_hypercube_sample(n_samples, PARAM_BOUNDS, seed=seed)
    args_list = [
        (i, *samples[i].tolist())
        for i in range(n_samples)
    ]

    t0 = time.time()
    n_written = 0

    # Append mode — preserves any existing records
    existing = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                existing.add(line.strip())

    with open(output_path, 'a', encoding='utf-8') as fout:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for i, batch in enumerate(pool.imap_unordered(_spice_one, args_list, chunksize=1)):
                for record in (batch or []):
                    line = json.dumps(record)
                    if line not in existing:
                        fout.write(line + '\n')
                        n_written += 1
                if (i + 1) % max(1, n_samples // 10) == 0:
                    elapsed = time.time() - t0
                    pct = (i + 1) / n_samples * 100
                    eta = elapsed / (i + 1) * (n_samples - i - 1)
                    print(f"  [{pct:5.1f}%] {i+1}/{n_samples} param sets | "
                          f"{n_written} records | ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[data_gen] Done: {n_written} new records in {elapsed:.1f}s -> {output_path}")
    return n_written


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_surrogate(
    data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    hidden_dim: int = 256,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.2,
) -> Optional[TransistorSurrogateMLP]:
    """
    Train the surrogate MLP on transistor_repository.jsonl data.
    Returns the trained model, or None if insufficient data.
    """
    if data_path is None:
        clean_path = PROJECT_ROOT / "data" / "transistor_repository_clean.jsonl"
        data_path = str(clean_path if clean_path.exists() else DATA_PATH)
    if model_path is None:
        model_path = str(MODEL_PATH)

    # Key alias map: accepts both legacy short keys (Wn, Rs, Cs, Itail, RL, Rdfe)
    # and new fully-qualified keys (Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm)
    _KEY_ALIASES = {
        'Wn_um':         ['Wn_um',  'Wn'],
        'Rs_ohm':        ['Rs_ohm', 'Rs'],
        'Cs_farad':      ['Cs_farad', 'Cs'],
        'Itail_half_ua': ['Itail_half_ua', 'Itail'],
        'RL_ohm':        ['RL_ohm', 'RL'],
        'Rdfe_ohm':      ['Rdfe_ohm', 'Rdfe'],
    }

    def _get(r, key):
        for alias in _KEY_ALIASES.get(key, [key]):
            if alias in r:
                return float(r[alias])
        raise KeyError(key)

    # Fallback defaults for outputs missing in old-format records.
    # Old records (pre eye_width_ui) lack these fields; filling with safe defaults
    # allows the full existing dataset to be used for training immediately.
    _OUTPUT_DEFAULTS = {
        'eye_width_ui': 0.0,   # unknown → 0.0 (model learns ~0 for old param ranges)
        'hd3_db':      -38.0,  # fast-path placeholder — overwritten by new records
    }

    # Load data
    records = []
    skipped_no_key = 0
    if not os.path.exists(data_path):
        print(f"[train] Data file not found: {data_path}")
        print("[train] Run with --generate first.")
        return None

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                row_in = [_get(r, k) for k in INPUT_KEYS]
                # Allow missing output keys — fall back to _OUTPUT_DEFAULTS
                row_out = []
                for k in OUTPUT_KEYS:
                    if k in r:
                        row_out.append(float(r[k]))
                    elif k in _OUTPUT_DEFAULTS:
                        row_out.append(_OUTPUT_DEFAULTS[k])
                    else:
                        raise KeyError(k)
                if any(np.isnan(v) or np.isinf(v) for v in row_in + row_out):
                    continue
                # Filter obviously-failed SPICE results (negative eye height or extreme noise)
                if row_out[3] < -500 or row_out[1] > 100.0:
                    continue
                records.append((row_in, row_out))
            except (KeyError, ValueError):
                skipped_no_key += 1
                continue
            except Exception:
                continue

    if skipped_no_key > 0:
        print(f"[train] Skipped {skipped_no_key} records with missing/incompatible keys")

    if len(records) < 50:
        print(f"[train] Only {len(records)} valid records — need at least 50. Run --generate first.")
        return None

    print(f"[train] Loaded {len(records)} valid records from {data_path}")

    X = np.array([r[0] for r in records], dtype=np.float32)
    Y = np.array([r[1] for r in records], dtype=np.float32)

    # Normalise
    x_mean = X.mean(axis=0)
    x_std  = X.std(axis=0) + 1e-8
    y_mean = Y.mean(axis=0)
    y_std  = Y.std(axis=0) + 1e-8
    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Fix: track validation indices explicitly so MAE report uses the correct rows.
    # random_split() shuffles internally — indexing X[n_train:] is WRONG because it
    # returns original-order rows, not the shuffled validation set.
    N = len(X)
    n_val   = max(1, int(N * val_split))
    n_train = N - n_val
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    X_t = torch.from_numpy(X_norm)
    Y_t = torch.from_numpy(Y_norm)
    from torch.utils.data import Subset
    dataset   = TensorDataset(X_t, Y_t)
    train_ds  = Subset(dataset, train_idx.tolist())
    val_ds    = Subset(dataset, val_idx.tolist())

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=(device.type == 'cuda'))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size * 4, shuffle=False)

    n_outputs = len(OUTPUT_KEYS)
    model = TransistorSurrogateMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=n_outputs).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    print(f"[train] Training {epochs} epochs, {n_train} train / {n_val} val samples, {n_outputs} outputs")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= n_val

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s")

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()

    # Accuracy report on val set (physical units) — uses correctly tracked val_idx
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    X_val_t = torch.from_numpy((X_val - x_mean) / x_std).to(device)
    with torch.no_grad():
        Y_pred_norm = model(X_val_t).cpu().numpy()
    Y_pred = Y_pred_norm * y_std + y_mean
    mae = np.abs(Y_pred - Y_val).mean(axis=0)
    print("\n[train] Validation MAE (physical units):")
    for name, err in zip(OUTPUT_KEYS, mae):
        unit = 'dB' if 'db' in name else ('UI' if 'ui' in name else ('mV' if 'mv' in name.lower() else 'mW'))
        print(f"  {name:<25} {err:.4f} {unit}")

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'state_dict':   best_state,
        'input_dim':    6,
        'hidden_dim':   hidden_dim,
        'output_dim':   n_outputs,
        'input_mean':   torch.from_numpy(x_mean),
        'input_std':    torch.from_numpy(x_std),
        'output_mean':  torch.from_numpy(y_mean),
        'output_std':   torch.from_numpy(y_std),
        'input_keys':   INPUT_KEYS,
        'output_keys':  OUTPUT_KEYS,
        'val_loss':     best_val,
        'n_records':    len(records),
    }, model_path)
    print(f"[train] Saved to {model_path}  (val_loss={best_val:.5f})")
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(description='Transistor surrogate MLP trainer')
    parser.add_argument('--generate', action='store_true',
                        help='Generate SPICE training data first')
    parser.add_argument('--train', action='store_true',
                        help='Train the surrogate MLP')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark surrogate vs SPICE')
    parser.add_argument('--n-samples', type=int, default=600,
                        help='LHS samples to generate (default: 600)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel SPICE workers (default: all cores, max 12)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden-dim', type=int, default=256)
    args = parser.parse_args()

    if not any([args.generate, args.train, args.benchmark]):
        # Default: generate + train
        args.generate = True
        args.train = True

    if args.generate:
        multiprocessing.freeze_support()
        n = generate_training_data(n_samples=args.n_samples, n_workers=args.workers)
        print(f"Generated {n} records.")

    if args.train:
        model = train_surrogate(epochs=args.epochs, hidden_dim=args.hidden_dim)
        if model is None:
            sys.exit(1)

    if args.benchmark:
        print("\n=== Surrogate vs SPICE benchmark ===")
        ok = load_transistor_surrogate()
        print(f"Surrogate loaded: {ok}")
        if ok:
            from circuit_transistor import simulate_transistor_fast
            test_params = [
                (4.0, 750.0, 1.8e-12, 400.0, 2200.0, 20000.0),
                (6.0, 500.0, 1.0e-12, 600.0, 1500.0, 10000.0),
                (8.0, 1200.0, 3.0e-12, 800.0, 3500.0, 30000.0),
            ]
            print(f"\n{'Params':<50} {'SPICE peak':>11} {'Surr peak':>10} {'Err':>7}")
            for p in test_params:
                spice = simulate_transistor_fast(*p)
                surr  = simulate_transistor_surrogate(*p)
                err   = abs(spice['peaking_db'] - surr['peaking_db'])
                label = f"Wn={p[0]:.0f} Rs={p[1]:.0f} Cs={p[2]*1e12:.1f}pF"
                print(f"  {label:<48} {spice['peaking_db']:>10.3f}  {surr['peaking_db']:>9.3f}  {err:>6.3f}dB")

            print("\n=== Speed benchmark ===")
            t0 = time.perf_counter()
            for _ in range(10000):
                simulate_transistor_surrogate(4.0, 750.0, 1.8e-12, 400.0, 2200.0, 20000.0)
            t1 = time.perf_counter()
            print(f"  10,000 surrogate calls: {(t1-t0)*1000:.1f} ms  ({(t1-t0)/10:.4f} ms avg)")

            # Batch benchmark
            N = 2048
            params_batch = np.column_stack([
                np.random.uniform(*PARAM_BOUNDS['Wn_um'], N),
                np.random.uniform(*PARAM_BOUNDS['Rs_ohm'], N),
                np.random.uniform(*PARAM_BOUNDS['Cs_farad'], N),
                np.random.uniform(*PARAM_BOUNDS['Itail_half_ua'], N),
                np.random.uniform(*PARAM_BOUNDS['RL_ohm'], N),
                np.random.uniform(*PARAM_BOUNDS['Rdfe_ohm'], N),
            ])
            t0 = time.perf_counter()
            simulate_batch_transistor(params_batch)
            t1 = time.perf_counter()
            print(f"  Batch {N} surrogate calls: {(t1-t0)*1000:.2f} ms  "
                  f"({(t1-t0)/N*1000:.4f} ms per sample)")
