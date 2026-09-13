"""
surrogate.py  -  Neural SPICE Surrogate Model
==============================================
Trains a compact 3-layer MLP on measured circuit-simulation data stored in
``data/circuit_repository.jsonl``.  After training the model and its
normalization statistics are saved together in ``models/surrogate.pt`` so
that any downstream component can call ``load_surrogate()`` and obtain
sub-microsecond predictions in place of a full ngspice run.

Schema expected in the JSONL file
----------------------------------
Inputs  (2):  Rs  [50-500 ohm],  Cs  [0.1e-12 - 10e-12 F]
Outputs (3):  peaking_db, noise_mvrms, eye_height_proxy_mv

Usage
-----
    python src/surrogate.py                  # train from scratch
    python -c "from src.surrogate import load_surrogate; m,n = load_surrogate('models/surrogate.pt')"
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---------------------------------------------------------------------------
# Paths (absolute, resolved relative to this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "circuit_repository.jsonl"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_PATH   = MODEL_DIR / "surrogate.pt"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
INPUT_COLS  = ["Rs", "Cs"]
OUTPUT_COLS = ["peaking_db", "noise_mvrms", "eye_height_proxy_mv"]

# Physical bounds used for min-max clipping before normalisation
INPUT_BOUNDS = {
    "Rs": (50.0,    500.0),
    "Cs": (0.1e-12, 10e-12),
}


# ===========================================================================
# Model
# ===========================================================================

class SurrogateMLP(nn.Module):
    """
    Three-hidden-layer Multi-Layer Perceptron for fast circuit-metric
    prediction.

    Parameters
    ----------
    input_dim  : int  - number of circuit design parameters (default 2)
    hidden_dim : int  - neurons per hidden layer              (default 128)
    output_dim : int  - number of predicted metrics           (default 3)
    """

    def __init__(
        self,
        input_dim:  int = 2,
        hidden_dim: int = 128,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # --- Layer 1 ---
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # --- Layer 2 ---
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # --- Layer 3 ---
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # --- Output head ---
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; x should be pre-normalised."""
        return self.net(x)


# ===========================================================================
# Data loading & preprocessing
# ===========================================================================

def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ``circuit_repository.jsonl`` and return clean (X, Y) arrays.

    - Rows where any required field is missing, NaN, or Inf are dropped.
    - Input values are clipped to their physical bounds before returning.

    Returns
    -------
    X : ndarray, shape (N, 2)   - [Rs, Cs]
    Y : ndarray, shape (N, 3)   - [peaking_db, noise_mvrms, eye_height_proxy_mv]
    """
    records = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] line {lineno}: JSON parse error - {exc}", file=sys.stderr)
                skipped += 1
                continue

            # Check all required keys exist
            if not all(k in row for k in INPUT_COLS + OUTPUT_COLS):
                skipped += 1
                continue

            # Extract numeric values
            try:
                vals = {k: float(row[k]) for k in INPUT_COLS + OUTPUT_COLS}
            except (ValueError, TypeError):
                skipped += 1
                continue

            # Drop NaN / Inf
            if any(not math.isfinite(v) for v in vals.values()):
                skipped += 1
                continue

            records.append(vals)

    if not records:
        raise RuntimeError(f"No valid records found in {path}")

    print(f"  Loaded {len(records):,} rows  ({skipped} skipped)")

    X_raw = np.array([[r[c] for c in INPUT_COLS]  for r in records], dtype=np.float32)
    Y_raw = np.array([[r[c] for c in OUTPUT_COLS] for r in records], dtype=np.float32)

    # Clip inputs to physical bounds (handle sensor noise / labelling artefacts)
    for col_idx, col in enumerate(INPUT_COLS):
        lo, hi = INPUT_BOUNDS[col]
        X_raw[:, col_idx] = np.clip(X_raw[:, col_idx], lo, hi)

    # Clip outputs to finite extremes (guard against rare measurement spikes)
    Y_raw = np.clip(Y_raw, -1e6, 1e6)

    return X_raw, Y_raw


# ===========================================================================
# Normalisation helpers
# ===========================================================================

def min_max_scale(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale each column of X to [0, 1] using per-column min and max.

    Returns
    -------
    X_scaled : ndarray  - normalised values in [0, 1]
    x_min    : ndarray  - per-column minimum
    x_max    : ndarray  - per-column maximum  (for inverse transform)
    """
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    # Avoid division by zero if a column is constant
    denom = np.where(x_max - x_min == 0, 1.0, x_max - x_min)
    return (X - x_min) / denom, x_min, x_max


def standard_scale(
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardise each column of Y to zero-mean / unit-variance.

    Returns
    -------
    Y_scaled : ndarray  - standardised values
    y_mean   : ndarray  - per-column mean
    y_std    : ndarray  - per-column std  (for inverse transform)
    """
    y_mean = Y.mean(axis=0)
    y_std  = Y.std(axis=0)
    y_std  = np.where(y_std == 0, 1.0, y_std)   # guard against zero-variance
    return (Y - y_mean) / y_std, y_mean, y_std


# ===========================================================================
# Training pipeline
# ===========================================================================

def train_surrogate(
    data_path:  Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    hidden_dim: int  = 128,
    epochs:     int  = 200,
    batch_size: int  = 512,
    lr:         float = 1e-3,
    val_frac:   float = 0.20,
    seed:       int   = 42,
    device:     str   = None,
) -> None:
    """
    Full training pipeline.

    Steps
    -----
    1. Load & clean data from *data_path*.
    2. Normalise inputs (min-max) and outputs (mean/std).
    3. Build ``SurrogateMLP`` and optimise with Adam + MSELoss.
    4. Print train/val loss every 20 epochs.
    5. Save best-val-loss model + norm stats to *model_path*.
    6. Print per-output MAE on 100 random validation samples.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Neural SPICE Surrogate - training on {device.upper()}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading dataset ...")
    X_np, Y_np = load_dataset(data_path)
    print(f"  X shape: {X_np.shape}   Y shape: {Y_np.shape}")

    # ------------------------------------------------------------------
    # 2. Normalise
    # ------------------------------------------------------------------
    print("[2/6] Normalising ...")
    X_scaled, x_min, x_max = min_max_scale(X_np)
    Y_scaled, y_mean, y_std = standard_scale(Y_np)

    # Store as float32 tensors
    X_t = torch.from_numpy(X_scaled).float()
    Y_t = torch.from_numpy(Y_scaled).float()

    # ------------------------------------------------------------------
    # 3. Train / val split
    # ------------------------------------------------------------------
    print("[3/6] Splitting train / val ...")
    dataset   = TensorDataset(X_t, Y_t)
    n_val     = max(1, int(len(dataset) * val_frac))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"  Train: {n_train:,}   Val: {n_val:,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # 4. Model, optimiser, loss
    # ------------------------------------------------------------------
    print("[4/6] Building model ...")
    model = SurrogateMLP(
        input_dim  = len(INPUT_COLS),
        hidden_dim = hidden_dim,
        output_dim = len(OUTPUT_COLS),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"\n[5/6] Training for {epochs} epochs ...\n")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}")

    best_val_loss  = float("inf")
    best_state     = None

    for epoch in range(1, epochs + 1):

        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()
            train_loss_sum += loss.item() * len(xb)

        train_loss = train_loss_sum / n_train

        # ---- validate ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item() * len(xb)

        val_loss = val_loss_sum / n_val
        scheduler.step()

        # Track best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Log every 20 epochs (and epoch 1)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}")

    print(f"\n  Best val loss: {best_val_loss:.6f}")

    # ------------------------------------------------------------------
    # 6. Save model + normalisation stats
    # ------------------------------------------------------------------
    print(f"\n[6/6] Saving model to {model_path} ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # input_mean / input_std kept as (min, range) so callers using the
    # standard z-score formula  (x - mean) / std  get min-max scaling.
    x_range = x_max - x_min
    x_range = np.where(x_range == 0, 1.0, x_range)

    save_dict: Dict[str, Any] = {
        "state_dict":   best_state,
        # --- input normalisation (min-max; stored in two equivalent forms) ---
        "input_min":    torch.from_numpy(x_min).float(),
        "input_max":    torch.from_numpy(x_max).float(),
        "input_mean":   torch.from_numpy(x_min).float(),    # alias: min
        "input_std":    torch.from_numpy(x_range).float(),  # alias: range
        # --- output normalisation (z-score) ---
        "output_mean":  torch.from_numpy(y_mean).float(),
        "output_std":   torch.from_numpy(y_std).float(),
        # --- metadata ---
        "input_cols":   INPUT_COLS,
        "output_cols":  OUTPUT_COLS,
        "hidden_dim":   hidden_dim,
    }
    torch.save(save_dict, model_path)
    print(f"  Saved  OK  ({model_path.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # 7. Quick accuracy check on 100 random val samples
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Accuracy check - 100 random validation samples")
    print("="*60)

    # Re-gather all val tensors from the loader
    val_X_list, val_Y_list = [], []
    for xb, yb in val_loader:
        val_X_list.append(xb)
        val_Y_list.append(yb)
    val_X_all = torch.cat(val_X_list, dim=0)
    val_Y_all = torch.cat(val_Y_list, dim=0)

    n_check = min(100, len(val_X_all))
    idx     = torch.randperm(len(val_X_all))[:n_check]
    check_X = val_X_all[idx].to(device)
    check_Y = val_Y_all[idx]          # normalised targets (CPU)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_norm = model(check_X).cpu()

    # De-normalise back to original physical units
    y_mean_t = torch.from_numpy(y_mean).float()
    y_std_t  = torch.from_numpy(y_std).float()
    pred_orig   = pred_norm * y_std_t + y_mean_t
    target_orig = check_Y  * y_std_t + y_mean_t

    mae = (pred_orig - target_orig).abs().mean(dim=0).numpy()

    units = ["dB", "mV_rms", "mV"]
    print(f"\n  {'Output':<26}  {'MAE':>10}  Unit")
    print(f"  {'-'*26}  {'-'*10}  {'-'*8}")
    for col, err, unit in zip(OUTPUT_COLS, mae, units):
        print(f"  {col:<26}  {err:>10.4f}  {unit}")
    print()


# ===========================================================================
# Public API
# ===========================================================================

def load_surrogate(path) -> Tuple["SurrogateMLP", Dict[str, Any]]:
    """
    Load a trained surrogate model from *path*.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pt`` file produced by :func:`train_surrogate`.

    Returns
    -------
    model : SurrogateMLP
        Model in eval mode with best weights loaded (on CPU).
    norm_stats : dict
        Dictionary with keys:

        - ``"input_mean"``   tensor (2,)  - min of each input (for min-max)
        - ``"input_std"``    tensor (2,)  - range of each input
        - ``"input_min"``    tensor (2,)
        - ``"input_max"``    tensor (2,)
        - ``"output_mean"``  tensor (3,)  - mean of each output (z-score)
        - ``"output_std"``   tensor (3,)  - std of each output
        - ``"input_cols"``   list[str]
        - ``"output_cols"``  list[str]

    Example
    -------
    >>> model, stats = load_surrogate("models/surrogate.pt")
    >>> x_raw = torch.tensor([[200.0, 5e-12]])
    >>> x_norm = (x_raw - stats["input_mean"]) / stats["input_std"]
    >>> with torch.no_grad():
    ...     y_norm = model(x_norm)
    >>> y = y_norm * stats["output_std"] + stats["output_mean"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Surrogate checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    hidden_dim  = checkpoint.get("hidden_dim", 128)
    n_in        = len(checkpoint.get("input_cols",  INPUT_COLS))
    n_out       = len(checkpoint.get("output_cols", OUTPUT_COLS))

    model = SurrogateMLP(input_dim=n_in, hidden_dim=hidden_dim, output_dim=n_out)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    norm_keys = (
        "input_mean", "input_std",
        "input_min",  "input_max",
        "output_mean", "output_std",
        "input_cols",  "output_cols",
    )
    norm_stats: Dict[str, Any] = {k: checkpoint[k] for k in norm_keys if k in checkpoint}
    return model, norm_stats


# ===========================================================================
# Entry-point
# ===========================================================================

if __name__ == "__main__":
    train_surrogate(
        data_path  = DATA_PATH,
        model_path = MODEL_PATH,
        hidden_dim = 128,
        epochs     = 200,
        batch_size = 512,
        lr         = 1e-3,
        val_frac   = 0.20,
        seed       = 42,
    )
