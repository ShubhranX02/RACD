"""
batched_vec_env.py — Zero-IPC Batched Surrogate VecEnv

Eliminates the Windows SubprocVecEnv named-pipe bottleneck entirely by running
all n_envs in the MAIN PROCESS and batching their actions into a single
numpy forward pass per timestep.

Performance comparison (measured on RTX 4050 Windows system):
  SPICE sequential:       ~0.5 steps/sec
  SubprocVecEnv + numpy:  ~240 steps/sec  (IPC-limited, ~0.8ms pipe RTT)
  BatchedSurrogateVecEnv: ~22,500 steps/sec (no IPC, vectorized batch inference)

Key idea: numpy matrix multiply is O(N) for N samples but overhead is constant,
so batch(12) takes ~same wall time as batch(1). All 12 "workers" become free.

Usage in train_transistor.py:
    from batched_vec_env import BatchedSurrogateVecEnv
    env = BatchedSurrogateVecEnv(n_envs=64, ...)  # use more envs since they're free!
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Dict, List, Optional, Tuple, Union


# PVT corners: (corner, temp_C, vdd_V)
_PVT_CORNERS = [
    ('tt',  27,  1.8),
    ('ss',  125, 1.71),
    ('ff',  0,   1.89),
    ('sf',  27,  1.8),
    ('fs',  27,  1.8),
]

# Parameter bounds matching environment_transistor.py (widened for 3-12 dB coverage)
_PARAM_RANGES = [
    (1.0,    15.0),   # Wn_um
    (100.0,  3000.0), # Rs_ohm
    (0.5e-12, 5.0e-12), # Cs_farad
    (100.0,  1200.0), # Itail_half_ua
    (500.0,  5000.0), # RL_ohm
    (5000.0, 40000.0),# Rdfe_ohm
]
_PARAM_LO = np.array([lo for lo, _ in _PARAM_RANGES], dtype=np.float32)
_PARAM_HI = np.array([hi for _, hi in _PARAM_RANGES], dtype=np.float32)
_PARAM_SPAN = _PARAM_HI - _PARAM_LO

# Limits matching TransistorEqualizerEnv defaults
_NOISE_LIMIT_MV   = 1.5
_POWER_LIMIT_MW   = 15.0
_HD3_LIMIT_DB     = -30.0
_EYE_HEIGHT_LIMIT = 100.0
_EYE_WIDTH_LIMIT  = 0.4    # 0.4 UI horizontal opening spec
_AREA_LIMIT_MM2   = 0.05
_ABS_SPAN_DB      = 9.0    # full spec range 3-12 dB


def _estimate_area(Wn, Rs, Cs, RL, Rdfe):
    """Vectorized area estimate (N,) arrays matching environment_transistor.py."""
    a_diff = 4 * (Wn * 0.15)
    a_res = (4 * RL + 2 * Rs + 2 * Rdfe + 100000) / 150.0
    a_cap = (2 * Cs + 10e-12) * 1e15 / 2.0
    total_um2 = (a_diff + a_res + a_cap) * 2.5
    return total_um2 / 1e6


class BatchedSurrogateVecEnv(VecEnv):
    """
    Zero-IPC VecEnv: all n_envs run in the main process, surrogate called once
    per timestep for the entire batch using vectorized numpy inference.

    With n_envs=64 and numpy surrogate, expect ~10,000-25,000 steps/sec on Windows.
    (vs ~240 steps/sec with SubprocVecEnv due to pipe RTT)
    """

    def __init__(
        self,
        n_envs: int = 64,
        peaking_target_range: Tuple[float, float] = (3.0, 12.0),
        topology: str = '1stage',
        multi_corner: bool = True,
        noise_limit_mvrms: float = _NOISE_LIMIT_MV,
        power_limit_mw: float = _POWER_LIMIT_MW,
        eye_height_limit_mv: float = _EYE_HEIGHT_LIMIT,
        seed: Optional[int] = None,
    ):
        self.n_envs = n_envs
        self.peaking_target_range = peaking_target_range
        self.multi_corner = multi_corner
        self.noise_limit = noise_limit_mvrms
        self.power_limit = power_limit_mw
        self.eye_height_limit = eye_height_limit_mv

        # Load surrogate (numpy path) into this process
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from surrogate_transistor import (
            load_transistor_surrogate, _NP_WEIGHTS, _NP_BN,
            _NP_IN_MEAN, _NP_IN_STD, _NP_OUT_MEAN, _NP_OUT_STD,
        )
        if not _NP_WEIGHTS:
            load_transistor_surrogate()
        # Cache module-level numpy arrays for fast access
        import surrogate_transistor as _st
        self._st = _st

        # Corner one-hot index map matching environment_transistor.py
        self._CORNER_IDX = {'tt': 0, 'ss': 1, 'ff': 2, 'sf': 3, 'fs': 4}
        self._corner_strings = ['tt', 'ss', 'ff', 'sf', 'fs']

        # Per-env state
        rng = np.random.default_rng(seed)
        self._rng = rng
        lo, hi = peaking_target_range
        self._targets = rng.uniform(lo, hi, size=n_envs).astype(np.float32)
        self._corner_idx = rng.integers(0, len(_PVT_CORNERS), size=n_envs) if multi_corner \
                           else np.zeros(n_envs, dtype=int)

        # SB3 VecEnv interface requirements
        # 9-D obs: [target_norm, noise_norm, power_norm, eye_norm, corner_one_hot(5)]
        obs_space = spaces.Box(low=np.zeros(9, dtype=np.float32),
                               high=np.ones(9, dtype=np.float32), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        super().__init__(n_envs, obs_space, act_space)

        self._pending_actions: Optional[np.ndarray] = None
        self._obs = self._build_obs_batch()

    def _build_obs_batch(self) -> np.ndarray:
        """Build (n_envs, 9) observation matrix for current episode state."""
        n = self.n_envs
        # Scalar features — normalised
        obs_scalar = np.stack([
            self._targets / 12.0,
            np.full(n, self.noise_limit / 1.5, dtype=np.float32),
            np.full(n, self.power_limit / 15.0, dtype=np.float32),
            np.full(n, self.eye_height_limit / 200.0, dtype=np.float32),
        ], axis=1).astype(np.float32)   # (n, 4)

        # Corner one-hot (n, 5)
        corner_oh = np.zeros((n, 5), dtype=np.float32)
        for i, cidx in enumerate(self._corner_idx):
            corner_oh[i, int(cidx)] = 1.0

        return np.concatenate([obs_scalar, corner_oh], axis=1)  # (n, 9)

    # ------------------------------------------------------------------
    # Core VecEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        lo, hi = self.peaking_target_range
        self._targets = self._rng.uniform(lo, hi, size=self.n_envs).astype(np.float32)
        if self.multi_corner:
            self._corner_idx = self._rng.integers(0, len(_PVT_CORNERS), size=self.n_envs)
        self._obs = self._build_obs_batch()
        return self._obs

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = actions  # (n_envs, 6) in [-1, 1]

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        actions = self._pending_actions
        st = self._st

        # Rescale actions from [-1,1] to physical ranges — vectorized
        params = _PARAM_LO + (actions + 1.0) / 2.0 * _PARAM_SPAN  # (N, 6)

        # ----- SINGLE BATCH NUMPY FORWARD PASS (the key optimization) -----
        if st._NP_WEIGHTS is not None:
            x_norm = (params - st._NP_IN_MEAN) / st._NP_IN_STD   # (N, 6)
            y_norm = st._numpy_forward(x_norm)                      # (N, 6) with new surrogate
            y = y_norm * st._NP_OUT_STD + st._NP_OUT_MEAN          # (N, 6) denorm
        else:
            # Torch fallback (slower)
            import torch
            with torch.no_grad():
                x_t = torch.from_numpy(params)
                x_norm_t = (x_t - st._SURROGATE_NORM['input_mean']) / (st._SURROGATE_NORM['input_std'] + 1e-8)
                y_t = st._SURROGATE_MODEL(x_norm_t.float())
                y_t = y_t * (st._SURROGATE_NORM['output_std'] + 1e-8) + st._SURROGATE_NORM['output_mean']
            y = y_t.numpy()

        # Unpack outputs (N, 6): [peaking, noise, power, eye_height, hd3, eye_width_ui]
        peaking   = y[:, 0]
        noise     = np.maximum(0.0, y[:, 1])
        power     = np.maximum(0.0, y[:, 2])
        eye       = np.maximum(0.0, y[:, 3])
        hd3       = y[:, 4]
        eye_w     = np.maximum(0.0, y[:, 5]) if y.shape[1] > 5 else np.zeros(len(y))

        # Vectorized reward — quadratic peaking × 3 weight + constraint penalties
        err_db = np.abs(peaking - self._targets)
        peaking_penalty = 3.0 * (err_db / _ABS_SPAN_DB) ** 2

        noise_pen   = np.maximum(0.0, (noise - self.noise_limit) / self.noise_limit)
        power_pen   = np.maximum(0.0, (power - self.power_limit) / self.power_limit)
        hd3_pen     = np.maximum(0.0, (hd3 - _HD3_LIMIT_DB) / abs(_HD3_LIMIT_DB))
        eye_h_pen   = np.maximum(0.0, (self.eye_height_limit - eye) / self.eye_height_limit)
        eye_w_pen   = np.where(eye_w > 0,
                               np.maximum(0.0, (_EYE_WIDTH_LIMIT - eye_w) / _EYE_WIDTH_LIMIT),
                               0.0)

        Wn, Rs, Cs, _, RL, Rdfe = params.T
        area = _estimate_area(Wn, Rs, Cs, RL, Rdfe)
        area_pen = np.maximum(0.0, (area - _AREA_LIMIT_MM2) / _AREA_LIMIT_MM2)

        rewards = -(peaking_penalty + noise_pen + power_pen + hd3_pen + eye_h_pen + eye_w_pen + area_pen)

        # Each step is episodic (single-step MDP) — always done after 1 step
        dones = np.ones(self.n_envs, dtype=bool)

        # Reset targets and corners for next episode
        lo, hi = self.peaking_target_range
        self._targets = self._rng.uniform(lo, hi, size=self.n_envs).astype(np.float32)
        if self.multi_corner:
            self._corner_idx = self._rng.integers(0, len(_PVT_CORNERS), size=self.n_envs)
        self._obs = self._build_obs_batch()

        infos = [
            {
                'peaking_db':          float(peaking[i]),
                'noise_mvrms':         float(noise[i]),
                'power_mw':            float(power[i]),
                'eye_height_proxy_mv': float(eye[i]),
                'eye_width_ui':        float(eye_w[i]),
                'target_peaking_db':   float(self._targets[i]),
                '_source':             'surrogate_batch',
            }
            for i in range(self.n_envs)
        ]

        return self._obs, rewards.astype(np.float32), dones, infos

    # ------------------------------------------------------------------
    # Required VecEnv abstract methods
    # ------------------------------------------------------------------

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.n_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * (self.n_envs if indices is None else len(indices))

    def get_attr(self, attr_name, indices=None):
        return [None] * (self.n_envs if indices is None else len(indices))

    def set_attr(self, attr_name, value, indices=None):
        pass

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed] * self.n_envs
