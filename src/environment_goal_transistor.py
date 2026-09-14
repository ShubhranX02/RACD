"""
environment_goal_transistor.py — GoalEnv wrapper for transistor-level CTLE+DFE sizing.

Uses the Gymnasium GoalEnv interface compatible with Stable-Baselines3 SAC+HER.
HER (Hindsight Experience Replay) relabels failed episodes by treating the actual
achieved metrics as if they were the desired goal — dramatically improving sample
efficiency in the sparse multi-spec analog design problem.

State (GoalEnv Dict):
    observation:    [Wn_norm, Rs_norm, Cs_norm, Itail_norm, RL_norm, Rdfe_norm]  — 6D (current params, normalised)
    desired_goal:   [target_peaking_norm, noise_limit_norm, power_limit_norm, eye_h_norm]  — 4D
    achieved_goal:  [peaking_norm, noise_norm, power_norm, eye_h_norm]  — 4D (from last simulation)

Action: 6D continuous in [-1, 1]  →  [Wn, Rs, Cs, Itail, RL, Rdfe]

Reward (static method for HER relabelling):
    -1.0 if any spec is violated (binary), 0.0 if all satisfied.
    Tolerances: peaking ±1.5 dB, noise/power/eye within limit.

Simulation backend: surrogate (fast) or SPICE (ground truth).

Usage with SAC+HER:
    from stable_baselines3 import SAC
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
    from environment_goal_transistor import GoalTransistorEnv

    env = GoalTransistorEnv()
    model = SAC(
        'MultiInputPolicy', env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={'n_sampled_goal': 4, 'goal_selection_strategy': 'future'},
        learning_rate=3e-4, buffer_size=200000, batch_size=512,
    )
    model.learn(total_timesteps=500_000)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PVT corners
_PVT_CORNERS = [
    ('tt', 27,  1.8),
    ('ss', 125, 1.71),
    ('ff', 0,   1.89),
    ('sf', 27,  1.8),
    ('fs', 27,  1.8),
]

# Parameter ranges — must match surrogate_transistor.PARAM_BOUNDS
_PARAM_RANGES = [
    (1.0,    15.0),     # Wn_um
    (100.0,  3000.0),   # Rs_ohm
    (0.5e-12, 5.0e-12), # Cs_farad
    (100.0,  1200.0),   # Itail_half_ua
    (500.0,  5000.0),   # RL_ohm
    (5000.0, 40000.0),  # Rdfe_ohm
]
_PARAM_LO = np.array([lo for lo, _ in _PARAM_RANGES], dtype=np.float64)
_PARAM_HI = np.array([hi for _, hi in _PARAM_RANGES], dtype=np.float64)
_PARAM_SPAN = _PARAM_HI - _PARAM_LO


def _try_load_surrogate() -> bool:
    try:
        from surrogate_transistor import load_transistor_surrogate
        return load_transistor_surrogate()
    except Exception:
        return False


class GoalTransistorEnv(gym.Env):
    """
    GoalEnv (HER-compatible) for transistor-level CTLE+DFE sizing.

    Achieved goal = [peaking_norm, noise_norm, power_norm, eye_h_norm]
    Desired goal  = [target_peaking_norm, noise_limit_norm, power_limit_norm, eye_h_limit_norm]

    Binary sparse reward: 0.0 if all specs met, -1.0 otherwise.
    HER relabels failed episodes using achieved goals as new desired goals.

    Spec tolerances (compute_reward):
        peaking: |achieved - desired| ≤ 0.125 (= 1.5 dB / 12 dB)
        noise:   achieved ≤ desired
        power:   achieved ≤ desired
        eye_h:   achieved ≥ desired (inverted — higher eye is better)
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        peaking_target_range: Tuple[float, float] = (3.0, 12.0),
        noise_limit_mvrms: float = 1.5,
        power_limit_mw: float = 15.0,
        eye_height_limit_mv: float = 100.0,
        topology: str = '2stage',
        use_surrogate: Optional[bool] = None,
        multi_corner: bool = True,
    ):
        super().__init__()
        self.peaking_target_range  = peaking_target_range
        self.noise_limit_mvrms     = noise_limit_mvrms
        self.power_limit_mw        = power_limit_mw
        self.eye_height_limit_mv   = eye_height_limit_mv
        self.topology              = topology
        self.multi_corner          = multi_corner
        self.current_corner        = _PVT_CORNERS[0]
        self._last_achieved        = np.zeros(4, dtype=np.float32)
        self._current_params_norm  = np.zeros(6, dtype=np.float32)

        # Normalisation constants (for goal space)
        self._peak_scale   = 12.0
        self._noise_scale  = 1.5
        self._power_scale  = 15.0
        self._eye_h_scale  = 200.0

        # Spaces
        # observation: normalised current sizing params
        obs_low  = np.zeros(6, dtype=np.float32)
        obs_high = np.ones(6, dtype=np.float32)

        # goal: [peaking_norm, noise_norm, power_norm, eye_h_norm]
        goal_low  = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        goal_high = np.array([1.5, 5.0, 5.0, 10.0], dtype=np.float32)

        self.observation_space = spaces.Dict({
            'observation':   spaces.Box(low=obs_low,  high=obs_high,  dtype=np.float32),
            'desired_goal':  spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            'achieved_goal': spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Target (randomised each reset)
        self._target_peaking_db = 6.0
        self._desired_goal_vec  = self._make_desired_goal(self._target_peaking_db)

        if use_surrogate is None:
            self.use_surrogate = _try_load_surrogate()
        else:
            self.use_surrogate = bool(use_surrogate)
            if self.use_surrogate:
                _try_load_surrogate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale action from [-1, 1] to physical parameter values."""
        return _PARAM_LO + (action + 1.0) / 2.0 * _PARAM_SPAN

    def _params_to_norm(self, params: np.ndarray) -> np.ndarray:
        """Normalise physical params to [0, 1] for observation."""
        return ((params - _PARAM_LO) / _PARAM_SPAN).astype(np.float32)

    def _make_desired_goal(self, target_peaking_db: float) -> np.ndarray:
        return np.array([
            target_peaking_db  / self._peak_scale,
            self.noise_limit_mvrms   / self._noise_scale,
            self.power_limit_mw      / self._power_scale,
            self.eye_height_limit_mv / self._eye_h_scale,
        ], dtype=np.float32)

    def _make_achieved_goal(self, result: dict) -> np.ndarray:
        return np.array([
            float(result.get('peaking_db',          0.0)) / self._peak_scale,
            float(result.get('noise_mvrms',          0.0)) / self._noise_scale,
            float(result.get('power_mw',             0.0)) / self._power_scale,
            float(result.get('eye_height_proxy_mv',  0.0)) / self._eye_h_scale,
        ], dtype=np.float32)

    def _simulate(self, Wn, Rs, Cs, Itail, RL, Rdfe) -> dict:
        corner_str, temp, vdd = self.current_corner
        if self.use_surrogate:
            from surrogate_transistor import simulate_transistor_surrogate
            return simulate_transistor_surrogate(Wn, Rs, Cs, Itail, RL, Rdfe)
        else:
            from circuit_transistor import simulate_transistor_fast
            return simulate_transistor_fast(
                Wn, Rs, Cs, Itail, RL, Rdfe,
                corner=corner_str, temp=temp, vdd=vdd,
                topology=self.topology,
            )

    # ------------------------------------------------------------------
    # GoalEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        lo, hi = self.peaking_target_range
        self._target_peaking_db = float(self.np_random.uniform(lo, hi))
        self._desired_goal_vec  = self._make_desired_goal(self._target_peaking_db)

        if self.multi_corner:
            idx = int(self.np_random.integers(0, len(_PVT_CORNERS)))
            self.current_corner = _PVT_CORNERS[idx]
        else:
            self.current_corner = _PVT_CORNERS[0]

        # Start from a neutral (mid-range) parameter point
        self._current_params_norm = np.full(6, 0.5, dtype=np.float32)
        self._last_achieved = np.zeros(4, dtype=np.float32)

        obs = {
            'observation':   self._current_params_norm.copy(),
            'desired_goal':  self._desired_goal_vec.copy(),
            'achieved_goal': self._last_achieved.copy(),
        }
        return obs, {}

    def step(self, action: np.ndarray):
        params = self._rescale_action(np.clip(action, -1.0, 1.0))
        Wn, Rs, Cs, Itail, RL, Rdfe = params

        try:
            result = self._simulate(Wn, Rs, Cs, Itail, RL, Rdfe)
        except Exception:
            result = {'peaking_db': 0.0, 'noise_mvrms': 99.0,
                      'power_mw': 99.0, 'eye_height_proxy_mv': 0.0}

        self._current_params_norm = self._params_to_norm(params)
        self._last_achieved       = self._make_achieved_goal(result)

        reward = float(self.compute_reward(
            self._last_achieved, self._desired_goal_vec, {}
        ))

        obs = {
            'observation':   self._current_params_norm.copy(),
            'desired_goal':  self._desired_goal_vec.copy(),
            'achieved_goal': self._last_achieved.copy(),
        }
        info = {
            'peaking_db':          float(result.get('peaking_db', 0)),
            'noise_mvrms':         float(result.get('noise_mvrms', 0)),
            'power_mw':            float(result.get('power_mw', 0)),
            'eye_height_proxy_mv': float(result.get('eye_height_proxy_mv', 0)),
            'target_peaking_db':   self._target_peaking_db,
            'is_success':          reward == 0.0,
        }
        return obs, reward, True, False, info

    @staticmethod
    def compute_reward(
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        **kwargs,
    ) -> np.ndarray:
        """
        Binary sparse reward for HER compatibility.
        Returns 0.0 if all specs satisfied, -1.0 otherwise.

        Tolerances (normalised space):
          peaking:  |achieved[0] - desired[0]| ≤ 0.125  (= 1.5 dB / 12 dB)
          noise:    achieved[1] ≤ desired[1]
          power:    achieved[2] ≤ desired[2]
          eye_h:    achieved[3] ≥ desired[3]            (higher is better)
        """
        ag = np.atleast_2d(achieved_goal)
        dg = np.atleast_2d(desired_goal)

        peaking_ok = np.abs(ag[:, 0] - dg[:, 0]) <= 0.125
        noise_ok   = ag[:, 1] <= dg[:, 1] * 1.05   # 5% margin
        power_ok   = ag[:, 2] <= dg[:, 2] * 1.05
        eye_h_ok   = ag[:, 3] >= dg[:, 3] * 0.95   # 5% margin

        all_ok = peaking_ok & noise_ok & power_ok & eye_h_ok
        reward = np.where(all_ok, 0.0, -1.0).astype(np.float32)
        if reward.shape[0] == 1:
            return float(reward[0])
        return reward

    def render(self):
        pass

    def close(self):
        pass
