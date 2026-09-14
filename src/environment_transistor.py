"""
environment_transistor.py — Gymnasium Environment for Active Transistor-Level CTLE + DFE Equalizer Sizing.

Optimized for:
- Cascaded 2-Stage CTLE + 1-Tap DFE architecture
- SkyWater 130 nm PDK BSIM4 models
- Surrogate MLP simulator (~0.01 ms/call) or full SPICE (~2-4 s/call)
- Multi-corner PVT-aware training: corner randomized each episode for robust policies
- Dense physics-informed reward balancing Peaking, Linearity (HD3), Integrated Noise, Power, and Eye Height
"""

import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Surrogate auto-detection (lazy-load per process — safe for SubprocVecEnv)
# ---------------------------------------------------------------------------

_SURROGATE_LOADED = False
_SURROGATE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'surrogate_transistor.pt')


def _try_load_surrogate():
    global _SURROGATE_LOADED
    if _SURROGATE_LOADED:
        return True
    try:
        from surrogate_transistor import load_transistor_surrogate
        if os.path.exists(_SURROGATE_PATH):
            ok = load_transistor_surrogate(_SURROGATE_PATH)
            _SURROGATE_LOADED = ok
            return ok
    except ImportError:
        pass
    return False


# ---------------------------------------------------------------------------
# PVT corners for multi-corner training
# ---------------------------------------------------------------------------

_PVT_CORNERS = [
    ('tt',  27,  1.8),    # typical-typical
    ('ss',  125, 1.71),   # slow-slow, hot, low-voltage
    ('ff',  0,   1.89),   # fast-fast, cold, high-voltage
    ('sf',  27,  1.8),    # slow-nfet, fast-pfet
    ('fs',  27,  1.8),    # fast-nfet, slow-pfet
]

# ---------------------------------------------------------------------------
# Transistor repository logging (for DAgger refinement of surrogate)
# ---------------------------------------------------------------------------

_TRANSISTOR_REPO_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'transistor_repository.jsonl'
)


def _append_transistor_record(params: dict, result: dict):
    """Append one validated SPICE result to transistor_repository.jsonl."""
    try:
        os.makedirs(os.path.dirname(_TRANSISTOR_REPO_PATH), exist_ok=True)
        record = {**params, **result}
        with open(_TRANSISTOR_REPO_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Area estimation
# ---------------------------------------------------------------------------

def estimate_area_2stage_mm2(Wn, Rs, Cs, RL, Rdfe):
    """Estimate total layout area in mm2 for 2-stage CTLE."""
    a_diff = 4 * (Wn * 0.15)  # 4 differential input transistors
    a_res = (4 * RL + 2 * Rs + 2 * Rdfe + 100000) / 150.0  # poly resistors
    a_cap = (2 * Cs + 10e-12) * 1e15 / 2.0  # MOM capacitors
    total_um2 = (a_diff + a_res + a_cap) * 2.5  # 2.5x routing overhead
    return total_um2 / 1e6


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class TransistorEqualizerEnv(gym.Env):
    """
    Continuous control environment for transistor sizing.

    State (9-D):
        [target_peaking_norm,   # target peaking / 12.0
         noise_limit_norm,      # noise_limit / 1.5
         power_limit_norm,      # power_limit / 15.0
         eye_limit_norm,        # eye_limit / 200.0
         corner_tt, corner_ss, corner_ff, corner_sf, corner_fs,  # one-hot (5)
        ]
        Total: 4 + 5 = 9 dimensions.
        The 9-D observation lets the agent learn PVT-adaptive sizing policies.

    Action: 6 continuous dims normalized to [-1, 1]:
            [Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm]

    Simulation backends (in priority order):
        1. Surrogate MLP  — ~0.01 ms/call (if models/surrogate_transistor.pt exists)
        2. Full ngspice   — ~2-4 s/call   (always available, used for DAgger refinement)
    """

    # Index of each corner in the one-hot encoding
    _CORNER_IDX = {'tt': 0, 'ss': 1, 'ff': 2, 'sf': 3, 'fs': 4}
    OBS_DIM = 9

    def __init__(self,
                 peaking_target_range=(3.0, 12.0),
                 noise_limit_mvrms=1.5,
                 hd3_limit_db=-30.0,
                 power_limit_mw=15.0,
                 eye_height_limit_mv=100.0,
                 eye_width_limit_ui=0.4,
                 area_limit_mm2=0.05,
                 w_range=(1.0, 15.0),
                 rs_range=(100.0, 3000.0),
                 cs_range=(0.5e-12, 5.0e-12),
                 itail_range=(100.0, 1200.0),
                 rl_range=(500.0, 5000.0),
                 rdfe_range=(5000.0, 40000.0),
                 topology='2stage',
                 use_surrogate=None,       # None = auto-detect from file presence
                 multi_corner=True,        # randomize PVT corner each episode
                 spice_validate_every=0):  # run real SPICE every N steps for DAgger; 0=disabled
        super().__init__()
        self.peaking_target_range   = peaking_target_range
        self.noise_limit_mvrms      = noise_limit_mvrms
        self.hd3_limit_db           = hd3_limit_db
        self.power_limit_mw         = power_limit_mw
        self.eye_height_limit_mv    = eye_height_limit_mv
        self.eye_width_limit_ui     = eye_width_limit_ui
        self.area_limit_mm2         = area_limit_mm2
        self.topology               = topology
        self.multi_corner           = multi_corner
        self.spice_validate_every   = spice_validate_every
        self._step_count            = 0
        self.current_corner         = _PVT_CORNERS[0]  # (corner_str, temp, vdd)

        self.ranges = [w_range, rs_range, cs_range, itail_range, rl_range, rdfe_range]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.ranges),), dtype=np.float32)

        # 9-D observation: [target_peaking_norm, noise_norm, power_norm, eye_norm,
        #                   corner_tt, corner_ss, corner_ff, corner_sf, corner_fs]
        self.observation_space = spaces.Box(
            low=np.zeros(self.OBS_DIM, dtype=np.float32),
            high=np.ones(self.OBS_DIM, dtype=np.float32),
            dtype=np.float32
        )
        self.target_peaking_db = None

        # Resolve surrogate availability (lazy — each subprocess loads its own copy)
        if use_surrogate is None:
            self.use_surrogate = _try_load_surrogate()
        else:
            self.use_surrogate = bool(use_surrogate)
            if self.use_surrogate:
                _try_load_surrogate()

    def _build_obs(self) -> np.ndarray:
        """Build the 9-D normalised observation vector for the current episode state."""
        corner_str, temp, vdd = self.current_corner
        corner_one_hot = np.zeros(5, dtype=np.float32)
        corner_one_hot[self._CORNER_IDX.get(corner_str, 0)] = 1.0

        obs = np.array([
            float(self.target_peaking_db) / 12.0,          # normalised [0,1] over 0-12 dB
            self.noise_limit_mvrms / 1.5,                  # normalised (1.0 = spec limit)
            self.power_limit_mw / 15.0,                    # normalised
            self.eye_height_limit_mv / 200.0,              # normalised
        ], dtype=np.float32)
        return np.concatenate([obs, corner_one_hot])

    def _rescale_action(self, action):
        return [lo + (a + 1) / 2 * (hi - lo)
                for a, (lo, hi) in zip(action, self.ranges)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low, high = self.peaking_target_range
        self.target_peaking_db = float(self.np_random.uniform(low, high))

        if self.multi_corner:
            idx = int(self.np_random.integers(0, len(_PVT_CORNERS)))
            self.current_corner = _PVT_CORNERS[idx]
        else:
            self.current_corner = _PVT_CORNERS[0]  # TT always

        return self._build_obs(), {}

    def step(self, action):
        Wn, Rs, Cs, Itail, RL, Rdfe = self._rescale_action(action)
        self._step_count += 1
        corner_str, temp, vdd = self.current_corner

        try:
            if self.use_surrogate:
                from surrogate_transistor import simulate_transistor_surrogate
                result = simulate_transistor_surrogate(Wn, Rs, Cs, Itail, RL, Rdfe)

                # DAgger: periodically validate with real SPICE and save to transistor repo
                if self.spice_validate_every > 0 and self._step_count % self.spice_validate_every == 0:
                    from circuit_transistor import simulate_transistor_fast
                    spice_result = simulate_transistor_fast(
                        Wn, Rs, Cs, Itail, RL, Rdfe,
                        corner=corner_str, temp=temp, vdd=vdd,
                        topology=self.topology, compute_hd3=True,
                    )
                    _append_transistor_record(
                        {'Wn_um': Wn, 'Rs_ohm': Rs, 'Cs_farad': Cs,
                         'Itail_half_ua': Itail, 'RL_ohm': RL, 'Rdfe_ohm': Rdfe,
                         'corner': corner_str, 'temp': temp, 'vdd': vdd},
                        spice_result
                    )
            else:
                from circuit_transistor import simulate_transistor_fast
                result = simulate_transistor_fast(
                    Wn, Rs, Cs, Itail, RL, Rdfe,
                    corner=corner_str, temp=temp, vdd=vdd,
                    topology=self.topology,
                )
        except Exception:
            return self._build_obs(), -10.0, True, False, {}

        # --- Reward computation ---
        # Peaking: quadratic penalty weighted 3× for sharp gradient near target.
        # Use full spec range (3-12 dB = 9 dB) for normalization.
        _ABS_SPAN = 9.0  # full spec range width in dB
        peaking_error_db   = abs(result['peaking_db'] - self.target_peaking_db)
        peaking_error_norm = (peaking_error_db / _ABS_SPAN) ** 2

        noise_penalty  = max(0.0, (result['noise_mvrms'] - self.noise_limit_mvrms) / self.noise_limit_mvrms)
        hd3_penalty    = max(0.0, (result['hd3_db'] - self.hd3_limit_db) / abs(self.hd3_limit_db))
        power_penalty  = max(0.0, (result['power_mw'] - self.power_limit_mw) / self.power_limit_mw)
        eye_h_penalty  = max(0.0, (self.eye_height_limit_mv - result['eye_height_proxy_mv']) / self.eye_height_limit_mv)

        # Eye width penalty — spec: > 0.4 UI; only penalise when below limit
        eye_w = result.get('eye_width_ui', 0.0)
        eye_w_penalty = max(0.0, (self.eye_width_limit_ui - eye_w) / self.eye_width_limit_ui) if eye_w > 0 else 0.0

        area = estimate_area_2stage_mm2(Wn, Rs, Cs, RL, Rdfe)
        area_penalty = max(0.0, (area - self.area_limit_mm2) / self.area_limit_mm2)

        # Peaking accuracy weighted 3× over constraint penalties
        reward = -(3.0 * peaking_error_norm
                   + noise_penalty + hd3_penalty + power_penalty
                   + eye_h_penalty + eye_w_penalty + area_penalty)

        obs  = self._build_obs()
        info = {
            'Wn': float(Wn), 'Rs': float(Rs), 'Cs': float(Cs),
            'Itail': float(Itail), 'RL': float(RL), 'Rdfe': float(Rdfe),
            'target_peaking_db':   float(self.target_peaking_db),
            'peaking_db':          float(result['peaking_db']),
            'noise_mvrms':         float(result['noise_mvrms']),
            'hd3_db':              float(result.get('hd3_db', -38.0)),
            'power_mw':            float(result.get('power_mw', 0.0)),
            'eye_height_proxy_mv': float(result['eye_height_proxy_mv']),
            'eye_width_ui':        float(result.get('eye_width_ui', 0.0)),
            'area_mm2':            float(area),
            'topology':            self.topology,
            'corner':              corner_str,
            '_source':             result.get('_source', 'spice'),
        }
        return obs, reward, True, False, info
