"""
environment_fast.py — Drop-in faster EqualizerEnv using circuit_fast.simulate().

Identical reward structure and observation space as environment.py,
but uses circuit_fast.simulate() instead of circuit.simulate().
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from circuit_fast import simulate


class EqualizerEnvFast(gym.Env):
    """
    Fast Gymnasium environment wrapping circuit_fast.simulate().
    API-compatible with EqualizerEnv — same action/obs spaces and reward.
    """

    def __init__(self,
                 peaking_target_range=(3.0, 11.0),
                 noise_limit_mvrms=1.5,
                 eye_height_limit_mv=100.0,
                 rs_range=(50, 500),
                 cs_range=(0.1e-12, 10e-12),
                 peaking_weight=1.0,
                 noise_weight=1.0,
                 eye_weight=1.0):
        super().__init__()
        self.peaking_target_range = peaking_target_range
        self.noise_limit_mvrms = noise_limit_mvrms
        self.eye_height_limit_mv = eye_height_limit_mv
        self.rs_range = rs_range
        self.cs_range = cs_range
        self.peaking_weight = peaking_weight
        self.noise_weight = noise_weight
        self.eye_weight = eye_weight
        self.target_peaking_db = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32)

    def _rescale_action(self, action):
        rs_low, rs_high = self.rs_range
        cs_low, cs_high = self.cs_range
        Rs = rs_low + (action[0] + 1) / 2 * (rs_high - rs_low)
        Cs = cs_low + (action[1] + 1) / 2 * (cs_high - cs_low)
        return Rs, Cs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low, high = self.peaking_target_range
        self.target_peaking_db = float(self.np_random.uniform(low, high))
        obs = np.array([self.target_peaking_db], dtype=np.float32)
        return obs, {}

    def step(self, action):
        Rs, Cs = self._rescale_action(action)
        result = simulate(Rs, Cs)

        peaking_low, peaking_high = self.peaking_target_range
        peaking_span = peaking_high - peaking_low
        if peaking_span == 0:
            peaking_span = 1.0

        peaking_error_norm = abs(result['peaking_db'] - self.target_peaking_db) / peaking_span
        noise_penalty_norm = max(
            0.0, (result['noise_mvrms'] - self.noise_limit_mvrms) / self.noise_limit_mvrms
        )
        eye_penalty_norm = max(
            0.0, (self.eye_height_limit_mv - result['eye_height_proxy_mv']) / self.eye_height_limit_mv
        )

        reward = -(
            self.peaking_weight * peaking_error_norm +
            self.noise_weight * noise_penalty_norm +
            self.eye_weight * eye_penalty_norm
        )

        obs = np.array([self.target_peaking_db], dtype=np.float32)
        info = {
            'Rs': float(Rs), 'Cs': float(Cs),
            'target_peaking_db': float(self.target_peaking_db),
            'peaking_db': float(result['peaking_db']),
            'noise_mvrms': float(result['noise_mvrms']),
            'eye_height_proxy_mv': float(result['eye_height_proxy_mv']),
            'pulse_width_proxy_ps': float(result['pulse_width_proxy_ps']),
        }
        return obs, reward, True, False, info


if __name__ == '__main__':
    import time
    env = EqualizerEnvFast()
    obs, _ = env.reset()
    action = env.action_space.sample()

    t0 = time.time()
    N = 1000
    for _ in range(N):
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    elapsed = time.time() - t0
    print(f"{N} steps in {elapsed:.3f}s = {N/elapsed:.0f} steps/sec")
    print("Sample info:", {k: f"{v:.4f}" if isinstance(v, float) else v
                           for k, v in info.items() if k not in ('Cs',)})
