import numpy as np
import gymnasium as gym
from gymnasium import spaces
from circuit import simulate


class EqualizerEnv(gym.Env):
    """
    Gymnasium environment wrapping the Stage A circuit simulator.
    One episode = one action = one full parameter proposal + one reward
    (single-step episodes, not incremental/multi-step adjustment).

    Reward combines three differently-shaped spec terms, each normalized
    to a comparable [0, ~1+] scale before weighting:
      - peaking: must hit an exact TARGET (distance-based penalty)
      - noise: must stay under a CEILING (penalty only if exceeded)
      - eye height: must stay over a FLOOR (penalty only if under)
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

        # Action: 2 values in [-1, 1] -> rescaled to real Rs/Cs in _rescale_action.
        # Using a small symmetric range instead of raw physical units is
        # standard practice -- neural networks train more reliably this way.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: the target spec itself (goal-conditioned RL -- the
        # agent must learn a policy that generalizes across the whole
        # target range, not memorize one fixed answer).
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
            peaking_span = 1.0  # avoid division by zero for single-target envs
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
        terminated = True
        truncated = False
        info = {
            'Rs': float(Rs), 'Cs': float(Cs),
            'target_peaking_db': float(self.target_peaking_db),
            'peaking_db': float(result['peaking_db']),
            'noise_mvrms': float(result['noise_mvrms']),
            'eye_height_proxy_mv': float(result['eye_height_proxy_mv']),
            'pulse_width_proxy_ps': float(result['pulse_width_proxy_ps']),
        }
        return obs, reward, terminated, truncated, info


if __name__ == '__main__':
    env = EqualizerEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("obs:", obs, "reward:", reward, "info:", info)
