import numpy as np
import gymnasium as gym
from gymnasium import spaces
from circuit_transistor import simulate_transistor_fast as simulate_transistor_level

class TransistorEqualizerEnv(gym.Env):
    def __init__(self,
                 peaking_target_range=(3.0, 11.0),
                 noise_limit_mvrms=1.5,
                 hd3_limit_db=-30.0,
                 power_limit_mw=15.0,
                 eye_height_limit_mv=100.0,
                 w_range=(1.0, 20.0),
                 rs_range=(50.0, 1000.0),
                 cs_range=(0.1e-12, 5e-12),
                 itail_range=(100.0, 2000.0),
                 rl_range=(500.0, 5000.0),
                 rdfe_range=(1000.0, 50000.0)):
        super().__init__()
        self.peaking_target_range = peaking_target_range
        self.noise_limit_mvrms = noise_limit_mvrms
        self.hd3_limit_db = hd3_limit_db
        self.power_limit_mw = power_limit_mw
        self.eye_height_limit_mv = eye_height_limit_mv
        
        self.ranges = [w_range, rs_range, cs_range, itail_range, rl_range, rdfe_range]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.ranges),), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32)
        self.target_peaking_db = None

    def _rescale_action(self, action):
        rescaled = []
        for a, (low, high) in zip(action, self.ranges):
            rescaled.append(low + (a + 1) / 2 * (high - low))
        return rescaled

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low, high = self.peaking_target_range
        self.target_peaking_db = float(self.np_random.uniform(low, high))
        obs = np.array([self.target_peaking_db], dtype=np.float32)
        return obs, {}

    def step(self, action):
        Wn, Rs, Cs, Itail, RL, Rdfe = self._rescale_action(action)
        try:
            result = simulate_transistor_level(Wn, Rs, Cs, Itail, RL, Rdfe, corner='tt', temp=27, vdd=1.8)
        except Exception as e:
            # Failed simulation (e.g. non-convergent) gets large penalty
            return np.array([self.target_peaking_db], dtype=np.float32), -10.0, True, False, {}

        peaking_low, peaking_high = self.peaking_target_range
        peaking_span = max(1.0, peaking_high - peaking_low)
        peaking_error_norm = abs(result['peaking_db'] - self.target_peaking_db) / peaking_span

        noise_penalty = max(0.0, (result['noise_mvrms'] - self.noise_limit_mvrms) / self.noise_limit_mvrms)
        hd3_penalty = max(0.0, (result['hd3_db'] - self.hd3_limit_db) / abs(self.hd3_limit_db))
        power_penalty = max(0.0, (result['power_mw'] - self.power_limit_mw) / self.power_limit_mw)
        eye_penalty = max(0.0, (self.eye_height_limit_mv - result['eye_height_proxy_mv']) / self.eye_height_limit_mv)

        reward = -(peaking_error_norm + noise_penalty + hd3_penalty + power_penalty + eye_penalty)

        obs = np.array([self.target_peaking_db], dtype=np.float32)
        info = {
            'Wn': float(Wn), 'Rs': float(Rs), 'Cs': float(Cs), 
            'Itail': float(Itail), 'RL': float(RL), 'Rdfe': float(Rdfe),
            'target_peaking_db': float(self.target_peaking_db),
            'peaking_db': float(result['peaking_db']),
            'noise_mvrms': float(result['noise_mvrms']),
            'hd3_db': float(result['hd3_db']),
            'power_mw': float(result['power_mw']),
            'eye_height_proxy_mv': float(result['eye_height_proxy_mv'])
        }
        return obs, reward, True, False, info
