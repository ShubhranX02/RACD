"""
environment_goal.py -- GoalEnv-compatible Gymnasium environment for SB3 HER.

Wraps circuit_fast.simulate() in the Dict observation-space contract required
by Stable-Baselines3's HindsightExperienceReplay (HER) wrapper:

    observation = {
        'observation':    Box(2,)   -- normalised [Rs_norm, Cs_norm] in [-1, 1]
        'desired_goal':   Box(3,)   -- [target_peaking_db, noise_limit_mvrms,
                                        eye_height_limit_mv]
        'achieved_goal':  Box(3,)   -- [peaking_db, noise_mvrms,
                                        eye_height_proxy_mv] from last sim
    }

HER requires compute_reward() to accept numpy arrays in batch form so that it
can relabel goals from the replay buffer.  The method is intentionally a
staticmethod (no self) so SB3 can call it on a freshly unpickled copy.
"""

import os
import numpy as np
import gymnasium
from gymnasium import spaces

# ---------------------------------------------------------------------------
# ngspice PATH fix (Windows standard install location)
# ---------------------------------------------------------------------------
if os.path.exists(r"C:\Spice64\bin") and r"C:\Spice64\bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = r"C:\Spice64\bin;" + os.environ.get("PATH", "")

from circuit_fast import simulate  # noqa: E402 -- import after PATH patch


# ---------------------------------------------------------------------------
# GoalEqualizerEnv
# ---------------------------------------------------------------------------

class GoalEqualizerEnv(gymnasium.Env):
    """
    Gymnasium GoalEnv wrapper for the passive RC equaliser circuit.

    Compatible with SB3's HER wrapper.  The desired goal is a randomly
    sampled peaking target (in dB); noise and eye-height limits are fixed
    per-episode but included in the goal vector so HER can relabel them.

    Action space
    ------------
    Box(low=-1, high=1, shape=(2,))
        action[0] -> Rs (rescaled to rs_range)
        action[1] -> Cs (rescaled to cs_range)

    Observation space (Dict)
    ------------------------
    'observation'   : normalised [Rs_norm, Cs_norm] in [-1, 1]
    'desired_goal'  : [target_peaking_db, noise_limit_mvrms, eye_height_limit_mv]
    'achieved_goal' : [peaking_db, noise_mvrms, eye_height_proxy_mv]

    Reward
    ------
    Negative sum of normalised errors (same formula as EqualizerEnvFast).
    compute_reward() is a staticmethod accepting arrays -- required by HER.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        peaking_target_range: tuple = (3.0, 11.0),
        noise_limit_mvrms: float = 1.5,
        eye_height_limit_mv: float = 100.0,
        rs_range: tuple = (50, 500),
        cs_range: tuple = (0.1e-12, 10e-12),
    ):
        super().__init__()

        self.peaking_target_range = peaking_target_range
        self.noise_limit_mvrms = float(noise_limit_mvrms)
        self.eye_height_limit_mv = float(eye_height_limit_mv)
        self.rs_range = rs_range
        self.cs_range = cs_range

        # Internal episode state (set during reset)
        self.target_peaking_db: float = float(np.mean(peaking_target_range))
        self._last_Rs: float = float(np.mean(rs_range))
        self._last_Cs: float = float(np.mean(cs_range))

        # ------------------------------------------------------------------
        # Action space: normalised [-1, 1] for (Rs, Cs)
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ------------------------------------------------------------------
        # Observation space: SB3 HER requires a Dict with exactly these keys
        # ------------------------------------------------------------------
        obs_low  = np.array([-1.0, -1.0], dtype=np.float32)
        obs_high = np.array([ 1.0,  1.0], dtype=np.float32)

        # Generous bounds for the goal dimensions so no clipping occurs
        goal_low  = np.array([0.0,   0.0,   0.0  ], dtype=np.float32)
        goal_high = np.array([30.0,  10.0,  3000.0], dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(low=obs_low,  high=obs_high,  dtype=np.float32),
            "desired_goal":  spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            "achieved_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rescale_action(self, action: np.ndarray):
        """Map normalised [-1, 1] action to physical (Rs, Cs) values."""
        rs_low, rs_high = self.rs_range
        cs_low, cs_high = self.cs_range
        Rs = rs_low + (float(action[0]) + 1.0) / 2.0 * (rs_high - rs_low)
        Cs = cs_low + (float(action[1]) + 1.0) / 2.0 * (cs_high - cs_low)
        return float(Rs), float(Cs)

    def _normalise_params(self, Rs: float, Cs: float):
        """Map physical (Rs, Cs) back to normalised [-1, 1] for observation."""
        rs_low, rs_high = self.rs_range
        cs_low, cs_high = self.cs_range
        Rs_norm = 2.0 * (Rs - rs_low) / (rs_high - rs_low) - 1.0
        Cs_norm = 2.0 * (Cs - cs_low) / (cs_high - cs_low) - 1.0
        return float(np.clip(Rs_norm, -1.0, 1.0)), float(np.clip(Cs_norm, -1.0, 1.0))

    def _build_obs_dict(
        self,
        Rs: float,
        Cs: float,
        peaking_db: float,
        noise_mvrms: float,
        eye_height_proxy_mv: float,
    ) -> dict:
        """Assemble the Dict observation required by HER."""
        Rs_norm, Cs_norm = self._normalise_params(Rs, Cs)
        return {
            "observation": np.array([Rs_norm, Cs_norm], dtype=np.float32),
            "desired_goal": np.array(
                [self.target_peaking_db, self.noise_limit_mvrms, self.eye_height_limit_mv],
                dtype=np.float32,
            ),
            "achieved_goal": np.array(
                [peaking_db, noise_mvrms, eye_height_proxy_mv],
                dtype=np.float32,
            ),
        }

    # ------------------------------------------------------------------
    # HER-required compute_reward (staticmethod -- works on batches)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_reward(
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info,  # unused but required by the HER interface
    ) -> np.ndarray:
        """
        Compute reward for a batch of (achieved_goal, desired_goal) pairs.

        HER calls this with arrays of shape (N, 3) to relabel transitions in
        the replay buffer, so we must handle both scalar and batch inputs.

        achieved_goal[:, 0] = peaking_db
        achieved_goal[:, 1] = noise_mvrms
        achieved_goal[:, 2] = eye_height_proxy_mv

        desired_goal[:, 0]  = target_peaking_db
        desired_goal[:, 1]  = noise_limit_mvrms
        desired_goal[:, 2]  = eye_height_limit_mv

        Returns negative sum of normalised errors (lower = better), shape (N,).
        """
        achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
        desired_goal  = np.asarray(desired_goal,  dtype=np.float64)

        # Ensure 2-D so slicing works uniformly for both single and batch
        single = achieved_goal.ndim == 1
        if single:
            achieved_goal = achieved_goal[np.newaxis, :]
            desired_goal  = desired_goal[np.newaxis, :]

        peaking_db           = achieved_goal[:, 0]
        noise_mvrms          = achieved_goal[:, 1]
        eye_height_proxy_mv  = achieved_goal[:, 2]

        target_peaking_db    = desired_goal[:, 0]
        noise_limit_mvrms    = desired_goal[:, 1]
        eye_height_limit_mv  = desired_goal[:, 2]

        # Peaking error normalised by the full target range span (8 dB)
        peaking_span = 8.0  # dB  (11.0 - 3.0)
        peaking_error_norm = np.abs(peaking_db - target_peaking_db) / peaking_span

        # Noise over-limit penalty, normalised by the limit itself
        noise_penalty_norm = np.maximum(
            0.0,
            (noise_mvrms - noise_limit_mvrms) / np.where(noise_limit_mvrms != 0, noise_limit_mvrms, 1.0),
        )

        # Eye-height under-limit penalty, normalised by the limit
        eye_penalty_norm = np.maximum(
            0.0,
            (eye_height_limit_mv - eye_height_proxy_mv) / np.where(eye_height_limit_mv != 0, eye_height_limit_mv, 1.0),
        )

        rewards = -(peaking_error_norm + noise_penalty_norm + eye_penalty_norm)

        return float(rewards[0]) if single else rewards.astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Sample a fresh random peaking target and return the initial obs dict.

        The initial observation uses the mid-range (Rs, Cs) values as a
        neutral starting point; the agent takes the first action before any
        simulation is run.
        """
        super().reset(seed=seed)

        # Sample target peaking from the allowed range
        low, high = self.peaking_target_range
        self.target_peaking_db = float(self.np_random.uniform(low, high))

        # Neutral physical starting point (mid-range)
        self._last_Rs = float(np.mean(self.rs_range))
        self._last_Cs = float(np.mean(self.cs_range))

        # Run a quick simulation so achieved_goal is meaningful from step 0
        result = simulate(self._last_Rs, self._last_Cs)

        obs = self._build_obs_dict(
            Rs=self._last_Rs,
            Cs=self._last_Cs,
            peaking_db=float(result["peaking_db"]),
            noise_mvrms=float(result["noise_mvrms"]),
            eye_height_proxy_mv=float(result["eye_height_proxy_mv"]),
        )
        return obs, {}

    def step(self, action: np.ndarray):
        """
        Execute one design step.

        Parameters
        ----------
        action : array-like, shape (2,)
            Normalised [-1, 1] values for (Rs, Cs).

        Returns
        -------
        obs_dict : dict
            Keys: 'observation', 'desired_goal', 'achieved_goal'.
        reward : float
        terminated : bool  (always True -- single-step episode)
        truncated : bool   (always False)
        info : dict
            Diagnostic values for logging / HER bookkeeping.
        """
        Rs, Cs = self._rescale_action(action)
        self._last_Rs, self._last_Cs = Rs, Cs

        result = simulate(Rs, Cs)

        peaking_db          = float(result["peaking_db"])
        noise_mvrms         = float(result["noise_mvrms"])
        eye_height_proxy_mv = float(result["eye_height_proxy_mv"])

        obs = self._build_obs_dict(Rs, Cs, peaking_db, noise_mvrms, eye_height_proxy_mv)

        reward = float(
            GoalEqualizerEnv.compute_reward(
                obs["achieved_goal"], obs["desired_goal"], {}
            )
        )

        info = {
            "Rs":                  Rs,
            "Cs":                  Cs,
            "target_peaking_db":   self.target_peaking_db,
            "peaking_db":          peaking_db,
            "noise_mvrms":         noise_mvrms,
            "eye_height_proxy_mv": eye_height_proxy_mv,
        }

        return obs, reward, True, False, info


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    print("=== GoalEqualizerEnv smoke test ===\n")

    env = GoalEqualizerEnv()
    obs, reset_info = env.reset(seed=42)

    print("reset() observation:")
    for k, v in obs.items():
        print(f"  {k}: {v}")
    print()

    action = env.action_space.sample()
    print(f"Random action: {action}")
    obs2, reward, terminated, truncated, info = env.step(action)

    print("\nstep() observation:")
    for k, v in obs2.items():
        print(f"  {k}: {v}")
    print(f"\nreward     = {reward:.6f}")
    print(f"terminated = {terminated}")
    print(f"truncated  = {truncated}")
    print("\ninfo:")
    pprint.pprint({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in info.items()})

    # Verify compute_reward works on a batch
    print("\n--- Batch compute_reward check ---")
    import numpy as _np
    achieved = _np.stack([obs2["achieved_goal"]] * 4)
    desired  = _np.stack([obs2["desired_goal"]]  * 4)
    batch_rewards = GoalEqualizerEnv.compute_reward(achieved, desired, {})
    print(f"Batch rewards (4 identical pairs): {batch_rewards}")
    assert batch_rewards.shape == (4,), "Expected shape (4,)"
    print("All checks passed.")
