"""
train_sac.py — SAC + HER training pipeline for RACD.

Key improvements over train_fast.py (PPO):
  1. SAC (Soft Actor-Critic) — off-policy, replay buffer reuses all historical data
  2. HerReplayBuffer — relabels ~50% of transitions with achieved goals (free data aug)
  3. Surrogate MLP simulator — sub-0.01ms per call (from circuit_surrogate.py)
  4. torch.compile on policy — 15-40% faster gradient updates
  5. SPICE validation pass after training for ground-truth accuracy check

Usage:
    cd src
    python train_sac.py                            # 50k steps, surrogate mode
    python train_sac.py --timesteps 200000         # longer run
    python train_sac.py --mode spice --timesteps 20000  # SPICE-only (accurate but slow)
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

from logging_utils import log_episode, load_repository
from retrieval import retrieve_k_nearest, rs_cs_to_action


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class SpeedCallback(BaseCallback):
    def __init__(self, total_timesteps, log_interval=2000, verbose=1):
        super().__init__(verbose)
        self._total = total_timesteps
        self._t0 = None
        self._interval = log_interval

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self._interval == 0 and self._t0:
            elapsed = time.time() - self._t0
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0
            remaining = (self._total - self.num_timesteps) / sps if sps > 0 else 0
            pct = 100 * self.num_timesteps / self._total
            print(f"  [{pct:5.1f}%] {self.num_timesteps:>7d}/{self._total}  "
                  f"{sps:>6.0f} steps/sec  ETA {remaining:.0f}s", flush=True)
        return True


# ---------------------------------------------------------------------------
# Logging wrapper around GoalEqualizerEnv
# ---------------------------------------------------------------------------

def make_logging_goal_env(mode='surrogate'):
    """Create a GoalEqualizerEnv that logs episodes and uses specified sim backend."""
    from environment_goal import GoalEqualizerEnv

    class LoggingGoalEnv(GoalEqualizerEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            if info:
                log_episode(info)
            return obs, reward, terminated, truncated, info

    # Configure simulation backend
    if mode == 'surrogate':
        import circuit_surrogate as sim_module
        from environment_goal import GoalEqualizerEnv as _Base
        _Base._simulate = staticmethod(sim_module.simulate)
    elif mode == 'analytical':
        import circuit_fast as sim_module
        sim_module.USE_ANALYTICAL = True
        from environment_goal import GoalEqualizerEnv as _Base
        _Base._simulate = staticmethod(sim_module.simulate)

    return LoggingGoalEnv()


# ---------------------------------------------------------------------------
# Main SAC + HER training function
# ---------------------------------------------------------------------------

def train_sac(
    total_timesteps: int = 50000,
    mode: str = 'surrogate',
    use_warmstart: bool = True,
    learning_starts: int = 1000,
    batch_size: int = 512,
    save_path: str = None,
):
    """
    Train RACD with SAC + HER.

    Args:
        total_timesteps: Total environment interactions.
        mode: 'surrogate' | 'analytical' | 'spice'
        use_warmstart: Pre-populate replay buffer with repository data.
        learning_starts: Steps before gradient updates begin.
        batch_size: Replay buffer sample size per gradient step.
        save_path: Where to save the trained model.
    """
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'sac_equalizer'
        )

    print(f"Mode: {mode}")
    print(f"Algorithm: SAC + HER (off-policy + hindsight relabeling)")

    # Build environment
    env = make_logging_goal_env(mode=mode)

    # SAC with HER replay buffer
    model = SAC(
        'MultiInputPolicy',
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,           # 4 HER relabelings per real transition
            goal_selection_strategy='future',  # relabel with future achieved goals
        ),
        verbose=0,
        learning_starts=learning_starts,
        batch_size=batch_size,
        learning_rate=3e-4,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        buffer_size=100_000,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    # torch.compile the actor/critic networks (PyTorch 2.0+ with CUDA available)
    try:
        import torch
        if torch.cuda.is_available():
            model.actor = torch.compile(model.actor, backend='inductor')
            model.critic = torch.compile(model.critic, backend='inductor')
            print("torch.compile applied to actor + critic (inductor backend)")
        else:
            print("torch.compile skipped (CUDA not available — run under venv312 for GPU)")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    # Warm-start: pre-populate replay buffer with repository data
    if use_warmstart:
        repository = load_repository()
        good = [r for r in repository
                if abs(r.get('peaking_db', 0) - r.get('target_peaking_db', 0)) < 1.0]
        if good:
            print(f"Pre-populating replay buffer with {len(good)} good episodes...")
            # Add as fake transitions — obs, action, reward, next_obs
            for ep in good[:min(len(good), 5000)]:
                try:
                    Rs, Cs = ep['Rs'], ep['Cs']
                    target = ep['target_peaking_db']
                    # Convert to action
                    a0 = 2 * (Rs - 50) / (500 - 50) - 1
                    a1 = 2 * (Cs - 0.1e-12) / (9.9e-12) - 1
                    action = np.array([a0, a1], dtype=np.float32)
                    obs, _ = env.reset()
                    # Override target to match episode
                    env.target_peaking_db = target
                    env.noise_limit_mvrms = ep.get('noise_mvrms', 1.5)
                    obs['desired_goal'] = np.array(
                        [target, env.noise_limit_mvrms, env.eye_height_limit_mv],
                        dtype=np.float32
                    )
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    model.replay_buffer.add(obs, next_obs, action, reward,
                                            terminated, [info])
                except Exception:
                    continue
            print(f"Replay buffer seeded: {model.replay_buffer.size()} transitions")
        else:
            print("No good episodes in repository yet — cold start")

    # Train
    print(f"\nTraining SAC+HER for {total_timesteps:,} steps...\n")
    t0 = time.time()
    callback = SpeedCallback(total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback,
                reset_num_timesteps=True)
    elapsed = time.time() - t0
    sps = total_timesteps / elapsed
    print(f"\nTraining complete: {total_timesteps:,} steps in {elapsed:.1f}s = {sps:.0f} steps/sec")

    # SPICE validation
    print("\nRunning SPICE validation on trained SAC policy...")
    from environment_goal import GoalEqualizerEnv
    from circuit_fast import _spice_simulate
    val_env = GoalEqualizerEnv()

    val_rewards = []
    for _ in range(10):
        obs, _ = val_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        target = float(obs['desired_goal'][0])
        Rs = 50 + (float(action[0]) + 1) / 2 * 450
        Cs = 0.1e-12 + (float(action[1]) + 1) / 2 * 9.9e-12
        try:
            spice_result = _spice_simulate(Rs, Cs)
            achieved = spice_result['peaking_db']
            reward_val = -abs(achieved - target) / 8.0
        except Exception:
            achieved, reward_val = float('nan'), -1.0
        val_rewards.append(reward_val)
        print(f"  target={target:.1f} dB  achieved={achieved:.2f} dB  "
              f"err={abs(achieved-target):.3f} dB  reward={reward_val:.4f}")

    print(f"Mean SPICE-validated reward: {np.mean(val_rewards):.4f}")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    if hasattr(env, 'close'):
        env.close()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch  # noqa — imported here so torch.compile is available above

    parser = argparse.ArgumentParser(description='RACD SAC+HER training')
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--mode', choices=['surrogate', 'analytical', 'spice'],
                        default='surrogate')
    parser.add_argument('--no-warmstart', action='store_true')
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    train_sac(
        total_timesteps=args.timesteps,
        mode=args.mode,
        use_warmstart=not args.no_warmstart,
        batch_size=args.batch_size,
    )
