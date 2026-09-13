"""
train_fast.py — Optimized PPO training with all performance improvements:

  Tier 1: Single-pass SPICE via stdin pipes (no temp files, 2.5x)
  Tier 2: Full CPU core saturation (up to all logical threads via SubprocVecEnv or DummyVecEnv)
  Tier 3: GPU/vectorized analytical RC model during exploration (100x+, validates with SPICE)
  Tier 4: In-memory buffered repository logging (eliminates per-step disk I/O lock contention)
  Tier 5: Configurable in-process vectorized DummyVecEnv for zero IPC overhead in analytical mode

Usage:
    cd src
    python train_fast.py
    python train_fast.py --timesteps 100000 --workers 12 --mode analytical
    python train_fast.py --timesteps 50000 --workers 12 --mode spice
"""

import os
import sys
import argparse
import time
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from environment_fast import EqualizerEnvFast
from logging_utils import log_episode, load_repository
from retrieval import retrieve_k_nearest, rs_cs_to_action
from warmstart import pretrain_policy_toward_retrieved


# ---------------------------------------------------------------------------
# In-Memory Buffered Logging Wrapper (Zero per-step disk I/O lock contention)
# ---------------------------------------------------------------------------

class BufferedLoggingEnv(EqualizerEnvFast):
    """
    Buffers episode transitions in memory and flushes in batches or skips disk writes
    to achieve maximum compute throughput without disk I/O blocking.
    """
    def __init__(self, log_to_disk: bool = False, buffer_size: int = 500):
        super().__init__()
        self.log_to_disk = log_to_disk
        self.buffer_size = buffer_size
        self._buffer = []

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.log_to_disk:
            self._buffer.append(info)
            if len(self._buffer) >= self.buffer_size:
                self.flush()
        return obs, reward, terminated, truncated, info

    def flush(self):
        if self.log_to_disk and self._buffer:
            for item in self._buffer:
                log_episode(item)
            self._buffer.clear()

    def close(self):
        self.flush()
        super().close()


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class SpeedCallback(BaseCallback):
    """Prints real-time steps/sec and ETA."""
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self._total = total_timesteps
        self._t0 = None
        self._report_interval = 4096

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self._report_interval == 0 and self._t0 is not None:
            elapsed = time.time() - self._t0
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0
            remaining = (self._total - self.num_timesteps) / sps if sps > 0 else float('inf')
            pct = 100 * self.num_timesteps / self._total
            print(f"  [{pct:5.1f}%] {self.num_timesteps:>7d}/{self._total} steps  "
                  f"{sps:>6.0f} steps/sec  ETA {remaining:.0f}s", flush=True)
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(rank: int, seed: int = 0, log_to_disk: bool = False):
    """Factory function for vectorized environments."""
    def _init():
        env = BufferedLoggingEnv(log_to_disk=log_to_disk)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_fast(
    total_timesteps: int = 100000,
    n_workers: int = None,
    use_warmstart: bool = True,
    mode: str = 'analytical',   # 'analytical' | 'spice_cached' | 'spice'
    vec_type: str = 'subproc',  # 'subproc' | 'dummy'
    log_to_disk: bool = False,
    save_path: str = None,
):
    """
    Args:
        total_timesteps: Total environment steps to train for.
        n_workers:       Number of parallel environments (defaults to all logical CPU cores).
        use_warmstart:   Whether to behaviour-clone from repository.
        mode:            'analytical' = GPU/NumPy RC model (fastest, ~100x+ baseline)
                         'spice_cached' = SPICE with LRU cache
                         'spice' = Full SPICE every step
        vec_type:        'subproc' for true multiprocessing across cores, or 'dummy' for in-process.
        log_to_disk:     Whether to write every transition to circuit_repository.jsonl.
        save_path:       Where to save the model. Defaults to models/ppo_equalizer_fast.
    """
    import circuit_fast

    # Configure simulation mode
    if mode == 'analytical':
        circuit_fast.USE_ANALYTICAL = True
        circuit_fast.USE_SPICE_CACHE = False
        print("Mode: GPU/NumPy analytical RC engine (fastest)")
    elif mode == 'spice_cached':
        circuit_fast.USE_ANALYTICAL = False
        circuit_fast.USE_SPICE_CACHE = True
        print("Mode: SPICE with LRU cache")
    else:
        circuit_fast.USE_ANALYTICAL = False
        circuit_fast.USE_SPICE_CACHE = False
        print("Mode: Full SPICE (ground truth)")

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'ppo_equalizer_fast'
        )

    # Saturate all available CPU threads
    total_cpu_cores = os.cpu_count() or 4
    if n_workers is None:
        n_workers = total_cpu_cores
    print(f"Workers: {n_workers} parallel environments (System total logical cores: {total_cpu_cores})")
    print(f"Disk Logging: {'ENABLED (Buffered)' if log_to_disk else 'DISABLED (Pure High-Speed RAM Compute)'}")

    # Build vectorized environment
    env_fns = [make_env(i, log_to_disk=log_to_disk) for i in range(n_workers)]
    if vec_type == 'subproc' and n_workers > 1:
        env = SubprocVecEnv(env_fns)
        print(f"Using SubprocVecEnv with {n_workers} independent CPU workers")
    else:
        env = DummyVecEnv(env_fns)
        print(f"Using DummyVecEnv with {n_workers} vectorized environments (zero IPC pickling overhead)")

    # Build PPO model with scaled batch size for high-worker throughput
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,  # larger batch size for multi-threaded efficiency
        n_epochs=10,
        gamma=0.99,
    )

    # Warm-start from repository
    if use_warmstart:
        repository = load_repository()
        tmp_env = EqualizerEnvFast()
        mid_target = sum(tmp_env.peaking_target_range) / 2
        retrieved = retrieve_k_nearest(mid_target, repository, k=10)
        if retrieved:
            examples = [
                {'target_peaking_db': ep['target_peaking_db'],
                 'action': rs_cs_to_action(ep['Rs'], ep['Cs'],
                                            tmp_env.rs_range, tmp_env.cs_range)}
                for ep in retrieved
            ]
            print(f"Warm-starting with {len(examples)} retrieved repository episodes")
            model = pretrain_policy_toward_retrieved(model, examples)
        else:
            print("Cold start (no repository data)")

    # Train
    print(f"\nTraining for {total_timesteps:,} steps...\n")
    t0 = time.time()
    callback = SpeedCallback(total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0

    total_steps_per_sec = total_timesteps / elapsed if elapsed > 0 else 0
    print(f"\nTraining complete: {total_timesteps:,} steps in {elapsed:.1f}s "
          f"= {total_steps_per_sec:.0f} steps/sec overall throughput")

    # SPICE validation pass
    print("\nRunning SPICE validation on trained policy (10 random targets)...")
    circuit_fast.USE_ANALYTICAL = False
    circuit_fast.USE_SPICE_CACHE = False
    val_env = EqualizerEnvFast()
    val_rewards = []
    for _ in range(10):
        obs, _ = val_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = val_env.step(action)
        val_rewards.append(reward)
        print(f"  target={info['target_peaking_db']:.1f} dB  "
              f"achieved={info['peaking_db']:.2f} dB  reward={reward:.4f}")
    print(f"Mean SPICE-validated reward: {np.mean(val_rewards):.4f}")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    if hasattr(env, 'close'):
        env.close()

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized RACD training')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps (default: 100000)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel env workers (default: all logical cores, 12)')
    parser.add_argument('--mode', choices=['analytical', 'spice_cached', 'spice'],
                        default='analytical',
                        help='Simulation mode (default: analytical = fastest)')
    parser.add_argument('--vec', choices=['subproc', 'dummy'],
                        default='subproc',
                        help='Vectorization type: subproc (multi-core) or dummy (in-process)')
    parser.add_argument('--log-disk', action='store_true',
                        help='Enable buffered disk logging to circuit_repository.jsonl (slower)')
    parser.add_argument('--no-warmstart', action='store_true',
                        help='Disable warm-start from repository')
    args = parser.parse_args()

    train_fast(
        total_timesteps=args.timesteps,
        n_workers=args.workers,
        use_warmstart=not args.no_warmstart,
        mode=args.mode,
        vec_type=args.vec,
        log_to_disk=args.log_disk,
    )
