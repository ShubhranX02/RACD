"""
train_transistor.py — Transistor-Level RL Training Pipeline (High-Speed Edition)

Simulation Modes
----------------
Surrogate (default) : ~0.01 ms/call  ->  12-worker DummyVecEnv (no IPC)  ->  ~1,000+ steps/sec
SPICE (--spice)     : ~2-4 s/call    ->  single-env sequential            ->  ~0.5 steps/sec

Workflow
--------
1. If surrogate model doesn't exist yet:
   a. Generate 600 SPICE training points (parallel, ~5 min)
   b. Train surrogate MLP (CPU, ~2 min)
2. Train PPO on surrogate-backed env (12 workers, ~50k steps default)
3. Validate trained policy on real SPICE across 3 targets
4. Save model to models/ppo_transistor.zip

Usage
-----
    # Full auto (generate + train surrogate + train policy):
    python src/train_transistor.py

    # Skip data-gen if transistor_repository.jsonl already has data:
    python src/train_transistor.py --skip-datagen

    # Force pure SPICE mode (slow, for final ground-truth training):
    python src/train_transistor.py --spice --steps 500

    # Long surrogate run:
    python src/train_transistor.py --steps 200000
"""

import os
import sys
import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TransistorSpeedCallback(BaseCallback):
    """Prints real-time steps/sec, ETA, and surrogate vs SPICE source ratio."""

    def __init__(self, total_timesteps, report_interval=2000, verbose=1):
        super().__init__(verbose)
        self._total = total_timesteps
        self._t0 = None
        self._interval = report_interval
        self._surrogate_steps = 0
        self._spice_steps = 0

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self) -> bool:
        # Track source mix
        for info in (self.locals.get('infos') or []):
            src = info.get('_source', 'spice')
            if 'surrogate' in str(src):
                self._surrogate_steps += 1
            else:
                self._spice_steps += 1

        if self.num_timesteps % self._interval == 0 and self._t0 is not None:
            elapsed = time.time() - self._t0
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0
            remaining = (self._total - self.num_timesteps) / sps if sps > 0 else float('inf')
            pct = 100 * self.num_timesteps / self._total
            total_src = self._surrogate_steps + self._spice_steps
            surr_pct = 100 * self._surrogate_steps / total_src if total_src > 0 else 0
            print(
                f"  [{pct:5.1f}%] {self.num_timesteps:>7d}/{self._total}  "
                f"{sps:>6.0f} steps/sec  ETA {remaining:.0f}s  "
                f"surrogate={surr_pct:.0f}%",
                flush=True
            )
        return True


# ---------------------------------------------------------------------------
# Surrogate bootstrap
# ---------------------------------------------------------------------------

def _ensure_surrogate(n_samples: int = 600, n_workers: int = None, skip_datagen: bool = False):
    """
    Check if surrogate model exists; if not, generate training data and train it.
    Returns True if surrogate is now available.
    """
    from pathlib import Path
    model_path = Path(__file__).resolve().parent.parent / 'models' / 'surrogate_transistor.pt'
    data_path  = Path(__file__).resolve().parent.parent / 'data'  / 'transistor_repository.jsonl'

    if model_path.exists():
        print(f"[surrogate] Found existing model: {model_path}")
        return True

    print("[surrogate] No surrogate model found — building from scratch...")

    # Step 1: data generation
    if not skip_datagen:
        existing_count = 0
        if data_path.exists():
            with open(data_path) as f:
                existing_count = sum(1 for _ in f)
        if existing_count >= 100:
            print(f"[surrogate] Using existing {existing_count} records in transistor_repository.jsonl")
        else:
            import multiprocessing
            multiprocessing.freeze_support()
            from surrogate_transistor import generate_training_data
            if n_workers is None:
                n_workers = min(os.cpu_count() or 4, 12)
            print(f"[surrogate] Generating {n_samples} SPICE samples with {n_workers} workers...")
            n_written = generate_training_data(n_samples=n_samples, n_workers=n_workers)
            print(f"[surrogate] {n_written} records generated.")
    else:
        print("[surrogate] Skipping data generation (--skip-datagen)")

    # Step 2: train MLP
    from surrogate_transistor import train_surrogate
    print("[surrogate] Training surrogate MLP...")
    model = train_surrogate()
    if model is None:
        print("[surrogate] WARNING: Surrogate training failed — will fall back to SPICE mode")
        return False

    print("[surrogate] Surrogate ready!")
    return True


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_transistor(
    total_timesteps: int = 50000,
    save_path: str = None,
    use_surrogate: bool = True,
    n_workers: int = None,
    multi_corner: bool = True,
    spice_validate_every: int = 200,
    n_samples: int = 600,
    skip_datagen: bool = False,
):
    """
    Train the transistor-level PPO policy.

    Args:
        total_timesteps:       RL training steps (50k surrogate ≈ 30 sec; 300 SPICE ≈ 30 min).
        use_surrogate:         Use surrogate MLP for rollouts (True) or full SPICE (False).
        n_workers:             Parallel SubprocVecEnv workers in surrogate mode (default: 12).
        multi_corner:          Randomize PVT corner each episode.
        spice_validate_every:  Run real SPICE every N steps for DAgger refinement (0=off).
        n_samples:             SPICE samples for surrogate bootstrap if model missing.
        skip_datagen:          Skip SPICE data generation even if surrogate missing.
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor')

    print("=" * 75)
    print(f"  Training Transistor-Level RL Agent (2-Stage CTLE, Sky130 PDK)")
    print(f"  Steps: {total_timesteps:,} | Surrogate: {use_surrogate} | "
          f"Multi-corner: {multi_corner} | Workers: {n_workers or 12}")
    print("=" * 75)

    # Ensure surrogate is ready (builds it if missing)
    if use_surrogate:
        surrogate_ok = _ensure_surrogate(
            n_samples=n_samples, skip_datagen=skip_datagen,
            n_workers=n_workers or min(os.cpu_count() or 4, 12)
        )
        if not surrogate_ok:
            print("[train] Falling back to SPICE mode (surrogate unavailable)")
            use_surrogate = False

    # Build environment factory
    actual_workers = (n_workers or min(os.cpu_count() or 4, 12)) if use_surrogate else 1

    def make_env(rank):
        def _init():
            from environment_transistor import TransistorEqualizerEnv
            env = TransistorEqualizerEnv(
                topology='1stage',
                use_surrogate=use_surrogate,
                multi_corner=multi_corner,
                spice_validate_every=spice_validate_every if rank == 0 else 0,
            )
            env.reset(seed=rank * 100)
            return env
        return _init

    if use_surrogate and actual_workers > 1:
        # SubprocVecEnv: 12 workers run in parallel OS processes.
        # Each subprocess independently runs surrogate inference (numpy, ~0.05ms).
        # Parallel execution: 1 timestep = max(12 workers) ≈ 0.05ms vs
        # DummyVecEnv sequential: 1 timestep = 12 × 0.05ms = 0.6ms.
        # Reverted from DummyVecEnv (which was 25% SLOWER due to losing parallelism).
        print(f"  Backend: Surrogate MLP ({actual_workers}-worker SubprocVecEnv, numpy inference)")
        env = SubprocVecEnv([make_env(i) for i in range(actual_workers)])
    else:
        print("  Backend: Full SPICE ngspice (sequential)")
        from environment_transistor import TransistorEqualizerEnv
        env = TransistorEqualizerEnv(
            topology='1stage',
            use_surrogate=False,
            multi_corner=multi_corner,
        )

    # PPO hyperparameters: SubprocVecEnv + numpy surrogate + CPU gradient updates
    # - device='cpu': faster than GPU for 102k-param MLP (no transfer overhead)
    # - n_steps=2048: larger rollout amortizes SubprocVecEnv IPC overhead better
    # - n_epochs=10: standard for PPO, balanced with batch_size
    # - batch_size=512: efficient for CPU matmul on 24576-sample rollouts
    if use_surrogate:
        lr, n_steps, batch_size = 3e-4, 2048, 512
        ppo_device = 'cpu'
        n_epochs = 10
    else:
        lr, n_steps, batch_size = 5e-4, 128, 64
        ppo_device = 'auto'
        n_epochs = 10

    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        ent_coef=0.01,
        device=ppo_device,
    )

    callback = TransistorSpeedCallback(total_timesteps, report_interval=max(500, total_timesteps // 50))
    t0 = time.time()
    print(f"\nTraining for {total_timesteps:,} steps...\n")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0
    sps = total_timesteps / elapsed if elapsed > 0 else 0
    print(f"\n[OK] Training done: {total_timesteps:,} steps in {elapsed:.1f}s = {sps:.0f} steps/sec")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"[OK] Model saved: {save_path}.zip")

    # SPICE validation on 3 targets
    print("\n--- SPICE Validation (ground truth) ---")
    from environment_transistor import TransistorEqualizerEnv
    from circuit_transistor import simulate_transistor_fast
    for target in [4.5, 6.0, 8.0]:
        val_env = TransistorEqualizerEnv(
            peaking_target_range=(target, target),
            topology='1stage',
            use_surrogate=False,   # always validate with real SPICE
            multi_corner=False,
        )
        obs, _ = val_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = val_env.step(action)
        print(
            f"  target={target:.1f} dB -> achieved={info['peaking_db']:.2f} dB  "
            f"noise={info['noise_mvrms']:.3f} mV  power={info['power_mw']:.2f} mW  "
            f"reward={reward:.3f}"
        )

    if hasattr(env, 'close'):
        env.close()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='Train transistor CTLE RL policy')
    parser.add_argument('--steps', type=int, default=50000,
                        help='RL training timesteps (default: 50000 with surrogate)')
    parser.add_argument('--spice', action='store_true',
                        help='Force full SPICE mode (slow; use for final ground-truth run)')
    parser.add_argument('--workers', type=int, default=None,
                        help='SubprocVecEnv workers in surrogate mode (default: all cores, max 12)')
    parser.add_argument('--no-multicorner', action='store_true',
                        help='Disable PVT corner randomization (train at TT only)')
    parser.add_argument('--skip-datagen', action='store_true',
                        help='Skip SPICE data generation (use existing transistor_repository.jsonl)')
    parser.add_argument('--n-samples', type=int, default=600,
                        help='LHS SPICE samples for surrogate bootstrap (default: 600)')
    parser.add_argument('--dagger-every', type=int, default=200,
                        help='Run real SPICE every N steps for DAgger refinement (0=off)')
    parser.add_argument('--batched', action='store_true',
                        help='Use BatchedSurrogateVecEnv (zero IPC, ~22500 steps/sec) instead of SubprocVecEnv')
    parser.add_argument('--batched-envs', type=int, default=256,
                        help='Number of parallel envs in batched mode (default: 256, free since no IPC)')
    parser.add_argument('--checkpoint-every', type=int, default=500000,
                        help='Save model checkpoint every N steps in batched mode (default: 500000, 0=off)')
    args = parser.parse_args()

    if args.batched and not args.spice:
        # Zero-IPC batched mode: import here to avoid import at top-level
        from batched_vec_env import BatchedSurrogateVecEnv
        from surrogate_transistor import load_transistor_surrogate
        load_transistor_surrogate()

        print("=" * 75)
        print(f"  Training Transistor-Level RL Agent (1-Stage CTLE, Sky130 PDK)")
        print(f"  Steps: {args.steps:,} | Backend: BatchedSurrogateVecEnv x{args.batched_envs} | ZERO IPC")
        print("=" * 75)

        vec_env = BatchedSurrogateVecEnv(
            n_envs=args.batched_envs,
            multi_corner=not args.no_multicorner,
        )
        # NO VecNormalize — tested and confirmed to make results WORSE (MAE 1.58 vs 0.875 dB).
        # Root cause: running stats are unstable early in training; surrogate region-dependent
        # biases aren't fixed by normalization. Raw [3-10] dB obs works better in practice.

        # Match the hyperparams that produced the best result (1M SubprocVecEnv, MAE=0.875 dB):
        # - n_steps=128 × 256 envs = 32,768 samples/update (vs 12×2048=24,576 that worked)
        # - n_epochs=10 (exact match)
        # - No VecNormalize, no big net_arch changes
        _ppo_device = 'cpu'

        model = PPO(
            'MlpPolicy', vec_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=32,            # 32 × 256 envs = 8,192 samples/update (4× faster gradient steps)
            batch_size=512,        # 16 mini-batches/epoch
            n_epochs=4,            # reduced to avoid CPU-gradient bottleneck; ~same data efficiency
            gamma=0.99,
            ent_coef=0.01,
            device=_ppo_device,
        )
        import time, os as _os
        _save_path = _os.path.join(_os.path.dirname(__file__), '..', 'models', 'ppo_transistor')
        _os.makedirs(_os.path.dirname(_save_path), exist_ok=True)
        _ckpt_every = args.checkpoint_every

        class _CkptCallback(TransistorSpeedCallback):
            def __init__(self, total, report_interval, ckpt_every, save_path):
                super().__init__(total, report_interval)
                self._ckpt_every = ckpt_every
                self._save_path = save_path
                self._last_ckpt = 0
            def _on_step(self):
                super()._on_step()
                if self._ckpt_every > 0 and self.num_timesteps - self._last_ckpt >= self._ckpt_every:
                    ckpt = f"{self._save_path}_ckpt{self.num_timesteps//1000}k"
                    self.model.save(ckpt)
                    # Also save VecNormalize stats so the checkpoint can be loaded correctly
                    if hasattr(self.training_env, 'save'):
                        self.training_env.save(f"{ckpt}_vecnorm.pkl")
                    print(f"  [ckpt] Saved {ckpt}.zip")
                    self._last_ckpt = self.num_timesteps
                return True

        callback = _CkptCallback(
            args.steps,
            report_interval=max(500, args.steps // 100),
            ckpt_every=_ckpt_every,
            save_path=_save_path,
        )
        t0 = time.time()
        print(f"\nTraining for {args.steps:,} steps (device={_ppo_device}, {args.batched_envs} envs)...\n")
        model.learn(total_timesteps=args.steps, callback=callback)
        elapsed = time.time() - t0
        sps = args.steps / elapsed
        print(f"\n[OK] Training done: {args.steps:,} steps in {elapsed:.1f}s = {sps:.0f} steps/sec")
        model.save(_save_path)
        print(f"[OK] Model saved: {_save_path}.zip")
        vec_env.close()

    else:
        train_transistor(
            total_timesteps=args.steps,
            use_surrogate=not args.spice,
            n_workers=args.workers,
            multi_corner=not args.no_multicorner,
            spice_validate_every=args.dagger_every,
            n_samples=args.n_samples,
            skip_datagen=args.skip_datagen,
        )

