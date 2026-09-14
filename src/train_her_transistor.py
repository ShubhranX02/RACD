"""
train_her_transistor.py — SAC + Hindsight Experience Replay for transistor CTLE sizing.

Why SAC+HER instead of PPO here?
  - SAC is off-policy (better sample efficiency, can replay past experiences)
  - HER relabels failed episodes using achieved metrics as new desired goals
  - Ideal for the sparse multi-spec reward: most random designs fail ALL specs
  - PPO (on-policy) discards every episode; HER reuses each one up to 4× via relabelling

Estimated training time (surrogate mode):
  - SAC is slower per step than PPO (off-policy replay overhead)
  - Target: 200k–500k steps, ~30 min on CPU with surrogate

Usage:
    # Default: SAC+HER surrogate mode, 300k steps
    python src/train_her_transistor.py

    # More steps for better convergence
    python src/train_her_transistor.py --steps 500000

    # Pure SPICE (slow, ground-truth)
    python src/train_her_transistor.py --spice --steps 2000
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))


def train_her(
    total_timesteps: int = 40_000,
    use_surrogate: bool = True,
    multi_corner: bool = True,
    save_path: str = None,
    n_sampled_goal: int = 4,
    goal_selection_strategy: str = 'future',
):
    """Train SAC+HER policy on GoalTransistorEnv."""
    from stable_baselines3 import SAC
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    from environment_goal_transistor import GoalTransistorEnv
    from surrogate_transistor import load_transistor_surrogate

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sac_her_transistor')

    print("=" * 70)
    print(f"  SAC+HER Transistor CTLE Training")
    print(f"  Steps: {total_timesteps:,} | Surrogate: {use_surrogate} | Multi-corner: {multi_corner}")
    print(f"  HER: n_sampled_goal={n_sampled_goal}, strategy={goal_selection_strategy}")
    print("=" * 70)

    if use_surrogate:
        loaded = load_transistor_surrogate()
        if not loaded:
            print("[HER] Surrogate not found — falling back to SPICE mode")
            use_surrogate = False

    env = GoalTransistorEnv(
        topology='2stage',
        use_surrogate=use_surrogate,
        multi_corner=multi_corner,
    )

    model = SAC(
        'MultiInputPolicy',
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            'n_sampled_goal': n_sampled_goal,
            'goal_selection_strategy': goal_selection_strategy,
        },
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        learning_starts=500,
        tau=0.005,
        gamma=0.98,
        train_freq=(16, "step"),
        gradient_steps=16,
        ent_coef='auto',
        verbose=1,
        device='cpu',
    )

    print(f"\nTraining SAC+HER for {total_timesteps:,} steps...\n")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, log_interval=2000)
    elapsed = time.time() - t0
    sps = total_timesteps / elapsed if elapsed > 0 else 0
    print(f"\n[OK] Training done: {total_timesteps:,} steps in {elapsed:.1f}s = {sps:.0f} steps/sec")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"[OK] Model saved: {save_path}.zip")

    # Quick validation across peaking targets
    print("\n--- SAC+HER Validation ---")
    from environment_goal_transistor import GoalTransistorEnv as _GTE
    val_env = _GTE(use_surrogate=False, multi_corner=False)  # always validate with SPICE
    for target in [4.0, 6.0, 8.0, 10.0, 12.0]:
        val_env._target_peaking_db = target
        val_env._desired_goal_vec  = val_env._make_desired_goal(target)
        obs, _ = val_env.reset()
        obs['desired_goal'] = val_env._desired_goal_vec
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = val_env.step(action)
        status = "PASS [OK]" if info.get('is_success') else "MISS [X]"
        print(
            f"  target={target:.1f}dB  achieved={info['peaking_db']:.2f}dB  "
            f"noise={info['noise_mvrms']:.3f}mV  power={info['power_mw']:.2f}mW  {status}"
        )

    env.close()
    val_env.close()
    return model


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='SAC+HER transistor CTLE training')
    parser.add_argument('--steps', type=int, default=40_000,
                        help='Total training timesteps (default: 40000)')
    parser.add_argument('--spice', action='store_true',
                        help='Force SPICE mode (slow, use for final ground-truth run)')
    parser.add_argument('--no-multicorner', action='store_true',
                        help='Disable PVT corner randomization')
    parser.add_argument('--n-goals', type=int, default=4,
                        help='HER n_sampled_goal (default: 4)')
    parser.add_argument('--strategy', type=str, default='future',
                        choices=['future', 'final', 'episode'],
                        help="HER goal selection strategy (default: future)")
    args = parser.parse_args()

    train_her(
        total_timesteps=args.steps,
        use_surrogate=not args.spice,
        multi_corner=not args.no_multicorner,
        n_sampled_goal=args.n_goals,
        goal_selection_strategy=args.strategy,
    )
