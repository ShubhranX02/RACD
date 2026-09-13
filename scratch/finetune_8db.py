"""
finetune_8db.py — Targeted fine-tuning of the PPO transistor model
to close the 0.71 dB gap at 8.0 dB target peaking.

Loads the existing best model, narrows the target range to [7.0, 9.0],
and runs 5,000 additional surrogate steps. Then validates on SPICE.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO
from environment_transistor import TransistorEqualizerEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
BEST_MODEL = os.path.join(MODEL_DIR, 'ppo_transistor_best')
SAVE_PATH = os.path.join(MODEL_DIR, 'ppo_transistor')

# 1. Load existing converged model
print("=" * 60)
print("  Targeted Fine-Tuning: 7.0–9.0 dB range")
print("=" * 60)

if os.path.exists(BEST_MODEL + '.zip'):
    print(f"[1/3] Loading best model from {BEST_MODEL}.zip")
    model = PPO.load(BEST_MODEL)
else:
    print(f"[1/3] Loading model from {SAVE_PATH}.zip")
    model = PPO.load(SAVE_PATH)

# 2. Create environment with narrowed target range
print("[2/3] Creating environment with target range [7.0, 9.0] dB")

from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    def _init():
        return TransistorEqualizerEnv(
            peaking_target_range=(7.0, 9.0),
            use_surrogate=True,
        )
    return _init

n_envs = 12  # Use 12 parallel workers for speed
env = DummyVecEnv([make_env() for _ in range(n_envs)])

# Reload model with new env (handles n_envs mismatch)
if os.path.exists(BEST_MODEL + '.zip'):
    model = PPO.load(BEST_MODEL, env=env)
else:
    model = PPO.load(SAVE_PATH, env=env)

# 3. Fine-tune
STEPS = 5000
print(f"[3/3] Fine-tuning for {STEPS} steps on surrogate...")
model.learn(total_timesteps=STEPS, reset_num_timesteps=False)

# Save
model.save(SAVE_PATH)
print(f"\n[OK] Fine-tuned model saved to {SAVE_PATH}.zip")

# 4. Validate on real SPICE
print("\n" + "=" * 60)
print("  Validation: Testing 7.0–9.0 dB on real SPICE")
print("=" * 60)

val_env = TransistorEqualizerEnv(use_surrogate=False)
targets = [7.0, 7.5, 8.0, 8.5, 9.0]

for t in targets:
    val_env.peaking_target_range = (t, t)
    obs, _ = val_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, info = val_env.step(action)
    error = abs(info['peaking_db'] - t)
    print(f"  Target: {t:.1f} dB | Achieved: {info['peaking_db']:.2f} dB | Error: {error:.2f} dB | Power: {info['power_mw']:.2f} mW")

print("\nDone!")
