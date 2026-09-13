import os
import json
from stable_baselines3 import PPO
from environment_transistor import TransistorEqualizerEnv

def log_transistor_episode(info, log_path=None):
    if log_path is None:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transistor_repository.jsonl')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not info:
        return
    with open(log_path, 'a') as f:
        f.write(json.dumps(info) + '\n')

class LoggingTransistorEnv(TransistorEqualizerEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        log_transistor_episode(info)
        return obs, reward, terminated, truncated, info

def train_transistor(total_timesteps=5000, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor')

    print(f"Training Transistor-level RL agent for {total_timesteps} steps...")
    env = LoggingTransistorEnv()
    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model

if __name__ == '__main__':
    train_transistor(total_timesteps=20000)
