from stable_baselines3 import PPO
from environment import EqualizerEnv
from logging_utils import log_episode, load_repository
from retrieval import retrieve_k_nearest, rs_cs_to_action
from warmstart import pretrain_policy_toward_retrieved
import os


class LoggingEqualizerEnv(EqualizerEnv):
    """Identical to EqualizerEnv, but logs every episode to the repository
    -- this is what actually populates data/circuit_repository.jsonl for
    future retrieval."""
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        log_episode(info)
        return obs, reward, terminated, truncated, info


def train(total_timesteps=20000, use_warmstart=True, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_equalizer')

    env = LoggingEqualizerEnv()
    model = PPO('MlpPolicy', env, verbose=1)

    if use_warmstart:
        repository = load_repository()
        mid_target = sum(env.peaking_target_range) / 2
        retrieved = retrieve_k_nearest(mid_target, repository, k=10)
        if retrieved:
            examples = [
                {'target_peaking_db': ep['target_peaking_db'],
                 'action': rs_cs_to_action(ep['Rs'], ep['Cs'], env.rs_range, env.cs_range)}
                for ep in retrieved
            ]
            print(f"Warm-starting with {len(examples)} retrieved examples")
            model = pretrain_policy_toward_retrieved(model, examples)
        else:
            print("No retrieved examples found -- cold start (this is expected on the very first run)")

    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model


if __name__ == '__main__':
    train()
