import numpy as np
from stable_baselines3 import PPO
try:
    from environment_fast import EqualizerEnvFast as EqualizerEnv
except ImportError:
    from environment import EqualizerEnv
from logging_utils import load_repository
from retrieval import retrieve_k_nearest, rs_cs_to_action
from warmstart import pretrain_policy_toward_retrieved


def compare_warmstart_vs_random(target_peaking_db, repository, timesteps=2000):
    env_config = dict(rs_range=(50, 500), cs_range=(0.1e-12, 10e-12))

    # Condition A: network-level warm start via behavior cloning
    retrieved = retrieve_k_nearest(target_peaking_db, repository, k=10)
    warm_env = EqualizerEnv(peaking_target_range=(target_peaking_db, target_peaking_db), **env_config)
    warm_model = PPO('MlpPolicy', warm_env, verbose=0)
    if retrieved:
        examples = [
            {'target_peaking_db': ep['target_peaking_db'],
             'action': rs_cs_to_action(ep['Rs'], ep['Cs'], env_config['rs_range'], env_config['cs_range'])}
            for ep in retrieved
        ]
        warm_model = pretrain_policy_toward_retrieved(warm_model, examples)
    warm_model.learn(total_timesteps=timesteps)

    # Condition B: random-init, no retrieval, no pretraining
    cold_env = EqualizerEnv(peaking_target_range=(target_peaking_db, target_peaking_db), **env_config)
    cold_model = PPO('MlpPolicy', cold_env, verbose=0)
    cold_model.learn(total_timesteps=timesteps)

    def evaluate(model, env, n=20):
        rewards = []
        for _ in range(n):
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        return float(np.mean(rewards))

    warm_reward = evaluate(warm_model, warm_env)
    cold_reward = evaluate(cold_model, cold_env)
    print(f"target={target_peaking_db:.1f}  warm-started={warm_reward:.4f}  random-init={cold_reward:.4f}")
    return warm_reward, cold_reward


if __name__ == '__main__':
    repository = load_repository()
    print("Running warm-start vs random-init comparison across multiple targets...")
    results = []
    for target in [4.0, 6.0, 8.0, 10.0, 11.0]:
        results.append((target, *compare_warmstart_vs_random(target, repository)))

    print("\n--- Summary ---")
    for target, warm, cold in results:
        improvement = warm - cold
        print(f"target={target:5.1f}  warm={warm:8.4f}  cold={cold:8.4f}  "
              f"improvement={improvement:+.4f} {'(better)' if improvement > 0 else '(worse)'}")
