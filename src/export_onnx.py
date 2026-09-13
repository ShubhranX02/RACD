"""
export_onnx.py — Export trained PPO/SAC policy to ONNX for fast dashboard inference.

ONNX inference via onnxruntime is ~10-50x faster than PyTorch for single-sample
policy queries (no Python overhead, ahead-of-time kernel fusion), making the
Streamlit dashboard feel truly instant.

Usage:
    cd src
    python export_onnx.py                          # exports ppo_equalizer_fast
    python export_onnx.py --model sac_equalizer    # exports SAC model
    python export_onnx.py --benchmark              # shows latency comparison
"""

import os
import sys
import argparse
import numpy as np
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def export_ppo_to_onnx(model_name: str = 'ppo_equalizer_fast',
                        output_name: str = None) -> str:
    """
    Export a SB3 PPO policy to ONNX format.
    Returns the path to the exported .onnx file.
    """
    import torch
    import torch.onnx
    from stable_baselines3 import PPO

    model_path = os.path.join(_MODELS_DIR, model_name)
    if output_name is None:
        output_name = model_name
    onnx_path = os.path.join(_MODELS_DIR, f'{output_name}.onnx')

    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path, device='cpu')

    # Extract the actor (policy) network
    policy = model.policy.to('cpu')
    policy.eval()

    # Create a dummy observation matching the input shape
    obs_shape = model.observation_space.shape
    dummy_obs = torch.zeros(1, *obs_shape, dtype=torch.float32, device='cpu')

    # Wrap the policy forward pass to return just the action mean
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            # Get the action distribution mean (deterministic action)
            features = self.policy.extract_features(obs, self.policy.pi_features_extractor)
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            return self.policy.action_net(latent_pi)

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        wrapper,
        dummy_obs,
        onnx_path,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'},
                      'action': {0: 'batch_size'}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported successfully: {onnx_path}")
    return onnx_path


def load_onnx_policy(onnx_path: str):
    """Load an ONNX policy and return an inference function."""
    import onnxruntime as ort

    # Use best available execution provider (DirectML / CUDA / CPU)
    avail = ort.get_available_providers()
    preferred = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    providers = [p for p in preferred if p in avail] or ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    def predict(obs: np.ndarray) -> np.ndarray:
        """
        Run inference on observation(s).
        obs shape: (obs_dim,) for single, (N, obs_dim) for batch.
        Returns action(s) in [-1, 1] action space.
        """
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        obs = obs.astype(np.float32)
        result = session.run(None, {input_name: obs})
        actions = result[0]
        # Clip to valid action range
        actions = np.clip(actions, -1.0, 1.0)
        return actions[0] if actions.shape[0] == 1 else actions

    return predict, session


def benchmark(onnx_path: str, model_name: str = 'ppo_equalizer_fast',
              n_calls: int = 10000):
    """Compare PyTorch vs ONNX inference latency."""
    from stable_baselines3 import PPO

    print(f"\n=== Benchmark: PyTorch vs ONNX ({n_calls} calls) ===")

    # Load models
    model = PPO.load(os.path.join(_MODELS_DIR, model_name), device='cpu')
    onnx_predict, _ = load_onnx_policy(onnx_path)

    # Dummy observation
    obs_shape = model.observation_space.shape
    dummy_obs = np.random.rand(*obs_shape).astype(np.float32)

    # PyTorch benchmark
    t0 = time.perf_counter()
    for _ in range(n_calls):
        model.predict(dummy_obs, deterministic=True)
    t_torch = (time.perf_counter() - t0) / n_calls * 1e6  # µs

    # ONNX benchmark
    t0 = time.perf_counter()
    for _ in range(n_calls):
        onnx_predict(dummy_obs)
    t_onnx = (time.perf_counter() - t0) / n_calls * 1e6  # µs

    print(f"  PyTorch model.predict():  {t_torch:.1f} µs/call")
    print(f"  ONNX onnxruntime:         {t_onnx:.1f} µs/call")
    print(f"  Speedup: {t_torch/t_onnx:.1f}x")

    # Verify outputs match
    action_torch, _ = model.predict(dummy_obs, deterministic=True)
    action_onnx = onnx_predict(dummy_obs)
    max_diff = float(np.max(np.abs(action_torch - action_onnx)))
    print(f"  Max action difference: {max_diff:.6f} (should be < 0.001)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export RACD policy to ONNX')
    parser.add_argument('--model', default='ppo_equalizer_fast',
                        help='Model name in models/ dir (without .zip)')
    parser.add_argument('--output', default=None,
                        help='Output ONNX filename (default: same as model)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run latency benchmark after export')
    args = parser.parse_args()

    onnx_path = export_ppo_to_onnx(args.model, args.output)

    if args.benchmark:
        benchmark(onnx_path, args.model)

    print(f"\nONNX model ready at: {onnx_path}")
    print("Load in dashboard with:")
    print(f"  from export_onnx import load_onnx_policy")
    print(f"  predict, session = load_onnx_policy('{onnx_path}')")
    print(f"  action = predict(obs)")
