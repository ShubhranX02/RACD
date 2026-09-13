import numpy as np
from src.environment_transistor import TransistorEqualizerEnv
from src.export_onnx import load_onnx_policy
import os

ONNX_PATH = "models/ppo_transistor.onnx"
if not os.path.exists(ONNX_PATH):
    print("ONNX model not found!")
    exit(1)

onnx_predict, _ = load_onnx_policy(ONNX_PATH)
env = TransistorEqualizerEnv()

targets = np.linspace(4.0, 10.0, 10)
errors = []
violations = 0

print("Testing Model Convergence across Target Peaking Range (4.0 dB to 10.0 dB):")
print("-" * 75)
print(f"{'Target (dB)':<15} | {'Achieved (dB)':<15} | {'Error (dB)':<15} | {'Constraints Met?'}")
print("-" * 75)

for target in targets:
    env.peaking_target_range = (target, target)
    obs, _ = env.reset()
    action = onnx_predict(obs)
    obs, reward, _, _, info = env.step(action)
    
    error = abs(info['peaking_db'] - target)
    errors.append(error)
    
    constraints_met = (
        info['power_mw'] < 15.0 and 
        info['hd3_db'] < -30.0 and 
        info['noise_mvrms'] < 1.5 and 
        info['eye_height_proxy_mv'] > 100.0
    )
    if not constraints_met:
        violations += 1
        
    print(f"{target:<15.2f} | {info['peaking_db']:<15.2f} | {error:<15.2f} | {'Yes' if constraints_met else 'No'}")

print("-" * 75)
print(f"Mean Absolute Error: {np.mean(errors):.3f} dB")
print(f"Total Constraint Violations: {violations} / 10")
