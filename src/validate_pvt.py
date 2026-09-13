import os
from circuit_transistor import simulate_transistor_level_full as simulate_transistor_level
from stable_baselines3 import PPO
from environment_transistor import TransistorEqualizerEnv

def run_pvt_validation(target_peaking_db=8.0):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor')
    if not os.path.exists(model_path + '.zip'):
        print(f"Model not found at {model_path}. Please run train_transistor.py first.")
        return

    model = PPO.load(model_path)
    env = TransistorEqualizerEnv(peaking_target_range=(target_peaking_db, target_peaking_db))
    
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    
    # Rescale action to get raw parameters
    Wn, Rs, Cs, Itail, RL, Rdfe = env._rescale_action(action)
    print(f"Candidate Design (Target: {target_peaking_db} dB):")
    print(f"Wn={Wn:.2f}u, Rs={Rs:.1f}Ω, Cs={Cs*1e12:.2f}pF, Itail={Itail:.1f}uA, RL={RL:.1f}Ω, Rdfe={Rdfe:.1f}Ω")
    print("\n--- PVT Validation Matrix ---")

    corners = ['tt', 'ss', 'ff', 'sf', 'fs']
    volts = [1.71, 1.8, 1.89] # +/- 5%
    temps = [0, 27, 125]

    for c in corners:
        for v in volts:
            for t in temps:
                res = simulate_transistor_level(Wn, Rs, Cs, Itail, RL, Rdfe, corner=c, temp=t, vdd=v)
                print(f"Corner: {c:2s}, VDD: {v:.2f}V, Temp: {t:3d}C -> Peaking: {res['peaking_db']:5.2f} dB, Noise: {res['noise_mvrms']:4.2f} mVrms, HD3: {res['hd3_db']:6.1f} dB, Power: {res['power_mw']:5.2f} mW, Eye: {res['eye_height_proxy_mv']:5.1f} mV")

if __name__ == '__main__':
    run_pvt_validation(target_peaking_db=8.0)
