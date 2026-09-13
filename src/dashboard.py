import streamlit as st
from stable_baselines3 import PPO
from environment import EqualizerEnv
from logging_utils import load_repository
from retrieval import retrieve_closest
from orchestrator import parse_spec_request
import os

st.title("RACD — Retrieval-Augmented Circuit Design")
st.write("Automated equalizer sizing with retrieval-warm-started RL")

mode = st.radio("Input method", ["Slider", "Natural language"])

if mode == "Slider":
    target = st.slider("Target peaking (dB)", 3.0, 11.0, 8.0)
else:
    user_text = st.text_input("Describe what you need:", "moderate boost, low noise")
    target = 8.0
    if user_text:
        spec = parse_spec_request(user_text)
        target = spec['target_peaking_db']
        st.write(f"Parsed target: {target:.1f} dB")

if st.button("Design Circuit"):
    repository = load_repository()
    retrieved = retrieve_closest(target, repository)

    if retrieved:
        st.info(
            f"Warm-starting from a similar past design: "
            f"Rs={retrieved['Rs']:.1f}Ω, Cs={retrieved['Cs']*1e12:.2f}pF "
            f"(previously achieved {retrieved['peaking_db']:.2f} dB)"
        )
    else:
        st.warning("No similar past design found — cold start")

    with st.spinner("Evaluating..."):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_equalizer')
        model = PPO.load(model_path)
        env = EqualizerEnv(peaking_target_range=(target, target))
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(action)

    st.success("Design complete")
    col1, col2 = st.columns(2)
    col1.metric("Rs", f"{info['Rs']:.1f} Ω")
    col1.metric("Cs", f"{info['Cs']*1e12:.2f} pF")
    col2.metric("Achieved peaking", f"{info['peaking_db']:.2f} dB",
                delta=f"{info['peaking_db'] - target:.2f}")
    col2.metric("Noise", f"{info['noise_mvrms']:.4f} mVrms")
    st.metric("Eye height proxy", f"{info['eye_height_proxy_mv']:.1f} mV")
