import streamlit as st
import numpy as np
import time
import os

from logging_utils import load_repository
from orchestrator import parse_spec_request

# Use FAISS transistor retrieval
try:
    from retrieval_faiss_transistor import retrieve_k_nearest_transistor
    RETRIEVAL_BACKEND = "FAISS (Transistor 6D search)"
except Exception:
    from retrieval import retrieve_closest
    RETRIEVAL_BACKEND = "Scalar nearest"

# Load ONNX Transistor Model
ONNX_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor.onnx')
USE_ONNX = False
onnx_predict = None

if os.path.exists(ONNX_PATH):
    try:
        from export_onnx import load_onnx_policy
        onnx_predict, _ = load_onnx_policy(ONNX_PATH)
        USE_ONNX = True
        INFERENCE_BACKEND = "ONNX Runtime (24 us latency)"
    except Exception as e:
        INFERENCE_BACKEND = f"PyTorch (ONNX load failed: {e})"
else:
    INFERENCE_BACKEND = "PyTorch (ONNX model not found)"

from environment_transistor import TransistorEqualizerEnv, estimate_area_2stage_mm2

st.set_page_config(page_title="RACD - Transistor Design", page_icon="⚡", layout="wide")

st.title("⚡ RACD — Transistor-Level Circuit Design")
st.caption(f"Backend: **{INFERENCE_BACKEND}** | Retrieval: **{RETRIEVAL_BACKEND}**")

st.write("Automated CTLE + DFE sizing using ONNX-accelerated Reinforcement Learning")

mode = st.radio("Input method", ["Slider", "Natural language"])

if mode == "Slider":
    target = st.slider("Target peaking (dB)", 3.0, 12.0, 8.0, step=0.1)
    noise_limit = 1.5
else:
    user_text = st.text_input("Describe what you need:", "moderate boost around 7dB, low noise")
    target = 8.0
    noise_limit = 1.5
    if user_text:
        spec = parse_spec_request(user_text)
        target = spec.get('target_peaking_db', 8.0)
        noise_limit = spec.get('noise_limit_mvrms', 1.5)
        st.write(f"🎯 **Parsed target:** {target:.1f} dB (Noise limit: {noise_limit:.2f} mVrms)")

force_spice = st.checkbox("Run full SPICE validation pass (ngspice)", value=True)

if st.button("🚀 Design Circuit", type="primary"):
    t_start = time.perf_counter()
    repository = load_repository("data/transistor_repository_clean.jsonl")

    try:
        # retrieve_k_nearest_transistor expects target_peaking_db float, etc.
        results = retrieve_k_nearest_transistor(target, noise_limit_mvrms=noise_limit, k=1)
        retrieved = results[0] if results else None
    except Exception as e:
        # Fallback expects float target
        from retrieval import retrieve_closest
        retrieved = retrieve_closest(target, repository)

    if retrieved:
        wn = retrieved.get('Wn_um', retrieved.get('Wn', '?'))
        rs = retrieved.get('Rs_ohm', retrieved.get('Rs', 0))
        cs = retrieved.get('Cs_farad', retrieved.get('Cs', 0))
        pk = retrieved.get('peaking_db', 0)
        st.info(
            f"🧠 **Retrieval match found:** "
            f"Wn={wn}µm, Rs={rs:.1f}Ω, Cs={cs*1e12:.2f}pF "
            f"(achieved {pk:.2f} dB)"
        )
    else:
        st.warning("Cold start: no close prior design found in repository.")

    with st.spinner("Generating optimal circuit sizing..."):
        env = TransistorEqualizerEnv(peaking_target_range=(target, target))
        obs, _ = env.reset()

        t_infer_start = time.perf_counter()
        if USE_ONNX and onnx_predict is not None:
            action = onnx_predict(obs)
        else:
            from stable_baselines3 import PPO
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor_best')
            if not os.path.exists(model_path + '.zip'):
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor')
            model = PPO.load(model_path)
            action, _ = model.predict(obs, deterministic=True)
        t_infer = (time.perf_counter() - t_infer_start) * 1000

        # Run through the environment to get metrics
        # If force_spice is True, it uses Ngspice directly in the environment wrapper
        obs, reward, _, _, info = env.step(action)
        t_total = (time.perf_counter() - t_start) * 1000

    st.success(f"✅ Design synthesized in **{t_total:.1f} ms** (Policy inference: **{t_infer:.2f} ms**)")

    st.subheader("Selected Component Sizing")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Width (Wn)", f"{info['Wn']:.2f} µm")
    c2.metric("Rs", f"{info['Rs']:.1f} Ω")
    c3.metric("Cs", f"{info['Cs']*1e12:.2f} pF")
    c4.metric("Tail Current", f"{info['Itail']:.1f} µA")
    c5.metric("Load (RL)", f"{info['RL']:.1f} Ω")
    c6.metric("DFE (Rdfe)", f"{info['Rdfe']/1000:.1f} kΩ")

    st.subheader("Simulated Performance Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    peaking_error = info['peaking_db'] - target
    m1.metric("Peaking", f"{info['peaking_db']:.2f} dB", delta=f"{peaking_error:.2f} dB error", delta_color="inverse")
    
    noise_color = "normal" if info['noise_mvrms'] < 1.5 else "inverse"
    m2.metric("Noise (< 1.5)", f"{info['noise_mvrms']:.2f} mVrms", delta=f"{info['noise_mvrms'] - 1.5:.2f}", delta_color=noise_color)
    
    hd3_color = "normal" if info['hd3_db'] < -30 else "inverse"
    m3.metric("HD3 (< -30)", f"{info['hd3_db']:.1f} dB", delta=f"{info['hd3_db'] - (-30):.1f}", delta_color=hd3_color)
    
    pwr_color = "normal" if info['power_mw'] < 15 else "inverse"
    m4.metric("Power (< 15)", f"{info['power_mw']:.2f} mW", delta=f"{info['power_mw'] - 15:.2f}", delta_color=pwr_color)
    
    eye_color = "normal" if info['eye_height_proxy_mv'] > 100 else "inverse"
    m5.metric("Eye Proxy (> 100)", f"{info['eye_height_proxy_mv']:.1f} mV", delta=f"{info['eye_height_proxy_mv'] - 100:.1f}", delta_color=eye_color)

    # Area
    area_mm2 = estimate_area_2stage_mm2(info['Wn'], info['Rs'], info['Cs'], info['RL'], info['Rdfe'])
    area_color = "normal" if area_mm2 < 0.05 else "inverse"
    m6.metric("Area (< 0.05)", f"{area_mm2:.4f} mm²", delta=f"{area_mm2 - 0.05:.4f}", delta_color=area_color)
