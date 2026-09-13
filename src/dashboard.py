import streamlit as st
import numpy as np
import time
import os

from logging_utils import load_repository
from orchestrator import parse_spec_request

# Try importing FAISS retrieval, fall back to scalar
try:
    from retrieval_faiss import retrieve_closest_faiss as retrieve_closest_fn
    RETRIEVAL_BACKEND = "FAISS (5D vector search)"
except Exception:
    from retrieval import retrieve_closest as retrieve_closest_fn
    RETRIEVAL_BACKEND = "Scalar nearest"

# Try importing ONNX runtime, fall back to PyTorch PPO
ONNX_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_equalizer_fast.onnx')
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

from environment_fast import EqualizerEnvFast

st.set_page_config(page_title="RACD - Circuit Design", page_icon="⚡", layout="centered")

st.title("⚡ RACD — Retrieval-Augmented Circuit Design")
st.caption(f"Backend: **{INFERENCE_BACKEND}** | Retrieval: **{RETRIEVAL_BACKEND}**")

st.write("Automated CTLE equalizer sizing with retrieval-warm-started RL")

mode = st.radio("Input method", ["Slider", "Natural language"])

if mode == "Slider":
    target = st.slider("Target peaking (dB)", 3.0, 11.0, 8.0, step=0.1)
    noise_limit = 1.5
else:
    user_text = st.text_input("Describe what you need:", "moderate boost around 7dB, low noise")
    target = 8.0
    noise_limit = 1.5
    if user_text:
        spec = parse_spec_request(user_text)
        target = spec['target_peaking_db']
        noise_limit = spec.get('noise_limit_mvrms', 1.5)
        st.write(f"🎯 **Parsed target:** {target:.1f} dB (Noise limit: {noise_limit:.2f} mVrms)")

force_spice = st.checkbox("Run full SPICE validation pass (ngspice)", value=False)

if st.button("🚀 Design Circuit", type="primary"):
    t_start = time.perf_counter()
    repository = load_repository()

    query_spec = {'target_peaking_db': target, 'noise_limit_mvrms': noise_limit}
    try:
        retrieved = retrieve_closest_fn(query_spec, repository)
    except Exception:
        from retrieval import retrieve_closest
        retrieved = retrieve_closest(target, repository)

    if retrieved:
        st.info(
            f"🧠 **Retrieval match found:** "
            f"Rs = {retrieved['Rs']:.1f} Ohm, Cs = {retrieved['Cs']*1e12:.2f} pF "
            f"(achieved {retrieved['peaking_db']:.2f} dB)"
        )
    else:
        st.warning("Cold start: no close prior design found in repository.")

    with st.spinner("Generating optimal circuit sizing..."):
        env = EqualizerEnvFast(peaking_target_range=(target, target))
        obs, _ = env.reset()

        t_infer_start = time.perf_counter()
        if USE_ONNX and onnx_predict is not None:
            action = onnx_predict(obs)
        else:
            from stable_baselines3 import PPO
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_equalizer_fast')
            if not os.path.exists(model_path + '.zip'):
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_equalizer')
            model = PPO.load(model_path)
            action, _ = model.predict(obs, deterministic=True)
        t_infer = (time.perf_counter() - t_infer_start) * 1000

        if force_spice:
            import circuit_fast
            circuit_fast.USE_ANALYTICAL = False
            circuit_fast.USE_SPICE_CACHE = False

        obs, reward, _, _, info = env.step(action)
        t_total = (time.perf_counter() - t_start) * 1000

    st.success(f"✅ Design synthesized in **{t_total:.1f} ms** (Policy inference: **{t_infer:.2f} ms**)")

    col1, col2 = st.columns(2)
    col1.metric("Optimal Rs", f"{info['Rs']:.1f} Ohm")
    col1.metric("Optimal Cs", f"{info['Cs']*1e12:.2f} pF")

    peaking_delta = info['peaking_db'] - target
    col2.metric("Achieved Peaking", f"{info['peaking_db']:.2f} dB",
                delta=f"{peaking_delta:+.2f} dB")
    col2.metric("Input-Referred Noise", f"{info['noise_mvrms']:.4f} mVrms")
    st.metric("Eye Height Proxy", f"{info['eye_height_proxy_mv']:.1f} mV")

    # Render AC Frequency Response (Bode Plot)
    import matplotlib.pyplot as plt
    import circuit_fast
    curve_data = circuit_fast.simulate(info['Rs'], info['Cs'], return_curve=True)
    if 'frequency_hz' in curve_data and 'gain_db_curve' in curve_data:
        st.subheader("📈 Frequency Response (Bode Plot)")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        freqs_ghz = np.array(curve_data['frequency_hz']) / 1e9
        ax.semilogx(freqs_ghz, curve_data['gain_db_curve'], color='#0284c7', lw=2.2, label='CTLE Transfer |H(f)|')
        ax.axvline(2.5, color='#e11d48', linestyle='--', lw=1.5, label='Nyquist (2.5 GHz)')
        ax.set_xlabel('Frequency (GHz)', fontsize=9)
        ax.set_ylabel('Gain (dB)', fontsize=9)
        ax.set_title(f'Synthesized CTLE Frequency Peaking: {info["peaking_db"]:.2f} dB (Target: {target:.1f} dB)', fontsize=10)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
