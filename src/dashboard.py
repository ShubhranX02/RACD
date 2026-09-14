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

from environment_transistor import TransistorEqualizerEnv, estimate_area_1stage_mm2

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

col_corner, col_spice = st.columns([1, 1])
with col_corner:
    corner_opt = st.selectbox(
        "PVT Evaluation Corner",
        ["TT (Nominal, 27°C, 1.8V)", "SS (Slow, 125°C, 1.71V)", "FF (Fast, 0°C, 1.89V)", "SF (27°C, 1.8V)", "FS (27°C, 1.8V)"]
    )
    selected_corner = corner_opt.split()[0].lower()

with col_spice:
    st.write("") # spacing
    force_spice = st.checkbox(
        "Run full SPICE validation pass (ngspice)",
        value=False,
        help="Default (unchecked): instantaneous surrogate synthesis (<100 ms). Checked: transistor-level ngspice AC + transient pulse sign-off (~25s)."
    )

_CORNER_MAP = {
    'tt': ('tt', 27, 1.8),
    'ss': ('ss', 125, 1.71),
    'ff': ('ff', 0, 1.89),
    'sf': ('sf', 27, 1.8),
    'fs': ('fs', 27, 1.8),
}

if st.button("🚀 Design Circuit", type="primary"):
    t_start = time.perf_counter()
    repository = load_repository("data/transistor_repository.jsonl")

    retrieved = None
    try:
        results = retrieve_k_nearest_transistor(
            target, noise_limit_mvrms=noise_limit,
            corner=selected_corner, k=1
        )
        retrieved = results[0] if results else None
    except Exception:
        try:
            from retrieval import retrieve_closest
            retrieved = retrieve_closest(target, repository)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Show retrieval match and compute how close it is to the target
    # -----------------------------------------------------------------------
    USE_RETRIEVAL_DIRECT = False
    if retrieved:
        wn  = retrieved.get('Wn_um',     retrieved.get('Wn', None))
        rs  = retrieved.get('Rs_ohm',    retrieved.get('Rs', 0))
        cs  = retrieved.get('Cs_farad',  retrieved.get('Cs', 0))
        pk  = retrieved.get('peaking_db', 0)
        retrieval_error = abs(pk - target)
        st.info(
            f"🧠 **Retrieval match found:** "
            f"Wn={wn:.2f}µm, Rs={rs:.1f}Ω, Cs={cs*1e12:.2f}pF "
            f"(achieved {pk:.2f} dB at {selected_corner.upper()}, Δ={retrieval_error:.2f} dB from target)"
        )
        # Use retrieved sizing as warm-start when the match is close enough (< 1 dB)
        if wn is not None and retrieval_error < 1.0:
            USE_RETRIEVAL_DIRECT = True
    else:
        st.warning(f"Cold start: no prior design found in repository for corner {selected_corner.upper()}.")

    with st.spinner("Generating optimal circuit sizing..."):
        t_infer_start = time.perf_counter()

        # Build env with the explicit chosen corner (no random corner roll)
        env = TransistorEqualizerEnv(
            peaking_target_range=(target, target),
            use_surrogate=not force_spice,
            multi_corner=False,
        )
        env.current_corner = _CORNER_MAP.get(selected_corner, ('tt', 27, 1.8))
        obs, _ = env.reset()
        env.current_corner = _CORNER_MAP.get(selected_corner, ('tt', 27, 1.8))
        env.target_peaking_db = float(target)
        obs = env._build_obs()

        def _normalize(val, lo, hi):
            return float(np.clip(2.0 * (val - lo) / (hi - lo) - 1.0, -1.0, 1.0))

        if USE_RETRIEVAL_DIRECT:
            # -----------------------------------------------------------------
            # Fast path: retrieved design is within 1 dB — use as warm-start
            # -----------------------------------------------------------------
            itail = retrieved.get('Itail_half_ua', retrieved.get('Itail', 100.0))
            rl    = retrieved.get('RL_ohm',         retrieved.get('RL',    500.0))
            rdfe  = retrieved.get('Rdfe_ohm',       retrieved.get('Rdfe',  5000.0))
            action = np.array([
                _normalize(wn,    *env.ranges[0]),
                _normalize(rs,    *env.ranges[1]),
                _normalize(cs,    *env.ranges[2]),
                _normalize(itail, *env.ranges[3]),
                _normalize(rl,    *env.ranges[4]),
                _normalize(rdfe,  *env.ranges[5]),
            ], dtype=np.float32)
            init_source = "retrieval"
        else:
            # -----------------------------------------------------------------
            # RL policy path: PPO gives initial action
            # -----------------------------------------------------------------
            if USE_ONNX and onnx_predict is not None:
                action = onnx_predict(obs)
            else:
                from stable_baselines3 import PPO
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor_best')
                if not os.path.exists(model_path + '.zip'):
                    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_transistor')
                model = PPO.load(model_path)
                action, _ = model.predict(obs, deterministic=True)
            init_source = "RL policy"

        t_infer = (time.perf_counter() - t_infer_start) * 1000

        # -----------------------------------------------------------------
        # Refinement Decision:
        # If retrieval is already verified within <= 0.4 dB of target, use it directly!
        # Do NOT let an approximate neural surrogate corrupt a SPICE-verified sizing.
        # Otherwise, run Trust-Region bounded optimization on the surrogate.
        # -----------------------------------------------------------------
        if USE_RETRIEVAL_DIRECT and retrieval_error <= 0.4:
            surrogate_peaking_after = pk
            method_label = f"SPICE-verified retrieval (Δ = {retrieval_error:.2f} dB, exact match)"
        else:
            from scipy.optimize import minimize
            from surrogate_transistor import simulate_transistor_surrogate

            PARAM_LO   = np.array([lo for lo, hi in env.ranges], dtype=np.float64)
            PARAM_HI   = np.array([hi for lo, hi in env.ranges], dtype=np.float64)
            PARAM_SPAN = PARAM_HI - PARAM_LO

            def _action_to_physical(a):
                return PARAM_LO + (np.clip(a, -1.0, 1.0) + 1.0) / 2.0 * PARAM_SPAN

            init_action = action.copy()

            def _refine_objective(a):
                Wn, Rs, Cs, Itail, RL, Rdfe = _action_to_physical(a)
                pred = simulate_transistor_surrogate(float(Wn), float(Rs), float(Cs),
                                                      float(Itail), float(RL), float(Rdfe))
                peaking_err = (pred['peaking_db'] - target) ** 2
                noise_pen = max(0.0, pred['noise_mvrms'] - 1.5) * 20.0
                power_pen = max(0.0, pred['power_mw']    - 15.0) * 10.0
                # Trust penalty: penalize drift away from verified/policy prior
                trust_pen = np.sum((a - init_action) ** 2) * 5.0
                return peaking_err + noise_pen + power_pen + trust_pen

            # Trust-region bounds: allow at most +/- 0.25 normalized step from initial sizing
            trust_bounds = [(max(-1.0, float(a - 0.25)), min(1.0, float(a + 0.25))) for a in init_action]

            t_refine_start = time.perf_counter()
            refine_result = minimize(
                _refine_objective,
                x0=init_action.astype(np.float64),
                method='L-BFGS-B',
                bounds=trust_bounds,
                options={'maxiter': 100, 'ftol': 1e-6},
            )
            t_refine = (time.perf_counter() - t_refine_start) * 1000

            cand_action = np.clip(refine_result.x, -1.0, 1.0).astype(np.float32)
            init_pred = simulate_transistor_surrogate(*[float(v) for v in _action_to_physical(init_action)])
            cand_pred = simulate_transistor_surrogate(*[float(v) for v in _action_to_physical(cand_action)])

            if abs(cand_pred['peaking_db'] - target) < abs(init_pred['peaking_db'] - target):
                action = cand_action
                surrogate_peaking_after = cand_pred['peaking_db']
                method_label = f"{init_source} → trust-region refinement ({refine_result.nit} iters, {t_refine:.1f} ms)"
            else:
                action = init_action
                surrogate_peaking_after = init_pred['peaking_db']
                method_label = f"{init_source} (prior retained, Δ = {abs(init_pred['peaking_db'] - target):.2f} dB)"

        obs, reward, _, _, info = env.step(action)

        # When using a verified retrieval match directly in fast mode (no ngspice),
        # preserve the ground-truth SPICE metrics from the repository instead of
        # overwriting them with the surrogate's approximation.
        if USE_RETRIEVAL_DIRECT and retrieval_error <= 0.4 and not force_spice and retrieved:
            info['peaking_db']          = float(retrieved.get('peaking_db', info['peaking_db']))
            info['noise_mvrms']         = float(retrieved.get('noise_mvrms', info['noise_mvrms']))
            info['power_mw']            = float(retrieved.get('power_mw', info['power_mw']))
            info['eye_height_proxy_mv'] = float(retrieved.get('eye_height_proxy_mv', info['eye_height_proxy_mv']))
            info['eye_width_ui']        = float(retrieved.get('eye_width_ui', info['eye_width_ui']))

    t_total = (time.perf_counter() - t_start) * 1000

    st.success(
        f"✅ Design synthesized in **{t_total:.1f} ms** "
        f"(Inference: **{t_infer:.2f} ms** · source: *{method_label}*)"
    )
    st.caption(
        f"Surrogate prediction after refinement: **{surrogate_peaking_after:.2f} dB** "
        f"(target {target:.1f} dB, Δ = {surrogate_peaking_after - target:+.2f} dB)"
    )

    st.subheader("Selected Component Sizing")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Width (Wn)", f"{info['Wn']:.2f} µm")
    c2.metric("Rs", f"{info['Rs']:.1f} Ω")
    c3.metric("Cs", f"{info['Cs']*1e12:.2f} pF")
    c4.metric("Tail Current", f"{info['Itail']:.1f} µA")
    c5.metric("Load (RL)", f"{info['RL']:.1f} Ω")
    c6.metric("DFE (Rdfe)", f"{info['Rdfe']/1000:.1f} kΩ")

    st.subheader("Simulated Performance Metrics")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

    peaking_error = info['peaking_db'] - target
    peaking_pass = abs(peaking_error) <= 0.5
    if peaking_pass:
        pk_color = "normal" if peaking_error >= 0 else "inverse"
    else:
        pk_color = "inverse" if peaking_error >= 0 else "normal"
    m1.metric(
        f"Peaking (Target {target:.1f})",
        f"{info['peaking_db']:.2f} dB",
        delta=f"{peaking_error:+.2f} dB error",
        delta_color=pk_color,
    )

    noise_color = "normal" if info['noise_mvrms'] < 1.5 else "inverse"
    m2.metric("Noise (< 1.5)", f"{info['noise_mvrms']:.2f} mVrms", delta=f"{info['noise_mvrms'] - 1.5:.2f}", delta_color=noise_color)

    hd3_color = "normal" if info['hd3_db'] < -30 else "inverse"
    m3.metric("HD3 (< -30)", f"{info['hd3_db']:.1f} dB", delta=f"{info['hd3_db'] - (-30):.1f}", delta_color=hd3_color)

    pwr_color = "normal" if info['power_mw'] < 15 else "inverse"
    m4.metric("Power (< 15)", f"{info['power_mw']:.2f} mW", delta=f"{info['power_mw'] - 15:.2f}", delta_color=pwr_color)

    eye_h_color = "normal" if info['eye_height_proxy_mv'] > 100 else "inverse"
    m5.metric("Eye-H (> 100 mV)", f"{info['eye_height_proxy_mv']:.1f} mV", delta=f"{info['eye_height_proxy_mv'] - 100:.1f}", delta_color=eye_h_color)

    # Eye width: spec > 0.4 UI (horizontal opening at 5 Gbps NRZ)
    eye_w = info.get('eye_width_ui', 0.0)
    eye_w_color = "normal" if eye_w > 0.4 else "inverse"
    eye_w_str = f"{eye_w:.3f} UI" if eye_w > 0 else "N/A (surrogate)"
    m6.metric("Eye-W (> 0.4 UI)", eye_w_str, delta=f"{eye_w - 0.4:.3f}" if eye_w > 0 else None, delta_color=eye_w_color)

    # Area — 1-stage estimator matches spec topology
    area_mm2 = estimate_area_1stage_mm2(info['Wn'], info['Rs'], info['Cs'], info['RL'], info['Rdfe'])
    area_color = "normal" if area_mm2 < 0.05 else "inverse"
    m7.metric("Area (< 0.05)", f"{area_mm2:.4f} mm²", delta=f"{area_mm2 - 0.05:.4f}", delta_color=area_color)

