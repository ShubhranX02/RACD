"""
circuit_transistor.py — SkyWater 130 nm Transistor-Level Continuous-Time Linear Equalizer (CTLE)
and 1-Tap Decision Feedback Equalizer (DFE) Simulation Engine.

Supports:
- Topology A: Single-stage degenerated differential pair CTLE + 1-Tap DFE
- Topology B: Cascaded 2-stage CTLE (Stage 1 + Stage 2 booster) + 1-Tap DFE
- Fast AC+OP+Noise evaluation mode for RL training
- Full multi-corner transient Fourier (HD3) and pulse-response (Eye opening) validation
"""

import subprocess
import re
import numpy as np
import os

# Ensure ngspice is discoverable if installed in standard location
if os.path.exists(r"C:\Spice64\bin") and r"C:\Spice64\bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = r"C:\Spice64\bin;" + os.environ.get("PATH", "")

_DEFAULT_PDK_PATH = os.path.expanduser("~/.ciel/ciel/sky130/versions/bdc9412b3e468c102d01b7cf6337be06ec6e9c9a/sky130A/libs.tech/ngspice/sky130.lib.spice").replace('\\', '/')
PDK_LIB_PATH = os.environ.get("PDK_LIB_PATH", _DEFAULT_PDK_PATH)
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_TEMP_DIR = os.path.join(_SRC_DIR, '.tmp')
os.makedirs(_TEMP_DIR, exist_ok=True)  # create .tmp/ if it doesn't exist yet


def _run_ngspice(netlist_content, label="sim"):
    temp_file = os.path.join(_TEMP_DIR, f'_temp_{label}.cir')
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(netlist_content)
    result = subprocess.run(['ngspice', '-b', temp_file], capture_output=True, text=True)
    return result


def _read_wrdata(filepath):
    """Read a wrdata output file into (col0, col1) numpy arrays."""
    xs, ys = [], []
    if not os.path.exists(filepath):
        for suf in ['.data', '']:
            if os.path.exists(filepath + suf):
                filepath = filepath + suf
                break
    if not os.path.exists(filepath):
        return np.array([]), np.array([])
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and not line.startswith('#'):
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(xs), np.array(ys)


def _device_section_2stage(Wn1_um, Rs1_ohm, Cs1_farad, Itail1_half_ua, RL1_ohm,
                           Wn2_um, Rs2_ohm, Cs2_farad, Itail2_half_ua, RL2_ohm,
                           Rdfe_ohm, vdd=1.8):
    """
    Cascaded 2-Stage CTLE + 1-Tap DFE subcircuit:
    - Stage 1: Zero-boosting degenerated differential pair
    - Stage 2: Secondary gain/peaking booster driving DFE summing node
    - AC coupling with self-bias to maintain optimal overdrive across PVT
    """
    return f"""
* === Stage 1 CTLE ===
XM1 vm1_n vin_p s1 0 sky130_fd_pr__nfet_01v8 w={Wn1_um} l=0.15
XM2 vm1_p vin_n s2 0 sky130_fd_pr__nfet_01v8 w={Wn1_um} l=0.15
Rs1 s1 s2 {Rs1_ohm}
Cs1 s1 s2 {Cs1_farad}
Itail1_1 s1 0 DC {Itail1_half_ua}u
Itail1_2 s2 0 DC {Itail1_half_ua}u
RL1_1 vdd vm1_n {RL1_ohm}
RL1_2 vdd vm1_p {RL1_ohm}

* === AC Coupling & Biasing ===
Cac1 vm1_n vg2_n 5p
Cac2 vm1_p vg2_p 5p
Rbias1 vdd_half vg2_n 50k
Rbias2 vdd_half vg2_p 50k
Vmid vdd_half 0 {vdd/2}

* === Stage 2 Booster ===
XM3 voutn vg2_p s3 0 sky130_fd_pr__nfet_01v8 w={Wn2_um} l=0.15
XM4 voutp vg2_n s4 0 sky130_fd_pr__nfet_01v8 w={Wn2_um} l=0.15
Rs2 s3 s4 {Rs2_ohm}
Cs2 s3 s4 {Cs2_farad}
Itail2_1 s3 0 DC {Itail2_half_ua}u
Itail2_2 s4 0 DC {Itail2_half_ua}u
RL2_1 vdd voutn {RL2_ohm}
RL2_2 vdd voutp {RL2_ohm}

* === 1-Tap DFE ===
Bslice_p vslice_p 0 V = V(voutp,voutn) > 0 ? 0.1 : -0.1
Bslice_n vslice_n 0 V = V(voutp,voutn) < 0 ? 0.1 : -0.1
Tdelay_p vslice_p 0 vdelayed_p 0 Z0=50 TD=200p
Tdelay_n vslice_n 0 vdelayed_n 0 Z0=50 TD=200p
Rdfe1 vdelayed_p voutp {Rdfe_ohm}
Rdfe2 vdelayed_n voutn {Rdfe_ohm}
"""


def _device_section_1stage(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm):
    """Legacy single-stage CTLE subcircuit."""
    return f"""
XM1 voutn vin_p s1 0 sky130_fd_pr__nfet_01v8 w={Wn_um} l=0.15
XM2 voutp vin_n s2 0 sky130_fd_pr__nfet_01v8 w={Wn_um} l=0.15
Rs s1 s2 {Rs_ohm}
Cs s1 s2 {Cs_farad}
Itail1 s1 0 DC {Itail_half_ua}u
Itail2 s2 0 DC {Itail_half_ua}u
RL1 vdd voutn {RL_ohm}
RL2 vdd voutp {RL_ohm}
Bslice_p vslice_p 0 V = V(voutp,voutn) > 0 ? 0.1 : -0.1
Bslice_n vslice_n 0 V = V(voutp,voutn) < 0 ? 0.1 : -0.1
Tdelay_p vslice_p 0 vdelayed_p 0 Z0=50 TD=200p
Tdelay_n vslice_n 0 vdelayed_n 0 Z0=50 TD=200p
Rdfe1 vdelayed_p voutp {Rdfe_ohm}
Rdfe2 vdelayed_n voutn {Rdfe_ohm}
"""


def simulate_transistor_fast(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm=2000, Rdfe_ohm=10000,
                             corner='tt', temp=27, vdd=1.8, topology='1stage', worker_id=None,
                             compute_hd3=False):
    """
    Fast simulation for RL training loop: AC + OP + Noise + (optional) HD3 + eye transient.

    Args:
        worker_id:   Optional integer (e.g. os.getpid()) that makes temp file names unique
                     per process. Required when calling from multiprocessing workers to
                     prevent concurrent writes to the same _temp_ac_data file.
        compute_hd3: If True, run an additional transient Fourier simulation to obtain a
                     real HD3 value (adds ~50 ms).  Set True during surrogate training data
                     generation; leave False (default) during RL rollouts for speed.

    Returns eye_height_proxy_mv via a 2 ns pulse transient (replaces non-physical AC-gain
    proxy), plus eye_width_ui (horizontal opening at 5 Gbps, spec: > 0.4 UI).
    """
    if topology == '2stage':
        devices = _device_section_2stage(
            Wn1_um=Wn_um, Rs1_ohm=Rs_ohm, Cs1_farad=Cs_farad, Itail1_half_ua=Itail_half_ua, RL1_ohm=RL_ohm,
            Wn2_um=Wn_um, Rs2_ohm=Rs_ohm, Cs2_farad=Cs_farad, Itail2_half_ua=Itail_half_ua, RL2_ohm=RL_ohm,
            Rdfe_ohm=Rdfe_ohm, vdd=vdd
        )
    else:
        devices = _device_section_1stage(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm)

    # Use process-unique temp file names when worker_id is provided (multiprocessing safety)
    suffix = f"_{worker_id}" if worker_id is not None else ""
    ac_data_path  = os.path.join(_TEMP_DIR, f'_temp_ac_data{suffix}')
    eye_data_path = os.path.join(_TEMP_DIR, f'_temp_eye_data{suffix}')
    label = f'fast_train{suffix}'

    # -------------------------------------------------------------------------
    # Single-Pass SPICE Execution: AC + OP + Noise + Eye Transient
    # Combines both runs into one ngspice process to avoid reloading the huge
    # Sky130 PDK library twice (cuts simulation time in half).
    # -------------------------------------------------------------------------
    UI_s = 200e-12
    netlist = f"""Combined Single-Pass SPICE Simulation
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp} scale=1u nomod nopage method=gear

Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} AC 0.5 PULSE({vdd/2} {vdd/2 + 0.1} 0 10p 10p 200p 5n)
Vinn vin_n 0 DC {vdd/2} AC -0.5 PULSE({vdd/2} {vdd/2 - 0.1} 0 10p 10p 200p 5n)
{devices}

.control
ac dec 10 10meg 10g
wrdata {ac_data_path} vdb(voutp,voutn)
op
print i(Vdd)
noise v(voutp,voutn) vinp dec 10 10meg 5g
print inoise_total
tran 10p 1n
wrdata {eye_data_path} v(voutp,voutn)
.endc
.end
"""
    res = _run_ngspice(netlist, label)
    freqs, gains = _read_wrdata(ac_data_path)
    if len(freqs) == 0:
        raise ValueError(f"Failed to parse AC data:\n{res.stdout}\n{res.stderr}")

    lf_idx = (np.abs(freqs - 100e6)).argmin()
    hf_idx = (np.abs(freqs - 2.5e9)).argmin()
    low_freq_gain  = float(gains[lf_idx])
    high_freq_gain = float(gains[hf_idx])
    peaking_db = high_freq_gain - low_freq_gain

    match_dc = re.search(r'i\(vdd\)\s*=\s*([\d.eE+-]+)', res.stdout, re.IGNORECASE)
    power_mw = abs(float(match_dc.group(1))) * vdd * 1000 if match_dc else 100.0

    match_n = re.search(r'inoise_total\s*=\s*([\d.eE+-]+)', res.stdout, re.IGNORECASE)
    noise_mvrms = float(match_n.group(1)) * 1000 if match_n else 10.0

    t_arr, v_arr = _read_wrdata(eye_data_path)

    if len(v_arr) > 4:
        eye_height_mv = float(np.max(np.abs(v_arr))) * 1000.0
        # Eye width: fraction of time waveform magnitude exceeds 10% of peak
        peak_v = float(np.max(np.abs(v_arr)))
        threshold = 0.1 * peak_v
        above = np.abs(v_arr) > threshold
        # Skip first 25% of waveform (settling transient)
        t_start_idx = max(1, int(len(t_arr) * 0.25))
        above_win = above[t_start_idx:]
        t_win     = t_arr[t_start_idx:]
        if above_win.any() and (~above_win).any():
            rising_idx  = int(np.argmax(above_win))
            falling_idx = int(len(above_win) - 1 - np.argmax(above_win[::-1]))
            t_open  = float(t_win[rising_idx])
            t_close = float(t_win[falling_idx])
            eye_width_ui = max(0.0, (t_close - t_open) / UI_s)
        else:
            eye_width_ui = 0.0
    else:
        # Transient failed — fall back to AC-gain proxy and mark width as unknown
        eye_height_mv = 10**(high_freq_gain / 20.0) * 100.0
        eye_width_ui  = 0.0

    # -------------------------------------------------------------------------
    # Run 3 (optional): Transient Fourier HD3
    # compute_hd3=False during RL rollouts (surrogate predicts HD3).
    # compute_hd3=True during surrogate training data generation (real values).
    # -------------------------------------------------------------------------
    if compute_hd3:
        hd3_data_path = os.path.join(_TEMP_DIR, f'_temp_hd3_data{suffix}')
        netlist_hd3 = f"""HD3 Transient Fourier
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp} scale=1u

Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} SINE({vdd/2} 0.05 100Meg)
Vinn vin_n 0 DC {vdd/2} SINE({vdd/2} -0.05 100Meg)
{devices}
.control
tran 10p 50n
fourier 100Meg v(voutp,voutn)
.endc
.end
"""
        res_hd3 = _run_ngspice(netlist_hd3, f'hd3{suffix}')
        # Match Harmonic 3 normalised magnitude from ngspice Fourier table:
        # Format: "3  3e+08  mag  phase  norm_mag  norm_phase"
        match_hd3 = re.search(
            r'^\s*3\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)',
            res_hd3.stdout, re.MULTILINE
        )
        if not match_hd3:
            match_hd3 = re.search(r'3\s+3e\+08\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)', res_hd3.stdout)
        if match_hd3:
            hd3_norm = float(match_hd3.group(1))
            hd3_db = float(20 * np.log10(max(hd3_norm, 1e-12)))
        else:
            hd3_db = -38.0  # Fourier parse failed — conservative proxy
    else:
        hd3_db = -38.0  # Fast path: proxy (surrogate predicts real HD3 during RL rollouts)

    return {
        'peaking_db':          float(peaking_db),
        'low_freq_gain_db':    float(low_freq_gain),
        'high_freq_gain_db':   float(high_freq_gain),
        'noise_mvrms':         float(noise_mvrms),
        'hd3_db':              float(hd3_db),
        'power_mw':            float(power_mw),
        'eye_height_proxy_mv': float(eye_height_mv),
        'eye_width_ui':        float(eye_width_ui),
        'corner':              corner,
        'temp':                temp,
        'vdd':                 vdd,
    }


def simulate_transistor_level(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm=2000, Rdfe_ohm=10000,
                             corner='tt', temp=27, vdd=1.8, topology='1stage'):
    """
    Full comprehensive multi-mode simulation:
    - Run 1: AC + OP + Noise
    - Run 2: Transient Fourier HD3 analysis
    - Run 3: Transient pulse-response eye-height estimation
    """
    if topology == '2stage':
        devices = _device_section_2stage(
            Wn1_um=Wn_um, Rs1_ohm=Rs_ohm, Cs1_farad=Cs_farad, Itail1_half_ua=Itail_half_ua, RL1_ohm=RL_ohm,
            Wn2_um=Wn_um, Rs2_ohm=Rs_ohm, Cs2_farad=Cs_farad, Itail2_half_ua=Itail_half_ua, RL2_ohm=RL_ohm,
            Rdfe_ohm=Rdfe_ohm, vdd=vdd
        )
    else:
        devices = _device_section_1stage(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm)

    ac_data_path = os.path.join(_TEMP_DIR, '_temp_ac_data')

    # Run 1: AC + OP + Noise
    netlist1 = f"""Combined AC/OP/Noise
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp} scale=1u

Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} AC 0.5
Vinn vin_n 0 DC {vdd/2} AC -0.5
{devices}
.control
ac dec 10 10meg 10g
wrdata {ac_data_path} vdb(voutp,voutn)
op
print i(Vdd)
noise v(voutp,voutn) vinp dec 10 10meg 5g
print inoise_total
.endc
.end
"""
    res1 = _run_ngspice(netlist1, 'full_ac')
    freqs, gains = _read_wrdata(ac_data_path)
    if len(freqs) == 0:
        raise ValueError(f"Failed to parse AC data:\n{res1.stdout}\n{res1.stderr}")

    lf_idx = (np.abs(freqs - 100e6)).argmin()
    hf_idx = (np.abs(freqs - 2.5e9)).argmin()
    low_freq_gain = float(gains[lf_idx])
    high_freq_gain = float(gains[hf_idx])
    peaking_db = high_freq_gain - low_freq_gain

    match_dc = re.search(r'i\(vdd\)\s*=\s*([\d.eE+-]+)', res1.stdout, re.IGNORECASE)
    power_mw = abs(float(match_dc.group(1))) * vdd * 1000 if match_dc else 100.0

    match_n = re.search(r'inoise_total\s*=\s*([\d.eE+-]+)', res1.stdout, re.IGNORECASE)
    noise_mvrms = float(match_n.group(1)) * 1000 if match_n else 10.0

    # Run 2: HD3 Transient Fourier
    netlist2 = f"""HD3 Transient
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp} scale=1u

Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} SINE({vdd/2} 0.05 100Meg)
Vinn vin_n 0 DC {vdd/2} SINE({vdd/2} -0.05 100Meg)
{devices}
.control
tran 10p 50n
fourier 100Meg v(voutp,voutn)
.endc
.end
"""
    res2 = _run_ngspice(netlist2, 'full_hd3')
    # Match Harmonic 3 from Fourier table: "3   3e+08   mag   phase   norm_mag   norm_phase"
    match_hd3 = re.search(r'^\s*3\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)', res2.stdout, re.MULTILINE)
    if not match_hd3:
        # Fallback to any line starting with 3 followed by 4 numerical columns
        match_hd3 = re.search(r'3\s+3e\+08\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)', res2.stdout)
    
    hd3_norm = float(match_hd3.group(1)) if match_hd3 else 0.03
    hd3_db = 20 * np.log10(hd3_norm + 1e-12)

    # Run 3: Eye Proxy
    eye_data_path = os.path.join(_TEMP_DIR, '_temp_eye_t_data')
    netlist3 = f"""Eye Proxy
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp} scale=1u

Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} PULSE({vdd/2} {vdd/2 + 0.1} 0 10p 10p 200p 10n)
Vinn vin_n 0 DC {vdd/2} PULSE({vdd/2} {vdd/2 - 0.1} 0 10p 10p 200p 10n)
{devices}
.control
tran 2p 2n
wrdata {eye_data_path} v(voutp,voutn)
.endc
.end
"""
    _run_ngspice(netlist3, 'full_eye')
    _, volts = _read_wrdata(eye_data_path)
    eye_height_proxy_mv = float(np.max(volts)) * 1000 if len(volts) > 0 else 0.0

    return {
        'peaking_db': float(peaking_db),
        'low_freq_gain_db': float(low_freq_gain),
        'high_freq_gain_db': float(high_freq_gain),
        'noise_mvrms': float(noise_mvrms),
        'hd3_db': float(hd3_db),
        'power_mw': float(power_mw),
        'eye_height_proxy_mv': float(eye_height_proxy_mv),
        'corner': corner,
        'temp': temp,
        'vdd': vdd
    }


simulate_transistor_level_full = simulate_transistor_level
