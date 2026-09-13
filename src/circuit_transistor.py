import subprocess
import re
import numpy as np
import os

PDK_LIB_PATH = "/Users/shubhransharma/.ciel/ciel/sky130/versions/bdc9412b3e468c102d01b7cf6337be06ec6e9c9a/sky130A/libs.tech/ngspice/sky130.lib.spice"
_TEMP_DIR = os.path.dirname(os.path.abspath(__file__))

def _run_ngspice(netlist_content, label="sim"):
    temp_file = os.path.join(_TEMP_DIR, f'_temp_{label}.cir')
    with open(temp_file, 'w') as f:
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

def _device_section(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm):
    """Returns the core device netlist section (shared by all analyses)."""
    return f"""
XM1 voutn vin_p s1 0 sky130_fd_pr__nfet_01v8 w={Wn_um} l=0.15
XM2 voutp vin_n s2 0 sky130_fd_pr__nfet_01v8 w={Wn_um} l=0.15

Rs s1 s2 {Rs_ohm}
Cs s1 s2 {Cs_farad}

Itail1 s1 0 DC {Itail_half_ua}u
Itail2 s2 0 DC {Itail_half_ua}u

RL1 vdd voutn {RL_ohm}
RL2 vdd voutp {RL_ohm}

* 1-Tap DFE (Fully Differential)
Bslice_p vslice_p 0 V = V(voutp,voutn) > 0 ? 0.1 : -0.1
Bslice_n vslice_n 0 V = V(voutp,voutn) < 0 ? 0.1 : -0.1
Tdelay_p vslice_p 0 vdelayed_p 0 Z0=50 TD=200p
Tdelay_n vslice_n 0 vdelayed_n 0 Z0=50 TD=200p
Rdfe1 vdelayed_p voutp {Rdfe_ohm}
Rdfe2 vdelayed_n voutn {Rdfe_ohm}
"""


def simulate_transistor_fast(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm=2000, Rdfe_ohm=10000, corner='tt', temp=27, vdd=1.8):
    """
    FAST simulation for RL training: single ngspice call with AC + OP + Noise.
    Skips transient HD3 and eye proxy (those are validated post-training).
    Eye height is estimated from AC gain at Nyquist.
    HD3 is estimated as 0 (assumed ok; validated properly later).
    """
    devices = _device_section(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm)
    ac_data_path = os.path.join(_TEMP_DIR, '_temp_ac_data')

    netlist = f"""Combined AC/OP/Noise (fast training mode)
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp}
.option scale=1u

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
    res = _run_ngspice(netlist, 'fast_train')

    # Parse AC
    freqs, gains = _read_wrdata(ac_data_path)
    if len(freqs) == 0:
        raise ValueError(f"Failed to parse AC data:\n{res.stdout}\n{res.stderr}")

    lf_idx = (np.abs(freqs - 100e6)).argmin()
    hf_idx = (np.abs(freqs - 2.5e9)).argmin()
    low_freq_gain = float(gains[lf_idx])
    high_freq_gain = float(gains[hf_idx])
    peaking_db = high_freq_gain - low_freq_gain

    # Parse DC power
    match_dc = re.search(r'i\(vdd\)\s*=\s*([\d.eE+-]+)', res.stdout, re.IGNORECASE)
    power_mw = abs(float(match_dc.group(1))) * vdd * 1000 if match_dc else 100.0

    # Parse noise
    match_n = re.search(r'inoise_total\s*=\s*([\d.eE+-]+)', res.stdout, re.IGNORECASE)
    noise_mvrms = float(match_n.group(1)) * 1000 if match_n else 10.0

    # Estimated eye height from AC gain at Nyquist
    eye_height_proxy_mv = 10**(high_freq_gain / 20.0) * 100.0

    # HD3 not measured in fast mode (too expensive); default to passing
    hd3_db = -40.0

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


def simulate_transistor_level(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm=2000, Rdfe_ohm=10000, corner='tt', temp=27, vdd=1.8):
    """
    Full simulation: AC + OP + Noise + HD3 transient.
    Used for validation and PVT sweeps.
    """
    devices = _device_section(Wn_um, Rs_ohm, Cs_farad, Itail_half_ua, RL_ohm, Rdfe_ohm)
    ac_data_path = os.path.join(_TEMP_DIR, '_temp_ac_data')

    # Run 1: AC + OP + Noise
    netlist1 = f"""Combined AC/OP/Noise
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp}
.option scale=1u

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

    # Run 2: HD3 transient
    netlist2 = f"""HD3 Transient
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp}
.option scale=1u

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
    match_hd3 = re.search(r'3\s+3e\+08\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)', res2.stdout, re.IGNORECASE)
    hd3_norm = float(match_hd3.group(1)) if match_hd3 else 1.0
    hd3_db = 20 * np.log10(hd3_norm + 1e-12)

    # Run 3: Eye proxy (pulse response)
    eye_data_path = os.path.join(_TEMP_DIR, '_temp_eye_t_data')
    netlist3 = f"""Eye Proxy
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp}
.option scale=1u

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


# Keep backward compat alias
simulate_transistor_level_full = simulate_transistor_level


if __name__ == '__main__':
    import time

    print("=== Fast mode (for RL training) ===")
    t0 = time.time()
    res = simulate_transistor_fast(Wn_um=4, Rs_ohm=200, Cs_farad=2e-12, Itail_half_ua=400, RL_ohm=2000, Rdfe_ohm=10000)
    t1 = time.time()
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"  Time: {t1-t0:.2f}s")

    print("\n=== Full mode (for validation) ===")
    t0 = time.time()
    res = simulate_transistor_level(Wn_um=4, Rs_ohm=200, Cs_farad=2e-12, Itail_half_ua=400, RL_ohm=2000, Rdfe_ohm=10000)
    t1 = time.time()
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"  Time: {t1-t0:.2f}s")
