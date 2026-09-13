import subprocess
import re
import numpy as np
import os
import tempfile

# Fixed lossy channel model (approximates PCB trace loss ahead of the
# equalizer). NOT tunable by the RL agent -- represents a given physical
# constraint, not a design choice.
CHANNEL_R = 50       # ohms
CHANNEL_C = 3e-12    # farads

# Directory for temporary SPICE netlists
_TEMP_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_ngspice(netlist_content, label="sim"):
    """Run an ngspice netlist and return stdout/stderr."""
    temp_file = os.path.join(_TEMP_DIR, f'_temp_{label}.cir')
    with open(temp_file, 'w') as f:
        f.write(netlist_content)
    result = subprocess.run(['ngspice', '-b', temp_file],
                             capture_output=True, text=True)
    return result


def _measure_noise_mvrms(Rs_ohm, Cs_farad, RL_ohm=200):
    """
    Measures input-referred noise via a raw netlist run through ngspice as
    a subprocess (deliberately NOT using PySpice's built-in noise() method,
    which has documented reliability issues).
    """
    netlist = f"""Passive equalizer - noise test
V1 in 0 AC 1
Rs in mid {Rs_ohm}
Cs in mid {Cs_farad}
RL mid 0 {RL_ohm}

.control
noise v(mid) v1 dec 10 1meg 10g
print inoise_total
.endc
.end
"""
    result = _run_ngspice(netlist, 'noise')

    match = re.search(r'inoise_total\s*=\s*([\d.eE+-]+)', result.stdout)
    if not match:
        raise ValueError(
            "Couldn't find inoise_total in ngspice output.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return float(match.group(1)) * 1000  # volts -> millivolts


def _measure_eye_proxy(Rs_ohm, Cs_farad, RL_ohm=200):
    """
    Sends a single isolated pulse through the fixed channel + tunable CTLE
    and measures the resulting pulse shape as a proxy for eye height/width.
    This is a deliberate simplification of a full statistical (PRBS-based)
    eye diagram, chosen for computational speed inside an RL training loop.
    """
    netlist = f"""Channel + Equalizer - Pulse Response
Vin in 0 PULSE(0 1 0 10p 10p 200p 10n)

Rch in ch_out {CHANNEL_R}
Cch ch_out 0 {CHANNEL_C}

Rs ch_out mid {Rs_ohm}
Cs ch_out mid {Cs_farad}
RL mid 0 {RL_ohm}

.control
tran 1p 2n
wrdata _temp_eye_data v(mid)
.endc
.end
"""
    result = _run_ngspice(netlist, 'eye')

    # Parse the wrdata output file
    data_file = os.path.join(_TEMP_DIR, '_temp_eye_data')
    if not os.path.exists(data_file):
        # Sometimes ngspice writes with .data extension
        for suffix in ['', '.data']:
            candidate = data_file + suffix
            if os.path.exists(candidate):
                data_file = candidate
                break

    try:
        times = []
        voltages = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('*'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        v = float(parts[1])
                        times.append(t)
                        voltages.append(v)
                    except ValueError:
                        continue

        if not times:
            raise ValueError("No data points parsed from eye proxy simulation")

        time = np.array(times)
        vout = np.array(voltages)

        peak_mv = float(np.max(vout) * 1000)
        half = np.max(vout) / 2
        above_half = time[vout >= half]
        width_ps = float((above_half[-1] - above_half[0]) * 1e12) if len(above_half) > 0 else 0.0

        return {'eye_height_proxy_mv': peak_mv, 'pulse_width_proxy_ps': width_ps}
    except Exception as e:
        # Fallback: try to parse from print output
        raise ValueError(
            f"Couldn't parse eye proxy data: {e}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


def _run_ac_simulation(Rs_ohm, Cs_farad, RL_ohm=200):
    """
    Runs AC analysis on the passive equalizer and returns parsed frequency/gain data.
    """
    netlist = f"""Passive CTLE - AC Analysis
Vin in 0 AC 1

Rs in mid {Rs_ohm}
Cs in mid {Cs_farad}
RL mid 0 {RL_ohm}

.control
ac dec 20 1meg 10g
wrdata _temp_ac_data vdb(mid)
.endc
.end
"""
    result = _run_ngspice(netlist, 'ac')

    data_file = os.path.join(_TEMP_DIR, '_temp_ac_data')
    if not os.path.exists(data_file):
        for suffix in ['', '.data']:
            candidate = data_file + suffix
            if os.path.exists(candidate):
                data_file = candidate
                break

    freqs = []
    gains = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('*'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    gain = float(parts[1])
                    freqs.append(freq)
                    gains.append(gain)
                except ValueError:
                    continue

    if not freqs:
        raise ValueError(
            f"Couldn't parse AC data.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    return np.array(freqs), np.array(gains)


def simulate(Rs_ohm, Cs_farad, RL_ohm=200, return_curve=False):
    """
    Stage A: passive Rs/Cs/RL equalizer, fed by a fixed lossy channel.
    This is the function the RL environment calls at every training step.
    Returns a flat dict of scalar specs (never raw SPICE analysis objects --
    see Part 0 rationale: scalars are cheap to log/reward on; raw curves
    are only needed occasionally for plotting, hence the opt-in return_curve).
    """
    freq, gain_db = _run_ac_simulation(Rs_ohm, Cs_farad, RL_ohm)

    low_freq_gain = float(gain_db[0])
    high_freq_gain = float(gain_db[-1])
    peaking_db = high_freq_gain - low_freq_gain

    noise_mvrms = _measure_noise_mvrms(Rs_ohm, Cs_farad, RL_ohm)
    eye_result = _measure_eye_proxy(Rs_ohm, Cs_farad, RL_ohm)

    result = {
        'peaking_db': peaking_db,
        'low_freq_gain_db': low_freq_gain,
        'high_freq_gain_db': high_freq_gain,
        'noise_mvrms': noise_mvrms,
        'eye_height_proxy_mv': eye_result['eye_height_proxy_mv'],
        'pulse_width_proxy_ps': eye_result['pulse_width_proxy_ps'],
    }
    if return_curve:
        result['frequency_hz'] = freq.tolist()
        result['gain_db_curve'] = gain_db.tolist()
    return result


if __name__ == '__main__':
    print(simulate(Rs_ohm=200, Cs_farad=2e-12))
