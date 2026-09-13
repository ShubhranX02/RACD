"""
scratch/test_stage2.py
Tests a Cascaded 2-Stage CTLE topology in SkyWater 130 nm PDK.
Stage 1 provides high-frequency zero boost with degeneration (Rs1, Cs1).
Stage 2 provides active boost + buffer driving the 1-Tap DFE.
"""
import sys, os, subprocess, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from circuit_transistor import PDK_LIB_PATH, _run_ngspice, _read_wrdata

def simulate_2stage(Wn1=4.0, Rs1=400.0, Cs1=1.0e-12, Itail1=400.0, RL1=1500.0,
                    Wn2=4.0, Rs2=400.0, Cs2=1.0e-12, Itail2=400.0, RL2=1500.0,
                    Rdfe=15000.0, corner='tt', temp=27, vdd=1.8):
    
    netlist = f"""Cascaded 2-Stage CTLE + 1-Tap DFE
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp}
.option scale=1u

Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} AC 0.5
Vinn vin_n 0 DC {vdd/2} AC -0.5

* ─── Stage 1: Continuous-Time Linear Equalizer ───
XM1 vm1_n vin_p s1 0 sky130_fd_pr__nfet_01v8 w={Wn1} l=0.15
XM2 vm1_p vin_n s2 0 sky130_fd_pr__nfet_01v8 w={Wn1} l=0.15
Rs1 s1 s2 {Rs1}
Cs1 s1 s2 {Cs1}
Itail1_1 s1 0 DC {Itail1}u
Itail1_2 s2 0 DC {Itail1}u
RL1_1 vdd vm1_n {RL1}
RL1_2 vdd vm1_p {RL1}

* ─── AC Coupling / Level Shift to Stage 2 ───
* Biased at mid-rail (vdd/2)
Cac1 vm1_n vg2_n 10p
Cac2 vm1_p vg2_p 10p
Rbias1 vdd_half vg2_n 100k
Rbias2 vdd_half vg2_p 100k
Vmid vdd_half 0 {vdd/2}

* ─── Stage 2: Second Booster + DFE Summing ───
XM3 voutn vg2_p s3 0 sky130_fd_pr__nfet_01v8 w={Wn2} l=0.15
XM4 voutp vg2_n s4 0 sky130_fd_pr__nfet_01v8 w={Wn2} l=0.15
Rs2 s3 s4 {Rs2}
Cs2 s3 s4 {Cs2}
Itail2_1 s3 0 DC {Itail2}u
Itail2_2 s4 0 DC {Itail2}u
RL2_1 vdd voutn {RL2}
RL2_2 vdd voutp {RL2}

* ─── 1-Tap DFE ───
Bslice_p vslice_p 0 V = V(voutp,voutn) > 0 ? 0.1 : -0.1
Bslice_n vslice_n 0 V = V(voutp,voutn) < 0 ? 0.1 : -0.1
Tdelay_p vslice_p 0 vdelayed_p 0 Z0=50 TD=200p
Tdelay_n vslice_n 0 vdelayed_n 0 Z0=50 TD=200p
Rdfe1 vdelayed_p voutp {Rdfe}
Rdfe2 vdelayed_n voutn {Rdfe}

.control
ac dec 10 10meg 10g
wrdata _temp_ac_2stage.data vdb(voutp,voutn)

op
print i(Vdd)

noise v(voutp,voutn) vinp dec 10 10meg 5g
print inoise_total
.endc
.end
"""
    res = _run_ngspice(netlist, '2stage_ac')
    freqs, gains = _read_wrdata(os.path.join(os.path.dirname(__file__), '..', 'src', '_temp_ac_2stage.data'))
    if len(freqs) == 0:
        # Check current working directory
        freqs, gains = _read_wrdata('_temp_ac_2stage.data')

    lf_idx = (np.abs(freqs - 100e6)).argmin()
    hf_idx = (np.abs(freqs - 2.5e9)).argmin()
    low_freq_gain = float(gains[lf_idx])
    high_freq_gain = float(gains[hf_idx])
    peaking_db = high_freq_gain - low_freq_gain

    match_dc = re.search(r'i\(vdd\)\s*=\s*([\d.eE+-]+)', res.stdout, re.IGNORECASE)
    power_mw = abs(float(match_dc.group(1))) * vdd * 1000 if match_dc else 100.0

    match_n = re.search(r'inoise_total\s*=\s*([\d.eE+-]+)', res.stdout, re.IGNORECASE)
    noise_mvrms = float(match_n.group(1)) * 1000 if match_n else 10.0

    print(f"2-Stage CTLE Simulated:")
    print(f"  Peaking:         {peaking_db:.2f} dB (Target: 3-12 dB)")
    print(f"  Low-Freq Gain:   {low_freq_gain:.2f} dB")
    print(f"  High-Freq Gain:  {high_freq_gain:.2f} dB")
    print(f"  Integrated Noise:{noise_mvrms:.3f} mVrms (Spec: < 1.5 mV)")
    print(f"  DC Power:        {power_mw:.2f} mW (Spec: < 15.0 mW)")
    return peaking_db, low_freq_gain, high_freq_gain, noise_mvrms, power_mw

if __name__ == '__main__':
    simulate_2stage()
