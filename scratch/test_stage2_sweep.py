"""
scratch/test_stage2_sweep.py
Sweeps 2-stage CTLE degeneration parameters to achieve 7.0 - 9.0 dB peaking
while measuring all 7 metrics (Peaking, Noise, HD3, Power, Eye, Area).
"""
import sys, os, subprocess, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from circuit_transistor import PDK_LIB_PATH, _run_ngspice, _read_wrdata

def estimate_area_2stage(Wn1, Rs1, Cs1, RL1, Wn2, Rs2, Cs2, RL2, Rdfe):
    a_diff = 2 * (Wn1 + Wn2) * 0.15
    a_res = (2*RL1 + Rs1 + 2*RL2 + Rs2 + 2*Rdfe + 200000) / 150.0  # including bias resistors
    a_cap = (Cs1 + Cs2 + 20e-12) * 1e15 / 2.0  # including 10pF AC coupling caps
    return (a_diff + a_res + a_cap) * 2.5 / 1e6  # mm2

def run_2stage_eval(Wn1=4.0, Rs1=500.0, Cs1=1.5e-12, Itail1=400.0, RL1=1800.0,
                    Wn2=4.0, Rs2=500.0, Cs2=1.5e-12, Itail2=400.0, RL2=1800.0,
                    Rdfe=15000.0, corner='tt', temp=27, vdd=1.8):
    
    devices = f"""
* Stage 1
XM1 vm1_n vin_p s1 0 sky130_fd_pr__nfet_01v8 w={Wn1} l=0.15
XM2 vm1_p vin_n s2 0 sky130_fd_pr__nfet_01v8 w={Wn1} l=0.15
Rs1 s1 s2 {Rs1}
Cs1 s1 s2 {Cs1}
Itail1_1 s1 0 DC {Itail1}u
Itail1_2 s2 0 DC {Itail1}u
RL1_1 vdd vm1_n {RL1}
RL1_2 vdd vm1_p {RL1}

* Coupling
Cac1 vm1_n vg2_n 5p
Cac2 vm1_p vg2_p 5p
Rbias1 vdd_half vg2_n 50k
Rbias2 vdd_half vg2_p 50k
Vmid vdd_half 0 {vdd/2}

* Stage 2
XM3 voutn vg2_p s3 0 sky130_fd_pr__nfet_01v8 w={Wn2} l=0.15
XM4 voutp vg2_n s4 0 sky130_fd_pr__nfet_01v8 w={Wn2} l=0.15
Rs2 s3 s4 {Rs2}
Cs2 s3 s4 {Cs2}
Itail2_1 s3 0 DC {Itail2}u
Itail2_2 s4 0 DC {Itail2}u
RL2_1 vdd voutn {RL2}
RL2_2 vdd voutp {RL2}

* 1-Tap DFE
Bslice_p vslice_p 0 V = V(voutp,voutn) > 0 ? 0.1 : -0.1
Bslice_n vslice_n 0 V = V(voutp,voutn) < 0 ? 0.1 : -0.1
Tdelay_p vslice_p 0 vdelayed_p 0 Z0=50 TD=200p
Tdelay_n vslice_n 0 vdelayed_n 0 Z0=50 TD=200p
Rdfe1 vdelayed_p voutp {Rdfe}
Rdfe2 vdelayed_n voutn {Rdfe}
"""
    # AC + Noise + Power
    netlist1 = f"""Combined AC
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp} scale=1u
Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} AC 0.5
Vinn vin_n 0 DC {vdd/2} AC -0.5
{devices}
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
    res1 = _run_ngspice(netlist1, '2s_ac')
    freqs, gains = _read_wrdata(os.path.join(os.path.dirname(__file__), '..', 'src', '_temp_ac_2stage.data'))
    if len(freqs) == 0:
        freqs, gains = _read_wrdata('_temp_ac_2stage.data')

    lf_idx = (np.abs(freqs - 100e6)).argmin()
    hf_idx = (np.abs(freqs - 2.5e9)).argmin()
    peaking_db = float(gains[hf_idx]) - float(gains[lf_idx])

    match_dc = re.search(r'i\(vdd\)\s*=\s*([\d.eE+-]+)', res1.stdout, re.IGNORECASE)
    power_mw = abs(float(match_dc.group(1))) * vdd * 1000 if match_dc else 100.0

    match_n = re.search(r'inoise_total\s*=\s*([\d.eE+-]+)', res1.stdout, re.IGNORECASE)
    noise_mvrms = float(match_n.group(1)) * 1000 if match_n else 10.0

    # HD3 Transient Fourier
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
    res2 = _run_ngspice(netlist2, '2s_hd3')
    match_hd3 = re.search(r'3\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)', res2.stdout, re.IGNORECASE)
    hd3_norm = float(match_hd3.group(1)) if match_hd3 else 0.05
    hd3_db = 20 * np.log10(hd3_norm + 1e-12)

    # Eye Proxy
    netlist3 = f"""Eye Proxy
.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp} scale=1u
Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} PULSE({vdd/2} {vdd/2 + 0.1} 0 10p 10p 200p 10n)
Vinn vin_n 0 DC {vdd/2} PULSE({vdd/2} {vdd/2 - 0.1} 0 10p 10p 200p 10n)
{devices}
.control
tran 2p 2n
wrdata _temp_eye_2stage.data v(voutp,voutn)
.endc
.end
"""
    _run_ngspice(netlist3, '2s_eye')
    _, volts = _read_wrdata(os.path.join(os.path.dirname(__file__), '..', 'src', '_temp_eye_2stage.data'))
    if len(volts) == 0:
        _, volts = _read_wrdata('_temp_eye_2stage.data')
    eye_mv = float(np.max(volts)) * 1000 if len(volts) > 0 else 0.0

    area = estimate_area_2stage(Wn1, Rs1, Cs1, RL1, Wn2, Rs2, Cs2, RL2, Rdfe)
    return peaking_db, noise_mvrms, hd3_db, power_mw, eye_mv, area

if __name__ == '__main__':
    candidates = [
        # (Rs1, Cs1, RL1, Rs2, Cs2, RL2)
        (600.0, 1.5e-12, 2000.0, 600.0, 1.5e-12, 2000.0),
        (700.0, 1.8e-12, 2200.0, 700.0, 1.8e-12, 2200.0),
        (800.0, 2.0e-12, 2500.0, 800.0, 2.0e-12, 2500.0),
    ]
    for Rs1, Cs1, RL1, Rs2, Cs2, RL2 in candidates:
        p, n, h, pwr, eye, a = run_2stage_eval(
            Wn1=4.0, Rs1=Rs1, Cs1=Cs1, Itail1=400.0, RL1=RL1,
            Wn2=4.0, Rs2=Rs2, Cs2=Cs2, Itail2=400.0, RL2=RL2,
            Rdfe=20000.0, corner='tt', temp=27, vdd=1.8
        )
        print(f"Rs={Rs1:.0f} RL={RL1:.0f} Cs={Cs1*1e12:.1f}p -> Peak={p:5.2f}dB, Noise={n:5.3f}mV, HD3={h:6.1f}dB, Pwr={pwr:5.2f}mW, Eye={eye:5.1f}mV, Area={a:.5f}mm2")
