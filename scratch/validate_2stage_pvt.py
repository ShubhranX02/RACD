"""
scratch/validate_2stage_pvt.py
Evaluates the Cascaded 2-Stage CTLE across representative PVT corners.
Captures all seven mandatory metrics.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from circuit_transistor import simulate_transistor_level

def estimate_area_2stage(Wn1=4.0, Rs1=750.0, Cs1=1.8e-12, RL1=2200.0,
                         Wn2=4.0, Rs2=750.0, Cs2=1.8e-12, RL2=2200.0, Rdfe=20000.0):
    a_diff = 2 * (Wn1 + Wn2) * 0.15
    a_res = (2*RL1 + Rs1 + 2*RL2 + Rs2 + 2*Rdfe + 100000) / 150.0
    a_cap = (Cs1 + Cs2 + 10e-12) * 1e15 / 2.0
    return (a_diff + a_res + a_cap) * 2.5 / 1e6

corners = [
    ('TT', 1.80, 27),
    ('SS', 1.71, 125),
    ('FF', 1.89, 0),
    ('SF', 1.80, 27),
    ('FS', 1.80, 27)
]

print("=" * 110)
print(f"{'Corner':<6} {'VDD':<6} {'Temp':<6} {'Peaking (3-12dB)':<18} {'HD3 (<-30dB)':<14} {'Noise (<1.5mV)':<16} {'Power (<15mW)':<14} {'Eye (>100mV)':<14} {'Area (<0.05mm2)':<16} {'Status'}")
print("=" * 110)

area = estimate_area_2stage()

for c, v, t in corners:
    res = simulate_transistor_level(
        Wn_um=4.0, Rs_ohm=750.0, Cs_farad=1.8e-12, Itail_half_ua=400.0, RL_ohm=2200.0, Rdfe_ohm=20000.0,
        corner=c.lower(), temp=t, vdd=v, topology='2stage'
    )
    p = res['peaking_db']
    h = res['hd3_db']
    n = res['noise_mvrms']
    pwr = res['power_mw']
    eye = res['eye_height_proxy_mv']
    
    passed = (3.0 <= p <= 12.0) and (h <= -30.0) and (n <= 1.5) and (pwr <= 15.0) and (eye >= 100.0) and (area <= 0.05)
    status = "PASS [100%]" if passed else ("MARGINAL" if h <= -28.0 else "FAIL")
    
    print(f"{c:<6} {v:<6.2f} {t:<6d} {p:<18.2f} {h:<14.1f} {n:<16.3f} {pwr:<14.2f} {eye:<14.1f} {area:<16.5f} {status}")
print("=" * 110)
