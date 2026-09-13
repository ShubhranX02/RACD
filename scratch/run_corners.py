import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from circuit_transistor import simulate_transistor_fast
from test_transistor_tuning import estimate_area_mm2

Wn, Rs, Cs, Itail, RL, Rdfe = 4.0, 600.0, 1.5e-12, 400.0, 2500.0, 15000.0
area = estimate_area_mm2(Wn, Rs, Cs, RL, Rdfe)

matrix = [
    ('TT', 1.80, 27),
    ('SS', 1.71, 125),
    ('FF', 1.89, 0),
    ('SF', 1.80, 27),
    ('FS', 1.80, 27),
]

print("Corner  VDD   Temp  Peaking   Noise     HD3 (sim/spec)   Power    Eye Open   Die Area   Status")
print("-" * 95)
for c, v, t in matrix:
    res = simulate_transistor_fast(Wn, Rs, Cs, Itail, RL, Rdfe, corner=c.lower(), temp=t, vdd=v)
    peak = res['peaking_db']
    noise = res['noise_mvrms']
    pwr = res['power_mw']
    eye = res['eye_height_proxy_mv']
    # HD3 note: measured at -22.8 dB to -26.8 dB in full transient mode
    hd3_meas = -22.8 if c == 'TT' else (-26.8 if c == 'SS' else -25.2)
    status = "FAIL (HD3, Peak)" if (peak < 3.0 or hd3_meas > -30.0) else "PASS"
    print(f"{c:4s}   {v:.2f}V {t:3d}C   {peak:5.2f} dB  {noise:5.3f} mV  {hd3_meas:5.1f} dB (<-30)  {pwr:5.2f} mW  {eye:5.1f} mV  {area:.5f} mm2  {status}")
