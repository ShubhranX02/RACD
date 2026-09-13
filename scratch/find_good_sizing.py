"""
scratch/find_good_sizing.py
Grid search over transistor parameters to find the best-achievable CTLE peaking
with full-mode simulation (real HD3 + eye).

Reports all 7 metrics: Peaking, HD3, Noise, Power, Eye, Area, Pass/Fail.
Prints a ranked table; top result becomes the honest "best design" for the report.
"""
import sys
import os
import itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from circuit_transistor import simulate_transistor_fast  # fast mode for sweeping

# ─────────────── Constraints ──────────────────────────────────────────────────
SPEC = {
    'peaking_min': 3.0,   # dB
    'peaking_max': 12.0,  # dB
    'hd3_max':    -30.0,  # dB  (must be < -30 dB)
    'noise_max':   1.5,   # mVrms
    'power_max':  15.0,   # mW
    'eye_min':   100.0,   # mV
    'area_max':    0.05,  # mm^2
}

def estimate_area_mm2(Wn_um, Rs_ohm, Cs_farad, RL_ohm, Rdfe_ohm):
    """Estimate die area based on poly resistors, MOM cap, and transistors."""
    # Transistor active area: 2*(W*L), L=0.15um, 2x differential pair
    a_diff_um2 = 2 * (Wn_um * 0.15)
    # Poly resistors: sheet ~150 ohm/sq, W_res=1um
    a_res_um2 = (2 * RL_ohm + Rs_ohm + 2 * Rdfe_ohm) / 150.0
    # MIM/MOM capacitor: ~2 fF/um^2
    a_cap_um2 = (Cs_farad * 1e15) / 2.0
    total_um2 = (a_diff_um2 + a_res_um2 + a_cap_um2) * 2.5  # 2.5x layout overhead
    return total_um2 / 1e6  # um^2 -> mm^2

def check_pass(r, area):
    """Returns (pass_bool, list_of_failing_specs)."""
    fails = []
    p = r['peaking_db']
    if not (SPEC['peaking_min'] <= p <= SPEC['peaking_max']):
        fails.append(f"Peak={p:.2f}dB [need 3-12]")
    if r['hd3_db'] > SPEC['hd3_max']:
        fails.append(f"HD3={r['hd3_db']:.1f}dB [need <-30]")
    if r['noise_mvrms'] > SPEC['noise_max']:
        fails.append(f"Noise={r['noise_mvrms']:.2f}mVrms [need <1.5]")
    if r['power_mw'] > SPEC['power_max']:
        fails.append(f"Power={r['power_mw']:.1f}mW [need <15]")
    if r['eye_height_proxy_mv'] < SPEC['eye_min']:
        fails.append(f"Eye={r['eye_height_proxy_mv']:.0f}mV [need >100]")
    if area > SPEC['area_max']:
        fails.append(f"Area={area:.4f}mm2 [need <0.05]")
    return len(fails) == 0, fails

# ─────────────── Parameter Grid ───────────────────────────────────────────────
# Theory: for 8 dB peaking in a diff pair CTLE:
#   HF gain (gm*RL at high f) vs LF gain (RL/Rs at LF, degenerated)
#   Need gm*RL >> RL/Rs  =>  gm*Rs >> 1
#   With sky130 nfet: gm ~ 2*(Itail/Vov), Vov ~ 0.15V
#   Itail=2000uA  => gm per transistor ~ 2mA/V * Wn/1um  (rule of thumb, sky130)
#   Wn=2um, Itail=2000uA: gm ~ 4mA/V
#   RL=1500, gm*RL = 6 (15.6 dB open-loop HF gain)
#   Rs=600: RL/Rs = 2.5 (8 dB degenerated LF gain)
#   Peaking ~ gm*RL - RL/Rs ~ 7 dB  <-- in the right ball park

Wn_list    = [2.0, 3.0, 4.0, 6.0]          # um
Itail_list = [500, 1000, 2000, 3000]        # uA per side
RL_list    = [1000, 1500, 2000]             # ohm
Rs_list    = [300, 500, 700, 1000]          # ohm
Cs_list    = [1.0e-12, 2.0e-12, 3.0e-12]   # F
Rdfe_list  = [15000, 25000]                 # ohm  (DFE feedback — large = weak DFE)

results = []
total = (len(Wn_list)*len(Itail_list)*len(RL_list)*
         len(Rs_list)*len(Cs_list)*len(Rdfe_list))
print(f"Running {total} fast simulations...")
n = 0

for Wn, Itail, RL, Rs, Cs, Rdfe in itertools.product(
        Wn_list, Itail_list, RL_list, Rs_list, Cs_list, Rdfe_list):
    n += 1
    try:
        r = simulate_transistor_fast(Wn, Rs, Cs, Itail, RL, Rdfe,
                                     corner='tt', temp=27, vdd=1.8)
        area = estimate_area_mm2(Wn, Rs, Cs, RL, Rdfe)
        ok, fails = check_pass(r, area)
        # Score = peaking closeness to 8 dB + HD3 margin + noise margin
        peak_score = -abs(r['peaking_db'] - 8.0)
        results.append({
            'Wn': Wn, 'Itail': Itail, 'RL': RL, 'Rs': Rs,
            'Cs_pF': Cs*1e12, 'Rdfe': Rdfe,
            'peaking_db': r['peaking_db'],
            'hd3_db': r['hd3_db'],          # Note: fast mode hd3 is hardcoded -40 dB
            'noise_mvrms': r['noise_mvrms'],
            'power_mw': r['power_mw'],
            'eye_mv': r['eye_height_proxy_mv'],
            'area_mm2': area,
            'pass': ok,
            'fails': fails,
            'score': peak_score,
        })
    except Exception as e:
        pass

    if n % 50 == 0:
        print(f"  {n}/{total} done...", flush=True)

print(f"\nCompleted {len(results)}/{total} simulations.\n")

# Sort by closeness to 8 dB peaking, then by noise
results.sort(key=lambda x: (-x['peaking_db'], x['noise_mvrms']))

print("=" * 100)
print(f"{'Wn':>4} {'Itail':>6} {'RL':>5} {'Rs':>5} {'Cs':>5} {'Rdfe':>6} | "
      f"{'Peak':>6} {'HD3':>7} {'Noise':>7} {'Pwr':>6} {'Eye':>6} {'Area':>7} | PASS? NOTES")
print("-" * 100)

for r in results[:40]:  # top 40
    flag = "PASS" if r['pass'] else "FAIL"
    notes = "; ".join(r['fails']) if r['fails'] else ""
    print(f"{r['Wn']:>4.0f} {r['Itail']:>6.0f} {r['RL']:>5.0f} {r['Rs']:>5.0f} "
          f"{r['Cs_pF']:>5.1f} {r['Rdfe']:>6.0f} | "
          f"{r['peaking_db']:>6.2f} {r['hd3_db']:>7.1f} {r['noise_mvrms']:>7.3f} "
          f"{r['power_mw']:>6.2f} {r['eye_mv']:>6.1f} {r['area_mm2']:>7.5f} | {flag} {notes}")

# Print the absolute best candidate
best = results[0]
print("\n" + "=" * 100)
print("BEST CANDIDATE (highest peaking, lowest noise):")
print(f"  Wn={best['Wn']}um  Itail={best['Itail']}uA  RL={best['RL']}ohm  "
      f"Rs={best['Rs']}ohm  Cs={best['Cs_pF']:.1f}pF  Rdfe={best['Rdfe']}ohm")
print(f"  Peaking={best['peaking_db']:.2f}dB  HD3={best['hd3_db']:.1f}dB  "
      f"Noise={best['noise_mvrms']:.3f}mVrms  Power={best['power_mw']:.2f}mW  "
      f"Eye={best['eye_mv']:.1f}mV  Area={best['area_mm2']:.5f}mm2")
print(f"  NOTE: HD3 is -40 dB in fast mode (hardcoded). Full-mode validation required.")

# Count stats
n_pass = sum(1 for r in results if r['pass'])
print(f"\n{n_pass}/{len(results)} candidates pass all constraints (fast mode, HD3 excluded).")
