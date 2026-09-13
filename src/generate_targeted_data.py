"""
generate_targeted_data.py — Screen candidate transistor parameters with the
existing surrogate to find candidates with peaking in [4.5, 9.5] dB, then
simulate them with true parallel ngspice to augment transistor_repository_clean.jsonl.
"""
import os, sys, time, json, multiprocessing
import numpy as np

sys.path.insert(0, r'C:\Users\Kio\Desktop\Dev\RACD2\src')

from circuit_transistor import simulate_transistor_fast
from surrogate_transistor import (
    PARAM_BOUNDS, _PVT_CORNERS, load_transistor_surrogate,
    simulate_transistor_surrogate, _latin_hypercube_sample
)

TARGET_CLEAN_PATH = r'C:\Users\Kio\Desktop\Dev\RACD2\data\transistor_repository_clean.jsonl'

def _spice_worker(args):
    idx, Wn, Rs, Cs, Itail, RL, Rdfe = args
    results = []
    for corner, temp, vdd in _PVT_CORNERS:
        try:
            r = simulate_transistor_fast(
                Wn, Rs, Cs, Itail, RL, Rdfe,
                corner=corner, temp=temp, vdd=vdd,
                topology='2stage', worker_id=idx,
            )
            record = {
                'Wn_um': float(Wn), 'Rs_ohm': float(Rs),
                'Cs_farad': float(Cs), 'Itail_half_ua': float(Itail),
                'RL_ohm': float(RL), 'Rdfe_ohm': float(Rdfe),
                'peaking_db':          float(r['peaking_db']),
                'noise_mvrms':         float(r['noise_mvrms']),
                'power_mw':            float(r['power_mw']),
                'eye_height_proxy_mv': float(r['eye_height_proxy_mv']),
                'hd3_db':              float(r.get('hd3_db', -38.0)),
                'corner': corner, 'temp': temp, 'vdd': vdd,
            }
            results.append(record)
        except Exception:
            pass
    return results


def main():
    multiprocessing.freeze_support()
    print("=" * 70)
    print("Targeted SPICE Data Generation for 4.5 - 9.5 dB Peaking")
    print("=" * 70)

    load_transistor_surrogate()

    # Step 1: Generate a large candidate pool of 8,000 LHS points
    print("[1/3] Screening 8,000 candidate designs with current surrogate...")
    candidates = _latin_hypercube_sample(8000, PARAM_BOUNDS, seed=123)

    filtered = []
    for row in candidates:
        Wn, Rs, Cs, Itail, RL, Rdfe = row.tolist()
        pred = simulate_transistor_surrogate(Wn, Rs, Cs, Itail, RL, Rdfe)
        # Filter for candidates predicted to produce 4.5 to 9.5 dB peaking
        if 4.5 <= pred['peaking_db'] <= 9.5 and pred['noise_mvrms'] <= 2.0:
            filtered.append(row)

    print(f"  Found {len(filtered)} candidates predicted in [4.5, 9.5] dB.")
    # Select up to 350 candidates (350 * 3 corners = ~1,050 SPICE simulations)
    n_selected = min(350, len(filtered))
    selected = filtered[:n_selected]
    print(f"  Selected {n_selected} designs for ngspice simulation across 3 corners ({n_selected * 3} calls).")

    # Step 2: Run parallel ngspice
    n_workers = min(os.cpu_count() or 4, 12)
    print(f"[2/3] Simulating with ngspice across {n_workers} workers...")
    args_list = [(i, *selected[i].tolist()) for i in range(n_selected)]

    t0 = time.time()
    new_records = []
    with multiprocessing.Pool(processes=n_workers) as pool:
        for i, res_batch in enumerate(pool.imap_unordered(_spice_worker, args_list, chunksize=1)):
            new_records.extend(res_batch)
            if (i + 1) % 25 == 0 or (i + 1) == n_selected:
                pct = (i + 1) / n_selected * 100
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_selected - i - 1)
                print(f"  [{pct:5.1f}%] {i+1}/{n_selected} designs done | {len(new_records)} records | ETA {eta:.0f}s", flush=True)

    print(f"[3/3] SPICE simulation completed in {time.time() - t0:.1f}s. Generated {len(new_records)} records.")

    # Check peaking distribution of new records
    peakings = [r['peaking_db'] for r in new_records]
    p_np = np.array(peakings)
    print(f"  New data peaking: min={p_np.min():.2f}, mean={p_np.mean():.2f}, max={p_np.max():.2f}")
    in_range = np.sum((p_np >= 4.5) & (p_np <= 9.5))
    print(f"  Actually in [4.5, 9.5] dB: {in_range}/{len(p_np)} ({in_range/len(p_np)*100:.1f}%)")

    # Append to transistor_repository_clean.jsonl
    with open(TARGET_CLEAN_PATH, 'a', encoding='utf-8') as f:
        for r in new_records:
            f.write(json.dumps(r) + '\n')

    print(f"[OK] Appended {len(new_records)} records to {TARGET_CLEAN_PATH}")


if __name__ == '__main__':
    main()
