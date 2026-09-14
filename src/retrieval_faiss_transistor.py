"""
retrieval_faiss_transistor.py — Fast FAISS-Based Nearest-Neighbour Retrieval for
Sky130 Transistor-Level 1-Stage CTLE + 1-Tap DFE Equalizers.

Indexes SPICE-verified circuit records from transistor_repository.jsonl.
Features:
  1. Instant (<0.1ms) retrieval of SPICE-verified sizing for target peaking, noise, power, eye.
  2. Persistent index caching to disk (models/transistor_faiss.index).
  3. Hybrid Retrieval-Augmented Sizing: retrieves exact match if error < threshold,
     or provides warm-start priors for RL policy fine-tuning.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    _FAISS_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Use the active 1-stage repository; fall back to clean archive only if primary missing
_PRIMARY_DATA   = PROJECT_ROOT / "data" / "transistor_repository.jsonl"
_FALLBACK_DATA  = PROJECT_ROOT / "data" / "transistor_repository_clean_2stage_archive.jsonl"
DATA_PATH = _PRIMARY_DATA if _PRIMARY_DATA.exists() else _FALLBACK_DATA
INDEX_DIR = PROJECT_ROOT / "models"
INDEX_PATH = INDEX_DIR / "transistor_faiss.index"
META_PATH  = INDEX_DIR / "transistor_faiss_meta.json"

# Normalization bounds for search space
_SPEC_SCALES = {
    "peaking_db": 10.0,
    "noise_mvrms": 1.5,
    "power_mw": 15.0,
    "eye_height_proxy_mv": 1000.0,
}

# Global singleton cache
_GLOBAL_INDEX = None
_GLOBAL_METADATA = None


def _extract_spec_vector(r: dict) -> np.ndarray:
    """Extract normalized float32 spec vector [peaking, noise, power, eye]."""
    peaking = float(r.get("peaking_db", 0.0)) / _SPEC_SCALES["peaking_db"]
    noise = float(r.get("noise_mvrms", 0.0)) / _SPEC_SCALES["noise_mvrms"]
    power = float(r.get("power_mw", 0.0)) / _SPEC_SCALES["power_mw"]
    eye = float(r.get("eye_height_proxy_mv", 0.0)) / _SPEC_SCALES["eye_height_proxy_mv"]
    # Weight peaking 3x higher in L2 distance than secondary constraints
    return np.array([3.0 * peaking, noise, power, eye], dtype=np.float32)


def build_transistor_faiss_index(
    data_path: Optional[str] = None,
    save_index: bool = True,
    force_rebuild: bool = False,
) -> Tuple[Optional["faiss.IndexFlatL2"], List[dict]]:
    """
    Build or load from disk a FAISS index over transistor_repository_clean.jsonl.
    Returns (index, metadata_list).
    """
    global _GLOBAL_INDEX, _GLOBAL_METADATA

    if not _FAISS_AVAILABLE:
        print("[transistor_faiss] WARNING: faiss is not installed.")
        return None, []

    if _GLOBAL_INDEX is not None and not force_rebuild:
        return _GLOBAL_INDEX, _GLOBAL_METADATA

    if data_path is None:
        data_path = str(DATA_PATH)

    # Check for cached on-disk index
    if (
        not force_rebuild
        and os.path.exists(INDEX_PATH)
        and os.path.exists(META_PATH)
        and os.path.exists(data_path)
        and os.path.getmtime(INDEX_PATH) > os.path.getmtime(data_path)
    ):
        try:
            t0 = time.perf_counter()
            index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            _GLOBAL_INDEX = index
            _GLOBAL_METADATA = metadata
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return index, metadata
        except Exception as ex:
            print(f"[transistor_faiss] Cache read failed ({ex}), rebuilding index...")

    if not os.path.exists(data_path):
        print(f"[transistor_faiss] Dataset not found: {data_path}")
        return None, []

    records = []
    vectors = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                # Basic quality gate: skip circuits with non-physical simulation results
                if r.get("peaking_db") is None or r.get("noise_mvrms", 999) > 10.0:
                    continue
                v = _extract_spec_vector(r)
                vectors.append(v)
                records.append(r)
            except Exception:
                continue

    if not records:
        print("[transistor_faiss] No valid records found to index.")
        return None, []

    mat = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
    dim = mat.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(mat)

    if save_index:
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f)

    _GLOBAL_INDEX = index
    _GLOBAL_METADATA = records
    print(f"[transistor_faiss] Built FAISS index with {index.ntotal} circuits (dim={dim})")
    return index, records


def retrieve_k_nearest_transistor(
    target_peaking_db: float,
    noise_limit_mvrms: float = 1.5,
    power_limit_mw: float = 15.0,
    eye_height_limit_mv: float = 100.0,
    corner: Optional[str] = "tt",
    k: int = 5,
) -> List[Dict]:
    """
    Retrieve top-k nearest transistor equalizers matching performance specifications.

    Parameters
    ----------
    target_peaking_db: float
        Desired high-frequency peaking in dB (e.g. 6.0, 8.0).
    noise_limit_mvrms: float
        Upper bound on input-referred noise (default 1.5 mV).
    power_limit_mw: float
        Upper bound on power consumption (default 15.0 mW).
    eye_height_limit_mv: float
        Lower bound on eye height proxy (default 100.0 mV).
    corner: Optional[str]
        If specified ('tt', 'ss', 'ff'), filters to that PVT corner.
    k: int
        Number of candidates to retrieve.

    Returns
    -------
    List of dicts containing full transistor sizing parameters and verified SPICE specs.
    """
    index, metadata = build_transistor_faiss_index()
    if index is None or not metadata:
        return []

    # Construct query vector in same normalized space
    query_spec = {
        "peaking_db": target_peaking_db,
        "noise_mvrms": noise_limit_mvrms,
        "power_mw": power_limit_mw,
        "eye_height_proxy_mv": eye_height_limit_mv,
    }
    query_vec = _extract_spec_vector(query_spec).reshape(1, -1)

    # Search for an expanded pool (4x) to allow post-filtering on constraints and corner
    fetch_k = min(max(k * 4, 30), len(metadata))
    distances, indices = index.search(query_vec, fetch_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        candidate = metadata[idx].copy()
        candidate["_faiss_distance"] = float(dist)
        candidate["_peaking_error_db"] = abs(candidate["peaking_db"] - target_peaking_db)

        # Apply corner filter if requested
        if corner is not None and candidate.get("corner") != corner:
            continue

        results.append(candidate)

    # Sort primarily by peaking error, then by noise
    results.sort(key=lambda c: (c["_peaking_error_db"], c.get("noise_mvrms", 999)))
    return results[:k]


def synthesize_transistor_racd(
    target_peaking_db: float,
    noise_limit_mvrms: float = 1.5,
    power_limit_mw: float = 15.0,
    eye_height_limit_mv: float = 100.0,
    corner: str = "tt",
    exact_match_threshold_db: float = 0.4,
) -> Dict:
    """
    Hybrid Retrieval-Augmented Circuit Design (RACD) Engine:
    1. Queries the FAISS index over SPICE-verified repository.
    2. If a verified circuit satisfies specs within exact_match_threshold_db, returns it (<0.1ms).
    3. Otherwise, loads the trained PPO policy to synthesize the sizing.
    """
    t0 = time.perf_counter()

    # Step 1: FAISS Retrieval
    candidates = retrieve_k_nearest_transistor(
        target_peaking_db, noise_limit_mvrms, power_limit_mw, eye_height_limit_mv,
        corner=corner, k=5
    )

    if candidates:
        from surrogate_transistor import simulate_transistor_surrogate
        for cand in candidates:
            err = cand["_peaking_error_db"]
            noise_ok = cand.get("noise_mvrms", 0.0) <= noise_limit_mvrms
            power_ok = cand.get("power_mw", 0.0) <= power_limit_mw
            if err <= exact_match_threshold_db and noise_ok and power_ok:
                # Neural surrogate gate: verifies that physical model agrees with database record
                pred = simulate_transistor_surrogate(
                    cand["Wn_um"], cand["Rs_ohm"], cand["Cs_farad"],
                    cand["Itail_half_ua"], cand["RL_ohm"], cand["Rdfe_ohm"]
                )
                if abs(pred["peaking_db"] - target_peaking_db) <= 1.5:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    return {
                        "source": "FAISS_SURROGATE_VERIFIED_RETRIEVAL",
                        "elapsed_ms": elapsed_ms,
                        "target_peaking_db": target_peaking_db,
                        "sizing": {
                            "Wn_um": cand["Wn_um"],
                            "Rs_ohm": cand["Rs_ohm"],
                            "Cs_farad": cand["Cs_farad"],
                            "Itail_half_ua": cand["Itail_half_ua"],
                            "RL_ohm": cand["RL_ohm"],
                            "Rdfe_ohm": cand["Rdfe_ohm"],
                        },
                        "verified_spice_metrics": {
                            "peaking_db": cand["peaking_db"],
                            "noise_mvrms": cand["noise_mvrms"],
                            "power_mw": cand["power_mw"],
                            "eye_height_proxy_mv": cand.get("eye_height_proxy_mv", 0.0),
                            "corner": cand.get("corner", corner),
                        },
                        "error_db": err,
                    }

    # Step 2: Fallback to Trained RL Policy
    try:
        from stable_baselines3 import PPO
        from environment_transistor import TransistorEqualizerEnv

        model_path = INDEX_DIR / "ppo_transistor"
        model = PPO.load(str(model_path), device="cpu")
        env = TransistorEqualizerEnv(peaking_target_range=(target_peaking_db, target_peaking_db))
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        Wn, Rs, Cs, Itail, RL, Rdfe = env._rescale_action(action)

        from surrogate_transistor import simulate_transistor_surrogate
        pred = simulate_transistor_surrogate(Wn, Rs, Cs, Itail, RL, Rdfe)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "source": "RL_POLICY_SYNTHESIS",
            "elapsed_ms": elapsed_ms,
            "target_peaking_db": target_peaking_db,
            "sizing": {
                "Wn_um": Wn, "Rs_ohm": Rs, "Cs_farad": Cs,
                "Itail_half_ua": Itail, "RL_ohm": RL, "Rdfe_ohm": Rdfe
            },
            "predicted_metrics": pred,
            "error_db": abs(pred["peaking_db"] - target_peaking_db),
            "nearest_faiss_prior": candidates[0] if candidates else None,
        }
    except Exception as ex:
        # Fallback to closest FAISS candidate if RL model is unavailable
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if candidates:
            c = candidates[0]
            return {
                "source": "FAISS_BEST_APPROX",
                "elapsed_ms": elapsed_ms,
                "target_peaking_db": target_peaking_db,
                "sizing": {
                    "Wn_um": c["Wn_um"], "Rs_ohm": c["Rs_ohm"], "Cs_farad": c["Cs_farad"],
                    "Itail_half_ua": c["Itail_half_ua"], "RL_ohm": c["RL_ohm"], "Rdfe_ohm": c["Rdfe_ohm"]
                },
                "verified_spice_metrics": c,
                "error_db": c["_peaking_error_db"],
            }
        raise RuntimeError(f"Both FAISS and RL synthesis failed: {ex}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing retrieval_faiss_transistor.py")
    print("=" * 70)

    # 1. Build index
    idx, meta = build_transistor_faiss_index(force_rebuild=True)
    print(f"Total indexed circuits: {len(meta)}")

    # 2. Test multiple targets across range
    test_targets = [3.0, 4.5, 6.0, 7.5, 8.5, 9.5]
    print("\nRetrieval Test Across Target Range:")
    for tgt in test_targets:
        res = synthesize_transistor_racd(tgt, corner="tt")
        src = res["source"]
        sz = res["sizing"]
        err = res["error_db"]
        ms = res["elapsed_ms"]
        print(f"  Target: {tgt:>4.1f} dB | Result: {src:<24} | Error: {err:>5.2f} dB | Latency: {ms:>6.2f} ms")
        print(f"    Sizing: Wn={sz['Wn_um']:.2f}um, Rs={sz['Rs_ohm']:.0f}R, Cs={sz['Cs_farad']*1e12:.2f}pF, RL={sz['RL_ohm']:.0f}R")
