"""
retrieval_faiss.py
==================
FAISS-based nearest-neighbour retrieval over the circuit episode repository.

Feature vector (dim=5, all normalised to roughly [0, 1]):
  [0]  peaking_db          / 11.0
  [1]  noise_mvrms         /  1.5
  [2]  eye_height_proxy_mv / 700.0
  [3]  (Rs - 50)           / 450.0
  [4]  (Cs - 0.1e-12)      / 9.9e-12

Falls back to the scalar functions in retrieval.py when FAISS is unavailable
or the repository is empty.
"""

import json
import os
import time

import numpy as np

# Attempt to import FAISS; degrade gracefully if not installed.
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    _FAISS_AVAILABLE = False

# Scalar fallback implementations.
from retrieval import retrieve_k_nearest, retrieve_closest

# ---------------------------------------------------------------------------
# Normalisation constants (must be kept in sync with the feature-vector
# construction in both build_faiss_index and retrieve_k_nearest_faiss).
# ---------------------------------------------------------------------------
_NORM = {
    "peaking_db":           11.0,
    "noise_mvrms":           1.5,
    "eye_height_proxy_mv": 700.0,
    "Rs_offset":            50.0,
    "Rs_scale":            450.0,
    "Cs_offset":             0.1e-12,
    "Cs_scale":              9.9e-12,
}


def _episode_to_vector(episode: dict) -> np.ndarray:
    """
    Convert a single episode dict to a normalised float32 feature vector.

    Missing optional fields default to zero (after normalisation).
    """
    peaking        = episode.get("peaking_db", 0.0)          / _NORM["peaking_db"]
    noise          = episode.get("noise_mvrms", 0.0)          / _NORM["noise_mvrms"]
    eye_height     = episode.get("eye_height_proxy_mv", 0.0)  / _NORM["eye_height_proxy_mv"]
    rs_norm        = (episode.get("Rs", 50.0) - _NORM["Rs_offset"]) / _NORM["Rs_scale"]
    cs_norm        = (episode.get("Cs", 0.1e-12) - _NORM["Cs_offset"]) / _NORM["Cs_scale"]
    return np.array([peaking, noise, eye_height, rs_norm, cs_norm], dtype=np.float32)


def _spec_to_query_vector(target_spec: dict) -> np.ndarray:
    """
    Convert a target-spec dict to the same normalised feature space used when
    indexing episodes.

    Required key:
      - 'target_peaking_db'  (float)

    Optional keys (default to 0.0 when absent, meaning they do not influence
    distance ranking for missing dimensions):
      - 'noise_limit_mvrms'   (float)
      - 'eye_height_limit_mv' (float)
      - 'Rs'                  (float)
      - 'Cs'                  (float)
    """
    peaking    = target_spec.get("target_peaking_db", 0.0)   / _NORM["peaking_db"]
    noise      = target_spec.get("noise_limit_mvrms", 0.0)   / _NORM["noise_mvrms"]
    eye_height = target_spec.get("eye_height_limit_mv", 0.0) / _NORM["eye_height_proxy_mv"]
    rs_norm    = (target_spec.get("Rs", 50.0)    - _NORM["Rs_offset"]) / _NORM["Rs_scale"]
    cs_norm    = (target_spec.get("Cs", 0.1e-12) - _NORM["Cs_offset"]) / _NORM["Cs_scale"]
    return np.array([peaking, noise, eye_height, rs_norm, cs_norm], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_faiss_index(repository: list, max_error_threshold: float = 1.0):
    """
    Build a FAISS flat-L2 index over the good episodes in *repository*.

    Parameters
    ----------
    repository : list[dict]
        List of episode dicts loaded from circuit_repository.jsonl.
    max_error_threshold : float
        Maximum allowed |achieved_peaking - target_peaking| in dB to consider
        an episode good (i.e. the agent actually hit its target).

    Returns
    -------
    tuple
        (index, good_episodes, vectors_array) on success.
        (None, [], None)                       if FAISS unavailable / repo empty.
    """
    if not _FAISS_AVAILABLE:
        print("[retrieval_faiss] WARNING: faiss not installed - index not built.")
        return None, [], None

    # Filter to episodes where the agent hit its target.
    good_episodes = [
        ep for ep in repository
        if abs(ep.get("peaking_db", float("inf")) - ep.get("target_peaking_db", 0.0))
           < max_error_threshold
    ]

    if not good_episodes:
        print("[retrieval_faiss] WARNING: no good episodes found - index not built.")
        return None, [], None

    # Build (N, 5) matrix.
    vectors = np.vstack([_episode_to_vector(ep) for ep in good_episodes])  # (N, 5)

    # FAISS requires C-contiguous float32.
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    dim = vectors.shape[1]  # 5
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    print(f"[retrieval_faiss] Built FAISS index: {index.ntotal} vectors, dim={dim}")
    return index, good_episodes, vectors


def retrieve_k_nearest_faiss(
    target_spec: dict,
    repository: list,
    k: int = 10,
    max_error_threshold: float = 1.0,
) -> list:
    """
    Return the k nearest episodes to *target_spec* using FAISS.

    Parameters
    ----------
    target_spec : dict
        Must contain 'target_peaking_db'; may optionally contain
        'noise_limit_mvrms', 'eye_height_limit_mv', 'Rs', 'Cs'.
    repository : list[dict]
        Full episode repository.
    k : int
        Number of neighbours to return.
    max_error_threshold : float
        Passed to build_faiss_index for good-episode filtering.

    Returns
    -------
    list[dict]
        Up to k episode dicts, ordered nearest to furthest.
        Falls back to retrieval.retrieve_k_nearest on any failure.
    """
    try:
        index, good_episodes, _ = build_faiss_index(repository, max_error_threshold)

        if index is None or not good_episodes:
            # Fall back to scalar retrieval.
            target_peaking = target_spec.get("target_peaking_db", 0.0)
            return retrieve_k_nearest(target_peaking, repository, k=k,
                                      max_error_threshold=max_error_threshold)

        query = _spec_to_query_vector(target_spec).reshape(1, -1)  # (1, 5)
        actual_k = min(k, len(good_episodes))
        distances, indices = index.search(query, actual_k)          # both (1, k)

        results = [good_episodes[i] for i in indices[0] if i >= 0]
        return results

    except Exception as exc:  # noqa: BLE001
        print(f"[retrieval_faiss] FAISS retrieval failed ({exc}); falling back to scalar.")
        target_peaking = target_spec.get("target_peaking_db", 0.0)
        return retrieve_k_nearest(target_peaking, repository, k=k,
                                  max_error_threshold=max_error_threshold)


def retrieve_closest_faiss(
    target_spec: dict,
    repository: list,
    max_error_threshold: float = 1.0,
):
    """
    Return the single closest episode to *target_spec*, or None if the
    repository is empty / no good episodes exist.

    Parameters
    ----------
    target_spec : dict
        Same format as retrieve_k_nearest_faiss.
    repository : list[dict]
        Full episode repository.
    max_error_threshold : float
        Passed to build_faiss_index for good-episode filtering.

    Returns
    -------
    dict or None
    """
    results = retrieve_k_nearest_faiss(
        target_spec, repository, k=1, max_error_threshold=max_error_threshold
    )
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Helper: load a JSONL repository file
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list:
    """Load a newline-delimited JSON file and return a list of dicts."""
    episodes = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


# ---------------------------------------------------------------------------
# __main__ - smoke test + timing comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    REPO_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "circuit_repository.jsonl"
    )
    REPO_PATH = os.path.normpath(REPO_PATH)

    print("=" * 60)
    print("retrieval_faiss.py - self-test")
    print("=" * 60)
    print(f"Repository path : {REPO_PATH}")
    print(f"FAISS available : {_FAISS_AVAILABLE}")

    if not os.path.exists(REPO_PATH):
        print(f"\nWARNING: repository file not found at {REPO_PATH}")
        print("Generating a synthetic repository for testing ...")

        rng = np.random.default_rng(42)
        synthetic = []
        for _ in range(500):
            target   = rng.uniform(1.0, 10.0)
            achieved = target + rng.normal(0, 0.4)
            rs       = rng.uniform(50, 500)
            cs       = rng.uniform(0.1e-12, 10e-12)
            synthetic.append({
                "target_peaking_db":   round(float(target),   3),
                "peaking_db":          round(float(achieved),  3),
                "noise_mvrms":         round(float(rng.uniform(0.1, 1.4)), 4),
                "eye_height_proxy_mv": round(float(rng.uniform(50, 680)),  2),
                "Rs":                  round(float(rs),  2),
                "Cs":                  round(float(cs),  6),
            })
        repository = synthetic
    else:
        repository = _load_jsonl(REPO_PATH)

    print(f"Episodes loaded  : {len(repository)}")

    TARGET_SPEC = {
        "target_peaking_db":   8.0,
        "noise_limit_mvrms":   0.5,
        "eye_height_limit_mv": 400.0,
    }
    print(f"\nQuery spec       : {TARGET_SPEC}")
    print("-" * 60)

    # ── FAISS retrieval ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    faiss_results = retrieve_k_nearest_faiss(TARGET_SPEC, repository, k=5)
    t_faiss = time.perf_counter() - t0

    print(f"\n[FAISS] Top-{len(faiss_results)} neighbours  (elapsed: {t_faiss*1e3:.3f} ms):")
    for i, ep in enumerate(faiss_results, 1):
        print(
            f"  {i}. peaking={ep.get('peaking_db', '?'):6.3f} dB  "
            f"target={ep.get('target_peaking_db', '?'):5.2f} dB  "
            f"Rs={ep.get('Rs', '?'):7.2f} Ohm  "
            f"Cs={ep.get('Cs', '?'):.4e} F"
        )

    # ── Scalar fallback retrieval ────────────────────────────────────────────
    t0 = time.perf_counter()
    scalar_results = retrieve_k_nearest(
        TARGET_SPEC["target_peaking_db"], repository, k=5
    )
    t_scalar = time.perf_counter() - t0

    print(f"\n[Scalar] Top-{len(scalar_results)} neighbours  (elapsed: {t_scalar*1e3:.3f} ms):")
    for i, ep in enumerate(scalar_results, 1):
        print(
            f"  {i}. peaking={ep.get('peaking_db', '?'):6.3f} dB  "
            f"target={ep.get('target_peaking_db', '?'):5.2f} dB  "
            f"Rs={ep.get('Rs', '?'):7.2f} Ohm  "
            f"Cs={ep.get('Cs', '?'):.4e} F"
        )

    # ── Closest episode ──────────────────────────────────────────────────────
    closest = retrieve_closest_faiss(TARGET_SPEC, repository)
    print(f"\n[Closest] {closest}")

    # ── Speed summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Timing summary (N={len(repository)} episodes):")
    print(f"  FAISS  : {t_faiss*1e3:.3f} ms")
    print(f"  Scalar : {t_scalar*1e3:.3f} ms")
    if _FAISS_AVAILABLE and t_faiss > 0:
        print(f"  Speedup: {t_scalar/t_faiss:.1f}x")
    else:
        print("  (FAISS unavailable - scalar used throughout)")
    print("=" * 60)
