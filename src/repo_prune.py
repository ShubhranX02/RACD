"""
repo_prune.py
=============
Prunes circuit_repository.jsonl to keep only high-quality episodes.

Pruning pipeline
----------------
1.  Load all records from data/circuit_repository.jsonl.
2.  Compute a scalar *reward* for every record:

        reward = -(peaking_error_norm + noise_penalty_norm + eye_penalty_norm)

    where
        peaking_error_norm  = abs(peaking_db - target_peaking_db) / 8.0
        noise_penalty_norm  = max(0, (noise_mvrms - 1.5) / 1.5)
        eye_penalty_norm    = max(0, (100.0 - eye_height_proxy_mv) / 100.0)

3.  Bucket records by round(target_peaking_db)  (integers 3-11).
4.  Within each bucket keep the top 20% by reward.
5.  Also unconditionally keep records where
        abs(peaking_db - target_peaking_db) < 0.5   (good quality hits)
6.  Deduplicate the union by (round(Rs), round(Cs * 1e13)).
7.  Write pruned records to data/circuit_repository_pruned.jsonl.
8.  Print a per-bucket summary table.
"""

import json
import math
import os
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = r"c:\Users\Kio\Desktop\Dev\RACD2"
REPO_IN  = os.path.join(_PROJECT_ROOT, "data", "circuit_repository.jsonl")
REPO_OUT = os.path.join(_PROJECT_ROOT, "data", "circuit_repository_pruned.jsonl")

# ---------------------------------------------------------------------------
# Pruning hyper-parameters
# ---------------------------------------------------------------------------
TOP_FRACTION        = 0.20   # keep top 20% per bucket
GOOD_HIT_THRESHOLD  = 0.5    # abs(peaking_db - target) < 0.5 dB -> always keep
PEAKING_SPAN_DB     = 8.0    # 11 - 3 = 8 dB  (normalisation denominator)
NOISE_BASELINE      = 1.5    # mV rms  (penalty starts above this)
EYE_BASELINE        = 100.0  # mV      (penalty starts below this)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_reward(rec: dict) -> float:
    """Return the scalar quality reward for *rec* (higher is better)."""
    peaking_error_norm = abs(rec["peaking_db"] - rec["target_peaking_db"]) / PEAKING_SPAN_DB
    noise_penalty_norm = max(0.0, (rec["noise_mvrms"] - NOISE_BASELINE) / NOISE_BASELINE)
    eye_penalty_norm   = max(0.0, (EYE_BASELINE - rec["eye_height_proxy_mv"]) / EYE_BASELINE)
    return -(peaking_error_norm + noise_penalty_norm + eye_penalty_norm)


def _dedup_key(rec: dict) -> tuple:
    """Coarse deduplication key: (rounded Rs [ohm], rounded Cs [x10^-13 F])."""
    return (round(rec["Rs"]), round(rec["Cs"] * 1e13))


def _bucket_id(rec: dict) -> int:
    """Bucket key: nearest integer dB target."""
    return int(round(rec["target_peaking_db"]))


# ---------------------------------------------------------------------------
# Core pruning function
# ---------------------------------------------------------------------------

def prune_repository(
    path_in: str = REPO_IN,
    path_out: str = REPO_OUT,
) -> dict:
    """
    Run the full pruning pipeline.

    Returns
    -------
    dict with keys:
        n_original  - total records loaded
        n_pruned    - records written after pruning
        before      - {bucket: count} before pruning
        after       - {bucket: count} after pruning
    """

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print(f"[repo_prune] Loading records from:\n  {path_in}")
    records: list = []
    with open(path_in, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Skipping malformed line {line_no}: {exc}")

    n_original = len(records)
    print(f"[repo_prune] Loaded {n_original:,} records.")

    # ------------------------------------------------------------------
    # 2. Annotate each record with reward and bucket
    # ------------------------------------------------------------------
    for rec in records:
        rec["_reward"] = _compute_reward(rec)
        rec["_bucket"] = _bucket_id(rec)

    # ------------------------------------------------------------------
    # 3. Bucket records and count
    # ------------------------------------------------------------------
    buckets: dict = defaultdict(list)
    for rec in records:
        buckets[rec["_bucket"]].append(rec)

    before_counts: dict = {b: len(v) for b, v in sorted(buckets.items())}

    # ------------------------------------------------------------------
    # 4 & 5. Select: top-K% per bucket  union  good quality hits
    # ------------------------------------------------------------------
    # Maps dedup_key -> best rec seen so far
    kept_set: dict = {}

    def _try_add(rec: dict) -> None:
        """Insert *rec* into kept_set; on collision keep the higher-reward record."""
        key = _dedup_key(rec)
        existing = kept_set.get(key)
        if existing is None or rec["_reward"] > existing["_reward"]:
            kept_set[key] = rec

    for bkt, bucket_recs in buckets.items():
        # Sort descending by reward so we can slice the top fraction
        bucket_recs.sort(key=lambda r: r["_reward"], reverse=True)

        # Step 4: top 20% (always keep at least 1)
        cutoff = max(1, math.ceil(len(bucket_recs) * TOP_FRACTION))
        for rec in bucket_recs[:cutoff]:
            _try_add(rec)

        # Step 5: all good-quality hits regardless of rank
        for rec in bucket_recs:
            if abs(rec["peaking_db"] - rec["target_peaking_db"]) < GOOD_HIT_THRESHOLD:
                _try_add(rec)

    # ------------------------------------------------------------------
    # 6. Deduplication is implicit - kept_set is keyed by dedup key
    # ------------------------------------------------------------------
    pruned_records: list = list(kept_set.values())

    # ------------------------------------------------------------------
    # 7. Write output (strip internal annotation fields)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    print(f"[repo_prune] Writing {len(pruned_records):,} pruned records to:\n  {path_out}")

    _INTERNAL_KEYS = {"_reward", "_bucket"}
    with open(path_out, "w", encoding="utf-8") as fh:
        for rec in pruned_records:
            clean = {k: v for k, v in rec.items() if k not in _INTERNAL_KEYS}
            fh.write(json.dumps(clean) + "\n")

    # ------------------------------------------------------------------
    # 8. Per-bucket counts after pruning
    # ------------------------------------------------------------------
    after_counts: dict = defaultdict(int)
    for rec in pruned_records:
        after_counts[rec["_bucket"]] += 1

    return {
        "n_original": n_original,
        "n_pruned":   len(pruned_records),
        "before":     before_counts,
        "after":      dict(after_counts),
    }


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def _print_summary(stats: dict) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 52)
    print("  REPOSITORY PRUNING SUMMARY")
    print("=" * 52)
    print(f"  Original records : {stats['n_original']:>10,}")
    print(f"  Pruned records   : {stats['n_pruned']:>10,}")
    if stats["n_original"] > 0:
        pct = 100.0 * stats["n_pruned"] / stats["n_original"]
        print(f"  Retention        : {pct:>9.1f} %")
    print()
    print(f"  {'Bucket':>8}  {'Before':>8}  {'After':>8}  {'Kept %':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    all_buckets = sorted(set(stats["before"].keys()) | set(stats["after"].keys()))
    for b in all_buckets:
        n_before = stats["before"].get(b, 0)
        n_after  = stats["after"].get(b, 0)
        pct_b    = (100.0 * n_after / n_before) if n_before else float("nan")
        print(f"  {b:>8}  {n_before:>8,}  {n_after:>8,}  {pct_b:>7.1f}%")

    print("=" * 52)
    print()


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stats = prune_repository()
    _print_summary(stats)
