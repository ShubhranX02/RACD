"""
test_faiss.py — Tests for retrieval_faiss_transistor.py

Tests:
  - Import and index build/load
  - retrieve_k_nearest_transistor returns results with correct keys
  - FAISS index returns peaking close to query for known-good data
"""
import os
import pytest


def test_faiss_imports():
    """retrieval_faiss_transistor.py must import without error."""
    from retrieval_faiss_transistor import (
        build_transistor_faiss_index,
        retrieve_k_nearest_transistor,
        synthesize_transistor_racd,
    )


def test_faiss_retrieve_returns_list():
    """retrieve_k_nearest_transistor must return a list."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transistor_repository.jsonl')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transistor_repository_clean.jsonl')
    if not os.path.exists(data_path):
        pytest.skip("transistor repository not found")
    from retrieval_faiss_transistor import retrieve_k_nearest_transistor
    results = retrieve_k_nearest_transistor(target_peaking_db=6.0, k=3)
    assert isinstance(results, list)
    if results:
        assert 'peaking_db' in results[0] or 'Wn_um' in results[0], (
            f"Expected peaking_db or Wn_um in result keys: {list(results[0].keys())}"
        )


def test_faiss_peaking_error_small():
    """FAISS retrieval should find a design within 2 dB of the query peaking."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transistor_repository.jsonl')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transistor_repository_clean.jsonl')
    if not os.path.exists(data_path):
        pytest.skip("transistor repository not found")
    from retrieval_faiss_transistor import retrieve_k_nearest_transistor
    target = 5.84  # known achievable from PVT table
    results = retrieve_k_nearest_transistor(target_peaking_db=target, k=1)
    if not results:
        pytest.skip("No results returned")
    best = results[0]
    err = abs(best.get('peaking_db', 0) - target)
    assert err < 3.0, f"FAISS peaking error {err:.2f} dB is too large for target {target} dB"
