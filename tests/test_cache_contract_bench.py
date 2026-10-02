# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Deterministic controls for the synthetic CPU source-contract exercise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from scripts.bench_cache_contract import attention_probe, make_trace, run_trial


def test_token_only_counterexample_and_invariant_containment():
    # One run under A, a repeated hit, then the same run under B and A.
    trace = make_trace(requests=4, unique_runs=1, seed=0)
    result = run_trial(trace, capacity=2)

    assert result["invariant_table"]["stats"] == {
        "hit": 2,
        "miss_cold": 1,
        "miss_cross_regime": 1,
        "admitted": 2,
        "evicted": 0,
        "rejected_cross_regime": 0,
    }
    assert result["invariant_table"]["wrong_regime_hits"] == 0
    assert result["invariant_table"]["audit_ok"] is True
    assert result["invariant_table"]["receipt_chain_ok"] is True
    assert result["invariant_table"]["receipt_rows"] == 6
    assert result["invariant_table"]["elapsed_seconds"] is None
    assert result["unsafe_token_only_fifo"] == {
        "hits": 3,
        "safe_hits": 2,
        "stale_cross_regime_hits": 1,
        "misses_recomputed": 1,
    }


def test_seeded_trace_has_deterministic_counts_and_eviction():
    first = run_trial(make_trace(100, 7, 41), capacity=2)
    second = run_trial(make_trace(100, 7, 41), capacity=2)
    assert first == second
    assert first["invariant_table"]["stats"]["evicted"] > 0
    assert first["invariant_table"]["audit_ok"] is True


def test_invalid_workload_is_rejected():
    with pytest.raises(ValueError):
        make_trace(3, 1, 0)
    with pytest.raises(ValueError):
        make_trace(4, 0, 0)
    with pytest.raises(ValueError):
        run_trial(make_trace(4, 1, 0), capacity=1)


def test_shuffled_pages_two_regimes_match_cpu_sdpa():
    result = attention_probe()
    assert result["device"] == "cpu"
    assert result["changed_regime_cross_misses"] == 2
    assert result["max_abs_vs_contiguous_sdpa"] <= result["tolerance"]
    assert result["stale_output_max_abs_difference"] > 0.1
