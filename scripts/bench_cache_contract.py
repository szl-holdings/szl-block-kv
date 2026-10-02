# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Reproducible CPU exercise of the source cache-addressing contract.

The token-only comparator is intentionally unsafe and is not vLLM (or any
other product). Timings cover Python lookup/admission/receipt work, not KV
construction, attention, GPU throughput, tokens/s, or energy.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))

from szl_block_kv import (  # noqa: E402
    InvariantBundle,
    InvariantKeyedBlockTable,
    PagedCache,
    ReceiptChain,
    paged_attn,
    reshape_and_cache,
)


def make_trace(requests: int, unique_runs: int, seed: int) -> list[tuple[tuple[int, ...], int]]:
    """A synthetic trace with a guaranteed same-token regime transition."""
    if requests < 4 or unique_runs < 1:
        raise ValueError("requests must be >= 4 and unique_runs must be >= 1")
    rng = random.Random(seed)
    catalog = [tuple((run * 17 + offset) % 32000 for offset in range(4)) for run in range(unique_runs)]
    trace = [(catalog[0], regime) for regime in (0, 0, 1, 0)]
    for index in range(4, requests):
        trace.append((catalog[rng.randrange(unique_runs)], (index // 11 + rng.randrange(3)) % 3))
    return trace


def run_trial(
    trace: Sequence[tuple[tuple[int, ...], int]], capacity: int, *, timed: bool = False
) -> dict:
    """Compare real invariant-keyed addressing with an unsafe token-only FIFO.

    The baseline stores the bundle that produced each block but intentionally
    excludes it from the key; a hit under another bundle is counted as stale.
    It never computes attention or claims to represent a deployed cache.
    """
    if capacity < 2:
        raise ValueError("capacity must be >= 2 to retain both sentinel regimes")
    bundles = [
        InvariantBundle(doctrine="synthetic-benchmark-v1", policy_class=f"regime-{index}")
        for index in range(3)
    ]
    chain = ReceiptChain()
    table = InvariantKeyedBlockTable(capacity, chain=chain)
    start = time.perf_counter_ns() if timed else None
    for run, regime in trace:
        bundle = bundles[regime]
        physical, _outcome = table.lookup(run, bundle)
        if physical is None:
            table.admit(run, bundle)
        elif table.bundle_of(physical) != bundle.id:
            raise AssertionError("invariant-keyed table served a foreign regime")
    elapsed = (time.perf_counter_ns() - start) / 1e9 if timed else None
    audit = table.audit()
    chain_ok, chain_depth, bad_row = chain.verify()
    if not audit["ok"] or not chain_ok:
        raise AssertionError(f"cache or receipt audit failed: {audit['problems']}, row {bad_row}")

    # Deliberately simpler than the real table: a token tuple is the entire
    # address. Count correctness, not a speed ratio against this toy mapping.
    baseline: dict[tuple[int, ...], str] = {}
    baseline_hits = baseline_safe_hits = baseline_stale_hits = baseline_misses = 0
    for run, regime in trace:
        bundle_id = bundles[regime].id
        prior_bundle = baseline.get(run)
        if prior_bundle is not None:
            baseline_hits += 1
            if prior_bundle == bundle_id:
                baseline_safe_hits += 1
            else:
                baseline_stale_hits += 1
        else:
            baseline_misses += 1
            if len(baseline) == capacity:
                baseline.pop(next(iter(baseline)))  # FIFO, like source v0
            baseline[run] = bundle_id

    if baseline_stale_hits < 1 or table.stats["miss_cross_regime"] < 1:
        raise AssertionError("synthetic trace failed to exercise the cross-regime counterexample")
    return {
        "capacity_blocks": capacity,
        "requests": len(trace),
        "regimes": len(bundles),
        "invariant_table": {
            "stats": dict(table.stats),
            "wrong_regime_hits": 0,
            "audit_ok": audit["ok"],
            "receipt_chain_ok": chain_ok,
            "receipt_rows": chain_depth,
            "elapsed_seconds": elapsed,
        },
        "unsafe_token_only_fifo": {
            "hits": baseline_hits,
            "safe_hits": baseline_safe_hits,
            "stale_cross_regime_hits": baseline_stale_hits,
            "misses_recomputed": baseline_misses,
        },
    }


def attention_probe() -> dict:
    """Check two regimes against CPU contiguous SDPA with distinct KV values.

    Different values are a synthetic counterexample: they show why a stale hit
    can be numerically wrong, not that real policy changes always alter KV.
    """
    import torch
    import torch.nn.functional as F

    generator = torch.Generator(device="cpu").manual_seed(20261002)
    q = torch.randn(1, 2, 5, 8, generator=generator)
    k_a = torch.randn(8, 2, 8, generator=generator)
    v_a = torch.randn(8, 2, 8, generator=generator)
    k_b, v_b = k_a + 0.25, v_a + 0.5
    cache = PagedCache(num_blocks=4, block_size=4, n_heads=2, d_head=8, device="cpu")
    reshape_and_cache(k_a, v_a, cache, torch.tensor([8, 9, 10, 11, 0, 1, 2, 3]))
    reshape_and_cache(k_b, v_b, cache, torch.tensor([4, 5, 6, 7, 12, 13, 14, 15]))
    bundles = [InvariantBundle(test_regime=regime) for regime in ("a", "b")]
    table = InvariantKeyedBlockTable(4)
    runs = [(101, 102, 103, 104), (105, 106, 107, 108)]
    for run, physical in zip(runs, (2, 0)):
        table.admit(run, bundles[0], physical=physical)
    if any(table.lookup(run, bundles[1]) != (None, "miss_cross_regime") for run in runs):
        raise AssertionError("changed regime reused a block")
    for run, physical in zip(runs, (1, 3)):
        table.admit(run, bundles[1], physical=physical)

    outputs = []
    errors = []
    for bundle, k, v in ((bundles[0], k_a, v_a), (bundles[1], k_b, v_b)):
        physical = [table.lookup(run, bundle)[0] for run in runs]
        if any(item is None for item in physical):
            raise AssertionError("admitted block was not addressable")
        actual = paged_attn(
            q, cache, torch.tensor([physical]), torch.tensor([8]), causal=False
        )
        reference = F.scaled_dot_product_attention(
            q, k.permute(1, 0, 2).unsqueeze(0), v.permute(1, 0, 2).unsqueeze(0),
            dropout_p=0.0, is_causal=False,
        )
        errors.append(float((actual - reference).abs().max().item()))
        outputs.append(actual)
    if max(errors) > 1e-5 or not table.audit()["ok"]:
        raise AssertionError(f"CPU SDPA reference mismatch: {errors}")
    stale_difference = float((outputs[0] - outputs[1]).abs().max().item())
    if stale_difference <= 0.1:
        raise AssertionError("synthetic KV change did not expose a stale-output difference")
    return {
        "device": "cpu",
        "dtype": "float32",
        "attention_mode": "noncausal_explicit",
        "tokens": 8,
        "max_abs_vs_contiguous_sdpa": max(errors),
        "stale_output_max_abs_difference": stale_difference,
        "changed_regime_cross_misses": 2,
        "tolerance": 1e-5,
        "what_not": "synthetic KV values; not a model, training, GPU, or causal-attention result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--unique-runs", type=int, default=128)
    parser.add_argument("--capacities", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args()
    trials = [
        {"seed": seed, **run_trial(make_trace(args.requests, args.unique_runs, seed), capacity,
                                   timed=not args.no_timing)}
        for capacity in args.capacities for seed in args.seeds
    ]
    repo = Path(__file__).resolve().parents[1]
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    source_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout)
    print(json.dumps({
        "label": "MEASURED synthetic CPU source-contract exercise",
        "scope": "Python invariant-keyed table lookup/admit with SHA3 receipt chain",
        "timing_scope": "wall time for real table lookup/admit/receipt emit only; no baseline speed ratio",
        "trace": "synthetic seeded token runs and three synthetic invariant bundles",
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "trials": trials,
        "attention_probe": attention_probe(),
        "what_not": "no real traffic, KV compute timing, GPU page kernel, tokens/s, joules, model training, or novelty proof",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
