# SPDX-License-Identifier: Apache-2.0
"""The benchmark harness runs on CPU against the in-tree package and matches its reference."""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))
import szl_block_kv  # noqa: E402

spec = importlib.util.spec_from_file_location("szl_bkv_bench", ROOT / "benchmarks" / "benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_harness_matches_reference_on_cpu():
    b = bench.PagedAttnNoncausal()
    b.kernel, b.device = szl_block_kv, "cpu"
    b.shape = dict(b=2, h=2, t=32, d=16, bs=8)
    b.setup()
    b.benchmark_paged_attn_decode()
    assert torch.allclose(b.out, b.verify_paged_attn_decode(), atol=1e-5)


def test_harness_ships_no_results():
    text = (ROOT / "benchmarks" / "benchmark.py").read_text()
    assert "ms" not in [w.strip(".,") for w in text.split()]  # no timing tables embedded
    assert not list((ROOT / "benchmarks").glob("*.json"))
