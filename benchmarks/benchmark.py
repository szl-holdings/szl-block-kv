# SPDX-License-Identifier: Apache-2.0
"""Benchmark harness for `kernels benchmark SZLHOLDINGS/szl-block-kv@v1` (kernels[benchmark]).

Workloads time the v0 torch-gather noncausal paged attention against a contiguous
SDPA reference and check that outputs match. This file contains no results; any
number it produces is a local measurement for the machine it ran on, and must not
be copied into a card until it is recorded with hardware, versions and raw output.
"""
import torch
import torch.nn.functional as F

try:
    from kernels.benchmark import Benchmark
except ImportError:  # harness importable for tests without the benchmark extra
    class Benchmark:  # type: ignore[no-redef]
        seed = 0
        device = "cpu"

        def __init__(self):
            self.kernel = None


def _case(kernel, device, b, h, t, d, bs, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, h, 1, d, generator=g).to(device)
    k = torch.randn(b * t, h, d, generator=g).to(device)
    v = torch.randn(b * t, h, d, generator=g).to(device)
    blocks_per_seq = (t + bs - 1) // bs
    cache = kernel.PagedCache(num_blocks=b * blocks_per_seq, block_size=bs, n_heads=h, d_head=d, device=device)
    slots = torch.arange(b * t, device=device)
    # sequence i owns slots [i*t, (i+1)*t); with t % bs == 0 those are blocks [i*bpb, (i+1)*bpb)
    kernel.reshape_and_cache(k, v, cache, slots)
    tables = torch.arange(b * blocks_per_seq, device=device).reshape(b, blocks_per_seq)
    lens = torch.full((b,), t, device=device)
    k_ref = k.reshape(b, t, h, d).permute(0, 2, 1, 3)
    v_ref = v.reshape(b, t, h, d).permute(0, 2, 1, 3)
    return q, cache, tables, lens, k_ref, v_ref


class PagedAttnNoncausal(Benchmark):
    seed = 0
    shape = dict(b=4, h=8, t=256, d=64, bs=16)

    def setup(self):
        self.q, self.cache, self.tables, self.lens, self.k_ref, self.v_ref = _case(
            self.kernel, self.device, seed=self.seed, **self.shape)
        self.out = torch.empty_like(self.q)

    def benchmark_paged_attn_decode(self):
        self.out = self.kernel.paged_attn(self.q, self.cache, self.tables, self.lens, causal=False)

    def verify_paged_attn_decode(self):
        return F.scaled_dot_product_attention(self.q, self.k_ref, self.v_ref, is_causal=False)
