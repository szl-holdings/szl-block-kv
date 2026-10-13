# SPDX-License-Identifier: Apache-2.0
"""Smoke hook for hf-kernel-publish (szl-block-kv).

`smoke(kernel)` receives the module that the `kernels` loader imported, from a local
build (check-local) or from the Kernel Hub (verify). It raises on any failure.
CPU-only, deterministic and fast; the returned summary is written to the receipt.
"""
from __future__ import annotations


def smoke(kernel):
    import torch

    gather = kernel.selfcheck()
    if gather.get("ok") is not True or gather.get("chain_ok") is not True:
        raise AssertionError(f"selfcheck failed: {gather}")
    invariant = kernel.selfcheck_invariant()
    if invariant.get("ok") is not True or invariant.get("cross_regime_lookup") != "miss_cross_regime":
        raise AssertionError(f"selfcheck_invariant failed: {invariant}")

    # Causal paged attention is not defined by v0 and must fail closed (#25).
    cache = kernel.PagedCache(num_blocks=1, block_size=2, n_heads=1, d_head=2)
    q = torch.zeros(1, 1, 1, 2)
    tables = torch.tensor([[0]])
    lens = torch.tensor([1])
    try:
        kernel.paged_attn(q, cache, tables, lens)
    except NotImplementedError:
        causal = "refused"
    else:
        raise AssertionError("paged_attn with the default causal=True must fail closed")

    return {
        "selfcheck": gather["ok"],
        "max_abs_vs_contiguous": gather["max_abs_vs_contiguous"],
        "selfcheck_invariant": invariant["ok"],
        "cross_regime_lookup": invariant["cross_regime_lookup"],
        "causal_default": causal,
        "version": getattr(kernel, "__version__", None),
    }
