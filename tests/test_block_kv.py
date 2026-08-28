# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))

import torch
import torch.nn.functional as F
from szl_block_kv import PagedCache, ReceiptChain, paged_attn, reshape_and_cache, selfcheck


def test_selfcheck():
    r = selfcheck()
    assert r["ok"] is True
    assert r["path"] == "torch_gather"
    assert r["lambda"] == "Conjecture 1"
    assert "ROADMAP" in r["note"]
    assert "no speedup" in r["note"].lower()


def test_shuffled_pages_match_contiguous():
    """Non-contiguous physical pages must match a contiguous KV reference."""
    torch.manual_seed(1)
    _b, h, t, d, bs = 1, 2, 8, 16, 4
    q = torch.randn(1, h, t, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    cache = PagedCache(num_blocks=4, block_size=bs, n_heads=h, d_head=d)
    # tokens 0..3 live on physical page 2; tokens 4..7 on physical page 0
    slots = torch.cat([torch.arange(bs) + 2 * bs, torch.arange(bs) + 0 * bs])
    reshape_and_cache(k, v, cache, slots)
    tables = torch.tensor([[2, 0, -1, -1]])
    y = paged_attn(q, cache, tables, torch.tensor([t]))
    k_ref = k.permute(1, 0, 2).unsqueeze(0)
    v_ref = v.permute(1, 0, 2).unsqueeze(0)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, dropout_p=0.0, is_causal=False)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)


def test_partial_context_lens_match_trimmed_contiguous():
    torch.manual_seed(2)
    h, t, d, bs = 2, 8, 16, 4
    q = torch.randn(1, h, t, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    cache = PagedCache(num_blocks=4, block_size=bs, n_heads=h, d_head=d)
    reshape_and_cache(k, v, cache, torch.arange(t))
    tables = torch.tensor([[0, 1, -1, -1]])
    clen = 5
    y = paged_attn(q, cache, tables, torch.tensor([clen]))
    k_ref = k[:clen].permute(1, 0, 2).unsqueeze(0)
    v_ref = v[:clen].permute(1, 0, 2).unsqueeze(0)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, dropout_p=0.0, is_causal=False)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)


def test_receipt_chain():
    torch.manual_seed(3)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(4, 1, 8)
    v = torch.randn(4, 1, 8)
    cache = PagedCache(num_blocks=2, block_size=4, n_heads=1, d_head=8)
    reshape_and_cache(k, v, cache, torch.arange(4))
    chain = ReceiptChain()
    paged_attn(q, cache, torch.tensor([[0, -1]]), torch.tensor([4]), chain=chain)
    ok, depth, brk = chain.verify()
    assert ok and depth == 1 and brk == -1
    assert chain.head() is not None and len(chain.head()) == 64


def test_cuda_skip_is_honest():
    """v0 has no Triton page kernel. Presence or absence of a GPU is not a failure."""
    r = selfcheck()
    assert r["path"] == "torch_gather"
    if not torch.cuda.is_available():
        return
    q = torch.randn(1, 1, 4, 8, device="cuda")
    k = torch.randn(4, 1, 8, device="cuda")
    v = torch.randn(4, 1, 8, device="cuda")
    cache = PagedCache(num_blocks=2, block_size=4, n_heads=1, d_head=8, device="cuda")
    reshape_and_cache(k, v, cache, torch.arange(4, device="cuda"))
    y = paged_attn(q, cache, torch.tensor([[0, -1]], device="cuda"), torch.tensor([4], device="cuda"))
    k_ref = k.permute(1, 0, 2).unsqueeze(0)
    v_ref = v.permute(1, 0, 2).unsqueeze(0)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, dropout_p=0.0, is_causal=False)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)
