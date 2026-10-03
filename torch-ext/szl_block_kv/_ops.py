# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Original paged KV via block tables. Not copied from vLLM .cu."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from ._chain import ReceiptChain

class PagedCache:
    def __init__(self, num_blocks: int, block_size: int, n_heads: int, d_head: int, device=None, dtype=torch.float32):
        self.block_size = int(block_size)
        self.k = torch.zeros(num_blocks, block_size, n_heads, d_head, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)

def _validate_index_tensor(indices: torch.Tensor, name: str, ndim: int) -> None:
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if (not isinstance(indices, torch.Tensor) or indices.dtype not in integer_dtypes
            or indices.layout != torch.strided or indices.ndim != ndim):
        raise ValueError(f"{name} must be a rank-{ndim} dense integer tensor")

def reshape_and_cache(k: torch.Tensor, v: torch.Tensor, cache: PagedCache, slot_mapping: torch.Tensor) -> None:
    """k,v: [T, H, D]; slot_mapping: [T] valid cache slots; no negative sentinel."""
    _validate_index_tensor(slot_mapping, "slot_mapping", 1)
    if (not isinstance(cache.k, torch.Tensor) or not isinstance(cache.v, torch.Tensor)
            or cache.k.ndim != 4 or cache.v.shape != cache.k.shape
            or cache.k.shape[1] != cache.block_size
            or cache.k.dtype != cache.v.dtype or cache.k.device != cache.v.device
            or cache.k.layout != torch.strided or cache.v.layout != torch.strided):
        raise ValueError("cache.k and cache.v must have matching rank-4 shape, dtype, device and dense layout")
    if (not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor)
            or k.ndim != 3 or v.shape != k.shape or k.shape[1:] != cache.k.shape[2:]
            or slot_mapping.shape[0] != k.shape[0]):
        raise ValueError("k and v must both have shape [T, H, D] matching the cache and slot_mapping")
    for name, value, destination in (("k", k, cache.k), ("v", v, cache.v)):
        if (value.dtype != destination.dtype or value.device != destination.device
                or value.layout != torch.strided):
            raise ValueError(f"{name} must match its cache dtype and device and have dense layout")
    bs = cache.block_size
    if bs <= 0:
        raise ValueError("cache.block_size must be positive")
    slots = slot_mapping.long()
    if torch.any((slots < 0) | (slots >= cache.k.shape[0] * bs)).item():
        raise ValueError("slot_mapping contains an out-of-range cache slot")
    blk = torch.div(slots, bs, rounding_mode="floor")
    off = slots % bs
    cache.k[blk, off] = k
    cache.v[blk, off] = v

def _gather_kv(cache: PagedCache, block_tables: torch.Tensor, context_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """block_tables: [B, max_blocks] -> contiguous K/V [B, H, Tmax, D] padded."""
    _validate_index_tensor(block_tables, "block_tables", 2)
    _validate_index_tensor(context_lens, "context_lens", 1)
    b, max_blocks = block_tables.shape
    bs = cache.block_size
    if bs <= 0 or context_lens.shape[0] != b:
        raise ValueError("context_lens must match the table batch and cache.block_size must be positive")
    contexts = context_lens.to(device=block_tables.device, dtype=torch.int64)
    if torch.any((contexts < 0) | (contexts > max_blocks * bs)).item():
        raise ValueError("context_lens contains a length outside the block-table capacity")
    nblocks = torch.div(contexts + bs - 1, bs, rounding_mode="floor")
    active = torch.arange(max_blocks, device=block_tables.device)[None, :] < nblocks[:, None]
    pages = block_tables.long()[active]
    if torch.any((pages < 0) | (pages >= cache.k.shape[0])).item():
        raise ValueError("block_tables contains an out-of-range active cache page")
    h, d = cache.k.shape[2], cache.k.shape[3]
    tmax = max_blocks * bs
    k_out = torch.zeros(b, h, tmax, d, device=cache.k.device, dtype=cache.k.dtype)
    v_out = torch.zeros_like(k_out)
    for bi in range(b):
        clen = int(context_lens[bi].item())
        nblk = (clen + bs - 1) // bs
        for j in range(nblk):
            blk = int(block_tables[bi, j].item())
            start = j * bs
            end = min(start + bs, clen)
            take = end - start
            k_out[bi, :, start:end] = cache.k[blk, :take].transpose(0, 1)
            v_out[bi, :, start:end] = cache.v[blk, :take].transpose(0, 1)
    return k_out, v_out

def paged_attn(q: torch.Tensor, cache: PagedCache, block_tables: torch.Tensor, context_lens: torch.Tensor,
              *, causal: bool = True, chain: Optional[ReceiptChain] = None, scale: Optional[float] = None) -> torch.Tensor:
    """Noncausal v0 torch gather for q: [B, H, Tq, D].

    Causal query positions are not defined by this API. The default causal=True
    therefore fails closed instead of silently returning noncausal attention.
    Triton page kernel is UNAVAILABLE.
    """
    if causal is not False:
        raise NotImplementedError(
            "causal paged attention is unavailable in v0; "
            "pass causal=False only for noncausal attention"
        )
    k, v = _gather_kv(cache, block_tables, context_lens)
    # trim to max context for SDPA; pad positions stay zeros and must be masked
    b, h, tq, d = q.shape
    tmax = k.shape[2]
    idx = torch.arange(tmax, device=q.device)[None, :]
    key_mask = idx < context_lens.to(q.device)[:, None]  # [B, Tkv]
    attn_mask = key_mask[:, None, None, :].expand(b, 1, tq, tmax)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=scale)
    if chain is not None:
        occupied = int((block_tables >= 0).sum().item())
        chain.emit({"op": "paged_attn", "num_blocks": int(cache.k.shape[0]), "block_size": cache.block_size,
                    "occupied_table_entries": occupied, "q_shape": list(q.shape),
                    "path": "torch_gather", "lambda": "Conjecture 1"})
    return y

def selfcheck() -> dict:
    torch.manual_seed(20260828)
    b, h, t, d, bs = 1, 2, 8, 16, 4
    q = torch.randn(b, h, t, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    cache = PagedCache(num_blocks=4, block_size=bs, n_heads=h, d_head=d)
    slots = torch.arange(t)
    reshape_and_cache(k, v, cache, slots)
    tables = torch.tensor([[0, 1, -1, -1]])
    clens = torch.tensor([t])
    chain = ReceiptChain()
    y = paged_attn(q, cache, tables, clens, causal=False, chain=chain)
    k_ref = k.permute(1, 0, 2).unsqueeze(0)  # [1,H,T,D]
    v_ref = v.permute(1, 0, 2).unsqueeze(0)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, dropout_p=0.0, is_causal=False)
    err = float((y - ref).abs().max().item())
    ok_c, depth, _ = chain.verify()
    ok = bool(err < 1e-5 and ok_c)
    return {"ok": ok, "max_abs_vs_contiguous": err, "chain_ok": ok_c, "chain_depth": depth,
            "path": "torch_gather", "lambda": "Conjecture 1",
            "note": "v0 gather matches contiguous KV; Triton page kernel UNAVAILABLE; no speedup claimed"}
