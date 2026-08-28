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

def reshape_and_cache(k: torch.Tensor, v: torch.Tensor, cache: PagedCache, slot_mapping: torch.Tensor) -> None:
    """k,v: [T, H, D]; slot_mapping: [T] linear slot = block * block_size + offset."""
    t = k.shape[0]
    bs = cache.block_size
    slots = slot_mapping.long()
    blk = torch.div(slots, bs, rounding_mode="floor")
    off = slots % bs
    cache.k[blk, off] = k
    cache.v[blk, off] = v

def _gather_kv(cache: PagedCache, block_tables: torch.Tensor, context_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """block_tables: [B, max_blocks] -> contiguous K/V [B, H, Tmax, D] padded."""
    b, max_blocks = block_tables.shape
    bs = cache.block_size
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
    """q: [B, H, Tq, D]. v0 torch gather; Triton page kernel is ROADMAP."""
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
    y = paged_attn(q, cache, tables, clens, chain=chain)
    k_ref = k.permute(1, 0, 2).unsqueeze(0)  # [1,H,T,D]
    v_ref = v.permute(1, 0, 2).unsqueeze(0)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, dropout_p=0.0, is_causal=False)
    err = float((y - ref).abs().max().item())
    ok_c, depth, _ = chain.verify()
    ok = bool(err < 1e-5 and ok_c)
    return {"ok": ok, "max_abs_vs_contiguous": err, "chain_ok": ok_c, "chain_depth": depth,
            "path": "torch_gather", "lambda": "Conjecture 1",
            "note": "v0 gather matches contiguous KV; Triton page kernel ROADMAP; no speedup claimed"}
