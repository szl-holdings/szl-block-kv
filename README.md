# szl-block-kv

Canonical GitHub source for `SZLHOLDINGS/szl-block-kv`.

**Original SZL construction in the paged-KV category.** Inspired by Kwon et al. PagedAttention SOSP 2023 https://arxiv.org/abs/2309.06180. **NOT a rehost of vLLM or kernels-community/paged-attention.** **Distinct from a11oy MODELED H2O eviction.**

Doctrine v11. **Λ = Conjecture 1 OPEN** (advisory; uniqueness unproven).

<!-- SZL-KERNEL-STATUS:import-LIVE:START -->
## Status

> **STATUS: import-LIVE** on CPU Kernel Hub `get_kernel` (kernels `0.16.1`). Triton page kernel is **UNAVAILABLE**.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| Kernel Hub `get_kernel` | **import-LIVE** | MEASURED 2026-08-28 2:29pm ET on kernels `0.16.1`. HEAD [`d3ede3e`](https://huggingface.co/kernels/SZLHOLDINGS/szl-block-kv/commit/d3ede3e471b51080492b1c69306283507dcf507e) (`d3ede3e471b51080492b1c69306283507dcf507e`). Legal name `szl-block-kv` (Python module `szl_block_kv`). Variants: `build/torch-universal` (default `get_kernel`) and `build/torch-cpu` (`backend="cpu"`). Working calls: `get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)` and the same with `backend="cpu"`. `selfcheck` **ok**. `max_abs_vs_contiguous=2.38e-07` (full `2.384185791015625e-07`), `path=torch_gather`, `chain_ok=true`. What-NOT: no tokens/s; no joules. |
| Triton page kernel | **UNAVAILABLE** | Not claimed LIVE. GPU / Triton page kernel is UNAVAILABLE. |

<!-- SZL-KERNEL-STATUS:import-LIVE:END -->

v0 is a **labeled torch gather** over a block table. GPU paged Triton is **UNAVAILABLE**. **No speedup claim.**

## Load

```python
from kernels import get_kernel

kv = get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)
print(kv.selfcheck())
```

Local checkout:

```python
from szl_block_kv import PagedCache, paged_attn, reshape_and_cache, selfcheck
print(selfcheck())
```

Correctness (documented): paged gather matches contiguous KV SDPA within atol/rtol `1e-5` on float32. Skip CUDA Triton — that kernel is not in v0.

Apache-2.0. Copyright 2026 SZL Holdings.
