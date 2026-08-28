# szl-block-kv

Canonical GitHub source for `SZLHOLDINGS/szl-block-kv`.

**Original SZL construction in the paged-KV category.** Inspired by Kwon et al. PagedAttention SOSP 2023 https://arxiv.org/abs/2309.06180. **NOT a rehost of vLLM or kernels-community/paged-attention.** **Distinct from a11oy MODELED H2O eviction.**

Doctrine v11. **Λ = Conjecture 1 OPEN** (advisory; uniqueness unproven).

v0 is a **labeled torch gather** over a block table. GPU paged Triton is **ROADMAP**. **No speedup claim.**

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
