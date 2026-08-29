---
library_name: kernels
license: apache-2.0
tags:
  - kernel
  - paged-attention
  - kv-cache
  - provenance
  - szl-holdings
---

# szl-block-kv

**Original SZL construction in the paged-KV category. Inspired by Kwon et al. PagedAttention SOSP 2023 https://arxiv.org/abs/2309.06180. NOT a rehost of vLLM or kernels-community/paged-attention. Distinct from a11oy MODELED H2O eviction.**

Doctrine v11. **Λ = Conjecture 1 OPEN** (advisory; uniqueness unproven — not proven trust).

GitHub bytes are the artifact. Hub is the publish mirror.

v0 path: labeled **torch gather** by block table. GPU Triton page kernel is **UNAVAILABLE**. **No speedup claim. No fabricated benchmark.**

## Load

```python
from kernels import get_kernel

kv = get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)
print(kv.selfcheck())
```

Local: put `torch-ext/` on `PYTHONPATH` and `from szl_block_kv import PagedCache, paged_attn, selfcheck`.

Apache-2.0. Copyright 2026 SZL Holdings.
