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

## Invariant-keyed blocks

A paged-KV block addressed by its token content alone is reusable by any request
that presents the same tokens. That is arithmetically sound and governance-unsound:
a block computed while a permissive invariant bundle was in force can be served to
a request running under a stricter one, and nothing in the cache layer can see the
difference.

`InvariantKeyedBlockTable` makes the governance regime part of the address:

```
block_key = SHA3-256( "szl.block-kv.invariant-key" || canonical(token_ids) || bundle_id )
```

Same tokens under a different bundle produce a different key, so the lookup misses
and the block is recomputed under the regime that asked for it. There is no code
path that returns a block keyed to another bundle.

```python
from szl_block_kv import InvariantBundle, InvariantKeyedBlockTable, ReceiptChain

permissive = InvariantBundle(doctrine="v11", policy_class="default-permissive",
                            receipts_required=True, verifier_budget=0)
strict = permissive.derive(policy_class="default-strict", verifier_budget=3)

chain = ReceiptChain()
table = InvariantKeyedBlockTable(num_blocks=1024, chain=chain)
run = [15496, 11, 995, 13]

table.lookup(run, permissive)      # (None, "miss_cold")
physical = table.admit(run, permissive)
table.lookup(run, permissive)      # (physical, "hit")
table.lookup(run, strict)          # (None, "miss_cross_regime")  <- containment
chain.verify()                     # (True, 4, -1)
```

A cross-regime miss is reported distinctly from a cold miss, because the two mean
different things operationally: a cold miss is capacity or novelty, a cross-regime
miss is governance doing its job. `table.audit()` asserts structurally that no
physical block serves two regimes, and every lookup, admission, rejection, and
eviction is appended to the same SHA3-256 receipt chain as `paged_attn`.

Self-check: `from szl_block_kv import selfcheck_invariant; selfcheck_invariant()`

| Thing | Label | Method / what-NOT |
|---|---|---|
| Cross-regime containment | **MEASURED** | 14 tests in `tests/test_invariant.py`, verified 2026-08-30. Covers canonical bundle ids, domain separation against boundary confusion, hit vs cold-miss vs cross-regime-miss, refusal to rebind a physical block across regimes, no-block-serves-two-regimes under 4 regimes x 3 runs, auditability across FIFO eviction, and receipt-chain depth. What-NOT: not a proof about attention arithmetic; no tokens/s; no joules. |
| Reuse fidelity under one regime | **MEASURED** | `test_reuse_under_the_same_regime_is_numerically_exact`: a served hit is `torch.equal` to recompute, and both match contiguous-KV SDPA within atol/rtol 1e-5 on float32. What-NOT: no speedup claim. |
| Eviction quality | **UNAVAILABLE** | v0 eviction is FIFO. No eviction-quality or hit-rate claim is made. |
| Cost of keying | **UNAVAILABLE** | The hit-rate cost of narrowing the keyspace by bundle is not measured here. Measure it on your own traffic before promotion. |

**No speedup claim. Λ = Conjecture 1 OPEN.**

## Load

```python
from kernels import get_kernel

kv = get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)
print(kv.selfcheck())
```

Local: put `torch-ext/` on `PYTHONPATH` and `from szl_block_kv import PagedCache, paged_attn, selfcheck`.

Apache-2.0. Copyright 2026 SZL Holdings.
