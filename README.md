# szl-block-kv
<!-- szl:header v1 -->
<!-- badges: add this repo's CI / release / status badges here -->
[![org: szl-holdings](https://img.shields.io/badge/org-szl--holdings-black)](https://github.com/szl-holdings)
[![doctrine](https://img.shields.io/badge/doctrine-control%20before%20action%20%C2%B7%20evidence%20after-blue)](https://a-11-oy.com)

**Control before action. Evidence after.**

Part of the [szl-holdings](https://github.com/szl-holdings) estate ·
Product: [a-11-oy.com](https://a-11-oy.com) ·
Proof: [a11oy.net](https://a11oy.net)
<!-- /szl:header -->

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

Local checkout:

```python
from szl_block_kv import PagedCache, paged_attn, reshape_and_cache, selfcheck
print(selfcheck())
```

Correctness (documented): paged gather matches contiguous KV SDPA within atol/rtol `1e-5` on float32. Skip CUDA Triton — that kernel is not in v0.

Apache-2.0. Copyright 2026 SZL Holdings.
