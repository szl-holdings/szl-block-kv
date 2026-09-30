---
license: apache-2.0
library_name: kernels
tags:
- kernel
- paged-attention
- kv-cache
- provenance
- szl-holdings
szl:
  source_repo: szl-holdings/szl-block-kv
  proof_url: https://github.com/szl-holdings/szl-block-kv
---
<!-- hf-card: type=kernel source=szl-holdings/szl-block-kv vars=hf/card.yaml -->
<!-- Rendered by szl-holdings/.github hf-card/render.py. Edit hf/card.yaml in szl-holdings/szl-block-kv; do not edit this card on the Hub. -->

# szl-block-kv

**Original SZL construction in the paged-KV category.** Inspired by Kwon et al. PagedAttention SOSP 2023 https://arxiv.org/abs/2309.06180. **NOT a rehost of vLLM or kernels-community/paged-attention.** **Distinct from a11oy MODELED H2O eviction.** v0 is a labeled torch gather over a block table with SHA3-256 receipts. The GPU Triton page kernel is **UNAVAILABLE**. No speedup claim.

## Kernel

| Field | Value |
| --- | --- |
| Hub id | `SZLHOLDINGS/szl-block-kv` |
| Library | `kernels` |
| Backends | cpu (torch gather) |

```python
import re

# Set only after owner qualification of this first-class Kernel Hub release.
KERNEL_REVISION = "REPLACE_WITH_OWNER_QUALIFIED_KERNEL_COMMIT"
if re.fullmatch(r"[0-9a-f]{40}", KERNEL_REVISION) is None:
    raise ValueError("An owner-qualified immutable Kernel Hub commit is required")

# This loads and executes remote Python code; review the pinned source first.
from kernels import get_kernel

kv = get_kernel("SZLHOLDINGS/szl-block-kv", revision=KERNEL_REVISION, trust_remote_code=True)
print(kv.selfcheck())
```

## Doctrine

Doctrine v11. **Λ = Conjecture 1 OPEN** (advisory; uniqueness unproven, not proven trust). GitHub bytes are the artifact; the Hub is the publish mirror.

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

## Local source tree

Put `torch-ext/` on `PYTHONPATH`, then:

```python
from szl_block_kv import PagedCache, paged_attn, reshape_and_cache, selfcheck
print(selfcheck())
```

## Source and Hub release scope

Loading with `trust_remote_code=True` executes code from the selected first-class
Kernel Hub repository. Review that immutable source and qualify a compatible
`kernels` client before running it. Set `KERNEL_REVISION` to the owner-qualified
Kernel Hub publication commit; this card does not establish one. A GitHub
source commit or model-twin revision is not the provider revision. The syntax
check in the example does not establish release qualification.

This is staged GitHub card source, not evidence that the described build is
currently published or qualified on either Hub twin. The imported Hub package predates the invariant-keyed source APIs described here. This staged card must be published together with a matching qualified build, never as a card-only update.

Publication/import context is recorded in [hf/README.md](https://github.com/szl-holdings/szl-block-kv/blob/dd91c1ca2431b7d1ef65dbbb261b8b402605f111/hf/README.md).
Historical Hub-only benchmark receipts remain REPORTED at their stated scope.

## Claims

| Label | Claim | Evidence |
| --- | --- | --- |
| MEASURED | Cross-regime containment. The 14 tests in `tests/test_invariant.py` cover canonical bundle ids, domain separation against boundary confusion, hit vs cold miss vs cross-regime miss, refusal to rebind a physical block across regimes, no physical block serving two regimes (4 regimes x 3 runs), auditability across FIFO eviction, and receipt-chain depth. CI runs them on every pull request and every push to main (`kernel-smoke.yml`). What-NOT: not a proof about attention arithmetic. | [receipt](https://github.com/szl-holdings/szl-block-kv/blob/49a96b6386b8ea22db20810c8d96ac0ac7083970/tests/test_invariant.py) |
| MEASURED | Reuse fidelity under one regime. `test_reuse_under_the_same_regime_is_numerically_exact` asserts that a served hit is `torch.equal` to recompute, and that both match contiguous-KV SDPA within atol/rtol 1e-5 on float32. | [receipt](https://github.com/szl-holdings/szl-block-kv/blob/49a96b6386b8ea22db20810c8d96ac0ac7083970/tests/test_invariant.py) |
| MEASURED | Paged gather correctness. `tests/test_block_kv.py` asserts that paged attention over shuffled pages and over partial context lengths matches contiguous-KV SDPA within atol/rtol 1e-5 on float32. | [receipt](https://github.com/szl-holdings/szl-block-kv/blob/49a96b6386b8ea22db20810c8d96ac0ac7083970/tests/test_block_kv.py) |
| REPORTED | CPU `get_kernel` load from the Kernel Hub (kernels 0.16.1, `build/torch-universal` and `build/torch-cpu`, `selfcheck` ok) and a local pytest run, recorded 2026-08-29 in the Hub-only `BENCH.laptop-blackwell.json` and `OPERATIONAL.json` against source commit 7cf99cca. No receipt for them is committed in this repository. | none linked |
| UNAVAILABLE | GPU Triton page kernel. It is not in v0; no cubin and no timed GPU run exist. | none linked |
| UNAVAILABLE | Eviction quality. v0 eviction is FIFO; no eviction-quality or hit-rate claim is made. | none linked |
| UNAVAILABLE | Cost of keying. The hit-rate cost of narrowing the keyspace by bundle is not measured here. Measure it on your own traffic before promotion. | none linked |
| NOT_CLAIMED | Any speed-up, tokens/s or energy figure. | none linked |

Labels follow the [SZL claim language](https://github.com/szl-holdings/.github/blob/main/docs/CLAIM_LANGUAGE.md). A claim is only as strong as the receipt it links.

## Limits

- Kernel, not weights. Not a drop-in vLLM replacement.
- Receipts prove integrity and declared origin only; they do not prove accuracy, readiness or performance.

## Source and provenance

| Field | Value |
| --- | --- |
| Source repository | [szl-holdings/szl-block-kv](https://github.com/szl-holdings/szl-block-kv) |
| Proof | <https://github.com/szl-holdings/szl-block-kv> |
| License | `apache-2.0` |

This card is written to the Hub only by the committed mirror workflow of szl-holdings/szl-block-kv.
