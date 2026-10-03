# Hugging Face source and CPU publication

This folder is the GitHub source for the two Hub twins of this repository:

- the Kernel Hub repo `kernels/SZLHOLDINGS/szl-block-kv` (what `get_kernel` loads);
- the model-type twin `SZLHOLDINGS/szl-block-kv`.

| Path | What it is |
| --- | --- |
| `card.yaml` | The card source for **both** twins (plan decision D9: one source, one template). |
| `../CARD.md` | Its rendering by the shared [`hf-card`](https://github.com/szl-holdings/.github/tree/main/hf-card) toolkit. This is the file a mirror publishes as `README.md`. |
| `hub-import/` | The historical 2026-09-29 import: card bytes, revisions, digests, and file lists of both twins. It is a baseline, not current publication state. |
| `releases/20261003-cache-index-cpu.json` | Source-bound CPU publication readback and conformance receipt for the repaired first-class kernel. |

CI keeps them honest:

- `.github/workflows/hf-card.yml` fails when `CARD.md` drifts from `card.yaml`, or when the card breaks the card contract: front matter (`license`, `library_name: kernels`, `szl.source_repo`, `szl.proof_url`), and decision D10 (every `MEASURED` claim links a receipt pinned to a commit of this repository).
- `tests/test_hf_card_source.py` checks, offline, that `hub-import/` still holds the imported bytes and that `CARD.md` names this repository and its Apache-2.0 license.

To edit the card, change `card.yaml` and re-render:

```bash
python <hf-card>/render.py hf/card.yaml --out CARD.md
```

To refresh the import (read-only, anonymous public Hub API, no token):

```bash
python scripts/hf_hub_import.py fetch
```

## First-class kernel CPU publication

The manual Git publication `54a19796ee2c4f761c9482f77dd1f989a773e17f` serves source
`217e3b24017cd5fadb2525a7b5207f7c89a46413` from this repository. It includes the cache
index admission repair, the invariant-keyed APIs, the rendered card, and Apache-2.0
license. It updates the existing `torch-cpu` and `torch-universal` Python variants;
qualification was run with `backend="cpu"` only.

**MEASURED publication readback:** all 12 declared files matched their byte hashes;
the source manifest and CPU receipt chain verified; `.gitattributes`,
`BENCH.laptop-blackwell.json`, and `OPERATIONAL.json` were preserved byte-for-byte.
**REPORTED CPU conformance:** a fresh `get_kernel` load using `kernels==0.16.1`,
Python 3.11.9, and PyTorch 2.10.0 passed all 92 source tests, `selfcheck()`, and
`selfcheck_invariant()`. The receipt records the source/harness digests, fixture
scope, seeds, CPU hardware, and results. Independent replay is **UNAVAILABLE**;
key trust remains **REPO_DECLARED**. No GPU, Triton, training, speedup, energy, or
scientific qualification is established.

The [release record](releases/20261003-cache-index-cpu.json) binds these observations
to the immutable first-class kernel revision. Review the pinned code before loading:

```python
from kernels import get_kernel

kv = get_kernel(
    "SZLHOLDINGS/szl-block-kv",
    revision="54a19796ee2c4f761c9482f77dd1f989a773e17f",
    backend="cpu",
    trust_remote_code=True,
)
print(kv.selfcheck())
print(kv.selfcheck_invariant())
```

`trust_remote_code=True` permits execution of the reviewed Python package; it is
not publisher certification. Causal attention still fails closed. The release did
not mutate the separately owned model-type twin or retire its cross-writer.

## Automated mirror: BLOCKED, not configured

No workflow in this repository automatically writes the Hub. The CPU release above
used a manual source-bound Git writer after green source checks and local package
qualification. Future source changes are not automatically published. The staged
shared-mirror plan still requires:

1. **A repository-scoped HF credential.** Prefer Trusted Publishers bound to a
   reviewed mirror on `main`, or a scoped repository secret. Local Git publishing
   authority does not establish CI authority.
2. **The shared mirror.** `reusable-hf-mirror.yml` for model and kernel targets is
   not available in the checked shared workflow inventory. Its trust-root review
   remains separate; no copied mirror is introduced here.
3. **A compatible kernel transport and witness.** Generic Hub commit methods do
   not establish first-class kernel write support. This release used Kernel Hub
   Git and separately loaded the exact provider revision with a compatible client.

Any future mirror must preserve Hub-only historical receipts and pair card changes
with a matching qualified build. The model twin's license, artifacts, and ownership
must be reconciled separately before a writer changes it.

## What the historical import showed

Read the historical numbers from `hub-import/IMPORT.json`; the observations below describe its 2026-09-29 baseline.

- **The imported Hub kernel build predates this tree.** `torch-ext/szl_block_kv/_invariant.py` (the invariant-keyed blocks the card describes) was absent from that imported build, and `__init__.py` and `_ops.py` differed. The CPU publication above closes this first-class kernel gap; the historical import remains unchanged.
- **Hub-only receipts.** `BENCH.laptop-blackwell.json` and `OPERATIONAL.json` exist only on the Hub. The staged card cites them as `REPORTED`, not `MEASURED`.
- **Other writers.** The model twin also carries `block_kv.py` and `chain.py`, uploaded by `szl-holdings/a11oy` `atelier-hub-publish.yml` (a cross-writer the plan retires). Its card thumbnail `og-card.png` exists only on the Hub; the staged card does not reference it.
