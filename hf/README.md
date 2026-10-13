# Hugging Face source and CPU publication

This folder is the GitHub source for the two Hub twins of this repository:

- the Kernel Hub repo `kernels/SZLHOLDINGS/szl-block-kv` (what `get_kernel` loads);
- the model-type twin `SZLHOLDINGS/szl-block-kv`.

| Path | What it is |
| --- | --- |
| `card.yaml` | The card source (plan decision D9: one source, one template). |
| `../CARD.md` | Its rendering by the shared [`hf-card`](https://github.com/szl-holdings/.github/tree/main/hf-card) toolkit. `hf-kernel-publish` publishes it as the Kernel Hub `README.md`. |
| `model-twin-pointer.md` | The README `hf-kernel-publish` writes to the model-type twin: a pointer to the Kernel Hub repo. |
| `kernel_smoke.py` | The smoke hook `hf-kernel-publish` runs on every loaded build (local and from the Hub). |
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

## Publication: `hf-kernel-publish`

`.github/workflows/hf-kernel-publish.yml` (with `scripts/szl_kernel_publish.py` and
`.github/hf-kernels.json`) is the only writer of both Hub twins.

- **Pull request:** builds `build/torch-cpu` and `build/torch-universal` from
  `torch-ext/szl_block_kv` (byte-identical, LF-normalised, generated `metadata.json`
  with a sha256 digest) and loads both with `kernels==0.17.2` and CPU torch, then runs
  `hf/kernel_smoke.py`. No secret, no Hub access.
- **Release published** (or a dispatch from `main`): publishes the same tree to the
  Kernel Hub repo on `main` **and** `v1` by fast-forward Git push (no force, no repo
  creation), reads it back byte for byte, and in a fresh job loads it anonymously with
  `get_kernel("SZLHOLDINGS/szl-block-kv", version=1, trust_remote_code=True)` and with
  the main branch, failing unless the served files equal the GitHub source. Only then
  does it replace the model twin's README with `model-twin-pointer.md` and remove the
  twin's stale `build/`, `torch-ext/`, code uploads and plates.
- **Weekly:** re-verifies what `get_kernel` serves and fails with `SOURCE_AHEAD` when the
  latest release changed the kernel but was never published.

The publisher owns `build/` on the Kernel Hub repo. It also writes `README.md` (from
`CARD.md`), `LICENSE` and `SZL_KERNEL_SOURCE.json` (the GitHub commit the build came
from), and removes the superseded Hub-only `OPERATIONAL.json`,
`BENCH.laptop-blackwell.json` and `SOURCE_BINDING.json`. Other Hub files, such as
`CPU_QUALIFICATION.json`, are kept. The credential is the org secret `HF_TOKEN`, used
only by the two writing steps.

The kernel major is `version` in `.github/hf-kernels.json`, not the release semver.
Bumping it starts `v2` and leaves `v1` frozen.

## Earlier manual CPU publication (historical)

Before this workflow existed, a manual Git publication
`54a19796ee2c4f761c9482f77dd1f989a773e17f` served source
`217e3b24017cd5fadb2525a7b5207f7c89a46413`, with the cache index admission repair, the
invariant-keyed APIs, the rendered card, and the Apache-2.0 license. Its
[release record](releases/20261003-cache-index-cpu.json) binds a publication readback
(all 12 declared files matched) and a **REPORTED** CPU conformance run (`kernels==0.16.1`,
Python 3.11.9, PyTorch 2.10.0: 92 source tests, `selfcheck()` and
`selfcheck_invariant()` passed) to that revision. Independent replay is **UNAVAILABLE**;
key trust remains **REPO_DECLARED**. No GPU, Triton, training, speedup, energy, or
scientific qualification is established. The record says `BLOCKED_NOT_CONFIGURED`
because no automatic writer existed then; it is kept unchanged as a historical receipt.

`trust_remote_code=True` permits execution of the reviewed Python package; it is
not publisher certification. Causal attention still fails closed.

## What the historical import showed

Read the historical numbers from `hub-import/IMPORT.json`; the observations below describe its 2026-09-29 baseline.

- **The imported Hub kernel build predates this tree.** `torch-ext/szl_block_kv/_invariant.py` (the invariant-keyed blocks the card describes) was absent from that imported build, and `__init__.py` and `_ops.py` differed. The CPU publication above closes this first-class kernel gap; the historical import remains unchanged.
- **Hub-only receipts.** `BENCH.laptop-blackwell.json` and `OPERATIONAL.json` exist only on the Hub. The staged card cites them as `REPORTED`, not `MEASURED`.
- **Other writers.** The model twin also carries `block_kv.py` and `chain.py`, uploaded by `szl-holdings/a11oy` `atelier-hub-publish.yml` (a cross-writer the plan retires). Its card thumbnail `og-card.png` exists only on the Hub; the staged card does not reference it.
