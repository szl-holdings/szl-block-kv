<!-- SZL-KERNEL-OPERATIONAL:START -->
## Operational (MEASURED laptop-Blackwell)

> **STATUS:** tests **PASS**. `get_kernel` **import-LIVE**. Unsloth/LoRA is the wrong tool. Receipted kernels, not silent CUDA.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| tests (`PYTHONPATH=torch-ext`) | **PASS** | MEASURED 2026-08-29T15:54:24Z host `betterwithage` Windows-10-10.0.26200-SP0. torch `2.10.0+cu128`. GPU `NVIDIA GeForce RTX 5050 Laptop GPU` arch `Blackwell`. pytest `10 passed in 2.52s`. Failed nodes: `none`. What-NOT: not a leaderboard. torch.compile fullgraph failures on Windows Blackwell (`cl is not found`) are MEASURED, not hidden. |
| Kernel Hub `get_kernel` | **import-LIVE** | kernels `0.16.1`. Default: `get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)` → `True`. `backend="cpu"` → `True`. trust_remote_code=False → `ValueError` (SZLHOLDINGS is not a trusted publisher). repo_type=kernel required (kernels 0.16). What-NOT: not a weight load; do not pickle/joblib.load. |
| formula-tax | **ADVISORY** | locked-8 `F1 F4 F7 F11 F12 F18 F19 F22`. registry_count=21. Λ geomean `1.0`. uniqueness **Conjecture 1** (never a theorem). |
| I1–I8 | **catalog** | `I1 receipt-chain-continuity; I2 ledger-failure-shape; I3 served-run-has-model; I4 signed-columns-atomic; I5 loop-steps-positive; I6 receipt-ed25519-verify; I7 receipt-columns-consistent; I8 flywheel-lineage`. Executed by `SZLHOLDINGS/szl-invariants`. Statuses never coerced. Λ untouched. |
| CUDA speedup / tokens/s / joules | **UNAVAILABLE** | Not claimed. Receipted kernels, not silent CUDA. |

GitHub source: [`szl-holdings/szl-block-kv`](https://github.com/szl-holdings/szl-block-kv) @ `7cf99cca9d2749482f5f57572c3affc2e7a5ebc7`. Artifacts: [`BENCH.laptop-blackwell.json`](./BENCH.laptop-blackwell.json), [`OPERATIONAL.json`](./OPERATIONAL.json).

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)
```

<!-- SZL-KERNEL-OPERATIONAL:END -->

# szl-block-kv — KERNEL SOURCE STATUS

Canonical GitHub source: [`szl-holdings/szl-block-kv`](https://github.com/szl-holdings/szl-block-kv). This Kernel Hub repo is the `get_kernel` publish surface (`build/torch-universal` + `build/torch-cpu`).

v0 is a labeled torch gather over a block table with SHA3 receipts. Inspired by Kwon et al. PagedAttention SOSP 2023. **NOT a rehost of vLLM or kernels-community/paged-attention.** Distinct from a11oy MODELED H2O eviction.

<!-- SZL-KERNEL-STATUS:import-LIVE:START -->
## Status

> **STATUS: import-LIVE** on CPU Kernel Hub `get_kernel` (kernels `0.16.1`). Triton page kernel is **UNAVAILABLE**.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| Kernel Hub `get_kernel` | **import-LIVE** | MEASURED 2026-08-28 2:29pm ET on kernels `0.16.1`. HEAD [`d3ede3e`](https://huggingface.co/kernels/SZLHOLDINGS/szl-block-kv/commit/d3ede3e471b51080492b1c69306283507dcf507e) (`d3ede3e471b51080492b1c69306283507dcf507e`). Legal name `szl-block-kv` (Python module `szl_block_kv`). Variants: `build/torch-universal` (default `get_kernel`) and `build/torch-cpu` (`backend="cpu"`). Working calls: `get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)` and the same with `backend="cpu"`. `selfcheck` **ok**. `max_abs_vs_contiguous=2.38e-07` (full `2.384185791015625e-07`), `path=torch_gather`, `chain_ok=true`. What-NOT: no tokens/s; no joules. |
| Triton page kernel | **UNAVAILABLE** | MEASURED 2026-08-28 7:01pm ET this session. Host `cursor` (Linux 6.12.94+ x86_64, Intel Xeon 8-core). `torch` `2.13.0+cu130` compiled CUDA 13.0. `torch.cuda.is_available()=false`. `nvidia-smi` UNAVAILABLE. `device_count=0`. Triton `3.7.1` present with no CUDA device. No cubin. No timed GPU run. No tokens/s. No joules. |

<!-- SZL-KERNEL-STATUS:import-LIVE:END -->

```python
from kernels import get_kernel

kv = get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)
print(kv.selfcheck())

kv_cpu = get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True, backend="cpu")
print(kv_cpu.selfcheck())
```

**Doctrine v11.** Λ = **Conjecture 1** (advisory). Apache-2.0. Copyright 2026 SZL Holdings.
