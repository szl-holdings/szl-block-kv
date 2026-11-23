# Leader-Study Upgrade Plan - W27 loop-7 - 2026-11-23

Status: PROPOSE-ONLY DRAFT - DO NOT MERGE. Owner-gated: merging requires governing approval and the estate signature-wall owner unlock.

Repo: szl-block-kv - Original SZL Triton paged KV cache with block-table SHA3 receipts. PagedAttention silhouette, not a rehost. Doctrine v11, Apache-2.0.

Frame: OUROBOROS loop-7 leader study. Governing seat: Kimi kimi-k3.1-pro (json_ok, think off). Advisory seats: Astra/Sol/Claude community rails (provenance unverified, advisory only). Leader evidence web-verified 2026-11-23.

## Leader moats to absorb
- vLLM PagedAttention: Fixed-size KV blocks at arbitrary physical addresses, block-table logical-to-physical mapping, reference-counted block sharing, hash-based prefix caching, and V1 block pool plus per-group KV cache managers. Moat: Production-grade memory management and prefix reuse integrated with a widely adopted serving engine; the README identifies it as the PagedAttention lineage and explicitly distinguishes this repo from a rehost.
- FlashInfer: Paged-attention kernels. Moat: Kernel-level paged-attention implementation and GPU execution depth; relevant because this repo's Triton page kernel is explicitly UNAVAILABLE.
- SGLang: Radix-tree prefix cache. Moat: Structured prefix sharing and cache lookup through a radix tree rather than only content-addressed block reuse.
- KV-cache layout/quantization lineage: KV-cache layout and quantization techniques. Moat: Memory-footprint and data-layout specialization; the facts name this lineage but provide no specific implementation or measured result to claim.

## Honest gaps (grounded in repo facts)
- The Triton page kernel is UNAVAILABLE, so the repo currently lacks a LIVE GPU paged-kernel execution path.
- Current v0 is a labeled torch gather over a block table, not a demonstrated full PagedAttention serving implementation.
- Only CPU import and selfcheck are import-LIVE; the facts do not establish GPU correctness, multi-request serving, or production integration.
- No tokens/s or joules are claimed, and no speedup claim is made; performance and energy competitiveness remain unmeasured.
- The historical Kernel Hub load is not an immutable-pin rerun, so current reproducibility at that pinned Hub revision is not established by the stated status.
- The README names reference-counted sharing, block pools, per-group managers, radix-tree caching, and quantization as leader capabilities, but does not state that this repo implements them.
- Lambda = Conjecture 1 is OPEN and uniqueness is unproven, so no uniqueness or exclusivity guarantee can be claimed for the construction.
- The supplied facts mention block-table SHA3 receipts and chain_ok=true, but do not provide enough detail to claim external receipt verification, persistence, or production audit integration.

## Dosed upgrade plan
- Dose 1: Freeze the current import-LIVE CPU baseline as a reproducible test matrix covering torch_gather, selfcheck, contiguous comparison, exact integer token/address rejection, cached-run rejection, bundle_of behavior, and receipt-chain checks; separately rerun the historical Hub commit by immutable pin before making any Hub claim.
- Dose 2: Specify and test the cache-state model before adding kernels: block-table mapping, physical bounds, admission atomicity, invariant-bundle separation, sharing/reference lifecycle, eviction, and receipt emission on every mutation path. Keep torch_gather as the reference implementation.
- Dose 3: Propose a minimal Triton page kernel behind an explicit backend selector and capability probe. Validate it against torch_gather across block sizes, partial blocks, empty and boundary inputs, sharing, and invalid-address cases; retain UNAVAILABLE status on unsupported systems and publish correctness evidence before any performance label.
- Dose 4: Propose prefix-index and block-pool improvements only after lifecycle invariants are executable tests. Compare content-keyed lookup with an optional radix index for hit/miss equivalence and cross-invariant isolation; do not make uniqueness claims while Conjecture 1 remains open.
- Dose 5: If GPU correctness is established, run separately labeled throughput, memory, and energy benchmarks with hardware, software, shapes, baselines, and immutable source pins. Treat layout or quantization as later proposals requiring numerical-correctness and receipt-compatibility tests, not assumed wins.
- Risk note: The main risk is overstating a CPU torch-gather reference as a production paged-attention system. GPU kernel work can introduce numerical, synchronization, bounds, and backend-availability failures; prefix sharing can cross invariant boundaries; lifecycle changes can invalidate receipts or mutate state before rejection; and quantization or benchmarking can create unsupported correctness or performance claims. Immutable pins, negative-path tests, differential validation, and conservative status labeling are required.
- Propose-only note: This is an audit-derived proposal, not evidence that these capabilities exist in the repository. No code, GPU result, benchmark, Hub rerun, or production readiness is claimed. Each dose should land behind tests and explicit status labels, with torch_gather retained as the CPU reference and no change to the OPEN status of Conjecture 1.

## Advisory notes (unverified rails, non-binding)
- astra: szl-quant-bench: torchao szl-block-kv: vLLM szl-energy-mcp: Home Assistant
- sol: szl-quant-bench: Hardware-specific quantization kernels for diverse accelerators szl-block-kv: Zero-copy memory mapping for heterogeneous GPU clusters szl-energy-mcp: Verified hardware-attested energy telemetry standards
- claude: szl-quant-bench: Hardware-specific quantization kernels (Triton/CUDA) for zero-shot accuracy validation. szl-block-kv: Dynamic memory fragmentation management for multi-tenant long-context serving. szl-energy-mcp: Hardware-level power rail telemetry integration for verifiable energy auditing.

## DO NOT MERGE
This plan is a proposal artifact of the 10-pass OUROBOROS loop (loop-7 of 10). No code, no CI, no behavior changes in this file. A reviewer with authority may convert to a work order later.