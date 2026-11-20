# Scoped KV reuse: source-only admission contract (2026-11-20)

**Status: RESEARCH / SOURCE-ONLY / NO TRAINING OR PRODUCTION AUTHORIZATION.**
This is an additive proof-of-contract for \`szl-block-kv\`. The existing
\`_invariant.py\` allows the application to choose any \`InvariantBundle\`.
It proves separation **only for differing supplied bundle identities**. It
does not require source, weights, tokenizer, adapter, policy and authorization
scope to be present. It is therefore not a tenant or model isolation
certificate for an arbitrary production caller.

## Leaders, research and the SZL adaptation

- [vLLM/PagedAttention (Kwon et al., SOSP 2023)](https://arxiv.org/abs/2309.06180),
  [source](https://github.com/vllm-project/vllm): reuse paged KV with careful
  physical-block accounting. SZL preserves its original explicit
  \`InvariantKeyedBlockTable\` rather than importing vLLM code.
- [LMCache](https://github.com/LMCache/LMCache): reusable, tiered, potentially
  cross-engine KV data require explicit cache provenance and observation.
  SZL's experiment adds subject-binding fields to **cache identity** before
  contemplating shared remote KV reuse; no LMCache integration is included.
- [FlashAttention-4 (Zadouri et al., 2026)](https://arxiv.org/abs/2603.05451),
  [source](https://github.com/Dao-AILab/flash-attention): CuTe DSL kernel
  optimizations target Hopper/Blackwell. There is **no transfer of published
  B200 performance** to the SZL paged-KV source or laptop GPU.
- [Hugging Face Kernel Hub requirements](https://huggingface.co/docs/kernels/kernel-requirements):
  explicit \`trust_remote_code\` for non-curated publisher code is an execution
  decision separate from a valid revision string. A kernel-hub release
  still needs immutable provider-byte readback and trust-qualified admission.

## Scoped key math

The original key is \`H = SHA3-256(domain || canonical({v:1,tokens,bundle}))\`.
The proposed \`subject_bundle\` keeps that established domain and identity
algorithm, but replaces the arbitrary bundle with a **new composite bundle**:

\`B' = InvariantBundle({contract:v1, source_commit, weights_sha256,
tokenizer_sha256, policy_sha256, authorization_scope_sha256,
adapter_sha256 or null, generation, regime_id:B.id})\`

\`K_scoped = block_key(token_ids,B')\`.

Every subject field is mandatory; \`adapter_sha256=None\` explicitly means
the caller declares no adapter. The 40-character source revision, six
additional commitments and generation are lower-case, nonzero, syntactically
validated identifiers. Same tokens with different **declared** model,
weights, tokenizer, policy, adapter, authorization scope, generation or
underlying regime cannot produce the same composite identity absent a
cryptographic collision.

**Critical limitation:** a caller-provided SHA-256 string is not an
independent source/weight readback, an owner signature, a secret-preserving
HMAC over a tenant identifier, or proof that a tenant is authorized.
\`authorization_scope_sha256\` must be a stable *trusted-controller-issued
opaque scope commitment*, not an unhashed private tenant ID. Never derive
or log it from raw credentials here. This module does not accept tenant
identifiers, JWTs, tokens, prompts, or model files and does not read them.

## Failure modes and adoption gates

The new \`ScopedBlockTable\` is opt-in and holds an internal strict
\`InvariantKeyedBlockTable\`. Same-subject reuses hit, while different
declared scopes miss. Invalid booleans, zero hashes, partial digests and
mutable revision placeholders reject. Explicit physical rebindings across
scopes are refused and still governed by the original receipt chain.

No production-serving caller currently imports this new module. Do not
assume the default kernel has been hardened, tenants are isolated, GPU
throughput improved, or cache memory safely shared between organizations.
No code is loaded from HF and no GPU is exercised.

**Before integration:** trusted principal/scope issuing authority; signed
source/adapter/model-byte evidence; tokenizer revision proof; immutable
publisher and version-pinned kernel runtime; negative cross-tenant tests
under real serving/auth context; concurrency and eviction stress; actual
TTFT and tokens/s A/B under one hardware/data/harness timeline; rollback
and explicit owner approval. Until then production state is **HOLD**.

CPU-only, source-only test: \`pytest -q tests/test_scoped_cache_subject.py\`.
Existing \`tests/test_invariant.py\` retains its original API compatibility.
