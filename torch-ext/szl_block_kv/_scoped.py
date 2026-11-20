# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Source-only, explicit subject binding for governed KV block reuse.

A bare policy bundle is insufficient to claim model, tokenizer, adapter, or
authorization isolation. These identities are caller-declared commitments, not
independently checked signatures, actual weight bytes, or tenant authorization.
This module is NOT wired into the serving engine or the Hub kernel entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Sequence

from ._chain import ReceiptChain
from ._invariant import InvariantBundle, InvariantKeyedBlockTable, block_key

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CONTRACT = "szl.kv.scoped-subject/v1"


def _declared_digest(name: str, value: object, width: int) -> None:
    pattern = _HEX40 if width == 40 else _HEX64
    if type(value) is not str or pattern.fullmatch(value) is None or set(value) == {"0"}:
        raise ValueError(f"{name} must be a nonzero lowercase {width}-hex commitment")


@dataclass(frozen=True, slots=True)
class CacheSubject:
    """Required, untrusted subject inputs; never a substitute for authorizer proof."""

    source_commit: str
    weights_sha256: str
    tokenizer_sha256: str
    policy_sha256: str
    authorization_scope_sha256: str
    adapter_sha256: Optional[str]
    generation: int

    def __post_init__(self) -> None:
        _declared_digest("source_commit", self.source_commit, 40)
        for name in (
            "weights_sha256",
            "tokenizer_sha256",
            "policy_sha256",
            "authorization_scope_sha256",
        ):
            _declared_digest(name, getattr(self, name), 64)
        if self.adapter_sha256 is not None:
            _declared_digest("adapter_sha256", self.adapter_sha256, 64)
        if type(self.generation) is not int or not 1 <= self.generation <= 1_000_000_000:
            raise ValueError("generation must be a positive bounded integer")


def subject_bundle(subject: CacheSubject, regime: InvariantBundle) -> InvariantBundle:
    """Bind all declared cache-reuse identities to an existing invariant regime.

    Use only with a trusted caller that independently attests source, weights,
    tokenizer, policy, adapter, authorization scope, and generation. A SHA
    string is not proof the named bytes exist or that the caller is authorized.
    """
    if type(subject) is not CacheSubject:
        raise TypeError("subject must be a CacheSubject")
    if not isinstance(regime, InvariantBundle):
        raise TypeError("regime must be an InvariantBundle")
    return InvariantBundle(
        contract=_CONTRACT,
        source_commit=subject.source_commit,
        weights_sha256=subject.weights_sha256,
        tokenizer_sha256=subject.tokenizer_sha256,
        policy_sha256=subject.policy_sha256,
        authorization_scope_sha256=subject.authorization_scope_sha256,
        adapter_sha256=subject.adapter_sha256,
        generation=subject.generation,
        regime_id=regime.id,
    )


def scoped_block_key(
    token_ids: Sequence[int], subject: CacheSubject, regime: InvariantBundle
) -> str:
    """Return the existing domain-separated block key with full subject binding."""
    return block_key(token_ids, subject_bundle(subject, regime))


class ScopedBlockTable:
    """Opt-in, fail-closed source adapter. Not a new serving or security runtime.

    The internal table is always strict. No unscoped lookup or direct storage
    manipulation is exported by this facade; this is an API design contract,
    not a sandbox against malicious Python callers.
    """

    def __init__(self, num_blocks: int, *, chain: Optional[ReceiptChain] = None) -> None:
        self.__table = InvariantKeyedBlockTable(num_blocks, chain=chain, strict=True)

    def lookup(
        self, token_ids: Sequence[int], subject: CacheSubject, regime: InvariantBundle
    ) -> tuple[Optional[int], str]:
        return self.__table.lookup(token_ids, subject_bundle(subject, regime))

    def admit(
        self,
        token_ids: Sequence[int],
        subject: CacheSubject,
        regime: InvariantBundle,
        *,
        physical: Optional[int] = None,
    ) -> int:
        return self.__table.admit(
            token_ids, subject_bundle(subject, regime), physical=physical
        )

    def audit(self) -> dict:
        report = self.__table.audit()
        return {
            **report,
            "scope_binding": "CALLER_DECLARED_NOT_AUTHENTICATED",
            "subject_bytes_verified": False,
            "authorization_verified": False,
            "tenant_isolation_proved": False,
            "runtime_integration_verified": False,
            "production_admission": "HOLD",
        }

    def __len__(self) -> int:
        return len(self.__table)
