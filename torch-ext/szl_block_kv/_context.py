# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Research-only model/context-bound KV addresses; not an active serving cache.

This module is intentionally not wired into v0 cache admission, Kernel Hub,
or production. Hashes identify caller-declared context bytes; they are NOT
proof of a provider's model, signatures, correctness, or token throughput.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import re
from typing import Sequence

from ._chain import _canon
from ._invariant import InvariantBundle, _validated_token_ids

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TOKENS = 4096
_MAX_LAYER_INDEX = 65535
_MAX_POSITION = (1 << 63) - 1
_DOMAIN = b"szl.block-kv.context-address.v1\x1f"


def _digest(value: object, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None or value == "0" * 64:
        raise ValueError(f"{name} must be a nonzero 64-hex SHA-256 digest")


@dataclass(frozen=True, slots=True)
class KVExecutionContext:
    """A declared candidate KV reuse domain, not independently verified.

    Each field is required, even when adapter_set_sha256 is None.
    parent_prefix_sha256 represents the full prior token prefix, not only
    this block's token IDs; hidden states can depend on earlier context.
    reuse_domain_sha256 binds the tenant/authorization isolation domain.
    """

    model_artifact_set_sha256: str
    adapter_set_sha256: str | None
    tokenizer_contract_sha256: str
    position_contract_sha256: str
    kv_layout_sha256: str
    parent_prefix_sha256: str
    reuse_domain_sha256: str
    layer_index: int
    absolute_first_token_index: int

    def __post_init__(self) -> None:
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if name == "adapter_set_sha256" and value is None:
                continue
            if name.endswith("_sha256"):
                _digest(value, name)
        for name, upper in (("layer_index", _MAX_LAYER_INDEX),
                            ("absolute_first_token_index", _MAX_POSITION)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= upper:
                raise ValueError(f"{name} must be a bounded nonnegative integer")


def context_bound_key(
    token_ids: Sequence[int], bundle: InvariantBundle, context: KVExecutionContext
) -> str:
    """Domain-separated SHA3-256 of tokens + policy + declared execution context.

    Pure and independent of GPU allocation or v0 InvariantKeyedBlockTable.
    A successful key calculation never admits reuse or grants authorization.
    """
    if not isinstance(bundle, InvariantBundle):
        raise TypeError("bundle must be an InvariantBundle")
    if not isinstance(context, KVExecutionContext):
        raise TypeError("context must be a KVExecutionContext")
    # Check the outer bound before walking possibly expensive token containers.
    try:
        size = len(token_ids)
    except (TypeError, ValueError) as exc:
        raise TypeError("token_ids must be a bounded finite sequence") from exc
    if not 1 <= size <= _MAX_TOKENS:
        raise ValueError("token_ids length must be 1..4096")
    ids = _validated_token_ids(token_ids)
    if len(ids) != size:
        raise ValueError("token_ids changed during validation")
    data = {
        "schema": "szl.block-kv.context-address/v1",
        "tokens": ids,
        "governance_bundle_sha3_256": bundle.id,
        "model_artifact_set_sha256": context.model_artifact_set_sha256,
        "adapter_set_sha256": context.adapter_set_sha256,
        "tokenizer_contract_sha256": context.tokenizer_contract_sha256,
        "position_contract_sha256": context.position_contract_sha256,
        "kv_layout_sha256": context.kv_layout_sha256,
        "parent_prefix_sha256": context.parent_prefix_sha256,
        "reuse_domain_sha256": context.reuse_domain_sha256,
        "layer_index": context.layer_index,
        "absolute_first_token_index": context.absolute_first_token_index,
    }
    return hashlib.sha3_256(_DOMAIN + _canon(data)).hexdigest()
