# SPDX-License-Identifier: Apache-2.0
"""CPU-only negative controls for the research KV execution-context key."""
from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
from szl_block_kv._context import KVExecutionContext, context_bound_key
from szl_block_kv._invariant import InvariantBundle, block_key

D = lambda char: char * 64
BUNDLE = InvariantBundle(doctrine="v11", policy_class="default-strict")


def context():
    return KVExecutionContext(
        model_artifact_set_sha256=D("1"),
        adapter_set_sha256=None,
        tokenizer_contract_sha256=D("2"),
        position_contract_sha256=D("3"),
        kv_layout_sha256=D("4"),
        parent_prefix_sha256=D("5"),
        reuse_domain_sha256=D("6"),
        layer_index=0,
        absolute_first_token_index=0,
    )


def test_identical_complete_context_is_deterministic_and_not_legacy_key():
    key = context_bound_key([1, 2, 3], BUNDLE, context())
    assert len(key) == 64
    assert key == context_bound_key([1, 2, 3], BUNDLE, context())
    assert key != block_key([1, 2, 3], BUNDLE)


@pytest.mark.parametrize("field,value", [
    ("model_artifact_set_sha256", D("7")),
    ("adapter_set_sha256", D("8")),
    ("tokenizer_contract_sha256", D("9")),
    ("position_contract_sha256", D("a")),
    ("kv_layout_sha256", D("b")),
    ("parent_prefix_sha256", D("c")),
    ("reuse_domain_sha256", D("d")),
    ("layer_index", 1),
    ("absolute_first_token_index", 1),
])
def test_one_context_difference_forces_different_key(field, value):
    original = context_bound_key([1, 2], BUNDLE, context())
    changed = context_bound_key([1, 2], BUNDLE, replace(context(), **{field: value}))
    assert original != changed


def test_token_difference_or_governance_regime_change_forces_new_key():
    baseline = context_bound_key([1, 2], BUNDLE, context())
    assert baseline != context_bound_key([1, 3], BUNDLE, context())
    assert baseline != context_bound_key([1, 2], BUNDLE.derive(policy_class="another"), context())


@pytest.mark.parametrize("field", [
    "model_artifact_set_sha256", "adapter_set_sha256", "tokenizer_contract_sha256",
    "position_contract_sha256", "kv_layout_sha256", "parent_prefix_sha256",
    "reuse_domain_sha256",
])
@pytest.mark.parametrize("invalid", ["", "0" * 64, "G" * 64, "f" * 63, 9, True])
def test_invalid_context_digest_fails_closed(field, invalid):
    with pytest.raises(ValueError):
        replace(context(), **{field: invalid})


@pytest.mark.parametrize("field", ["layer_index", "absolute_first_token_index"])
@pytest.mark.parametrize("invalid", [-1, 0.0, True, "0", 1 << 64])
def test_invalid_position_and_layer_rejected(field, invalid):
    with pytest.raises(ValueError):
        replace(context(), **{field: invalid})


@pytest.mark.parametrize("token_ids", [[], [True], [1.0], ["1"], [1] * 4097])
def test_invalid_token_identity_refused(token_ids):
    with pytest.raises((TypeError, ValueError)):
        context_bound_key(token_ids, BUNDLE, context())


def test_invalid_bundle_and_context_type_refused():
    with pytest.raises(TypeError):
        context_bound_key([1], {"doctrine": "v11"}, context())
    with pytest.raises(TypeError):
        context_bound_key([1], BUNDLE, {"model": D("1")})


def test_input_token_array_remains_unchanged_after_key_calculation():
    ids = [10, 11, 12]
    saved = ids[:]
    context_bound_key(ids, BUNDLE, context())
    assert ids == saved
