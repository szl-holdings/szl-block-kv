# SPDX-License-Identifier: Apache-2.0
"""Pure synthetic KV subject-scope tests; no Torch, model or network I/O."""
from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1] / "torch-ext" / "szl_block_kv"
PACKAGE = "_szl_scoped_kv_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
for name in ("_chain", "_invariant", "_scoped"):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

invariant = sys.modules[f"{PACKAGE}._invariant"]
scoped = sys.modules[f"{PACKAGE}._scoped"]
chain = sys.modules[f"{PACKAGE}._chain"]

def subject():
    return scoped.CacheSubject(
        source_commit="a" * 40,
        weights_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        policy_sha256="d" * 64,
        authorization_scope_sha256="e" * 64,
        adapter_sha256=None,
        generation=1,
    )

REGIME = invariant.InvariantBundle(doctrine="v11", policy_class="strict")


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_commit", "a" * 39),
        ("source_commit", "0" * 40),
        ("weights_sha256", "0" * 64),
        ("tokenizer_sha256", "C" * 64),
        ("policy_sha256", "d" * 63),
        ("authorization_scope_sha256", "tenant-plaintext"),
        ("adapter_sha256", "0" * 64),
        ("generation", True),
        ("generation", 0),
        ("generation", -1),
        ("generation", 1_000_000_001),
    ],
)
def test_missing_unpinned_or_placeholder_subject_fails_closed(field, value):
    with pytest.raises(ValueError):
        replace(subject(), **{field: value})


def test_dataclass_binding_is_frozen_and_no_unsafe_default_scope():
    with pytest.raises(FrozenInstanceError):
        subject().generation = 2
    with pytest.raises(TypeError):
        scoped.CacheSubject(source_commit="a" * 40)
    assert "authorization_scope_sha256" in scoped.CacheSubject.__dataclass_fields__


def test_untyped_scope_or_regime_is_not_accepted():
    with pytest.raises(TypeError, match="CacheSubject"):
        scoped.scoped_block_key([1, 2], {}, REGIME)
    with pytest.raises(TypeError, match="InvariantBundle"):
        scoped.scoped_block_key([1, 2], subject(), {"doctrine": "v11"})


def test_same_subject_stable_digest_and_cache_hit():
    s = subject()
    table = scoped.ScopedBlockTable(3)
    assert scoped.scoped_block_key([1, 2], s, REGIME) == scoped.scoped_block_key(
        (1, 2), s, REGIME
    )
    idx = table.admit([1, 2], s, REGIME)
    assert table.lookup([1, 2], s, REGIME) == (idx, "hit")
    assert table.audit()["production_admission"] == "HOLD"


@pytest.mark.parametrize(
    "change",
    [
        {"source_commit": "f" * 40},
        {"weights_sha256": "f" * 64},
        {"tokenizer_sha256": "f" * 64},
        {"policy_sha256": "f" * 64},
        {"authorization_scope_sha256": "f" * 64},
        {"adapter_sha256": "f" * 64},
        {"generation": 2},
    ],
)
def test_same_tokens_cannot_cross_declared_subject_dimension(change):
    first = subject()
    other = replace(first, **change)
    assert scoped.scoped_block_key([7], first, REGIME) != scoped.scoped_block_key(
        [7], other, REGIME
    )
    table = scoped.ScopedBlockTable(2)
    physical = table.admit([7], first, REGIME)
    assert table.lookup([7], other, REGIME) == (None, "miss_cross_regime")
    assert table.lookup([7], first, REGIME) == (physical, "hit")


def test_policy_regime_change_forces_miss_even_when_subject_unchanged():
    table = scoped.ScopedBlockTable(2)
    first = subject()
    table.admit([7], first, REGIME)
    assert table.lookup([7], first, REGIME.derive(policy_class="new")) == (
        None, "miss_cross_regime"
    )


def test_scoped_table_refuses_cross_subject_physical_rebinding():
    table = scoped.ScopedBlockTable(2)
    first = subject()
    physical = table.admit([7], first, REGIME, physical=0)
    assert physical == 0
    with pytest.raises(invariant.CrossRegimeError):
        table.admit(
            [7], replace(first, authorization_scope_sha256="f" * 64), REGIME, physical=0
        )
    assert table.lookup([7], first, REGIME) == (0, "hit")


def test_no_raw_scope_or_private_subject_in_receipt_metadata():
    receipt = chain.ReceiptChain()
    table = scoped.ScopedBlockTable(2, chain=receipt)
    s = subject()
    table.admit([9], s, REGIME)
    table.lookup([9], s, REGIME)
    output = json.dumps({"audit": table.audit(), "rows": receipt._rows})
    assert receipt.verify()[0]
    assert s.authorization_scope_sha256 not in output
    assert s.weights_sha256 not in output
    assert '"tenant"' not in output
    assert '"production_admission": "HOLD"' in output
    assert table.audit()["authorization_verified"] is False


def test_invalid_token_is_not_coerced_before_cache_lookup():
    table = scoped.ScopedBlockTable(2)
    s = subject()
    table.admit([1], s, REGIME)
    before = json.dumps(table.audit(), sort_keys=True)
    for value in (True, False, 1.0, 1.9, "1"):
        with pytest.raises(TypeError, match="token"):
            table.lookup([value], s, REGIME)
        assert json.dumps(table.audit(), sort_keys=True) == before


def test_adapter_is_explicit_none_and_not_inferred_from_name():
    first = subject()
    assert first.adapter_sha256 is None
    assert scoped.scoped_block_key([7], first, REGIME) != scoped.scoped_block_key(
        [7], replace(first, adapter_sha256="a" * 64), REGIME
    )


def test_eviction_keeps_subject_containment():
    table = scoped.ScopedBlockTable(1)
    first = subject()
    next_subject = replace(first, authorization_scope_sha256="f" * 64)
    table.admit([1], first, REGIME)
    table.admit([1], next_subject, REGIME)
    assert table.lookup([1], first, REGIME) == (None, "miss_cross_regime")
    assert table.audit()["ok"] is True
    assert len(table) == 1
