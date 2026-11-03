# SPDX-License-Identifier: Apache-2.0
"""Physical-address admission must not coerce an invalid address into a live block."""
import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "torch-ext" / "szl_block_kv"
PACKAGE = "_block_kv_physical_admission"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
for name in ("_chain", "_invariant"):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
invariant = sys.modules[f"{PACKAGE}._invariant"]
ReceiptChain = sys.modules[f"{PACKAGE}._chain"].ReceiptChain


def snapshot(table, chain):
    return copy.deepcopy(({k: v for k, v in table.__dict__.items() if k != "chain"},
                          chain.__dict__))


@pytest.mark.parametrize("bad", [0.9, 0.0, True, False, "0", "00"])
@pytest.mark.parametrize("tokens", [[7], [8]])
def test_invalid_physical_address_cannot_rebind_or_bypass_validation(bad, tokens):
    bundle = invariant.InvariantBundle(policy="strict")
    chain = ReceiptChain()
    table = invariant.InvariantKeyedBlockTable(2, chain=chain)
    table.admit([7], bundle, physical=0)
    before = snapshot(table, chain)

    with pytest.raises(TypeError, match="physical"):
        table.admit(tokens, bundle, physical=bad)

    assert snapshot(table, chain) == before
    assert table.lookup([7], bundle) == (0, "hit")


@pytest.mark.parametrize("bad", [0.5, 1.0, True, False, "2"])
def test_capacity_requires_an_integer_count(bad):
    with pytest.raises(TypeError, match="num_blocks"):
        invariant.InvariantKeyedBlockTable(bad)


@pytest.mark.parametrize("bad", [0.9, 0.0, True, False, "0"])
def test_bundle_lookup_does_not_alias_invalid_physical_address(bad):
    bundle = invariant.InvariantBundle(policy="strict")
    table = invariant.InvariantKeyedBlockTable(2)
    table.admit([7], bundle, physical=0)
    with pytest.raises(TypeError, match="physical"):
        table.bundle_of(bad)


@pytest.mark.parametrize("physical", [-1, 2])
@pytest.mark.parametrize("tokens", [[7], [8]])
def test_physical_bounds_checked_even_for_existing_key(physical, tokens):
    bundle = invariant.InvariantBundle(policy="strict")
    chain = ReceiptChain()
    table = invariant.InvariantKeyedBlockTable(2, chain=chain)
    table.admit([7], bundle, physical=0)
    before = snapshot(table, chain)
    with pytest.raises(ValueError, match="out of range"):
        table.admit(tokens, bundle, physical=physical)
    assert snapshot(table, chain) == before


def test_numpy_duration_cannot_be_capacity_or_physical_address():
    np = pytest.importorskip("numpy")
    duration = np.timedelta64(1, "ns")
    bundle = invariant.InvariantBundle(policy="strict")
    table = invariant.InvariantKeyedBlockTable(2)
    with pytest.raises(TypeError, match="num_blocks"):
        invariant.InvariantKeyedBlockTable(duration)
    with pytest.raises(TypeError, match="physical"):
        table.admit([7], bundle, physical=duration)


@pytest.mark.parametrize("dtype", ["int8", "int64", "uint64"])
def test_numpy_integer_addresses_remain_supported(dtype):
    np = pytest.importorskip("numpy")
    scalar = getattr(np, dtype)
    bundle = invariant.InvariantBundle(policy="strict")
    table = invariant.InvariantKeyedBlockTable(scalar(2))
    assert table.admit([7], bundle, physical=scalar(1)) == 1
    assert table.bundle_of(scalar(1)) == bundle.id
    assert table.bundle_of(2) is None  # Existing unknown-address lookup contract.


@pytest.mark.parametrize("dtype", ["int8", "int64", "uint64"])
def test_integer_tensor_addresses_remain_supported(dtype):
    torch = pytest.importorskip("torch")
    tensor = lambda value: torch.tensor(value, dtype=getattr(torch, dtype))
    bundle = invariant.InvariantBundle(policy="strict")
    table = invariant.InvariantKeyedBlockTable(tensor(2))
    assert table.admit([7], bundle, physical=tensor(1)) == 1
    assert table.bundle_of(tensor(1)) == bundle.id


@pytest.mark.parametrize("dtype,value", [("bool", True), ("float32", 0.9), ("float64", 0.0)])
def test_noninteger_tensor_addresses_rejected_before_state_change(dtype, value):
    torch = pytest.importorskip("torch")
    bad = torch.tensor(value, dtype=getattr(torch, dtype))
    bundle = invariant.InvariantBundle(policy="strict")
    chain = ReceiptChain()
    table = invariant.InvariantKeyedBlockTable(2, chain=chain)
    table.admit([7], bundle, physical=0)
    before = snapshot(table, chain)
    with pytest.raises(TypeError, match="physical"):
        table.admit([8], bundle, physical=bad)
    assert snapshot(table, chain) == before
