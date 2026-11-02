# SPDX-License-Identifier: Apache-2.0
"""Pure addressing-layer admission tests; no Torch or tensor qualification.

Load the canonical dependency-free modules under an isolated package name so
these checks can also run on hosts without the tensor runtime.
"""
import copy
import hashlib
import importlib.util
import json
import sys
import types
from enum import IntEnum
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "torch-ext" / "szl_block_kv"
PACKAGE = "_block_kv_token_admission"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
for name in ("_chain", "_invariant"):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
chain_module = sys.modules[f"{PACKAGE}._chain"]
invariant = sys.modules[f"{PACKAGE}._invariant"]


@pytest.mark.parametrize("bad", [1.9, 1.0, 0.9, True, False, "1", "01"])
@pytest.mark.parametrize("operation", ["block_key", "lookup", "admit", "regimes_for"])
def test_lossy_token_ids_rejected_before_state_or_receipts(bad, operation):
    bundle = invariant.InvariantBundle(policy="strict")
    chain = chain_module.ReceiptChain()
    table = invariant.InvariantKeyedBlockTable(1, chain=chain)
    physical = table.admit([1], bundle)
    before = copy.deepcopy(table.audit()), chain.head(), len(chain)

    with pytest.raises(TypeError, match="token"):
        if operation == "block_key":
            invariant.block_key([bad], bundle)
        elif operation == "regimes_for":
            table.regimes_for([bad])
        else:
            getattr(table, operation)([bad], bundle)

    assert (table.audit(), chain.head(), len(chain)) == before
    assert table.lookup([1], bundle) == (physical, "hit")
    assert chain.verify() == (True, 2, -1)


class Token(IntEnum):
    FIRST = 1


@pytest.mark.parametrize("tokens", [[], [0, 1, 2], (0, 1, 2), range(3),
                                   [-1, 2 ** 80], [Token.FIRST]])
def test_valid_integer_ids_preserve_existing_keys_and_containment(tokens):
    bundle = invariant.InvariantBundle(policy="strict")
    plain = [int(token) for token in tokens]
    body = json.dumps({"v": 1, "tokens": plain, "bundle": bundle.id},
                      sort_keys=True, separators=(",", ":")).encode()
    expected = hashlib.sha3_256(b"szl.block-kv.invariant-key\x1f" + body).hexdigest()
    assert invariant.block_key(tokens, bundle) == expected
    table = invariant.InvariantKeyedBlockTable(2)
    physical = table.admit(tokens, bundle)
    assert table.lookup(plain, bundle) == (physical, "hit")
    assert table.lookup(plain, bundle.derive(policy="other")) == (None, "miss_cross_regime")
    assert table.audit()["ok"]


def test_invariant_selfcheck_remains_valid_without_torch():
    assert invariant.selfcheck_invariant()["ok"]


@pytest.mark.parametrize("dtype", ["uint8", "int8", "int16", "int32", "int64"])
def test_integer_tensor_token_sequences_keep_existing_identity(dtype):
    torch = pytest.importorskip("torch")
    tokens = torch.tensor([0, 1, 2], dtype=getattr(torch, dtype))
    bundle = invariant.InvariantBundle(policy="tensor-tokenizer")
    assert invariant.block_key(tokens, bundle) == invariant.block_key([0, 1, 2], bundle)
    table = invariant.InvariantKeyedBlockTable(2)
    physical = table.admit(tokens, bundle)
    assert table.lookup([0, 1, 2], bundle) == (physical, "hit")
    assert table.lookup(tokens, bundle) == (physical, "hit")
    assert table.regimes_for(tokens) == [bundle.id]


@pytest.mark.parametrize("value,dtype", [(True, "bool"), (False, "bool"),
                                       (1.0, "float32"), (1.9, "float64")])
def test_noninteger_tensor_tokens_cannot_alias_integer_ids(value, dtype):
    torch = pytest.importorskip("torch")
    tokens = torch.tensor([value], dtype=getattr(torch, dtype))
    bundle = invariant.InvariantBundle(policy="tensor-tokenizer")
    chain = chain_module.ReceiptChain()
    table = invariant.InvariantKeyedBlockTable(1, chain=chain)
    table.admit([1], bundle)
    before = copy.deepcopy(table.audit()), chain.head(), len(chain)
    for operation in (lambda: invariant.block_key(tokens, bundle),
                      lambda: table.admit(tokens, bundle),
                      lambda: table.lookup(tokens, bundle),
                      lambda: table.regimes_for(tokens)):
        with pytest.raises(TypeError, match="token"):
            operation()
        assert (table.audit(), chain.head(), len(chain)) == before
