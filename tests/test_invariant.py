# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Invariant-keyed block addressing.

These tests establish a containment property of the addressing layer: a KV
block computed under one invariant bundle is unreachable from a request running
under another. Not a Lambda proof. Not a speedup claim.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))

import pytest
import torch
import torch.nn.functional as F

from szl_block_kv import (
    CrossRegimeError,
    InvariantBundle,
    InvariantKeyedBlockTable,
    PagedCache,
    ReceiptChain,
    block_key,
    paged_attn,
    reshape_and_cache,
    selfcheck_invariant,
)

PERMISSIVE = InvariantBundle(
    doctrine="v11", policy_class="default-permissive", receipts_required=True, verifier_budget=0
)
STRICT = PERMISSIVE.derive(policy_class="default-strict", verifier_budget=3)


def test_selfcheck_invariant():
    r = selfcheck_invariant()
    assert r["ok"] is True
    assert r["first_lookup"] == "miss_cold"
    assert r["same_regime_lookup"] == "hit"
    assert r["cross_regime_lookup"] == "miss_cross_regime"
    assert r["cross_regime_admit_rejected"] is True
    assert r["chain_ok"] is True
    assert r["lambda"] == "Conjecture 1"
    assert "no speedup claimed" in r["note"].lower()
    assert "ROADMAP" not in r["note"]


def test_bundle_id_is_canonical_and_order_independent():
    a = InvariantBundle(alpha=1, beta="two", gamma=[3, 4])
    b = InvariantBundle(gamma=[3, 4], beta="two", alpha=1)
    assert a.id == b.id
    assert a == b
    assert len(a.id) == 64


def test_empty_bundle_is_rejected():
    with pytest.raises(ValueError):
        InvariantBundle()


def test_derive_always_changes_the_id():
    assert STRICT.id != PERMISSIVE.id
    assert PERMISSIVE.derive(verifier_budget=1).id != PERMISSIVE.id


def test_block_key_depends_on_both_tokens_and_bundle():
    run_a = [1, 2, 3]
    run_b = [1, 2, 4]
    assert block_key(run_a, PERMISSIVE) == block_key(run_a, PERMISSIVE)
    assert block_key(run_a, PERMISSIVE) != block_key(run_b, PERMISSIVE)
    assert block_key(run_a, PERMISSIVE) != block_key(run_a, STRICT)


def test_block_key_rejects_a_bare_dict():
    with pytest.raises(TypeError):
        block_key([1, 2, 3], {"policy_class": "default-strict"})


def test_domain_separation_resists_boundary_confusion():
    """A token run and a bundle id must not be reinterpretable as a different
    split of the same bytes."""
    keys = {
        block_key([1, 23], PERMISSIVE),
        block_key([12, 3], PERMISSIVE),
        block_key([123], PERMISSIVE),
    }
    assert len(keys) == 3


def test_same_regime_hits_and_cross_regime_misses():
    table = InvariantKeyedBlockTable(num_blocks=8)
    run = [15496, 11, 995, 13]

    assert table.lookup(run, PERMISSIVE) == (None, "miss_cold")
    physical = table.admit(run, PERMISSIVE)
    assert table.lookup(run, PERMISSIVE) == (physical, "hit")

    found, outcome = table.lookup(run, STRICT)
    assert found is None
    assert outcome == "miss_cross_regime", "a cross-regime miss must be distinguishable"

    strict_physical = table.admit(run, STRICT)
    assert strict_physical != physical
    assert sorted(table.regimes_for(run)) == sorted([PERMISSIVE.id, STRICT.id])
    assert table.audit()["ok"] is True


def test_cross_regime_rebind_is_refused_in_strict_mode():
    table = InvariantKeyedBlockTable(num_blocks=4)
    run = [7, 7, 7]
    physical = table.admit(run, PERMISSIVE)
    with pytest.raises(CrossRegimeError):
        table.admit(run, STRICT, physical=physical)
    assert table.bundle_of(physical) == PERMISSIVE.id
    assert table.stats["rejected_cross_regime"] == 1
    assert table.audit()["ok"] is True


def test_no_physical_block_ever_serves_two_regimes():
    table = InvariantKeyedBlockTable(num_blocks=16)
    bundles = [PERMISSIVE.derive(verifier_budget=n) for n in range(4)]
    for n, bundle in enumerate(bundles):
        for run in ([1, 2], [3, 4], [5, 6]):
            table.admit(run, bundle)
            assert table.audit()["ok"] is True, f"regime leak after bundle {n}"
    physical_to_bundle = {}
    for run in ([1, 2], [3, 4], [5, 6]):
        for bundle in bundles:
            found, outcome = table.lookup(run, bundle)
            assert outcome == "hit"
            assert physical_to_bundle.setdefault(found, bundle.id) == bundle.id


def test_eviction_keeps_the_table_auditable():
    table = InvariantKeyedBlockTable(num_blocks=2)
    for i in range(6):
        table.admit([i, i + 1], PERMISSIVE)
        assert table.audit()["ok"] is True
    assert len(table) <= 2
    assert table.stats["evicted"] >= 4


def test_receipts_chain_over_every_lookup_and_admission():
    chain = ReceiptChain()
    table = InvariantKeyedBlockTable(num_blocks=4, chain=chain)
    run = [11, 12, 13]
    table.lookup(run, PERMISSIVE)
    table.admit(run, PERMISSIVE)
    table.lookup(run, PERMISSIVE)
    table.lookup(run, STRICT)
    ok, depth, bad = chain.verify()
    assert ok is True
    assert bad == -1
    assert depth == 4


def test_reuse_under_the_same_regime_is_numerically_exact():
    """The containment property must not cost arithmetic fidelity: a block
    served from an invariant-keyed hit is bit-identical to recomputing it."""
    torch.manual_seed(20260830)
    h, t, d, bs = 2, 8, 16, 4
    q = torch.randn(1, h, t, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)

    table = InvariantKeyedBlockTable(num_blocks=4)
    run = list(range(t))

    cache = PagedCache(num_blocks=4, block_size=bs, n_heads=h, d_head=d)
    reshape_and_cache(k, v, cache, torch.arange(t))
    tables = torch.tensor([[0, 1, -1, -1]])
    clens = torch.tensor([t])

    table.admit(run[:bs], PERMISSIVE, physical=0)
    table.admit(run[bs:], PERMISSIVE, physical=1)
    first = paged_attn(q, cache, tables, clens)

    found, outcome = table.lookup(run[:bs], PERMISSIVE)
    assert (found, outcome) == (0, "hit")
    second = paged_attn(q, cache, tables, clens)
    assert torch.equal(first, second)

    k_ref = k.permute(1, 0, 2).unsqueeze(0)
    v_ref = v.permute(1, 0, 2).unsqueeze(0)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, dropout_p=0.0, is_causal=False)
    assert torch.allclose(first, ref, atol=1e-5, rtol=1e-5)


def test_cross_regime_containment_forces_recompute_not_silent_reuse():
    """The whole point: under a changed regime the cache must not hand back the
    previously computed block, even though the tokens are identical."""
    table = InvariantKeyedBlockTable(num_blocks=8)
    run = [101, 102, 103, 104]
    permissive_block = table.admit(run, PERMISSIVE)

    found, outcome = table.lookup(run, STRICT)
    assert found is None and outcome == "miss_cross_regime"
    assert table.stats["hit"] == 0
    assert table.stats["miss_cross_regime"] == 1

    strict_block = table.admit(run, STRICT)
    assert strict_block != permissive_block
    assert table.bundle_of(permissive_block) == PERMISSIVE.id
    assert table.bundle_of(strict_block) == STRICT.id
