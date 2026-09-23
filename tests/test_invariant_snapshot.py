# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Regression tests for invariant-bundle identity snapshots."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))

from szl_block_kv import InvariantBundle


def test_bundle_identity_is_stable_against_caller_mutation():
    policy = {"guards": ["allow-a"]}
    bundle = InvariantBundle(policy=policy, verifier_budget=1)
    original_id = bundle.id

    policy["guards"].append("allow-b")

    assert bundle.id == original_id
    assert bundle.fields == {"policy": {"guards": ["allow-a"]}, "verifier_budget": 1}


def test_fields_property_cannot_mutate_bundle_identity_state():
    bundle = InvariantBundle(policy={"guards": ["allow-a"]}, verifier_budget=1)
    original_id = bundle.id
    exposed = bundle.fields

    exposed["policy"]["guards"].append("allow-b")
    exposed["verifier_budget"] = 99

    assert bundle.id == original_id
    assert bundle.fields == {"policy": {"guards": ["allow-a"]}, "verifier_budget": 1}


def test_derive_with_same_semantics_preserves_identity():
    bundle = InvariantBundle(policy_class="strict", verifier_budget=3)

    same = bundle.derive(verifier_budget=3)

    assert same.id == bundle.id
    assert same.fields == bundle.fields
