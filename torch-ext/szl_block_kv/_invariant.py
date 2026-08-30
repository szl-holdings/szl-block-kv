# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Invariant-keyed block addressing for paged KV.

A paged-KV block is normally addressed by its content alone -- the token ids
that produced it. That is sound for arithmetic but unsound for governance: a
block computed while one invariant bundle was in force can then be served to a
request running under a different bundle. The reuse is numerically correct and
governance-wrong, and nothing in the cache layer can see the difference.

This module makes the governance regime part of the address:

    block_key = SHA3-256( canonical(token_ids) || bundle_id )

Same tokens under a different bundle produce a different key, so the lookup
misses and the block is recomputed under the regime that asked for it. The
property is structural, not advisory -- there is no code path that returns a
block keyed to another bundle.

Labels. This is a MEASURED containment property of this module's addressing,
verified by the tests in tests/test_invariant.py. It is NOT a proof about
attention arithmetic, NOT a speedup claim, and NOT a Lambda result.
Lambda = Conjecture 1 OPEN.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._chain import ReceiptChain, _canon, _sha3

__all__ = [
    "InvariantBundle",
    "block_key",
    "InvariantKeyedBlockTable",
    "CrossRegimeError",
    "selfcheck_invariant",
]


class CrossRegimeError(RuntimeError):
    """Raised when a caller tries to force a block across governance regimes."""


class InvariantBundle:
    """A canonical, hashable governance regime.

    The bundle is whatever set of invariants, policy classes, verifier budgets,
    or doctrine versions must hold for a computed block to be reusable. It is
    canonicalized the same way receipts are, so the id is stable across
    processes and machines.
    """

    __slots__ = ("_fields", "_id")

    def __init__(self, **fields: Any) -> None:
        if not fields:
            raise ValueError("an invariant bundle must not be empty")
        self._fields: Dict[str, Any] = dict(fields)
        self._id: str = _sha3(_canon(self._fields))

    @property
    def id(self) -> str:
        """Full SHA3-256 hex digest of the canonicalized bundle."""
        return self._id

    @property
    def short_id(self) -> str:
        return self._id[:16]

    @property
    def fields(self) -> Dict[str, Any]:
        return dict(self._fields)

    def derive(self, **overrides: Any) -> "InvariantBundle":
        """A new bundle with fields changed. Always a different id."""
        merged = dict(self._fields)
        merged.update(overrides)
        return InvariantBundle(**merged)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InvariantBundle) and other._id == self._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self) -> str:
        return f"InvariantBundle(id={self.short_id}..., fields={sorted(self._fields)})"


def block_key(token_ids: Sequence[int], bundle: InvariantBundle) -> str:
    """Content address of a KV block under a governance regime.

    Domain-separated so that no concatenation of a token run and a bundle id can
    collide with a different split of the same bytes.
    """
    if not isinstance(bundle, InvariantBundle):
        raise TypeError("bundle must be an InvariantBundle")
    ids: List[int] = [int(t) for t in token_ids]
    body = _canon({"v": 1, "tokens": ids, "bundle": bundle.id})
    return hashlib.sha3_256(b"szl.block-kv.invariant-key\x1f" + body).hexdigest()


class InvariantKeyedBlockTable:
    """Maps invariant-keyed block addresses to physical block indices.

    Every admission and every lookup is receipted. Statistics separate a true
    hit from the two distinct kinds of miss, because they mean different things
    operationally: a `cold` miss is capacity or novelty, a `cross_regime` miss
    is governance doing its job.
    """

    def __init__(
        self,
        num_blocks: int,
        *,
        chain: Optional[ReceiptChain] = None,
        strict: bool = True,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = int(num_blocks)
        self.strict = bool(strict)
        self.chain = chain
        self._by_key: Dict[str, int] = {}
        self._meta: Dict[int, Dict[str, Any]] = {}
        self._free: List[int] = list(range(self.num_blocks - 1, -1, -1))
        self.stats: Dict[str, int] = {
            "hit": 0,
            "miss_cold": 0,
            "miss_cross_regime": 0,
            "admitted": 0,
            "evicted": 0,
            "rejected_cross_regime": 0,
        }
        # token-run digest -> set of bundle ids seen for it, so a cross-regime
        # miss can be reported as such instead of looking like a cold miss
        self._runs: Dict[str, set] = {}

    # -------------------------------------------------------------- internals
    @staticmethod
    def _run_digest(token_ids: Sequence[int]) -> str:
        return _sha3(_canon({"v": 1, "tokens": [int(t) for t in token_ids]}))

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.chain is not None:
            self.chain.emit(payload)

    # ------------------------------------------------------------------- api
    def lookup(self, token_ids: Sequence[int], bundle: InvariantBundle) -> Tuple[Optional[int], str]:
        """Return (physical_block, outcome).

        outcome is one of "hit", "miss_cold", "miss_cross_regime".
        """
        key = block_key(token_ids, bundle)
        run = self._run_digest(token_ids)
        physical = self._by_key.get(key)
        if physical is not None:
            self.stats["hit"] += 1
            outcome = "hit"
        elif run in self._runs and bundle.id not in self._runs[run]:
            # exact same tokens are cached, but only under another regime
            self.stats["miss_cross_regime"] += 1
            outcome = "miss_cross_regime"
        else:
            self.stats["miss_cold"] += 1
            outcome = "miss_cold"
        self._emit(
            {
                "op": "block_lookup",
                "key": key[:32],
                "run": run[:32],
                "bundle": bundle.short_id,
                "outcome": outcome,
                "physical": physical if physical is not None else -1,
                "lambda": "Conjecture 1",
            }
        )
        return physical, outcome

    def admit(
        self,
        token_ids: Sequence[int],
        bundle: InvariantBundle,
        *,
        physical: Optional[int] = None,
    ) -> int:
        """Bind a computed block to its invariant-keyed address.

        Returns the physical block index. Evicts the oldest admission when the
        table is full (FIFO in v0 -- no eviction-quality claim is made).
        """
        key = block_key(token_ids, bundle)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing

        if physical is None:
            if not self._free:
                self._evict_one()
            physical = self._free.pop()
        else:
            physical = int(physical)
            if not 0 <= physical < self.num_blocks:
                raise ValueError(f"physical block {physical} out of range")
            prior = self._meta.get(physical)
            if prior is not None:
                if self.strict and prior["bundle"] != bundle.id:
                    self.stats["rejected_cross_regime"] += 1
                    self._emit(
                        {
                            "op": "block_admit_rejected",
                            "key": key[:32],
                            "bundle": bundle.short_id,
                            "physical": physical,
                            "reason": "physical block already bound to another regime",
                            "lambda": "Conjecture 1",
                        }
                    )
                    raise CrossRegimeError(
                        f"physical block {physical} is bound to bundle "
                        f"{prior['bundle'][:16]}, refusing to rebind to {bundle.short_id}"
                    )
                self._release(physical)
            if physical in self._free:
                self._free.remove(physical)

        run = self._run_digest(token_ids)
        self._by_key[key] = physical
        self._meta[physical] = {
            "key": key,
            "run": run,
            "bundle": bundle.id,
            "n_tokens": len(list(token_ids)),
        }
        self._runs.setdefault(run, set()).add(bundle.id)
        self.stats["admitted"] += 1
        self._emit(
            {
                "op": "block_admit",
                "key": key[:32],
                "run": run[:32],
                "bundle": bundle.short_id,
                "physical": physical,
                "n_tokens": self._meta[physical]["n_tokens"],
                "lambda": "Conjecture 1",
            }
        )
        return physical

    def _release(self, physical: int) -> None:
        meta = self._meta.pop(physical, None)
        if meta is None:
            return
        self._by_key.pop(meta["key"], None)
        run_bundles = self._runs.get(meta["run"])
        if run_bundles is not None:
            # only forget the (run, bundle) pairing if no other live block holds it
            still_live = any(
                m["run"] == meta["run"] and m["bundle"] == meta["bundle"]
                for m in self._meta.values()
            )
            if not still_live:
                run_bundles.discard(meta["bundle"])
                if not run_bundles:
                    self._runs.pop(meta["run"], None)

    def _evict_one(self) -> None:
        if not self._meta:
            return
        physical = next(iter(self._meta))
        self._release(physical)
        self._free.append(physical)
        self.stats["evicted"] += 1
        self._emit({"op": "block_evict", "physical": physical, "policy": "fifo_v0",
                    "lambda": "Conjecture 1"})

    def bundle_of(self, physical: int) -> Optional[str]:
        meta = self._meta.get(int(physical))
        return meta["bundle"] if meta else None

    def regimes_for(self, token_ids: Sequence[int]) -> List[str]:
        """Bundle ids that currently hold a block for this exact token run."""
        return sorted(self._runs.get(self._run_digest(token_ids), set()))

    def audit(self) -> Dict[str, Any]:
        """Structural check: no physical block serves two regimes, and every
        address resolves to a block whose recorded bundle matches its key."""
        problems: List[str] = []
        for physical, meta in self._meta.items():
            if self._by_key.get(meta["key"]) != physical:
                problems.append(f"address/physical disagreement at block {physical}")
        seen: Dict[int, str] = {}
        for physical in self._by_key.values():
            bundle = self._meta[physical]["bundle"]
            if physical in seen and seen[physical] != bundle:
                problems.append(f"block {physical} serves two regimes")
            seen[physical] = bundle
        return {
            "ok": not problems,
            "problems": problems,
            "live_blocks": len(self._meta),
            "distinct_regimes": len({m["bundle"] for m in self._meta.values()}),
            "stats": dict(self.stats),
            "label": "MEASURED containment of this addressing layer",
            "what_not": "not a proof about attention arithmetic; no speedup claim",
            "lambda": "Conjecture 1",
        }

    def __len__(self) -> int:
        return len(self._meta)


def selfcheck_invariant() -> Dict[str, Any]:
    """Demonstrate the containment property end to end, with receipts.

    Reuse under the same regime hits. The identical token run under a stricter
    regime misses and is reported as cross-regime, not as a cold miss. No block
    is ever served across regimes.
    """
    chain = ReceiptChain()
    permissive = InvariantBundle(
        doctrine="v11",
        policy_class="default-permissive",
        receipts_required=True,
        verifier_budget=0,
    )
    strict = permissive.derive(policy_class="default-strict", verifier_budget=3)

    table = InvariantKeyedBlockTable(num_blocks=4, chain=chain)
    run = [15496, 11, 995, 13]  # arbitrary token run

    _, first = table.lookup(run, permissive)
    physical = table.admit(run, permissive)
    _, second = table.lookup(run, permissive)
    _, crossed = table.lookup(run, strict)

    rejected = False
    try:
        table.admit(run, strict, physical=physical)
    except CrossRegimeError:
        rejected = True

    strict_physical = table.admit(run, strict)
    audit = table.audit()
    chain_ok, depth, _bad = chain.verify()

    ok = bool(
        first == "miss_cold"
        and second == "hit"
        and crossed == "miss_cross_regime"
        and rejected
        and strict_physical != physical
        and block_key(run, permissive) != block_key(run, strict)
        and audit["ok"]
        and chain_ok
    )
    return {
        "ok": ok,
        "first_lookup": first,
        "same_regime_lookup": second,
        "cross_regime_lookup": crossed,
        "cross_regime_admit_rejected": rejected,
        "distinct_physical_blocks": strict_physical != physical,
        "permissive_bundle": permissive.short_id,
        "strict_bundle": strict.short_id,
        "chain_ok": chain_ok,
        "chain_depth": depth,
        "audit": audit,
        "label": "MEASURED containment property of invariant-keyed addressing",
        "note": (
            "Same tokens under a different invariant bundle address a different "
            "block, so cross-regime reuse cannot occur. This is a property of the "
            "addressing layer only. Lambda = Conjecture 1 OPEN. No speedup claimed; "
            "no tokens/s; no joules."
        ),
        "lambda": "Conjecture 1",
    }
