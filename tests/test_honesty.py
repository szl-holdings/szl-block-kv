# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Honesty labels. Not a Λ proof. Not a speedup claim."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "Doctrine v11",
    "Conjecture 1 OPEN",
    "Original SZL construction in the paged-KV category",
    "https://arxiv.org/abs/2309.06180",
    "NOT a rehost of vLLM or kernels-community/paged-attention",
    "Distinct from a11oy MODELED H2O eviction",
    'get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)',
    "library_name: kernels",
]


def test_card_honesty_phrases():
    card = (ROOT / "CARD.md").read_text(encoding="utf-8")
    for phrase in REQUIRED:
        assert phrase in card, f"missing from CARD.md: {phrase}"


def test_readme_load_and_no_speedup():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)' in readme
    assert "No speedup claim" in readme
    assert "ROADMAP" in readme
    assert "Copyright 2026 SZL Holdings" in readme


def test_license_szl_2026():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Copyright 2026 SZL Holdings" in text


def test_build_toml_edition_5_no_triton_backend():
    text = (ROOT / "build.toml").read_text(encoding="utf-8")
    assert "edition = 5" in text
    assert "[torch-noarch]" in text
    assert not any(line.strip().startswith("backend") for line in text.splitlines())
    assert 'repo-id = "SZLHOLDINGS/szl-block-kv"' in text


def test_no_vendored_vllm_cuda():
    cu = list(ROOT.rglob("*.cu")) + list(ROOT.rglob("*.cuh"))
    assert cu == [], f"vendored CUDA: {cu}"
