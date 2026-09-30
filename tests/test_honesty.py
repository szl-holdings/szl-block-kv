# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Honesty labels. Not a Λ proof. Not a speedup claim."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace

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
    assert "UNAVAILABLE" in card
    assert "ROADMAP" not in card


def test_readme_load_and_no_speedup():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert "No speedup claim" in readme
    assert "UNAVAILABLE" in readme
    assert "ROADMAP" not in readme
    assert "Copyright 2026 SZL Holdings" in readme
    assert "permits execution of the selected repository's Python" in normalized
    assert "does not verify hashes, publisher authorization or compatibility" in normalized

    examples = []
    for snippet in re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL):
        tree = ast.parse(snippet)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_kernel"
        ]
        for call in calls:
            assert call.args and isinstance(call.args[0], ast.Constant)
            assert call.args[0].value == "SZLHOLDINGS/szl-block-kv"
            keywords = {item.arg: item.value for item in call.keywords}
            trust = keywords.get("trust_remote_code")
            assert isinstance(trust, ast.Constant) and trust.value is True
            revision = keywords.get("revision")
            if isinstance(revision, ast.Constant):
                assert isinstance(revision.value, str)
                assert re.fullmatch(r"[0-9a-f]{40}", revision.value)
            else:
                assert isinstance(revision, ast.Name)
                # Exercise only the revision precondition, before any loader import.
                prefix = []
                for node in tree.body:
                    if isinstance(node, ast.ImportFrom) and node.module == "kernels":
                        break
                    if isinstance(node, ast.Import):
                        assert all(item.name in {"os", "re"} for item in node.names)
                        continue
                    prefix.append(node)
                assert prefix
                code = compile(ast.Module(body=prefix, type_ignores=[]), "<README revision guard>", "exec")
                for invalid in ["", "main", "v1", "a" * 39, "a" * 41, "A" * 40, "a" * 39 + "g"]:
                    namespace = {
                        "os": SimpleNamespace(environ={"SZL_BLOCK_KV_HF_REVISION": invalid}),
                        "re": re,
                    }
                    try:
                        exec(code, namespace)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Invalid revision reached the loader import")
            examples.append(call)
    assert examples, "README must retain a discoverable kernel example"


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
