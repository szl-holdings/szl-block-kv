# SPDX-License-Identifier: Apache-2.0
"""This repository's .github/hf-kernels.json builds a valid kernel payload. Offline.

Runs the real publisher against this checkout's HEAD, so a card that fails the
publisher's card lint, a missing smoke hook, or a package layout the `kernels`
loader cannot import fails every PR, not only PRs that touch the publisher paths.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "szl_kernel_publish.py"
_spec = importlib.util.spec_from_file_location("szl_kernel_publish_cfg", _SCRIPT)
kp = importlib.util.module_from_spec(_spec)
sys.modules["szl_kernel_publish_cfg"] = kp
_spec.loader.exec_module(kp)

_REPO_ID = "SZLHOLDINGS/szl-block-kv"


def _cfg():
    (cfg,) = kp.load_config(_ROOT)
    return cfg


def test_config_targets_this_kernel():
    cfg = _cfg()
    assert cfg.repo_id == _REPO_ID
    assert cfg.name == "szl-block-kv" and cfg.version == 1
    assert cfg.source_repository == "szl-holdings/szl-block-kv"
    assert [(p.src, p.name) for p in cfg.packages] == [("torch-ext/szl_block_kv", "szl_block_kv")]
    assert cfg.card == "CARD.md" and cfg.license_file == "LICENSE"
    assert cfg.twin is not None and cfg.twin.repo_id == _REPO_ID
    assert {"build/", "torch-ext/"} <= set(cfg.twin.prune_paths)


def test_payload_from_head_is_the_torch_ext_package():
    cfg = _cfg()
    head = kp.resolve_commit(_ROOT, "HEAD")
    payload = kp.build_payload(cfg, _ROOT, head)
    source = kp.read_tree_files(_ROOT, head, "torch-ext/szl_block_kv")
    for variant in kp.VARIANTS:
        files = {p[len(f"build/{variant}/szl_block_kv/"):]: d for p, d in payload.items()
                 if p.startswith(f"build/{variant}/szl_block_kv/")}
        assert files == {rel: data.replace(b"\r\n", b"\n") for rel, data in source.items()}
        meta = json.loads(payload[f"build/{variant}/metadata.json"])
        assert meta["name"] == "szl-block-kv" and meta["version"] == 1
        assert meta["backend"] == {"type": "cpu"}
        assert set(meta["digest"]["files"]) == {f"szl_block_kv/{rel}" for rel in source}
    assert not any(p.startswith("build/") and p.count("/") == 2 and p.endswith(".py") for p in payload)
    readme = payload["README.md"].decode("utf-8")
    kp.lint_card(cfg, readme, "CARD.md", require_usage=True)
    binding = json.loads(payload[kp.BINDING_PATH])
    assert binding["source_commit"] == head and binding["repo_type"] == "kernel"


def test_twin_pointer_and_smoke_hook_load():
    cfg = _cfg()
    head = kp.resolve_commit(_ROOT, "HEAD")
    text = kp.read_blob(_ROOT, head, cfg.twin.card).decode("utf-8")
    kp.lint_card(cfg, kp.render_card(cfg, text, cfg.twin.card), cfg.twin.card, require_usage=True)
    assert callable(kp.load_smoke(cfg, _ROOT, head))
