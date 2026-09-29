# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face card source checks. Offline; no Hub access.

hf/hub-import/ must still hold the exact bytes the Hub served when it was
imported (scripts/hf_hub_import.py verify), and CARD.md, the card this repo
will publish to both Hub twins, must name this repo and its Apache-2.0 license.
The full card contract (schema and D10 receipts) is enforced by
.github/workflows/hf-card.yml with the shared hf-card toolkit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = "szl-block-kv"


def _importer():
    spec = importlib.util.spec_from_file_location("hf_hub_import", _ROOT / "scripts" / "hf_hub_import.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hub_import_matches_recorded_digests():
    assert _importer().verify() == []


def test_card_front_matter_names_this_repo_and_license():
    card = (_ROOT / "CARD.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    front_matter = card.split("\n---\n", 1)[0].splitlines()
    assert "license: apache-2.0" in front_matter
    assert "library_name: kernels" in front_matter
    assert f"  source_repo: szl-holdings/{_REPO}" in front_matter
    license_text = (_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text and "Version 2.0" in license_text
