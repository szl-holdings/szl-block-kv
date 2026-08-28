# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
from szl_block_kv import selfcheck

def test_selfcheck():
    r = selfcheck()
    assert r["ok"] is True
    assert r["path"] == "torch_gather"
