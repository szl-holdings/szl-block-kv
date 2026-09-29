#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Import the live Hugging Face card and file list of this repo's Hub twins.

Read-only. Uses the anonymous public Hub API (no token, no write call), so it
can never change the Hub. It records what the Hub serves today so that the
GitHub source can take over the card (plan decision: import the Hub card, then
mirror from GitHub) without losing Hub-only content.

    python scripts/hf_hub_import.py fetch            # refresh from the targets in IMPORT.json
    python scripts/hf_hub_import.py fetch --target kernel:SZLHOLDINGS/<id> --target model:SZLHOLDINGS/<id>
    python scripts/hf_hub_import.py verify           # offline: committed bytes match IMPORT.json

fetch writes hf/hub-import/<type>.README.md (the card bytes, verbatim) and
hf/hub-import/IMPORT.json (revision, card digests, the full Hub file list and
how each Hub file compares with this working tree). Every downloaded card is
checked against the git blob id the Hub tree reports before it is written.

verify needs no network. It fails when a committed card no longer matches the
digests recorded for it, or when IMPORT.json is inconsistent with itself.

Exit codes: 0 ok, 1 verification or fetch failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMPORT_DIR = ROOT / "hf" / "hub-import"
IMPORT_JSON = IMPORT_DIR / "IMPORT.json"
SCHEMA = "szl.hf-hub-import.v1"
HUB = "https://huggingface.co"
API_KIND = {"model": "models", "kernel": "kernels"}
RESOLVE_PREFIX = {"model": "", "kernel": "kernels/"}
REPO_ID = re.compile(r"^SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
# The Hub card is README.md; in this repo the card source is CARD.md.
PATH_MAP = {"README.md": "CARD.md"}
# Kernel Hub builds are build/<variant>/<module>/...; the source is torch-ext/<module>/...
BUILD_FILE = re.compile(r"^build/[^/]+/([^/]+/.+)$")
TIMEOUT = 30


class HubImportError(RuntimeError):
    """Raised when the Hub answer cannot be recorded exactly."""


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _get(url: str) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": "szl-hf-hub-import/1 (read-only)"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 (fixed https host)
            return resp.read(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        raise HubImportError(f"GET {url} -> HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise HubImportError(f"GET {url} -> {exc.reason}") from None


def _get_json(url: str) -> tuple[Any, dict[str, str]]:
    body, headers = _get(url)
    try:
        return json.loads(body), headers
    except json.JSONDecodeError:
        raise HubImportError(f"GET {url} -> not JSON") from None


def _next_link(headers: dict[str, str]) -> str | None:
    match = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
    return match.group(1) if match else None


def hub_tree(kind: str, repo_id: str, revision: str) -> list[dict[str, Any]]:
    url: str | None = f"{HUB}/api/{API_KIND[kind]}/{repo_id}/tree/{revision}?recursive=true"
    entries: list[dict[str, Any]] = []
    while url:
        page, headers = _get_json(url)
        if not isinstance(page, list):
            raise HubImportError(f"tree listing for {kind}:{repo_id} is not a list")
        entries.extend(page)
        url = _next_link(headers)
    return entries


def local_counterpart(kind: str, hub_path: str) -> str:
    if hub_path in PATH_MAP:
        return PATH_MAP[hub_path]
    if kind == "kernel":
        match = BUILD_FILE.match(hub_path)
        if match:
            return f"torch-ext/{match.group(1)}"
    return hub_path


def stored_oid(rel: str) -> str:
    """Blob id this file would have in git (after the repo's eol filters).

    Hashing the checkout bytes directly would compare CRLF copies on Windows.
    `git hash-object --path` applies the same clean filters as `git add`; the
    raw-bytes id is the fallback when git is not available.
    """
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "hash-object", "--path", rel, "--", rel],
                             check=True, capture_output=True, text=True, timeout=TIMEOUT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out if SHA40.fullmatch(out) else git_blob_oid((ROOT / rel).read_bytes())


def compare(kind: str, hub_path: str, hub_oid: str) -> dict[str, str]:
    github_path = local_counterpart(kind, hub_path)
    if not (ROOT / github_path).is_file():
        return {"github_path": github_path, "status": "hub_only"}
    same = stored_oid(github_path) == hub_oid
    return {"github_path": github_path, "status": "same" if same else "differs"}


def head_sha() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True, timeout=TIMEOUT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out if SHA40.fullmatch(out) else None


def fetch_target(kind: str, repo_id: str) -> dict[str, Any]:
    info, _ = _get_json(f"{HUB}/api/{API_KIND[kind]}/{repo_id}")
    revision = info.get("sha")
    if not isinstance(revision, str) or not SHA40.fullmatch(revision):
        raise HubImportError(f"{kind}:{repo_id} has no 40-hex revision in its info")
    if info.get("private") or info.get("gated"):
        raise HubImportError(f"{kind}:{repo_id} is private or gated; the anonymous import does not cover it")
    tree = hub_tree(kind, repo_id, revision)
    files = []
    card_oid = None
    for entry in sorted(tree, key=lambda e: e.get("path", "")):
        if entry.get("type") != "file":
            continue
        path, oid = entry["path"], entry["oid"]
        record = {"path": path, "oid": oid, "size": entry.get("size")}
        if entry.get("lfs"):
            record["lfs_sha256"] = entry["lfs"].get("oid")
        record.update(compare(kind, path, oid))
        files.append(record)
        if path == "README.md":
            card_oid = oid
    card: dict[str, Any] | None = None
    if card_oid is not None:
        url = f"{HUB}/{RESOLVE_PREFIX[kind]}{repo_id}/resolve/{revision}/README.md"
        data, _ = _get(url)
        if git_blob_oid(data) != card_oid:
            raise HubImportError(f"{kind}:{repo_id} README.md bytes do not match the tree blob id {card_oid}")
        rel = f"hf/hub-import/{kind}.README.md"
        IMPORT_DIR.mkdir(parents=True, exist_ok=True)
        (ROOT / rel).write_bytes(data)
        card = {
            "path": rel,
            "blob_oid": card_oid,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "has_front_matter": data.startswith((b"---\n", b"---\r\n")),
            "line_endings": "crlf" if b"\r\n" in data else "lf",
            "source_url": url,
        }
    summary: dict[str, int] = {"files": len(files)}
    for status in ("same", "differs", "hub_only"):
        summary[status] = sum(1 for f in files if f["status"] == status)
    record: dict[str, Any] = {
        "repo_type": kind,
        "repo_id": repo_id,
        "revision": revision,
        "last_modified": info.get("lastModified"),
        "card": card,
        "files": files,
        "summary": summary,
    }
    if kind == "kernel":
        # Source files this repo has that no Hub build carries.
        mapped = {f["github_path"] for f in files}
        record["github_only"] = sorted(p for p in tracked("torch-ext") if p not in mapped)
    return record


def tracked(prefix: str) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", prefix], check=True,
                             capture_output=True, text=True, timeout=TIMEOUT).stdout
        return [line for line in out.splitlines() if line]
    except (OSError, subprocess.SubprocessError):
        base = ROOT / prefix
        return sorted(p.relative_to(ROOT).as_posix() for p in base.rglob("*") if p.is_file())


def parse_targets(values: list[str]) -> list[tuple[str, str]]:
    targets = []
    for value in values:
        kind, _, repo_id = value.partition(":")
        if kind not in API_KIND or not REPO_ID.fullmatch(repo_id):
            raise SystemExit(f"--target must be model:SZLHOLDINGS/<id> or kernel:SZLHOLDINGS/<id>, got {value!r}")
        targets.append((kind, repo_id))
    return targets


def cmd_fetch(args: argparse.Namespace) -> int:
    if args.target:
        targets = parse_targets(args.target)
    else:
        try:
            current = json.loads(IMPORT_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print("hf_hub_import: no IMPORT.json to refresh; pass --target", file=sys.stderr)
            return 2
        targets = [(t["repo_type"], t["repo_id"]) for t in current.get("targets", [])]
    try:
        records = [fetch_target(kind, repo_id) for kind, repo_id in targets]
    except HubImportError as exc:
        print(f"hf_hub_import: {exc}", file=sys.stderr)
        return 1
    document = {
        "schema": SCHEMA,
        "generator": "scripts/hf_hub_import.py",
        "method": "anonymous public Hub API (no token, read-only)",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "compared_against": {"git_head": head_sha(), "tree": "working tree at generation time"},
        "path_map": {"README.md": "CARD.md", "build/<variant>/<module>/<file>": "torch-ext/<module>/<file>"},
        "targets": records,
    }
    IMPORT_JSON.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    for rec in records:
        s = rec["summary"]
        print(f"{rec['repo_type']}:{rec['repo_id']}@{rec['revision'][:8]} files={s['files']} "
              f"same={s['same']} differs={s['differs']} hub_only={s['hub_only']}")
    return 0


def verify() -> list[str]:
    problems: list[str] = []
    try:
        doc = json.loads(IMPORT_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {IMPORT_JSON.relative_to(ROOT).as_posix()}: {exc}"]
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    targets = doc.get("targets")
    if not isinstance(targets, list) or not targets:
        return problems + ["targets must be a non-empty list"]
    for t in targets:
        name = f"{t.get('repo_type')}:{t.get('repo_id')}"
        if t.get("repo_type") not in API_KIND or not REPO_ID.fullmatch(str(t.get("repo_id"))):
            problems.append(f"{name}: bad repo_type or repo_id")
        if not SHA40.fullmatch(str(t.get("revision"))):
            problems.append(f"{name}: revision is not a 40-hex sha")
        files = t.get("files") or []
        summary = t.get("summary") or {}
        if summary.get("files") != len(files):
            problems.append(f"{name}: summary.files != len(files)")
        for status in ("same", "differs", "hub_only"):
            if summary.get(status) != sum(1 for f in files if f.get("status") == status):
                problems.append(f"{name}: summary.{status} does not match the file list")
        card = t.get("card")
        readme = [f for f in files if f.get("path") == "README.md"]
        if card is None:
            if readme:
                problems.append(f"{name}: Hub has README.md but no card was imported")
            continue
        if not readme or readme[0].get("oid") != card.get("blob_oid"):
            problems.append(f"{name}: card blob id does not match the README.md tree entry")
        path = ROOT / str(card.get("path"))
        try:
            data = path.read_bytes()
        except OSError:
            problems.append(f"{name}: missing imported card {card.get('path')}")
            continue
        if git_blob_oid(data) != card.get("blob_oid"):
            problems.append(f"{name}: {card.get('path')} bytes changed (git blob id mismatch)")
        if hashlib.sha256(data).hexdigest() != card.get("sha256") or len(data) != card.get("bytes"):
            problems.append(f"{name}: {card.get('path')} sha256 or size mismatch")
    return problems


def cmd_verify(_: argparse.Namespace) -> int:
    problems = verify()
    for p in problems:
        print(f"hf_hub_import verify: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"{IMPORT_JSON.relative_to(ROOT).as_posix()}: imported cards match their recorded digests")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="read the Hub (anonymous) and rewrite hf/hub-import/")
    fetch.add_argument("--target", action="append", default=[],
                       help="model:SZLHOLDINGS/<id> or kernel:SZLHOLDINGS/<id> (repeatable)")
    fetch.set_defaults(func=cmd_fetch)
    ver = sub.add_parser("verify", help="offline check of the committed import")
    ver.set_defaults(func=cmd_verify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
