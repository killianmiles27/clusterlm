#!/usr/bin/env python3
"""Fetch pinned upstream sources (third_party/upstream.json) into third_party/upstream/<name>.

Stdlib only. Each pin is fetched at the exact commit (git init + fetch --depth 1 origin <sha> + checkout).
Dirty existing checkouts are refused and never modified. `--check` only verifies, never touches the network.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / "third_party" / "upstream.json"
DEST = ROOT / "third_party" / "upstream"


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=check, capture_output=True, text=True)


def head(repo):
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def is_dirty(repo):
    return bool(git(repo, "status", "--porcelain").stdout.strip())


def verify(name, pin):
    repo = DEST / name
    if not (repo / ".git").exists():
        print(f"[{name}] MISSING: {repo}", file=sys.stderr)
        return False
    if is_dirty(repo):
        print(f"[{name}] DIRTY working tree: {repo}", file=sys.stderr)
        return False
    h = head(repo)
    if h != pin["commit"]:
        print(f"[{name}] HEAD {h} != pin {pin['commit']}", file=sys.stderr)
        return False
    lic = repo / pin.get("license_file", "LICENSE")
    if not lic.is_file():
        print(f"[{name}] license file {lic} not found", file=sys.stderr)
        return False
    print(f"[{name}] OK {h} ({pin['license']}) license: {lic}")
    return True


def fetch(name, pin):
    repo = DEST / name
    if (repo / ".git").exists():
        if is_dirty(repo):
            print(f"[{name}] refusing: dirty tree at {repo}", file=sys.stderr)
            return False
        if head(repo) == pin["commit"]:
            return verify(name, pin)
    else:
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, "init", "-q")
    existing = git(repo, "remote", "get-url", "origin", check=False)
    if existing.returncode != 0:
        git(repo, "remote", "add", "origin", pin["url"])
    elif existing.stdout.strip() != pin["url"]:
        git(repo, "remote", "set-url", "origin", pin["url"])
    r = git(repo, "fetch", "-q", "--depth", "1", "origin", pin["commit"], check=False)
    if r.returncode != 0:
        print(f"[{name}] fetch failed: {r.stderr.strip()}", file=sys.stderr)
        return False
    git(repo, "checkout", "-q", "--detach", pin["commit"])
    return verify(name, pin)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify existing checkouts only")
    ap.add_argument("names", nargs="*", help="subset of pins (default: all)")
    args = ap.parse_args()
    pins = json.loads(PINS.read_text())["pins"]
    names = args.names or list(pins)
    unknown = [n for n in names if n not in pins]
    if unknown:
        print(f"unknown pin(s): {unknown}", file=sys.stderr)
        return 2
    ok = True
    for n in names:
        ok &= (verify if args.check else fetch)(n, pins[n])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
