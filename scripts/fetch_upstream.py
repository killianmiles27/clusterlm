#!/usr/bin/env python3
"""Fetch pinned upstream sources (third_party/upstream.json) into third_party/upstream/<name>.

Stdlib only. Each pin is fetched at the exact commit (git init + fetch --depth 1 origin <sha> + checkout).
Dirty existing checkouts are refused and never modified -- except a checkout that is exactly "pin + its
ClusterLM patches", which is the expected state after --apply-patches.

Patches: a pin may name a patch directory ("patches": "third_party/patches/<name>"). Its `*.patch` files are
applied in file-name order by --apply-patches. Application is idempotent: a checkout that already equals
pin + every patch is left alone; a clean checkout is patched; anything else (a partial or foreign
modification) is refused. `--check` verifies the pin and, with --apply-patches, that the tree is exactly
pin + patches (no network, nothing modified).

The "exactly pin + patches" test is content-addressed: the expected tree is computed by applying every patch to
HEAD in a temporary index (`git apply --cached`), the actual tree by adding the working tree to another
temporary index; the two tree hashes must be equal.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / "third_party" / "upstream.json"
DEST = ROOT / "third_party" / "upstream"


def git(repo, *args, check=True, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], check=check, capture_output=True, text=True, env=env)


def head(repo):
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def is_dirty(repo):
    return bool(git(repo, "status", "--porcelain").stdout.strip())


def patch_files(pin):
    d = pin.get("patches")
    if not d:
        return []
    return sorted((ROOT / d).glob("*.patch"))


def _tree_with_index(repo, fill):
    """Run `fill(env)` against a fresh temporary index and return `git write-tree` for it."""
    fd, idx = tempfile.mkstemp(prefix="clusterlm-idx-")
    os.close(fd)
    os.unlink(idx)  # git creates it
    env = dict(os.environ, GIT_INDEX_FILE=idx)
    try:
        if not fill(env):
            return None
        return git(repo, "write-tree", env=env).stdout.strip()
    finally:
        if os.path.exists(idx):
            os.unlink(idx)


def expected_tree(repo, patches):
    def fill(env):
        git(repo, "read-tree", "HEAD", env=env)
        for p in patches:
            r = git(repo, "apply", "--cached", "--whitespace=nowarn", str(p), check=False, env=env)
            if r.returncode != 0:
                print(f"patch {p.name} does not apply to the pin: {r.stderr.strip()}", file=sys.stderr)
                return False
        return True
    return _tree_with_index(repo, fill)


def working_tree(repo):
    def fill(env):
        git(repo, "read-tree", "HEAD", env=env)
        git(repo, "add", "-A", env=env)
        return True
    return _tree_with_index(repo, fill)


def patch_state(name, pin):
    """'clean', 'patched', 'none' (no patches) or 'foreign' (dirty in some other way)."""
    repo = DEST / name
    patches = patch_files(pin)
    if not patches:
        return "none"
    if not is_dirty(repo):
        return "clean"
    want = expected_tree(repo, patches)
    have = working_tree(repo)
    if want is not None and want == have:
        return "patched"
    return "foreign"


def verify(name, pin, patched=False):
    repo = DEST / name
    if not (repo / ".git").exists():
        print(f"[{name}] MISSING: {repo}", file=sys.stderr)
        return False
    h = head(repo)
    if h != pin["commit"]:
        print(f"[{name}] HEAD {h} != pin {pin['commit']}", file=sys.stderr)
        return False
    state = patch_state(name, pin) if pin.get("patches") else ("clean" if not is_dirty(repo) else "foreign")
    if state == "foreign" or (state == "patched" and not patched) or (patched and state == "clean"):
        what = {"foreign": "DIRTY working tree (not pin + ClusterLM patches)",
                "patched": "patched (pin + ClusterLM patches); pass --apply-patches to accept",
                "clean": "not patched (run with --apply-patches)"}[state]
        print(f"[{name}] {what}: {repo}", file=sys.stderr)
        return False
    lic = repo / pin.get("license_file", "LICENSE")
    if not lic.is_file():
        print(f"[{name}] license file {lic} not found", file=sys.stderr)
        return False
    suffix = f" + {len(patch_files(pin))} ClusterLM patch(es)" if state == "patched" else ""
    print(f"[{name}] OK {h}{suffix} ({pin['license']}) license: {lic}")
    return True


def apply_patches(name, pin):
    repo = DEST / name
    patches = patch_files(pin)
    if not patches:
        return True
    state = patch_state(name, pin)
    if state == "patched":
        print(f"[{name}] patches already applied ({len(patches)})")
        return True
    if state != "clean":
        print(f"[{name}] refusing to patch: the tree is modified and is not pin + patches", file=sys.stderr)
        return False
    for p in patches:
        r = git(repo, "apply", "--whitespace=nowarn", str(p), check=False)
        if r.returncode != 0:
            print(f"[{name}] patch {p.name} failed: {r.stderr.strip()}", file=sys.stderr)
            # leave the pin clean, never half-patched (the tree was clean before, so anything untracked is ours)
            git(repo, "checkout", "-q", "--", ".", check=False)
            git(repo, "clean", "-fdq", check=False)
            return False
        print(f"[{name}] applied {p.name}")
    return True


def fetch(name, pin, patched):
    repo = DEST / name
    if (repo / ".git").exists():
        if head(repo) == pin["commit"]:
            if is_dirty(repo):
                state = patch_state(name, pin) if pin.get("patches") else "foreign"
                if state != "patched":
                    print(f"[{name}] refusing: dirty tree at {repo}", file=sys.stderr)
                    return False
            if patched and not apply_patches(name, pin):
                return False
            return verify(name, pin, patched)
        if is_dirty(repo):
            print(f"[{name}] refusing: dirty tree at {repo}", file=sys.stderr)
            return False
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
    if patched and not apply_patches(name, pin):
        return False
    return verify(name, pin, patched)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify existing checkouts only")
    ap.add_argument("--apply-patches", action="store_true",
                    help="apply (or with --check: require) the ClusterLM patches of each pin that has them")
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
        patched = args.apply_patches and bool(pins[n].get("patches"))
        ok &= verify(n, pins[n], patched) if args.check else fetch(n, pins[n], patched)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
