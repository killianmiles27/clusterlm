#!/usr/bin/env python3
"""Tests scripts/fetch_upstream.py's patch handling on a synthetic upstream (stdlib only, no network).

Checks: patches apply in order onto a fresh pinned checkout; applying again is a no-op (idempotent); --check accepts
exactly "pin + patches"; a foreign modification is refused and never patched over; a clean pin is reported as not
patched; a patch that does not apply leaves the pin clean (never half-patched). Registered with ctest.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args, cwd):
    subprocess.run(list(args), cwd=cwd, check=True, capture_output=True, text=True)


def git_env():
    env = dict(os.environ)
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return env


def load_module(root: Path):
    spec = importlib.util.spec_from_file_location("fetch_upstream_under_test", HERE / "fetch_upstream.py")
    fu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fu)
    fu.ROOT = root
    fu.DEST = root / "third_party" / "upstream"
    fu.PINS = root / "third_party" / "upstream.json"
    return fu


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


def main():
    os.environ.update(git_env())
    with tempfile.TemporaryDirectory(prefix="clusterlm-fetch-test-") as tmp:
        tmp = Path(tmp)
        origin = tmp / "origin"
        origin.mkdir()
        run("git", "init", "-q", cwd=origin)
        run("git", "config", "uploadpack.allowAnySHA1InWant", "true", cwd=origin)
        (origin / "LICENSE").write_text("MIT\n")
        (origin / "a.txt").write_text("one\ntwo\nthree\n")
        (origin / "b.txt").write_text("alpha\nbeta\n")
        run("git", "add", "-A", cwd=origin)
        run("git", "commit", "-qm", "pin", cwd=origin)
        pin_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=origin, check=True, capture_output=True,
                                 text=True).stdout.strip()

        # two patches made against the pin, applied in order (the second touches a file the first changed)
        work = tmp / "work"
        run("git", "clone", "-q", str(origin), str(work), cwd=tmp)
        patches = tmp / "third_party" / "patches" / "up"
        patches.mkdir(parents=True)
        (work / "a.txt").write_text("one\nTWO\nthree\n")
        (patches / "0001-a.patch").write_text(subprocess.run(["git", "diff"], cwd=work, check=True, capture_output=True,
                                                             text=True).stdout)
        run("git", "commit", "-qam", "p1", cwd=work)
        (work / "a.txt").write_text("one\nTWO\nthree\nfour\n")
        (work / "b.txt").write_text("alpha\nBETA\n")
        (patches / "0002-ab.patch").write_text(subprocess.run(["git", "diff"], cwd=work, check=True, capture_output=True,
                                                              text=True).stdout)

        pins = {"schema": 1, "pins": {"up": {"url": "file://" + str(origin), "commit": pin_sha, "license": "MIT",
                                             "license_file": "LICENSE", "patches": "third_party/patches/up"}}}
        (tmp / "third_party" / "upstream.json").write_text(json.dumps(pins))
        fu = load_module(tmp)
        pin = pins["pins"]["up"]
        repo = fu.DEST / "up"

        failures = []

        def expect(cond, what):
            if not cond:
                failures.append(what)

        expect(quiet(fu.fetch, "up", pin, True), "fresh fetch + patches")
        expect((repo / "a.txt").read_text() == "one\nTWO\nthree\nfour\n", "patches applied in order")
        expect((repo / "b.txt").read_text() == "alpha\nBETA\n", "second patch applied")
        expect(fu.patch_state("up", pin) == "patched", "state is pin + patches")
        before = fu.working_tree(repo)
        expect(quiet(fu.fetch, "up", pin, True), "re-running is accepted")
        expect(fu.working_tree(repo) == before, "re-running changes nothing (idempotent)")
        expect(quiet(fu.apply_patches, "up", pin), "apply on an applied tree is a no-op")
        expect(fu.working_tree(repo) == before, "apply twice changes nothing")
        expect(quiet(fu.verify, "up", pin, True), "--check --apply-patches accepts pin + patches")
        expect(not quiet(fu.verify, "up", pin, False), "--check without --apply-patches flags the patched tree")

        (repo / "b.txt").write_text("alpha\nlocal edit\n")
        expect(fu.patch_state("up", pin) == "foreign", "a local edit is foreign")
        expect(not quiet(fu.apply_patches, "up", pin), "patching over a foreign edit is refused")
        expect((repo / "b.txt").read_text() == "alpha\nlocal edit\n", "the foreign edit is left untouched")
        expect(not quiet(fu.verify, "up", pin, True), "--check refuses a foreign edit")
        (repo / "b.txt").write_text("alpha\nBETA\n")
        expect(fu.patch_state("up", pin) == "patched", "undoing the edit restores pin + patches")

        run("git", "checkout", "-q", "--", ".", cwd=repo)
        expect(fu.patch_state("up", pin) == "clean", "a reset tree is the clean pin")
        expect(not quiet(fu.verify, "up", pin, True), "--check --apply-patches flags an unpatched pin")
        expect(quiet(fu.apply_patches, "up", pin), "a clean pin is patched")
        expect(fu.working_tree(repo) == before, "patching the clean pin gives the same tree")

        # a patch that does not apply leaves the pin clean, never half-patched
        run("git", "checkout", "-q", "--", ".", cwd=repo)
        (patches / "0003-bad.patch").write_text("--- a/zzz.txt\n+++ b/zzz.txt\n@@ -1 +1 @@\n-nope\n+yes\n")
        expect(not quiet(fu.apply_patches, "up", pin), "a broken patch fails")
        expect(fu.patch_state("up", pin) == "clean", "a failed application leaves the pin clean")

        if failures:
            for f in failures:
                print("FAIL:", f)
            return 1
        print("fetch_upstream patch handling: all checks passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
