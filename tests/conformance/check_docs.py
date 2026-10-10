#!/usr/bin/env python3
"""Documentation and policy conformance checks (workstream G). Run: python3 -I tests/conformance/check_docs.py

1. Every relative Markdown link and image resolves to an existing file (anchors are not checked).
2. Licensing policy (owner decision pending, spec 02): no LICENSE/COPYING file at the repository root, and no user-facing
   document claims the project is "open source". Documents that discuss the question are exempt (EXEMPT_PREFIXES).

Exit status is the number of failures capped at 1. Output is one line per failure.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP_DIRS = {".git", "third_party", "build", "out", "node_modules", ".cache"}
# Files that legitimately mention the licensing question or quote the spec.
EXEMPT_PREFIXES = ("docs/spec/", "docs/status.md", "docs/cross-module-requests.md", "docs/plan.md", "docs/handoff/",
                   "docs/adr/", "docs/licensing/", "tests/conformance/")
LABELS = ["Supported and qualified", "Supported, awaiting hardware qualification", "Experimental", "Unsupported"]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*(```|~~~)")


def markdown_files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("build")]
        for f in files:
            if f.endswith(".md"):
                yield os.path.join(base, f)


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def main():
    failures = []
    for path in markdown_files():
        r = rel(path)
        in_fence = False
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if FENCE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                stripped = re.sub(r"`[^`]*`", "", line)  # ignore inline code
                for m in LINK.finditer(stripped):
                    target = m.group(1)
                    if re.match(r"^([a-z][a-z0-9+.-]*:|#|/mnt/)", target, re.I):
                        continue
                    target = target.split("#", 1)[0]
                    if not target:
                        continue
                    full = os.path.normpath(os.path.join(os.path.dirname(path), target))
                    if not os.path.exists(full):
                        failures.append(f"{r}:{n}: broken link: {m.group(1)}")
                if not r.startswith(EXEMPT_PREFIXES) and re.search(r"open[- ]source", stripped, re.I):
                    failures.append(f"{r}:{n}: claims 'open source' (license is an owner decision; see docs/licensing/)")
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md", "UNLICENSE"):
        if os.path.exists(os.path.join(ROOT, name)):
            failures.append(f"{name}: a license file exists; choosing a license is the owner's decision")
    for f in failures:
        print("FAIL", f)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
