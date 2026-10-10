#!/usr/bin/env python3
"""Offline-operation guard (guarantee G-12, workstream G). Run: python3 -I tests/conformance/check_offline.py

Static check over product sources (apps, orchestrator, node, runtime, ui, bench, tools): fails if they contain
  * a literal http:// or https:// URL (the product has no reason to name an internet endpoint; documentation URLs belong
    in docs, and the pinned-upstream fetcher is a build-time script, not product code), or
  * an include of an internet-client library (libcurl, WinHTTP, WinINet, cpp-httplib client use).
It is a guard, not a proof of offline behaviour: the dynamic check is "run the suite with the network namespace empty"
(see docs/testing.md). Local sockets (loopback, LAN) are the product's job and are not flagged.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIRS = ["apps", "orchestrator", "node", "runtime", "ui", "bench", "tools"]
EXT = (".cpp", ".cc", ".hpp", ".h", ".c", ".cu", ".cuh", ".mm")
URL = re.compile(r"https?://", re.I)
INCLUDE = re.compile(r'#\s*include\s*[<"](curl/[^>"]+|winhttp\.h|wininet\.h|urlmon\.h|httplib\.h)[>"]')
# Reviewed exceptions: (relative path, reason). Keep empty unless a reviewer approves an entry.
ALLOW = {}


def main():
    failures = []
    for d in DIRS:
        for base, dirs, files in os.walk(os.path.join(ROOT, d)):
            dirs[:] = [x for x in dirs if not x.startswith("build") and x != "third_party"]
            for f in files:
                if not f.endswith(EXT):
                    continue
                path = os.path.join(base, f)
                r = os.path.relpath(path, ROOT).replace(os.sep, "/")
                if r in ALLOW:
                    continue
                with open(path, encoding="utf-8", errors="replace") as fh:
                    for n, line in enumerate(fh, 1):
                        if URL.search(line):
                            failures.append(f"{r}:{n}: URL literal in product code")
                        m = INCLUDE.search(line)
                        if m:
                            failures.append(f"{r}:{n}: internet-client include {m.group(1)}")
    for f in failures:
        print("FAIL", f)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
