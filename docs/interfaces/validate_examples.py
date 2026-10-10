#!/usr/bin/env python3
"""Validate docs/interfaces/examples/*.json against docs/interfaces/schemas/*.json (needs the jsonschema package).

Checks only JSON Schema conformance plus the cross-field rules the schema cannot express (listed in
profile-schema-v1.md "Semantic validation"). It is a design-time guard, not the product validator; workstream A
implements the real one in C++.
"""
import glob, json, os, sys
from jsonschema import Draft202012Validator

here = os.path.dirname(os.path.abspath(__file__))
schemas = {"clusterlm.profile": "profile-v1", "clusterlm.routing-alias": "routing-alias-v1",
           "clusterlm.backend-descriptor": "backend-descriptor-v1"}
failures = 0
profiles = {}
for path in sorted(glob.glob(os.path.join(here, "examples", "*.json"))):
    doc = json.load(open(path))
    sname = schemas[doc["schema"]]
    schema = json.load(open(os.path.join(here, "schemas", sname + ".schema.json")))
    Draft202012Validator.check_schema(schema)
    errs = sorted(Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.path))
    for e in errs:
        failures += 1
        print(f"FAIL {os.path.basename(path)}: {'/'.join(map(str, e.path))}: {e.message}")
    if doc["schema"] == "clusterlm.profile":
        profiles[doc["id"]] = doc
        slots = [s["slot"] for s in doc["topology"]["slots"]]
        sem = []
        if len(set(slots)) != len(slots): sem.append("duplicate slot names")
        if doc["topology"]["slots"][0]["kind"] != "host": sem.append("slot 0 must be the host")
        if sum(s["kind"] == "host" for s in doc["topology"]["slots"]) != 1: sem.append("exactly one host slot")
        nw = sum(s["kind"] == "worker" for s in doc["topology"]["slots"])
        t = doc["topology"]
        if not (t.get("min_workers", 0) <= t.get("max_workers", 0) <= nw): sem.append("min<=max<=worker slots violated")
        if doc["context"]["default_tokens"] > doc["context"]["max_tokens"]: sem.append("default_tokens > max_tokens")
        for c in doc["context"].get("offered_profiles", []):
            if c > doc["context"]["max_tokens"]: sem.append(f"offered {c} > max_tokens")
        for m in sem:
            failures += 1; print(f"FAIL {os.path.basename(path)}: {m}")
# fallback chain: references resolve and are acyclic
for pid, doc in profiles.items():
    seen, cur = set(), doc
    while cur and cur.get("on_worker_loss", {}).get("then") == "fallback-profile":
        nxt = cur["on_worker_loss"].get("fallback_profile_id")
        if nxt not in profiles: failures += 1; print(f"FAIL {pid}: fallback {nxt} not found in examples"); break
        if nxt in seen or nxt == pid: failures += 1; print(f"FAIL {pid}: fallback cycle"); break
        seen.add(nxt); cur = profiles[nxt]
print("OK" if not failures else f"{failures} failure(s)")
sys.exit(1 if failures else 0)
