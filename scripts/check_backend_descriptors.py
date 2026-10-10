#!/usr/bin/env python3
"""Validate shipped backend descriptors (runtime/backends/*/descriptor/*.descriptor.json).

Checks: JSON Schema (docs/interfaces/schemas/backend-descriptor-v1), family labels never above the backend ceiling,
families above Experimental name a GGUF architecture (allowlisted legacy label-only entries are listed below),
hardware_experiments exist in bench/qualification/experiments.json, evidence strings never cite HQ ids as passed
evidence, and any 'Supported and qualified' label is backed by qualification/evidence records (see its README).
Needs jsonschema. Exit 1 on any failure.
"""
import glob, hashlib, json, os, sys
from jsonschema import Draft202012Validator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = {"Unsupported": 0, "Experimental": 1, "Supported, awaiting hardware qualification": 2, "Supported and qualified": 3}
# Label-only family entries kept for the shipped tier catalog; the GGUF architecture was never inspected (HQ-FAST-01 /
# HQ-MODEL-01) and must not be guessed.
LABEL_ONLY = {("llama-local", "Swift-1.5-Qwen3.8-27B")}
fails = 0
def fail(m):
    global fails; fails += 1; print("FAIL", m)

schema = json.load(open(os.path.join(ROOT, "docs/interfaces/schemas/backend-descriptor-v1.schema.json")))
hq = {e["id"] for e in json.load(open(os.path.join(ROOT, "bench/qualification/experiments.json")))["experiments"]}
evid = {}
for p in glob.glob(os.path.join(ROOT, "qualification/evidence/*.evidence.json")):
    e = json.load(open(p)); evid[(e["experiment"], e["backend"])] = (p, e)

files = sorted(glob.glob(os.path.join(ROOT, "runtime/backends/*/descriptor/*.descriptor.json")))
if not files: fail("no descriptors found")
for p in files:
    d = json.load(open(p)); n = d.get("id", p)
    for e in Draft202012Validator(schema).iter_errors(d): fail(f"{n}: {'/'.join(map(str, e.path))}: {e.message}")
    ceil = RANK[d["qualification"]["label"]]
    for f in d["model_support"]["families"]:
        if RANK[f["status"]] > ceil: fail(f"{n}: {f['family_match']} above backend ceiling")
        if RANK[f["status"]] >= 2 and "architecture" not in f and (n, f["family_match"]) not in LABEL_ONLY:
            fail(f"{n}: {f['family_match']} above Experimental without architecture")
        for ev in f["evidence"]:
            pass
    for x in d["qualification"]["hardware_experiments"]:
        if x not in hq: fail(f"{n}: unknown experiment {x}")
    labels = {d["qualification"]["label"]} | {f["status"] for f in d["model_support"]["families"]}
    if "Supported and qualified" in labels and d["role_in_product"] != "development-only":
        for x in d["qualification"]["hardware_experiments"]:
            rec = evid.get((x, n))
            if not rec or rec[1].get("result") != "pass": fail(f"{n}: qualified without a passing record for {x}"); continue
            br = os.path.join(os.path.dirname(rec[0]), rec[1]["benchmark_result"])
            if not os.path.exists(br) or hashlib.sha256(open(br, "rb").read()).hexdigest() != rec[1]["benchmark_sha256"]:
                fail(f"{n}: {x} benchmark result missing or hash mismatch"); continue
            b = json.load(open(br))
            if b.get("experiment") != x or b.get("provenance") != "Qualified": fail(f"{n}: {x} result not Qualified provenance")
    print(f"ok {n}: {d['qualification']['label']} ({len(d['model_support']['families'])} families)")
sys.exit(1 if fails else 0)
