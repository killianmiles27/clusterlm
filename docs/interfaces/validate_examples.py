#!/usr/bin/env python3
"""Validate docs/interfaces/examples/*.json against docs/interfaces/schemas/*.json (needs the jsonschema package).

Checks JSON Schema conformance plus the cross-field and cross-document rules the schemas cannot express
(profile-schema-v1.md section 3 level 1, the parts of level 2 that need only a backend descriptor, routing-alias rules
and descriptor consistency). It is a design-time guard, not the product validator; workstream A implements the real
one in C++. Contract revision v1.1 (ADR 0407).
"""
import fnmatch, glob, json, os, sys
from jsonschema import Draft202012Validator

here = os.path.dirname(os.path.abspath(__file__))
schemas = {"clusterlm.profile": "profile-v1", "clusterlm.routing-alias": "routing-alias-v1",
           "clusterlm.backend-descriptor": "backend-descriptor-v1"}
RESERVED_ALIAS_ONLY = {"auto", "default"}
RESERVED_PREFIX = "clusterlm-"
LABEL_RANK = {"Unsupported": 0, "Experimental": 1, "Supported, awaiting hardware qualification": 2,
              "Supported and qualified": 3}
MAX_DOC_BYTES = 1 << 20

failures = 0
def fail(name, msg):
    global failures
    failures += 1
    print(f"FAIL {name}: {msg}")

def cc(s):
    a, b = s.split(".")
    return (int(a), int(b))

docs = {}
for path in sorted(glob.glob(os.path.join(here, "examples", "*.json"))):
    name = os.path.basename(path)
    if os.path.getsize(path) > MAX_DOC_BYTES:
        fail(name, "document larger than 1 MiB"); continue
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") not in schemas:
        fail(name, f"unknown schema {doc.get('schema')!r}"); continue
    with open(os.path.join(here, "schemas", schemas[doc["schema"]] + ".schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    errs = sorted(Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.path))
    for e in errs:
        fail(name, f"{'/'.join(map(str, e.path))}: {e.message}")
    if not errs:
        docs[name] = doc

descriptors = {d["id"]: d for d in docs.values() if d["schema"] == "clusterlm.backend-descriptor"}
profiles = {d["id"]: (n, d) for n, d in docs.items() if d["schema"] == "clusterlm.profile"}
aliases = {d["id"]: (n, d) for n, d in docs.items() if d["schema"] == "clusterlm.routing-alias"}

# ---- descriptors -------------------------------------------------------------------------------------------
for bid, d in descriptors.items():
    n = f"descriptor {bid}"
    ceiling = LABEL_RANK[d["qualification"]["label"]]
    for fam in d["model_support"]["families"]:
        if LABEL_RANK[fam["status"]] > ceiling:
            fail(n, f"family {fam['family_match']} label above the backend ceiling")
    if "nvidia" in d["devices"]["gpu_vendors"] and "min_compute_capability" not in d["devices"]:
        fail(n, "nvidia listed without min_compute_capability")
    if d["role_in_product"] == "production" and not d["qualification"]["hardware_experiments"]:
        fail(n, "production backend without hardware_experiments")
    if d["execution"]["cross_machine"] and not d["execution"]["worker_roles"]:
        fail(n, "cross_machine without worker_roles")

def compat(desc, ident):
    """(label, max_workers) for a profile identity, per backend-capability-v1 section 3 (no library record here:
    the family label and dominant quant stand in for the record)."""
    if ident["quant"].upper() not in {t.upper() for t in desc["model_support"]["tensor_formats"]}:
        return "Unsupported", 0
    label = desc["model_support"]["unlisted_family_status"]
    for fam in desc["model_support"]["families"]:
        if fnmatch.fnmatchcase(ident["family"], fam["family_match"]):
            label = fam["status"]; break
    if LABEL_RANK[label] > LABEL_RANK[desc["qualification"]["label"]]:
        label = desc["qualification"]["label"]
    distributable = desc["execution"]["cross_machine"] and LABEL_RANK[label] >= 2
    return label, (desc["execution"]["validated_max_workers"] if distributable else 0)

# ---- profiles ----------------------------------------------------------------------------------------------
api_ids = {}
def claim_api_id(owner, api_id, alias):
    if api_id in api_ids:
        fail(owner, f"api_model_id {api_id!r} also used by {api_ids[api_id]}")
    api_ids[api_id] = owner
    if api_id.startswith(RESERVED_PREFIX):
        fail(owner, f"api_model_id {api_id!r} uses the reserved prefix {RESERVED_PREFIX!r}")
    if api_id in RESERVED_ALIAS_ONLY and not alias:
        fail(owner, f"api_model_id {api_id!r} is reserved for routing aliases")

for pid, (n, d) in profiles.items():
    slots = d["topology"]["slots"]
    names = [s["slot"] for s in slots]
    if len(set(names)) != len(names): fail(n, "duplicate slot names")
    if slots[0]["kind"] != "host": fail(n, "slot 0 must be the host")
    if sum(s["kind"] == "host" for s in slots) != 1: fail(n, "exactly one host slot")
    workers = [s for s in slots if s["kind"] == "worker"]
    if len(workers) > 8: fail(n, "more than 8 worker slots")
    required = sum(1 for s in workers if not s.get("optional", False))
    t = d["topology"]
    mn, mx = t.get("min_workers", required), t.get("max_workers", len(workers))
    if not (required <= mn <= mx <= len(workers)):
        fail(n, f"need non-optional({required}) <= min_workers({mn}) <= max_workers({mx}) <= worker slots({len(workers)})")
    sel_keys = [(s["select"]["mode"], s["select"].get("binding") or s["select"].get("machine"))
                for s in workers if s["select"]["mode"] != "requirements"]
    if len(set(sel_keys)) != len(sel_keys): fail(n, "two slots select the same binding/machine")
    c = d["context"]
    if c["default_tokens"] > c["max_tokens"]: fail(n, "default_tokens > max_tokens")
    off = c.get("offered_profiles", [])
    if off:
        if off != sorted(off): fail(n, "offered_profiles not ascending")
        if off[-1] != c["max_tokens"]: fail(n, "max_tokens must be the largest offered profile")
        if any(x > c["max_tokens"] for x in off): fail(n, "offered profile > max_tokens")
    lc = d.get("lifecycle", {})
    if lc.get("preparation") == "manual" and lc.get("prepare_when_available"):
        fail(n, "prepare_when_available with preparation=manual")
    exp = d.get("exposure", {})
    if "api_model_id" in exp: claim_api_id(n, exp["api_model_id"], alias=False)
    desc = descriptors.get(d["backend"]["id"])
    if desc is None:
        fail(n, f"example uses backend {d['backend']['id']!r} with no example descriptor"); continue
    if desc["role_in_product"] != "production": fail(n, "product profile on a development-only backend")
    if d["backend"].get("min_contract_version", 1) > desc["contract_version"]: fail(n, "min_contract_version too new")
    label, cmax = compat(desc, d["model"]["identity"])
    if label == "Unsupported": fail(n, f"model is Unsupported on {desc['id']}")
    if mx > cmax: fail(n, f"max_workers {mx} > CompatReport.max_workers {cmax} ({label})")
    if workers and not desc["execution"]["cross_machine"]: fail(n, "worker slots on a single-host backend")
    sp = d.get("speculation", {})
    q = sp.get("max_q", 1) if sp.get("enabled", True) else 1
    if q > desc["speculation"]["max_q"]: fail(n, f"max_q {q} > descriptor max_q")
    if q > 1 and not desc["speculation"]["window_commit_abort"]: fail(n, "q > 1 without window commit/abort")
    oschema = desc.get("options_schema", {})
    for k, v in d["backend"].get("options", {}).items():
        spec = oschema.get(k)
        if spec is None: fail(n, f"backend option {k!r} not declared by the descriptor"); continue
        ok = {"integer": isinstance(v, int) and not isinstance(v, bool), "boolean": isinstance(v, bool),
              "string": isinstance(v, str)}[spec["type"]]
        if not ok: fail(n, f"backend option {k!r} has the wrong type")
        if ok and spec["type"] == "integer" and not (spec.get("minimum", v) <= v <= spec.get("maximum", v)):
            fail(n, f"backend option {k!r} out of range")
        if ok and "enum" in spec and v not in spec["enum"]: fail(n, f"backend option {k!r} not in enum")
    pl = d.get("placement", {})
    if pl.get("mode") == "manual":
        if not desc["execution"]["manual_layer_ranges"]: fail(n, "manual placement on a backend without manual ranges")
        prev_end = 0
        for st in pl["manual"]["stages"]:
            if st["slot"] not in names: fail(n, f"manual stage names unknown slot {st['slot']!r}")
            if not (st["first_layer"] == prev_end and st["end_layer"] > st["first_layer"]):
                fail(n, "manual stages must be contiguous, ascending and non-empty from layer 0")
            prev_end = st["end_layer"]
    for s in workers:
        r = s["select"].get("requirements")
        if r and "min_compute_capability" in r and "min_compute_capability" in desc["devices"]:
            if cc(r["min_compute_capability"]) < cc(desc["devices"]["min_compute_capability"]) and not r.get("cpu_only_ok"):
                fail(n, f"slot {s['slot']} admits GPUs below the backend's min compute capability")

# fallback chain: references resolve, never self, acyclic
for pid, (n, d) in profiles.items():
    seen, cur = {pid}, d
    while cur.get("on_worker_loss", {}).get("then") == "fallback-profile":
        nxt = cur["on_worker_loss"]["fallback_profile_id"]
        if nxt not in profiles: fail(n, f"fallback {nxt} not found in examples"); break
        if nxt in seen: fail(n, "fallback cycle"); break
        seen.add(nxt); cur = profiles[nxt][1]

# ---- aliases -----------------------------------------------------------------------------------------------
for aid, (n, a) in aliases.items():
    claim_api_id(n, a["api_model_id"], alias=True)
    cids = [c["profile_id"] for c in a["candidates"]]
    if len(set(cids)) != len(cids): fail(n, "duplicate candidate profile ids")
    for c in a["candidates"]:
        if c["profile_id"] not in profiles: fail(n, f"candidate {c['profile_id']} not found"); continue
        p = profiles[c["profile_id"]][1]
        wslots = {s["slot"] for s in p["topology"]["slots"] if s["kind"] == "worker"}
        w = c.get("when", {})
        for sname in w.get("workers_available", []):
            if sname not in wslots: fail(n, f"{c['profile_id']}: {sname!r} is not a worker slot of that profile")
        if w.get("min_workers_available", 0) > len(wslots): fail(n, f"{c['profile_id']}: min_workers_available > slots")

print("OK" if not failures else f"{failures} failure(s)")
sys.exit(1 if failures else 0)
