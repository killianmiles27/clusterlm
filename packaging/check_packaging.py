#!/usr/bin/env python3
"""Static consistency checks of the packaging sources against the code and each other (runs on Linux CI).

The MSIs themselves are built and exercised on Windows (CI job windows-packaging); these checks catch the drift that
would only show up there: a custom action calling a flag the service no longer accepts, a smoke test expecting a
service or firewall rule name the code does not use, an install rule missing for a shipped executable, a signing
script that could fabricate a certificate.
"""
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIX_NS = "http://wixtoolset.org/schemas/v4/wxs"
errors = []


def check(ok, message):
    if not ok:
        errors.append(message)


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def tag(name):
    return "{%s}%s" % (WIX_NS, name)


def parse_wxs(name):
    # The preprocessor instructions (<?if?>) are not XML elements for ElementTree; they are skipped as PIs.
    return ET.parse(ROOT / "packaging" / "wix" / name).getroot()


node = parse_wxs("Node.wxs")
father = parse_wxs("Father.wxs")

# --- package identity ---------------------------------------------------------------------------------------------
codes = {}
for name, root in (("Node", node), ("Father", father)):
    pkg = root.find(tag("Package"))
    check(pkg is not None, "%s.wxs has no Package" % name)
    if pkg is None:
        continue
    code = pkg.get("UpgradeCode", "")
    try:
        check(str(uuid.UUID(code)) == code.lower(), "%s UpgradeCode is not a canonical GUID" % name)
    except ValueError:
        check(False, "%s UpgradeCode is not a GUID: %r" % (name, code))
    codes[name] = code.lower()
    check(pkg.get("Scope") == "perMachine", "%s package must be perMachine" % name)
    check(pkg.get("Version") == "$(Version)", "%s Version must come from the build" % name)
check(len(set(codes.values())) == len(codes), "Node and Father must have different UpgradeCodes")

# --- the service flags the custom actions use exist -----------------------------------------------------------------
service_main = read("apps/node-service/main.cpp")
service_flags = set(re.findall(r'args\.(?:has|get|all|integer|number)\("([a-z\-]+)"', service_main))
used_flags = set()
actions = {a.get("Id"): a for a in node.iter(tag("CustomAction"))}
for aid, a in actions.items():
    cmd = a.get("ExeCommand", "")
    check(a.get("Impersonate") == "no" or a.get("Execute") == "immediate", "CA %s must run as SYSTEM (Impersonate=no)" % aid)
    if "clusterlm-node-service.exe" in cmd:
        for flag in re.findall(r"--([a-z\-]+)", cmd):
            used_flags.add(flag)
            check(flag in service_flags, "CA %s uses --%s, which clusterlm-node-service does not accept" % (aid, flag))
for needed in ("install", "uninstall", "cleanup", "purge", "port", "remote"):
    check(needed in used_flags, "no custom action uses --%s" % needed)

# --- every sequenced action exists, and every defined action is sequenced ------------------------------------------------
sequenced = set()
for c in node.iter(tag("Custom")):
    sequenced.add(c.get("Action"))
    check(c.get("Action") in actions, "sequence refers to unknown custom action %s" % c.get("Action"))
    for anchor in ("After", "Before"):
        ref = c.get(anchor)
        if ref and ref not in actions:
            check(ref in {"InstallFiles", "RemoveFiles", "InstallServices", "StopServices"}, "unknown anchor %s" % ref)
check(sequenced == set(actions), "unsequenced custom actions: %s" % (set(actions) - sequenced))
check(any(a.get("Execute") == "rollback" for a in actions.values()), "the install needs a rollback custom action")

# --- names shared with code, smoke test and docs ------------------------------------------------------------------------
smoke = read("packaging/smoke-test.ps1")
docs = read("docs/packaging.md")
svc_name = re.search(r's\.name = "([A-Za-z]+)"', read("runtime/platform/src/service_host.cpp")).group(1)
for text, what in ((smoke, "smoke-test.ps1"), (read("packaging/wix/Node.wxs"), "Node.wxs"), (docs, "docs/packaging.md")):
    check(svc_name in text, "%s does not mention service %s" % (what, svc_name))
rule = re.search(r'kNodeRuleName = "([^"]+)"', read("runtime/platform/include/clusterlm/platform/firewall.hpp")).group(1)
check(rule in smoke, "smoke-test.ps1 does not check firewall rule %r" % rule)
check(rule in docs, "docs/packaging.md does not name firewall rule %r" % rule)
helper_value = re.search(r'run_key_value_name = "([A-Za-z]+)"', read("runtime/platform/include/clusterlm/platform/helper_startup.hpp")).group(1)
node_text = read("packaging/wix/Node.wxs")
check('Name="%s"' % helper_value in node_text, "Node.wxs Run value is not %s" % helper_value)
check(helper_value in smoke, "smoke-test.ps1 does not check the %s Run value" % helper_value)
check("ClusterLMFatherAgent" in read("packaging/wix/Father.wxs") and "ClusterLMFatherAgent" in smoke, "Father agent Run value mismatch")

# --- install rules cover every executable the packages name -----------------------------------------------------------------
install_rules = read("cmake/ClusterLMInstall.cmake")
for text in (node_text, read("packaging/wix/Father.wxs"), smoke):
    for exe in set(re.findall(r"(clusterlm-[a-z\-]+)\.exe", text)):
        check(exe in install_rules, "%s.exe is named by packaging but has no install rule" % exe)
# And every apps/ executable is shipped by one of the components.
for exe in re.findall(r"add_executable\((clusterlm-[a-z\-]+)", read("apps/CMakeLists.txt") + read("bench/CMakeLists.txt")):
    check(exe in install_rules, "%s is built but not installed by any package component" % exe)

# --- preprocessor variables the build script passes cover the ones the sources use ---------------------------------------
build = read("packaging/build-msi.ps1")
passed = set(re.findall(r'-d "([A-Za-z]+)=', build))
for name in ("Node.wxs", "Father.wxs"):
    used = set(re.findall(r"\$\(([A-Za-z]+)\)", read("packaging/wix/" + name)))
    check(used <= passed, "%s uses build variables %s that build-msi.ps1 does not define" % (name, used - passed))

# --- signing never fabricates a certificate --------------------------------------------------------------------------------
for script in ("sign.ps1", "build-msi.ps1", "smoke-test.ps1"):
    text = read("packaging/" + script)
    for bad in ("New-SelfSignedCertificate", "makecert", "/sm /a"):
        check(bad not in text, "%s must never create a certificate (%s)" % (script, bad))
sign = read("packaging/sign.ps1")
for secret in ("CLUSTERLM_SIGN_CERT_BASE64", "CLUSTERLM_SIGN_CERT_PASSWORD", "CLUSTERLM_SIGN_TIMESTAMP_URL"):
    check(secret in sign, "sign.ps1 does not use %s" % secret)
check("UNSIGNED" in build, "unsigned artifacts must say so in their file name")

# --- CI job -------------------------------------------------------------------------------------------------------------------
ci = read(".github/workflows/ci.yml")
check("windows-packaging:" in ci, "ci.yml has no windows-packaging job")
check(re.search(r"windows-packaging:\s*\n\s*needs:\s*\[?windows-msvc", ci) is not None, "windows-packaging must need windows-msvc")
for fragment in ("x64-windows-static", "smoke-test.ps1", "check-dependencies.ps1", "upload-artifact", "wix msi validate"):
    check(fragment in ci, "ci.yml packaging job lacks %s" % fragment)

if errors:
    print("packaging consistency check FAILED:")
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print("packaging consistency check passed")
