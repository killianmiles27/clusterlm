# Packaging: Windows installers

Two per-machine MSI packages, built with WiX Toolset v5 on `windows-latest` CI (job `windows-packaging`). Decisions:
`docs/adr/0290`-`0292`. Nothing here has been installed on real hardware yet: `HQ-INSTALL-01` is the procedure, and the
CI smoke test is the Linux-independent evidence that the MSIs install, configure and uninstall cleanly on a Windows runner.

## What each package installs

| | ClusterLM Father | ClusterLM Node |
|---|---|---|
| Directory | `C:\Program Files\ClusterLM Father` | `C:\Program Files\ClusterLM Node` |
| `bin\` | `clusterlm-father-agent.exe`, `clusterlm-father.exe`, `clusterlm-bench.exe`, `clusterlm-model-inspect.exe`, `clusterlm-father-ui.exe` | `clusterlm-node-service.exe`, `clusterlm-node.exe`, `clusterlm-node-helper.exe`, `clusterlm-node-ui.exe` |
| Other | `catalog\clusterlm-catalog.json` (the shipped tier catalog), `licenses\` | `licenses\` |
| Registered | Run value `ClusterLMFatherAgent` (optional feature), `bin\` on the machine PATH, Start menu shortcut (UI only) | Service `ClusterLMNode`, firewall rule, Run value `ClusterLMNodeHelper` |
| Runtime data | `%LOCALAPPDATA%\ClusterLM\{identity,models,logs}` of each user | `%ProgramData%\ClusterLM\Node\{staging,identity,logs}` |

`clusterlm-model-inspect.exe`, `clusterlm-father-ui.exe` and `clusterlm-node-ui.exe` are installed only when the build
contains those targets (`cmake/ClusterLMInstall.cmake` skips targets that do not exist). The UI workstreams add their
own files with `install(... COMPONENT father|node DESTINATION bin)`; the MSI packages everything staged under the
component prefix, and adds the Father Start menu shortcut when `bin\clusterlm-father-ui.exe` exists at build time.

Everything is in the MSI: no Python, Git, compiler, CUDA toolkit or Visual C++ Redistributable is needed, and nothing is
downloaded at install or run time. OpenSSL and the C runtime are linked statically (ADR 0292); CI fails if a binary
imports a DLL that Windows does not ship. A future Strata-enabled build may add the CUDA runtime and cuBLAS DLLs
(NVIDIA EULA redistributables) to `bin\` through the same install components; no CUDA is included now.

## Accounts, service, firewall

- **Service** `ClusterLMNode` (display name "ClusterLM Node"): `NT AUTHORITY\LocalService`, start type automatic
  (delayed), failure actions restart after 5 s / 30 s / 60 s with a 24 h reset period, preshutdown timeout 15 s,
  unrestricted service SID. These come from `platform::node_service_install_spec()`: the MSI runs
  `clusterlm-node-service.exe --install --port N` (default 47600; `NODE_PORT=` overrides) and never duplicates the values.
  The service is started at the end of the install; a failed start is logged but does not fail the install.
- **Firewall**: rule `ClusterLM Node data (TCP-In)` from `platform::firewall_rule_specs()`: inbound TCP, local port = the
  Node port, scoped to `...\bin\clusterlm-node.exe`, profiles Domain + Private (Public never), remote addresses
  `LocalSubnet` or, with `NODE_REMOTE=<ip>[,<ip>...]`, only those peers, edge traversal off. `clusterlm-node-service
  --print-firewall-specs --port N` shows the exact data. The Father package opens no port (it only dials out).
- **Helper**: `HKLM\Software\Microsoft\Windows\CurrentVersion\Run\ClusterLMNodeHelper` starts
  `clusterlm-node-helper.exe` at every logon (ADR 0132). Without a helper the Node stays Busy (fail closed).
- **Father agent**: `HKLM\...\Run\ClusterLMFatherAgent` starts the per-user agent at logon. Leave it out with
  `msiexec /i ClusterLM-Father-<v>-x64.msi ADDLOCAL=FatherCore`.
- **Node data directories** are created by the service on first start, owner (LocalService) + SYSTEM only, protected
  DACL (ADR 0133). The MSI does not create them, so their owner can only be the service account. Consequence: even an
  elevated Administrator cannot list them without taking ownership (an audited act). The smoke test inspects them as SYSTEM.

## Install, upgrade, uninstall

```powershell
msiexec /i ClusterLM-Node-0.1.0-x64.msi /qn /l*v node-install.log [NODE_PORT=47601] [NODE_REMOTE=192.168.1.10]
msiexec /x ClusterLM-Node-0.1.0-x64.msi /qn /l*v node-uninstall.log [CLUSTERLM_PURGE=1]
```

- **Upgrade**: installing a higher version removes the old product (MSI major upgrade) and installs the new one; the
  service is stopped first, `--uninstall`/`--cleanup` of the old version run, then `--install` of the new one. Pairing
  identity survives. Downgrades are refused.
- **Uninstall always** stops and deletes the service, removes the firewall rules and the helper Run value, runs
  `clusterlm-node-service --cleanup` (lease-store recovery, then deletes `%ProgramData%\ClusterLM\Node\staging`: no
  model fragment survives), and removes the program files.
- **Uninstall keeps** (Node) the pairing identity and logs under `%ProgramData%\ClusterLM\Node`, so reinstall or upgrade
  does not require pairing again; `CLUSTERLM_PURGE=1` removes them too (ADR 0291). (Father) `%LOCALAPPDATA%\ClusterLM` of every
  user, including the model store, is never touched; delete it by hand if wanted:
  `Remove-Item -Recurse "$env:LOCALAPPDATA\ClusterLM"`.
- Both packages can be installed on one machine (different folders, service and agent do not collide).

## Signing

`packaging/sign.ps1` signs every EXE and DLL in the stage, then the MSI, with `signtool` (SHA-256, RFC 3161
timestamp) when all of `CLUSTERLM_SIGN_CERT_BASE64` (base64 .pfx), `CLUSTERLM_SIGN_CERT_PASSWORD` and
`CLUSTERLM_SIGN_TIMESTAMP_URL` are set (repository secrets, the timestamp URL as an Actions variable). Otherwise **nothing is
signed and the build says so**: a warning block in the log, a `::warning::` annotation, `-UNSIGNED` in the MSI file names,
artifact `clusterlm-msi-unsigned`, `Signed: no` in the MSI summary, `"signed": false` in `packaging-manifest.json`. No
self-signed or test certificate is ever generated. `-RequireSigning` (for release builds) turns missing secrets into a failure.

## Building locally

```powershell
# Visual Studio developer prompt; vcpkg install openssl:x64-windows-static; dotnet tool install --global wix --version 5.0.2
wix extension add --global WixToolset.Util.wixext/5.0.2
cmake -S . -B build-pkg -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLUSTERLM_BUILD_TESTS=OFF `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static
cmake --build build-pkg
cmake --install build-pkg --config Release --component father --prefix stage/father
cmake --install build-pkg --config Release --component node   --prefix stage/node
packaging/check-dependencies.ps1 -Stage stage/father, stage/node
packaging/build-msi.ps1 -StageRoot stage -OpenSslLicense $env:VCPKG_ROOT/installed/x64-windows-static/share/openssl/copyright
wix msi validate packaging/out/ClusterLM-Node-0.1.0-x64-UNSIGNED.msi
packaging/smoke-test.ps1 -Package Node -Msi packaging/out/ClusterLM-Node-0.1.0-x64-UNSIGNED.msi
```

CI job `windows-packaging` runs exactly these steps after `windows-msvc` passes, uploads the MSI files (and a manifest with
SHA-256, signing state and staged file list), runs the install/uninstall smoke tests and uploads the `msiexec /l*v` logs
(`clusterlm-msi-logs`) even on failure; start with those logs when a step fails.

## What is verified where

| Property | Linux CI | Windows CI (`windows-packaging`) | Real hardware |
|---|---|---|---|
| `--cleanup` deletes crashed-lease and orphan files, keeps identity unless `--purge` | `test_packaging` | smoke test (orphan planted as SYSTEM) | `HQ-INSTALL-01` |
| WiX sources, service/rule/Run names and flags match the code, install rules cover every executable, no fabricated signature | `packaging_sources_consistent` (`packaging/check_packaging.py`) | - | - |
| MSI builds, ICE validation, no non-system DLL imports | - | build + `wix msi validate` + `check-dependencies.ps1` | - |
| Service config, firewall scope, Run values, staging ACL, clean uninstall | - | `smoke-test.ps1` (hosted runner) | `HQ-INSTALL-01` (G14, 3060, Father) |
| SmartScreen behaviour, Windows Security prompts, reboot persistence | - | - | `HQ-INSTALL-01` |

## Manual procedures

### HQ-INSTALL-01
Install both MSIs on the Father and both Nodes and verify service, firewall rule, helper and agent autostart, ACLs, upgrade and
uninstall cleanliness. The exact commands and pass criteria are the `measurements` and `acceptance` of the registry entry:
see `HARDWARE-QUALIFICATION.md#hq-install-01`. In short, per machine, from an elevated PowerShell in a repository checkout
(the script needs nothing else):

```powershell
.\packaging\smoke-test.ps1 -Package Node   -Msi .\ClusterLM-Node-0.1.0-x64.msi      # G14 and 3060
.\packaging\smoke-test.ps1 -Package Father -Msi .\ClusterLM-Father-0.1.0-x64.msi    # Father
```

Each check prints PASS or FAIL and the script uninstalls at the end. The by-hand parts the script cannot do (reboot persistence,
the SmartScreen and Windows Security prompts, a peer connecting through the rule, Public-profile blocking, upgrade over a
busy Node) are listed in the registry entry.

## Known limits

- The MSI sources were checked for XML and WiX compile errors only partly off Windows (WiX on Linux is unsupported and fails
  on directory names); real validation is the `windows-packaging` job. Expect the first CI runs to need small fixes.
- `NODE_PORT`/`NODE_REMOTE` are applied at install; changing them later means reinstalling or running
  `clusterlm-node-service.exe --install --port N --remote ADDR` elevated.
- The Father catalog is installed to `catalog\clusterlm-catalog.json`; `clusterlm-father-agent` resolves
  `<exe dir>\..\catalog\clusterlm-catalog.json` when `--catalog` is not given (then `<exe dir>\clusterlm-catalog.json`,
  the development layout).
- A `--cleanup` failure does not fail the uninstall (ADR 0291); `HQ-INSTALL-01` checks the result.
