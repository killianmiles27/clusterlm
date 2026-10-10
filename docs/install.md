# Installing ClusterLM

**Status:** pre-hardware. The installers have been built, validated and install/uninstall-tested on a Windows CI runner. They
have not been installed on the target machines (`HQ-INSTALL-01`), and no inference has run on a GPU. No signed release exists:
CI builds are named `-UNSIGNED` and Windows SmartScreen will warn about them. The project has no license yet
([licensing](licensing/license-comparison.md)), so there is no public download.

You need two roles. The **Host** (internally *Father*) is the machine you chat from and where models permanently live. A
**Worker** (internally *Node*) is any other Windows PC you let ClusterLM borrow while you are not using it. A single-machine
setup needs only the Host. Terminology note: today's apps still say Father and Node; the UI will move to Host and Worker
(ledger R-50).

## Requirements

* Windows 10/11 x64. Nothing else: no Python, Git, compiler, CUDA toolkit or Visual C++ Redistributable, and nothing is downloaded
  during install or use.
* Host and Workers on the same wired LAN, reachable by IP. Wi-Fi works but is not what the design targets.
* Enough disk on the Host for the models you use. Workers need only temporary space that is deleted after every session.
* Model files (GGUF) that you obtained yourself. ClusterLM never downloads models.

## Install

Run in an elevated PowerShell. Replace the file names with the MSIs you built or were given.

```powershell
# Host (Father): per-user agent, command-line tools and UI
msiexec /i ClusterLM-Father-<version>-x64.msi /qn /l*v father-install.log

# Each Worker (Node): Windows service, firewall rule scoped to the local subnet, logon helper
msiexec /i ClusterLM-Node-<version>-x64.msi /qn /l*v node-install.log
# optional: NODE_PORT=47601   NODE_REMOTE=192.168.1.10   (only accept this Host's address)
```

What each package installs, accounts, firewall scope, upgrade and uninstall behaviour: [packaging.md](packaging.md). Short
version: the Worker service runs as `LocalService`, the firewall rule never opens the Public profile, a Worker keeps no model data
after a session, and uninstall removes everything except the pairing identity and logs (add `CLUSTERLM_PURGE=1` to remove those too).
The Host's model store under `%LOCALAPPDATA%\ClusterLM` is never touched by uninstall.

## Verify

* Worker: `Get-Service ClusterLMNode` shows `Running`; the Worker window or tray icon reports its state.
* Host: `clusterlm-father --help` works from a new terminal (the installer adds its folder to PATH).

## Pair a Worker with the Host

1. On the Worker, choose **Pair with a Father...** (or `clusterlm-node-service --pair`). It shows a code, its address and a short
   fingerprint.
2. On the Host, enter the Worker's address and the code. Compare the fingerprint the Host shows with the Worker's. If they differ,
   cancel: someone else saw the code.
3. Repeat for each Worker. Details and what pairing does and does not protect against: [pairing.md](pairing.md).

## Uninstall

```powershell
msiexec /x ClusterLM-Node-<version>-x64.msi /qn [CLUSTERLM_PURGE=1]
msiexec /x ClusterLM-Father-<version>-x64.msi /qn
```

## Building from source

See the Build section of the [README](../README.md) and [testing.md](testing.md).

## Not available yet

Importing arbitrary models, creating your own profiles, and the OpenAI-compatible API are planned
([plan.md](plan.md), workstreams A, B) and do not exist in the code today. See the [walkthrough](walkthrough.md) for what an
ordinary user can do now and what is coming.
