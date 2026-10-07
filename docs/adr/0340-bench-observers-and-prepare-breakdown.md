# 0340: Bench observers (dynamic NVML, OS counters) and the PlanReady breakdown

Status: accepted (WP20)

## Context

Several qualification entries listed tool gaps that need no target hardware to implement: per-cycle RSS/commit and VRAM
series, Father NIC byte counters, GPU clock/power/thermal telemetry, the Windows power plan, a Father-read vs. transfer
vs. Node-hash split of provisioning, inter-token gaps, a cache/temp census, and for the expert-domain prototype a LAN
peer mode and the Strata IQ kernels. What cannot be verified here is the behaviour on the target machines.

## Decision

- **NVML is loaded at run time** through a small `std::function` table (`libnvidia-ml.so.1`, `nvml.dll`). No link-time
  dependency, no CUDA toolkit; absent NVML yields `unavailable: <reason>` (never zeros). The table is injectable, so the
  available path is tested with a fake NVML and the absent path against a nonexistent library.
- **Observers read OS counters only** and live in `bench/src/{system_probe,nvml_probe,storage_census}.cpp`; Win32 code
  (`GetProcessMemoryInfo`, `GetIfTable2`/`GetBestInterfaceEx`, `PowerGetActiveScheme`) sits behind `_WIN32` and is
  compile-checked by MinGW; text parsers (`/proc/...`) are exposed and tested with captured text. On Linux commit
  charge is approximated by VmData and is labelled as such.
- **`PlanReady` gains three trailing u64 durations** (`chunk_write_ns`, `seal_hash_ns`, `build_ns`) as an optional
  extension: decoders accept the old body, partial extensions are malformed, the privacy shape registry is updated.
  An old decoder rejects the longer body, which is acceptable because Father and Node ship together. Father measures
  its own source read (`stream_object` minus callback), digest and send time without a protocol change.
- **The census is snapshot-based** (names, sizes, mtimes; contents never read). It cannot replace ETW/procmon, which stays
  a documented gap on HQ-STORE-01.
- **Expert-domain peer mode** derives the same fixture and ownership in every process instead of adding a provisioning
  protocol to an experimental prototype. **`--expert-kernel iq3_s|iq2_xs`** uses the bench's `strata-cpu` provider on
  synthetic blobs: it measures cost, not accuracy, skips the FP32 reference comparison and says so, and fails with
  `kHardwareUnavailable` rather than falling back to FP32 in a build without the kernels.

## Consequences

Every closed gap has code, tests and a registry entry that names the real flags; what only hardware can answer (real
NVML values, a real NIC, Windows plan names, the IQ kernels at Flash-Next scale on the target CPUs) stays pending and
the results stay Synthetic wherever the model, cluster or link is simulated.
