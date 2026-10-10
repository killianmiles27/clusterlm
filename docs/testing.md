# Testing guide and test matrix

Owner: workstream G. This page says what each kind of test proves, how to run it, and which rows of the
[spec test matrix](spec/04-test-matrix.md) have coverage. **Real versus mocked is stated for every row.** The
requirement-level state lives in [status.md](status.md); this page does not duplicate it.

## Kinds of evidence

| Kind | Meaning | Examples |
|---|---|---|
| Real, CI | Ran on a GitHub-hosted runner against real code, real processes or real sockets (loopback) | unit tests, multi-process cluster tests, fuzz-corpus replay, MSI install on a Windows runner |
| Reference/fixture | Real code driving the deterministic reference backend or a generated fixture model; proves protocol and ownership logic, **not** model quality or GPU behaviour | `integration_*`, fault suite |
| Mocked | The OS or hardware boundary is replaced (WTS sessions, power events, NVML) | Windows session/power tests on Linux |
| Pending hardware | Test or procedure written, never run on the target machines | everything in [HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md) |

Synthetic numbers are never shown as benchmarks ([provenance.md](provenance.md)).

## Running the suites locally

| CI job | Local equivalent |
|---|---|
| `linux-gcc-release` | `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ && cmake --build build && ctest --test-dir build --output-on-failure` (add `-LE slow` to skip slow tests; the tokenizer vocab tests need `python3 scripts/fetch_upstream.py llama.cpp` first) |
| `linux-clang-sanitizers` | configure with `-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer"` and the same linker flag |
| `linux-clang-tidy` | `scripts/run_clang_tidy.sh build-tidy` after a build with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` |
| `linux-llama-cpu` | `python3 scripts/fetch_upstream.py llama.cpp`; configure with `-DCLUSTERLM_ENABLE_LLAMA=ON` |
| `linux-strata-cuda-build` | needs CUDA 12.x toolchain; compiles only, no GPU |
| `windows-msvc`, `windows-packaging` | Visual Studio developer prompt; see [packaging.md](packaging.md) |
| `windows-mingw-syntax` | `scripts/check_windows_compile.sh` |
| `qualification-doc` | `python3 scripts/gen_qualification_md.py --check` |
| `contracts-and-docs` | `pip install jsonschema` then `python3 -I docs/interfaces/validate_examples.py`, `python3 -I tests/conformance/check_docs.py`, `python3 -I tests/conformance/check_offline.py`, `python3 -I -m unittest discover -s tests/conformance -p "selftest_*.py"` |

## Spec test matrix coverage

States: **Covered** (real, CI, at the level stated), **Partial**, **Planned** (owner workstream named), **Pending HW**.

| Spec 04 item | Today | Where | Next |
|---|---|---|---|
| Single machine: import a generic llama.cpp model, select in GUI, generate on the Host | Partial. Host-only generation through `llama-local` is tested on a tiny random-weight GGUF (CI `linux-llama-cpu`). Import, library and GUI selection do not exist yet | `tests/backends_llama/` | A (library), E (GUI), G adds end-to-end once both land |
| Custom distributed profile: pair, assign, provision, infer, clean up | Partial. Pairing, provisioning and cleanup are covered with the reference backend and fixture model across real processes | `tests/integration`, `tests/pairing`, `tests/lease_store` | A/C (profiles), then G adds the profile-driven version |
| Topology logic: Host only, +1, +2 Workers, busy Worker, replacement, insufficient memory, missing model, unsupported backend, conflicts, context growth re-plan | Partial. Busy/replan/recosting exist for the fixed 3-machine shape; N=0,1,2 through one code path is [CMR-0007](cross-module-requests.md) | `tests/placement`, `tests/planning` | A, C |
| API clients (Pi, OpenCode, streaming, tools, cancel, errors, long context, SDK conformance) | Planned. No API exists. G owns `tests/clients/` and the conformance suite; the contract is [scheduler-admission-v1](interfaces/scheduler-admission-v1.md) | | B, then G |
| MCP discovery, schemas, permissions, read-only, authorization refusal | Planned | | D, then G |
| Failures: preserve and extend the fault-injection suite | Covered for the existing runtime (census 0 B after every scenario). Extensions follow the scheduler | `tests/integration`, `tests/bench` (faults) | C |
| Windows: GUI, service, process supervision, pairing, installers on Windows CI | Covered at install level: both MSIs build, validate (ICE), install, verify and uninstall on a Windows runner. Interactive GUI, real lock/suspend and LAN pairing are Pending HW (HQ-UI-01, HQ-WIN-*, HQ-PAIR-01) | `packaging/smoke-test.ps1`, `tests/windows` | F/E |
| GPU qualification (never synthetic) | Pending HW. No result exists | `bench/qualification` | Hardware |

## Cross-cutting guards owned by G (`tests/conformance/`)

* `check_docs.py`: every relative Markdown link resolves; no license file at the root; no license claim wording outside the
  documents that discuss the licensing question (the script spells out the exact rule).
* `check_offline.py`: no URL literals and no internet-client includes (libcurl, WinHTTP, WinINet) in product sources. This is
  a static guard for guarantee G-12, not a proof. The dynamic check, running the suite with no network namespace beyond
  loopback, is planned (needs a CI job with `unshare -n`; loopback-only tests already dominate the suite).
* `selftest_checks.py`: the scripts above must fail on planted violations.
* `docs/interfaces/validate_examples.py` (workstream 0's, run by G's CI job): examples against the frozen JSON Schemas.

## Security tests that exist

`tests/security` (hardening, peer role binding, replay, TLS policy), `tests/privacy` (no prompt text or token IDs in logs,
diagnostics, error replies or on the wire), `tests/fuzz` (13 targets replayed from corpora in every CI run). API-facing
security tests (scope enforcement, an API key can never reach Worker control, rate/size limits, CORS, redacted logs) are
specified by [auth-scopes-v1](interfaces/auth-scopes-v1.md) and become writable when B's server exists; G writes them then.
