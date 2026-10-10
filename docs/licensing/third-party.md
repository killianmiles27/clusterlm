# Third-party components and provenance

What the repository contains or fetches, where it came from, and under which license **as recorded by its own files**. This is
an inventory for the owner's license decision ([license-comparison.md](license-comparison.md)), not legal advice. The
installer-facing notices are in [packaging/licenses/THIRD-PARTY-NOTICES.txt](../../packaging/licenses/THIRD-PARTY-NOTICES.txt).

| Component | Version / pin | License (per upstream file) | How it enters | In shipped binaries? |
|---|---|---|---|---|
| nlohmann/json | vendored, `third_party/nlohmann/` (`LICENSE.MIT`) | MIT | vendored header | Yes |
| doctest | vendored, `third_party/doctest/` | MIT (`LICENSE.txt`) | vendored header | No (tests only) |
| Dear ImGui | v1.91.9, vendored unmodified, `third_party/imgui/` ([README-clusterlm.txt](../../third_party/imgui/README-clusterlm.txt)) | MIT (`LICENSE.txt`) | vendored sources | Yes (UI). Its embedded default font ships inside the library; confirm font terms in imgui's own docs before a release |
| Strata | commit `1735d647…` ([upstream.json](../../third_party/upstream.json)) | MIT | fetched at build time, never committed; ClusterLM patches in `third_party/patches/strata` | Only in a Strata-enabled build; current installers do not enable it |
| llama.cpp (+ ggml) | commit `6753a033…` (Host backend) and `3cf03257…` (ggml for Strata) | MIT | fetched at build time, never committed | Only in builds with `CLUSTERLM_ENABLE_LLAMA`/`STRATA`; current installers do not enable them |
| OpenSSL 3.x | vcpkg port `openssl:x64-windows-static` | Apache-2.0 | linked statically on Windows; system package on Linux | Yes (Windows installers) |
| Microsoft Visual C++ runtime | static (`/MT`) | Microsoft redistribution terms | linked statically | Yes |
| NVIDIA CUDA runtime / cuBLAS | not included | NVIDIA CUDA Toolkit EULA | would be bundled by a future Strata build | No |
| WiX Toolset 5.0.2, vcpkg, CMake, Ninja | build tools | various | CI only | No |

Conventions:

* Upstream sources are **pinned by commit** and fetched by `scripts/fetch_upstream.py`; they are never copied into the repository
  (except the small vendored libraries above). Patches to Strata are a reviewed series with ADR 0204.
* Model weights are not part of the repository or installers. Each model has its own license; the Host's model library (workstream A)
  records the artifact hash, not a license judgement.
* When a dependency is added or its pin changes: update this table, `third_party/upstream.json` if relevant, and
  `packaging/licenses/THIRD-PARTY-NOTICES.txt` in the same change.
