# Platform, model and backend matrices

Labels are exactly: **Supported and qualified**, **Supported, awaiting hardware qualification**, **Experimental**,
**Unsupported**. Today nothing is *Supported and qualified* because no hardware qualification has been run
([HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md)). The target model for this matrix is that labels are computed from backend
descriptors plus evidence ([backend-capability-v1](interfaces/backend-capability-v1.md)); descriptors are not implemented yet,
so this page is maintained by hand and may lag the code. When they disagree, the code and [status.md](status.md) win.

## Platforms

| Platform | Host (Father) | Worker (Node) | Evidence |
|---|---|---|---|
| Windows 10/11 x64 | Supported, awaiting hardware qualification | Supported, awaiting hardware qualification | MSVC build + tests and both MSIs install/uninstall on a Windows CI runner; interactive sessions, lock/suspend, LAN pairing and UI not exercised on real machines |
| Linux x64 (gcc 13 / clang 17) | Experimental (development and CI platform: CLI tools and tests; no installer, no service integration, no UI) | Experimental | Full test suite in CI |
| macOS, ARM64 | Unsupported | Unsupported | Not built, not tested |

## Backends

| Backend | Host-only | Distributed across Workers | Evidence |
|---|---|---|---|
| `reference` (CPU, deterministic) | Experimental (test backend) | Experimental (test backend) | Bitwise-identical logits for every split; fixture models only. Not for real inference |
| `llama-local` (pinned llama.cpp) | CPU: Supported, awaiting hardware qualification; CUDA offload: Experimental (compiled, never run on a GPU) | Unsupported (Host only; the llama.cpp RPC route is an investigation, `HQ-P0A-01`) | Generation equals llama.cpp's own decode on a tiny random-weight GGUF in CI |
| `strata-hybrid` (Flash-Next, MTP, hybrid CPU/GPU experts) | Supported, awaiting hardware qualification | Supported, awaiting hardware qualification, validated up to 2 Workers | CPU kernels verified against ggml-cpu; CUDA engine compiled, **never run on a GPU**; GPU claims pending (`HQ-GPU-*`, `HQ-MTP-*`) |

## Models

ClusterLM does not claim to run every GGUF. A model is supported only through a backend that lists its architecture and
quantization. Today:

| Model family / quantization | Backend | Label | Notes |
|---|---|---|---|
| Tiny generated fixture models | reference | Experimental | tests only |
| Tiny random-weight llama-architecture GGUF (F32/F16) | llama-local | Experimental | CI fixture, not a real model |
| Qwen-family models the example Fast/Strong/Ultra profiles name | llama-local, strata-hybrid | Supported, awaiting hardware qualification | tokenizer covers qwen2/qwen35 pre-tokenizers and ChatML only; no Jinja chat templates yet ([status](status.md) R-09); never run with the real artifacts |
| Any other architecture | any | Unsupported until a backend descriptor lists it | no silent substitution of a different model or quantization |

Tool calling, reasoning and vision support are reported per model only when reliably known; none is claimed today.
