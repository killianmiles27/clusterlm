## `01-product-vision.md`

**ClusterLM combines compatible compute resources across multiple computers into a configurable local model-serving platform.**

User flow: install on each computer → choose one Host → pair others as Workers → import supported models to the Host library → create execution profiles → define which model runs when which Workers are available → let ClusterLM measure and optimize placement → chat in the built-in UI or connect external clients via standard APIs → optionally manage via MCP → keep privacy, predictability and automatic resource release.

Useful for one, two or several heterogeneous computers. Support zero Workers as a normal deployment. Keep extensible toward more Workers, with **explicit validated per-backend limits**. The original 4060 Ti / 4070 Laptop / 3060 three-PC setup and its Swift/Flash-Next quantizations become an **importable example profile set**, not product logic. The Host might be a laptop; the same software should work with a 3090, 4090 or other supported NVIDIA GPU.

Naming: product **ClusterLM**, repo `clusterlm`, description *Distributed local LLM inference and model serving.* UI says **Host** and **Worker**; internal Father/Node terms stay (display-name layer only, no mass rename). Use plain terms: Models, Profiles, Workers, Runtime, API, Benchmarks, Diagnostics. No invented futuristic names. No unverified throughput claims.

