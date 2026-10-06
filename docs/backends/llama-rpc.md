# llama.cpp RPC (comparison arm)

Pin: `https://github.com/ggml-org/llama.cpp` @ `6753a033f058fbf778d282556ed9b16c78de7c71` (MIT, HEAD on
2026-10-06, commit date 2026-10-06). Paths are relative to `third_party/upstream/llama.cpp/`.
This is read-only analysis; nothing was built or run (no GPU, and the arm is only a comparison baseline).

## Security status (per `tools/rpc/README.md:3-5`)

"Currently in a proof-of-concept development stage ... the functionality is fragile and insecure. **Never run
the RPC server on an open network or in a sensitive environment!**" The server binds `127.0.0.1` by default
(`tools/rpc/rpc-server.cpp:95`) and prints a warning banner for any other host (`rpc-server.cpp:224-231`).
There is no authentication, no encryption and no per-client authorization in the protocol
(`ggml/src/ggml-rpc/ggml-rpc.cpp`, `rpc_serve_client` `:1822`; the only handshake is `RPC_CMD_HELLO`
version/capability exchange, `:1829-1845`). The server serves one client connection at a time (accept loop `ggml-rpc.cpp:2145-2153`; `listen(sockfd, 1)`,
`ggml/src/ggml-rpc/transport.cpp:736`). Protocol version at the pin: `RPC_PROTO_MAJOR/MINOR/PATCH = 7.0.0`
(`ggml/include/ggml-rpc.h:9-11`). Implication for ClusterLM: the arm may only be run on an isolated benchmark
network and must never share a LAN segment with real model data or the ClusterLM control plane; it offers no
isolation, lease or deletion semantics.

## Where the tensor cache is enabled / disabled

* **Server flag:** `-c` / `--cache` ("enable local file cache"), `rpc-server.cpp:110`, parsed at `:156-157`
  into `rpc_server_params::use_cache`, default `false` (`:97`). Off by default.
* **Cache directory:** `fs_get_cache_directory() / "rpc"` (`rpc-server.cpp:243`); `fs_get_cache_directory`
  (`:64-82`) honours `LLAMA_CACHE`, else `%LOCALAPPDATA%` (Windows), `~/Library/Caches` (macOS),
  `$XDG_CACHE_HOME` or `~/.cache` + `/llama.cpp`. The directory is created at start (`:244-251`). With the flag
  off, `cache_dir` is `nullptr`, the server prints `local cache : n/a` (`ggml-rpc.cpp:2102`) and both cache
  paths below are inert (`ggml-rpc.cpp:1471`, `:1486`).
* **Client side** (always compiled in, no flag): a weight tensor larger than `HASH_THRESHOLD = 10 MiB`
  (`ggml-rpc.cpp:87`) whose buffer usage is `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` goes through
  `RPC_CMD_SET_TENSOR_HASH` first (`rpc_use_hash_cache`, `:716-718`; sites `:724-739` and `:958-975`). Only
  weights are eligible, deliberately not activations (comment `:711-715`). If the server answers "have it",
  no data is sent; otherwise the client resends with `cache_flag = 1`, which makes the server persist the
  tensor (`ggml-rpc.cpp:1471-1480`).
* **Cache key and storage:** FNV-1a 64-bit hash of the tensor bytes (`fnv_hash`, `:248`), file name = 16 hex
  digits under the cache dir, raw tensor bytes, written with a plain `std::ofstream` (`:1472-1479`) and read by
  `get_cached_file` (`:1485-1500`). This is a non-cryptographic 64-bit key: no integrity check on read, no
  eviction, no deletion on disconnect, plaintext model weights persist on the node's disk.

ClusterLM requirement (ephemeral lease store, delete-on-release): the arm must run **without `-c`**, and any
benchmark that does enable it must treat `<cache dir>/rpc` as model data to be wiped after the run. The
per-buffer allocation-size cache at `ggml-rpc.cpp:892-893` (client-side hash of a key + endpoint) is
metadata only.

## What crosses the wire

Commands (`ggml-rpc.cpp:62-81`): `ALLOC_BUFFER, GET_ALIGNMENT, GET_MAX_SIZE, BUFFER_GET_BASE, FREE_BUFFER,
BUFFER_CLEAR, SET_TENSOR, SET_TENSOR_HASH, GET_TENSOR, COPY_TENSOR, GRAPH_COMPUTE, GET_DEVICE_MEMORY,
INIT_TENSOR, GET_ALLOC_SIZE, HELLO (=14, pinned by static_assert at :84), DEVICE_COUNT, GRAPH_RECOMPUTE,
MEMSET_TENSOR`.

* **Weights** are pushed by the client (the machine that has the GGUF) with `SET_TENSOR`: serialized as
  `rpc_tensor | cache_flag (1 B) | offset (8 B) | data` (`serialize_set_tensor`, `:701-711`). So the full
  model weights for the remote device cross the network in the clear at load time, unless the server's
  cache already holds them.
* **Compute** is driven by `GRAPH_COMPUTE`/`GRAPH_RECOMPUTE` (`:1050-1056`): the serialized ggml graph
  (tensor descriptors and op parameters) goes over the wire; inputs/outputs move via `SET_TENSOR`/`GET_TENSOR`
  and `COPY_TENSOR` between backends. Activations between split points therefore cross as raw ggml tensors,
  with layout and precision dictated by ggml (not ClusterLM's boundary ABI), and (inference from the design, not checked in code) **token IDs and logits can
  appear in graph inputs/outputs** since the split is driven from the client's scheduler. That breaks
  ClusterLM's "token IDs never leave Father" rule, so the arm is a throughput comparison only, not a product
  path.
* **Server-side bounds check** exists for `SET_TENSOR` (tensor region must lie in the buffer,
  `ggml-rpc.cpp:1458-1468`) but there is no overall hardening claim; treat all inputs as trusted-peer only.
* **Transport:** TCP, or RDMA negotiated in the handshake and used only if both peers support it
  (`README.md`, "RDMA transport"; `transport.cpp`, `transport-apple.cpp`); `GGML_RPC_NO_RDMA=1` forces TCP.
  `GGML_RPC_DEBUG=1` enables verbose logging (`ggml-rpc.cpp:26`) — such logs include tensor metadata and must not
  be collected in ClusterLM runs (privacy rule).

## Adapter implications (`runtime/backends/llama`)

* The arm should drive stock `llama-server`/`ggml-rpc-server` as external processes and measure end-to-end
  throughput; it should not link llama.cpp into the ClusterLM process (the licence is compatible, MIT, but the
  threading/allocation model is not the ExecutionDomain model).
* Run settings to record with every result: pin hash above, RPC protocol 7.0.0, `-c` off, TCP vs RDMA, split
  proportions (`--tensor-split`), and the benchmark network isolation.
* Not verified here: actual throughput, Windows build behaviour of the pinned RPC server, RPC behaviour with MoE
  expert offload flags.
