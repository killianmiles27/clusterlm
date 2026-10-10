# Ordinary-user walkthrough

From a clean Windows PC to a coding agent talking to your cluster. Each step says whether it works **today**, depends on a
workstream still in progress (**planned**), or needs hardware nobody has run yet (**pending hardware**). Where a step is planned,
the text describes the intended behaviour from the frozen contracts in [docs/interfaces/](interfaces/README.md); it will be
revised against the implementation when that lands (G keeps this page in step with [status.md](status.md)).

## 1. Install the Host and, optionally, Workers — today (installers CI-tested, not hardware-tested)

Follow [install.md](install.md). A one-PC setup needs only the Host.

## 2. Pair your Workers — today

Pairing is a one-time code exchange per Worker with a fingerprint check ([install.md](install.md#pair-a-worker-with-the-host),
[pairing.md](pairing.md)). Workers can be added or unpaired at any time.

## 3. Import a model — planned (workstream A)

Point the Host at a GGUF file or a folder containing one (including split files). The Host records the exact file hash, size,
detected architecture and quantization, and tells you which backends can run it and how (Host only, or spread over Workers),
with one of four honest labels: *Supported and qualified*, *Supported, awaiting hardware qualification*, *Experimental*,
*Unsupported*. Today the only models the product knows are the three in the shipped example catalog, found via their directories;
there is no general import.

## 4. Create a profile — planned (workstream A; UI in E)

A **profile** says which model and quantization to run, on which backend, with how much context, on which machines (the Host
alone, or the Host plus Workers chosen by name or by what they offer), and what to do if a Worker disappears. Start from the
shipped *Fast*, *Strong* and *Ultra* examples or duplicate and edit one. **Dry run** shows where each piece would be placed and
why, labelled with its provenance: a prediction from the cost model is `synthetic` until measured ([provenance.md](provenance.md)).
Today the three profiles exist as hard-wired tiers.

## 5. Prepare and chat — today for the three tiers (fixtures only on CI; real models pending hardware)

Choose a tier in the Host window and chat. If the tier spans Workers, the Host transfers only the needed parts to each (never to
disk unless you configured a temporary file) and releases them when the session ends or the Worker's owner returns. A tier shows
**Ready** only when every machine is ready. Real-model quality and speed are `pending hardware`.

## 6. Connect an external coding agent — planned (workstream B)

The Host will expose an OpenAI-compatible endpoint (`/v1/models`, `/v1/chat/completions` with streaming and tool calls) on
`127.0.0.1` by default. To allow other PCs, you must turn on LAN access and create a per-client key; keys are separate from Worker
pairing and can never control Workers ([auth-scopes-v1](interfaces/auth-scopes-v1.md)).
Pi and OpenCode configuration snippets will be added to `docs/clients/` by workstream B/G once the server exists and a scripted
agentic test passes; whether a real model does tool calling *well* is measured separately and may be `pending hardware`.
An agent addresses a model by its exact API model id; ClusterLM never silently swaps in a different model or quantization
([ADR 0402](adr/0402-routing-versus-substitution.md)).

## 7. Let MCP clients manage the cluster — planned (workstream D)

MCP is for management and inspection (read-only tools first); agents use the serving API for inference.

## 8. Go back to using your Worker PC — today

When you return to a Worker (activity, lock/unlock policy, battery per your settings), the Worker releases its allocation and
deletes what it held; you do not have to do anything. Real-session behaviour is verified by mocks and the installer smoke test,
not yet on real machines (`HQ-WIN-01..04`).
