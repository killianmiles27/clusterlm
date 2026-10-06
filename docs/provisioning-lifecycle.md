# Ephemeral provisioning lifecycle

**Invariant (addendum §10):** after a Node stops participating, no application-owned model weights, converted packs,
tensor fragments or inference-state files remain on its persistent storage. Father is the only permanent model store.

## Father side

1. **Canonical manifest.** Father opens the model's GGUF shards and builds the `ModelManifest`
   (`runtime/objects/gguf_manifest.hpp`; tool `clusterlm-model-inspect`). The manifest describes every object's exact
   source byte ranges. A routed expert is three ranges inside stacked tensors. Each object carries two digests: the
   source digest and the digest of the provisioned representation. See [model-manifest.md](model-manifest.md).
2. **Plan-scoped manifest.** For each Node, Father includes only the objects of that Node's layers, each with an
   allocation target: CPU-resident, GPU-resident or temporary backing. Embedding, PLE/lookup, head and MTP objects
   never leave Father.
3. **Bounded streaming.** `CanonicalModelStore::stream_object` reads each object's ranges in chunks of at most
   `provision_chunk_bytes`, hashing incrementally. Father never holds a whole object in memory. Transfers to several
   Nodes run concurrently, but they share Father's single egress link: the simulated `SimulatedLink`, or the real
   NIC.
4. **Resume.** If the bulk connection breaks inside a valid lease, Father reconnects. It reads the Node's
   `ProvisionStatus` and resends only the unsealed objects (`CoordinatorConfig::provision_retries`).
5. **Cancel.** `Coordinator::cancel_prepare()` stops the transfer. The partial lease is then released, and the Node
   deletes whatever arrived.

## Node side (`node/lease-store`)

| Step | What happens | Crash safety |
|---|---|---|
| `begin_lease` | A content-free journal record `B <gen>` is written durably, then the lease directory `leases/<gen>/` is created | Recovery finds the record |
| `create_object` | Budgets are checked. RAM objects use 64-byte-aligned process memory; disk objects (temporary backing) are files named `obj-<index>.part`. A journal `F` record is written before the file is created, and the file is preallocated, so a full disk fails here | Store-generated names only; nothing from the manifest becomes a path |
| `write_chunk` | Bounds, overlap and duplicate checks, plus the chunk's SHA-256 | |
| `seal` | Every byte must be present and the whole-object SHA-256 must match; disk objects are remapped read-only | A digest mismatch is `DATA_LOSS` |
| Domain prepare | The backend binds the objects through `ObjectResolver` (process-local spans only), allocates state for its own layers, and runs a synthetic execution check | No fallback to Father or any other weight source |
| Ready | Allocations persist across chat turns while the lease stays valid | |
| `release` | Stop work, release domains, unmap views, close handles, delete the lease tree without following links, run a census of the root (excluding the journal), then write the journal `R` record | `storage_cleaned` is reported only at a census of 0 bytes; otherwise the state is `CleanupPending`, which is retried before any new lease |
| Restart | Orphan recovery deletes every unreleased lease directory, and anything unrecorded under `leases/`, before the Node listens | Tested for crashes at every lifecycle phase |

**Release triggers.** A Node releases its lease when any of the following happens:
- Father sends `ReleaseLease`.
- The user becomes active locally (session helper → service → worker).
- The machine suspends.
- The Node is paused.
- The Father control connection is lost.
- A fault occurs.

**Release deadlines.** The service supervisor (`node/service`) enforces the cooperative release deadline (default
2 s). If the worker misses it, the supervisor terminates the worker's Job Object. The relaunched worker runs orphan
recovery before it can accept another lease.

## Verified here (development host)

- `test_lease_store`: chunk and seal validation, budgets, crashes at 8 phases with recovery, torn journal, and
  symlinks planted in the lease tree.
- `test_transactions`:
  - provisioning resume after a broken stream;
  - cancelled preparation;
  - temporary-file backing that executes and is deleted on release.
- `test_node_protocol`: corrupt chunks, model-mismatch seals, disk-full preallocation, stale leases, and Father
  loss.
- `clusterlm-bench faults`: process-level crashes during transfer, hashing, mapping, allocation, Ready, prefill,
  inference, commit and cleanup; local activity; link loss; a stalled Node; Father process loss; and 20-cycle
  release/reprepare. Every scenario ends with a census of 0 bytes.

## Pending on real hardware

- **Provisioning speed:** HQ-PROV-01.
- **Release latency under real drivers:** HQ-REL-01.
- **Full filesystem census including backend and driver caches:** HQ-STORE-01.
- **Windows ACL and delete-on-close behaviour:** HQ-WIN-04.
