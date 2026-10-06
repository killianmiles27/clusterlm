# ClusterLM

Distributed heterogeneous local LLM inference for Windows.

| Component | Role |
|---|---|
| **ClusterLM Father** | User-facing coordinator and primary inference machine; the only permanent model store. |
| **ClusterLM Node** | Idle-machine compute worker. Holds only lease-scoped, plan-assigned model objects. |
| **ClusterLM Runtime** | Heterogeneous distributed inference runtime (execution domains, stage boundary, transport). |
| **ClusterLM Bench** | Hardware, transport and runtime qualification suite. |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [HARDWARE-QUALIFICATION.md](HARDWARE-QUALIFICATION.md).
