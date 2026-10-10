---
name: Model / backend compatibility report
about: Report that a model or backend works, fails, or behaves differently than the matrix says
labels: compatibility
---

**Model** (exact file name, SHA-256, quantization, size):

**Backend and version** (llama-local, strata-hybrid, ...):

**Topology** (Host only, Host + N Workers; CPU/GPU/RAM of each):

**Result:** works / fails / wrong output. Paste the error code and message. **Do not paste prompts you consider private.**

**How measured:** (command and tool; throughput numbers need `measured` provenance, see docs/provenance.md)

The maintainers assign the compatibility label (`Supported and qualified`, `Supported, awaiting hardware qualification`,
`Experimental`, `Unsupported`); a single report is evidence, not a qualification.
