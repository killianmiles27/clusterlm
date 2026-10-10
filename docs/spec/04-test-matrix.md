## `04-test-matrix.md`

- **Single machine:** import a generic llama.cpp-supported model, select via GUI, generate on the Host.
- **Custom distributed profile:** pair a compatible Worker, assign, provision selected pieces, infer, clean up.
- **Topology logic:** Host only; +1 Worker; +2 Workers; a Worker goes busy; a different eligible Worker replaces it where supported; insufficient memory; missing model; unsupported backend; profile conflicts; context growth forcing re-plan.
- **API clients:** actual Pi and OpenCode configs where executable; streaming, normal chat, tool calls, tool results, cancellation, errors, long context; SDK conformance.
- **MCP:** discovery, schemas, permissions, read-only access, authorization refusal.
- **Failures:** preserve and extend the existing fault-injection suite.
- **Windows:** GUI, service behavior, process supervision, pairing, installers on Windows CI where possible.
- **GPU qualification:** never substitute synthetic results; mark pending if no hardware.

