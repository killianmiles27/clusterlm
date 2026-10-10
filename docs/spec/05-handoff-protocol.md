# PART 4: HANDOFF PROTOCOL (every thread)

Threads in a Project can search each other via past-chat search, but that is a convenience, not a contract. Durable state lives in the repo.

**On start:** read `docs/status.md`, `docs/interfaces/`, the latest relevant `docs/handoff/*.md`, and `docs/cross-module-requests.md`. State in one line what you are about to build.

**On finish (or when context runs long), write `docs/handoff/<workstream>-<YYYY-MM-DD>.md` containing:**
1. What was implemented, with commits and files.
2. What was tested, with real results, and what was only mocked.
3. What is pending hardware qualification.
4. Interface changes made (with ADR links) and requests for other workstreams.
5. Subagent usage and any model substitutions.
6. Known issues and the next concrete step.

Then update `docs/status.md`, commit, push, and end with a short summary for me.

**Context hygiene:** if a thread is getting long, stop, write the handoff, and tell me to start a fresh thread of the same name suffixed `(2)`. Don't carry on with degraded context.

**Conflict rule:** two threads must not edit the same files concurrently. If a thread needs a file owned by another workstream, it files a cross-module request instead.

