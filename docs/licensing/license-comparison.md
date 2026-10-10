# License comparison (MIT vs Apache-2.0) — no license has been chosen

**Status: the project has no license.** Choosing one is the owner's decision. This document informs that decision; it does
not make it, and it is not legal advice. Until a license file is committed by the owner, the code is "all rights reserved" by
default, nobody else may lawfully reuse it, and the project must not be described as open source. CI
(`tests/conformance/check_docs.py`) fails if a `LICENSE`/`COPYING` file appears or if user-facing docs claim a license.

## The two candidates

| | MIT | Apache-2.0 |
|---|---|---|
| Length / ceremony | One short paragraph; copyright + permission notice must accompany copies | Longer; license text must accompany copies, a `NOTICE` file's contents must be preserved, modified files should state their changes |
| Copyright and permission | Use, copy, modify, merge, publish, distribute, sublicense, sell | Same, plus explicit terms |
| Patent license | Not stated (often argued to be implied) | **Express** patent grant from each contributor, which terminates for a licensee who sues over the work for patent infringement |
| Trademark | Silent | Explicitly does not grant trademark rights |
| Contributions | No inbound terms (inbound = outbound by convention) | Section 5: contributions are under the license unless stated otherwise |
| Warranty/liability | Disclaimed | Disclaimed |
| Compatibility with GPLv2-only | Compatible | **Not** compatible (compatible with GPLv3) |
| Compatibility with GPLv3 / LGPLv3 | Compatible | Compatible |
| Typical in this ecosystem | llama.cpp, ggml, Strata, nlohmann/json, Dear ImGui are MIT | OpenSSL 3 is Apache-2.0 |
| Cost of compliance for downstream binary distributors | Include the notice | Include the license, preserve `NOTICE`, mark changes |

Practical differences for this project:

* ClusterLM statically links **OpenSSL 3 (Apache-2.0)** into the shipped executables. Distributing that binary already
  requires carrying the Apache-2.0 text for OpenSSL ([packaging/licenses](../../packaging/licenses/THIRD-PARTY-NOTICES.txt)),
  whichever license ClusterLM itself uses. Choosing Apache-2.0 for ClusterLM would make the obligations uniform; choosing MIT
  keeps ClusterLM's own terms minimal while OpenSSL's remain separate.
* Contributors who might hold patents (companies, hardware vendors) are often more comfortable with the explicit patent
  grant of Apache-2.0; individual hobby contributors often prefer MIT's brevity.
* Either choice allows later relicensing only with every contributor's consent (or a contributor agreement), so the choice is
  easier to make before outside contributions arrive.
* Neither license says anything about **model weights** a user loads; those carry their own licenses and are never part of
  this repository or the installers.

## Dependency compatibility

See [third-party.md](third-party.md) for the inventory. Summary: every dependency compiled into the shipped product is MIT or
Apache-2.0 (or a platform runtime with its own redistribution terms); **no GPL/LGPL/AGPL code is linked**, so both candidate
licenses are compatible with all current dependencies. A future CUDA runtime redistribution would add NVIDIA's EULA terms to
the *binary* package regardless of choice. This is an inventory made by reading each project's license file name and
metadata at the pinned versions; it should be re-checked by the owner (or counsel) before a release.

## What the owner needs to decide and do

1. Choose MIT, Apache-2.0, or something else (or decide not to publish).
2. Commit the license file and set the copyright holder line(s).
3. Decide on a contribution policy (inbound = outbound, DCO sign-off, or a CLA).
4. Only then may README/release text describe the licensing; update `check_docs.py` policy in the same change.

Nothing in the repository currently depends on the answer.
