# Release procedure

No release has been made. This is the procedure to follow so the first one is honest. Versions are not assigned yet; the MSI
sample name `0.1.0` in [packaging.md](packaging.md) is a placeholder.

## Preconditions (all must hold; any "no" blocks the release)

1. `main` CI is green for **every** job in `.github/workflows/ci.yml`, on the exact commit to be tagged. Link the run in the release notes.
2. `docs/status.md` is current; every row touched since the last release has evidence.
3. `HARDWARE-QUALIFICATION.md` is regenerated and in sync (`qualification-doc` job). The release notes list how many entries are still pending.
4. `CHANGELOG.md` has a section for the version.
5. `packaging/licenses/THIRD-PARTY-NOTICES.txt` and [docs/licensing/third-party.md](licensing/third-party.md) match the dependencies actually linked.
6. The owner has decided the project license. Until then there is **no** public source or binary release (see [licensing](licensing/license-comparison.md)).

## Steps

1. Create a release branch or tag candidate; set the version in `CMakeLists.txt` and the MSI version inputs.
2. Build the installers with `packaging/build-msi.ps1 -RequireSigning` (release builds must be signed; the pipeline refuses to
   sign with anything but real credentials and never generates a test certificate). An unsigned build is named `-UNSIGNED` and
   must not be published as a release.
3. Run the packaging smoke tests (CI `windows-packaging`) and, when hardware exists, `HQ-INSTALL-01` on a clean machine.
4. Attach `packaging-manifest.json` and `signing-status.json`; state in the notes whether the MSIs are signed.
5. Release notes must include: what changed, **known limitations**, which features are `Supported, awaiting hardware qualification`
   versus `Supported and qualified`, and that performance figures are not qualified unless `HQ-*` evidence is attached.
6. Tag, push the tag, publish. Do not make any statement about the project's license unless the owner has chosen one and committed it.

## After the release

Update `docs/status.md` and `CHANGELOG.md` (move *Unreleased* under the version). Security fixes follow [SECURITY.md](../SECURITY.md).
