# 0292 Static OpenSSL and C runtime; unsigned packages say so

## Context

End-user machines have no compiler, Python, Git, CUDA toolkit or Visual C++ Redistributable, and installation and
operation are offline. OpenSSL is needed for SHA-256, TLS and X.509. Code signing needs a certificate that only the
project owner holds.

## Decision

- The installer build links OpenSSL statically (vcpkg `x64-windows-static`) and uses the static MSVC runtime
  (`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`, `/MT`), so no OpenSSL DLL and no redistributable is installed.
  `packaging/check-dependencies.ps1` fails CI if any staged binary imports a non-system DLL. The regular
  `windows-msvc` job keeps the dynamic triplet; only the packaging job changes. Shipping DLLs was rejected because
  nothing needs them and DLL side-by-side placement in a per-machine install is a servicing liability. The OpenSSL
  license text (Apache-2.0) is copied from the vcpkg port into `licenses\`.
- CUDA: not included. A build with `CLUSTERLM_ENABLE_STRATA` may later install the CUDA runtime and cuBLAS DLLs
  (redistributable under the NVIDIA EULA) into the same `bin\` through `install(... COMPONENT father|node)`; the MSI
  picks up everything staged, and `check-dependencies.ps1` then needs an explicit allowlist entry for them.
- `packaging/sign.ps1` signs every EXE, DLL and MSI with `signtool` only when `CLUSTERLM_SIGN_CERT_BASE64`,
  `CLUSTERLM_SIGN_CERT_PASSWORD` and `CLUSTERLM_SIGN_TIMESTAMP_URL` are all present. Otherwise nothing is signed, the log
  and a GitHub annotation say so, the MSI file name contains `-UNSIGNED`, the artifact is named
  `clusterlm-msi-unsigned` and the MSI summary says `Signed: no`. The tooling never creates a self-signed or test
  signature; `check_packaging.py` forbids the relevant commands.

## Consequences

- Windows SmartScreen warns on unsigned packages; that is the truthful state until a certificate is configured.
- Static OpenSSL means OpenSSL security updates require a rebuild and a new package.
