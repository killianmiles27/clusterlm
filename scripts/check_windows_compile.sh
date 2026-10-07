#!/usr/bin/env bash
# Syntax-checks every Windows-reachable .cpp with the MinGW-w64 cross compiler (MSVC proxy) using the project's
# warning flags. Excludes runtime/backends (CUDA/vendor gated) and POSIX-only test helpers.
# There is no MinGW OpenSSL: the host's OpenSSL headers are staged in a temp dir so /usr/include itself never
# enters the include path.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cxx="${MINGW_CXX:-x86_64-w64-mingw32-g++-posix}"
command -v "$cxx" >/dev/null || { echo "missing $cxx (apt-get install g++-mingw-w64-x86-64-posix)" >&2; exit 2; }

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/inc"
cp -r /usr/include/openssl "$stage/inc/openssl"
for d in /usr/include/*-linux-gnu/openssl; do
  if [ -d "$d" ]; then cp -r "$d"/. "$stage/inc/openssl/"; fi
done

# CUDA/vendor-gated backends are not part of the Windows CPU build; vendored Dear ImGui is third-party code (its
# backends are still compiled for real through ui/common/win32, which includes their headers).
# CUDA-gated adapters only; runtime/backends/strata-layout is plain C++ and always built.
# Tests that compile against the fetched Strata/llama.cpp sources are gated the same way.
skip_re='^(runtime/backends/(strata|llama)/|tests/backends_llama/|bench/src/gpu_probe_cuda\.cpp$|tests/backends/test_strata_(cpu_kernels|convert|cuda)\.cpp$|third_party/imgui/)'

flags=(-std=c++20 -fsyntax-only -D_WIN32_WINNT=0x0A00 -DWIN32_LEAN_AND_MEAN -DNOMINMAX
       -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -Werror)
incs=(-isystem "$stage/inc" -isystem "$root/third_party" -isystem "$root/third_party/imgui" -I "$root/apps/common" -I "$root/tests/domain" -I "$root/bench/src" -DCLUSTERLM_SOURCE_DIR="\"$root\"" -DCLUSTERLM_NODE_BINARY="\"clusterlm-node.exe\"" -DCLUSTERLM_LLAMACPP_VOCAB_DIR="\"$root/third_party/upstream/llama.cpp/models\"")
cd "$root"
while IFS= read -r d; do incs+=(-I "$root/$d"); done < <(
  find . -type d -name include -not -path './third_party/*' -not -path './build*' -not -path './.*' \
    -not -path './runtime/backends/strata/*' -not -path './runtime/backends/llama/*' | sed 's|^\./||' | sort)

fail=0
n=0
while IFS= read -r f; do
  if [[ "$f" =~ $skip_re ]]; then continue; fi
  n=$((n + 1))
  # Private headers live beside the sources (src/ and src/<subdir>/).
  d="$(dirname "$f")"
  extra=(-I "$d")
  if [[ "$d" == */src/* ]]; then extra+=(-I "${d%/src/*}/src"); fi
  if ! "$cxx" "${flags[@]}" "${incs[@]}" "${extra[@]}" "$f"; then
    echo "FAIL: $f" >&2
    fail=$((fail + 1))
  fi
done < <(git ls-files '*.cpp' | sort)
echo "checked $n files, $fail failed"
[ "$fail" -eq 0 ]
