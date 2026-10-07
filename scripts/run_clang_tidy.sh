#!/usr/bin/env bash
# Run clang-tidy (policy: .clang-tidy) over the product translation units of a compile database.
# Used by CI (job linux-clang-tidy) and by developers. Tests, third_party and build trees are out of scope.
#
#   CC=clang CXX=clang++ cmake -S . -B build-tidy -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build-tidy        # generated sources / headers must exist before analysis
#   scripts/run_clang_tidy.sh [build-dir] [extra run-clang-tidy args...]
#
# Exit status is non-zero on any finding (WarningsAsErrors is '*').
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${1:-build-tidy}"
[ $# -gt 0 ] && shift
case "$build" in /*) ;; *) build="$root/$build" ;; esac
db="$build/compile_commands.json"
[ -f "$db" ] || { echo "run_clang_tidy.sh: $db not found (configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)" >&2; exit 2; }

runner="${RUN_CLANG_TIDY:-run-clang-tidy}"
command -v "$runner" >/dev/null || { echo "run_clang_tidy.sh: $runner not found" >&2; exit 2; }

# Product translation units: every entry of the database under the repository except third_party/, tests/, build*/.
mapfile -t files < <(python3 -I - "$db" "$root" <<'PY'
import json, os, sys
db, root = sys.argv[1], os.path.realpath(sys.argv[2])
seen = set()
for e in json.load(open(db)):
    f = os.path.realpath(os.path.join(e["directory"], e["file"]))
    rel = os.path.relpath(f, root)
    top = rel.split(os.sep)[0]
    if rel.startswith("..") or top in ("third_party", "tests") or top.startswith("build"):
        continue
    if f not in seen:
        seen.add(f)
        print(f)
PY
)
[ "${#files[@]}" -gt 0 ] || { echo "run_clang_tidy.sh: no product translation units selected" >&2; exit 2; }
echo "clang-tidy: ${#files[@]} translation units"

# run-clang-tidy takes the files as regexes; anchor them and escape regex metacharacters.
regexes=()
for f in "${files[@]}"; do regexes+=("^$(printf '%s' "$f" | sed 's/[][\.^$*+?(){}|]/\\&/g')\$"); done
exec "$runner" -p "$build" -j "${CLANG_TIDY_JOBS:-$(nproc)}" -quiet "$@" "${regexes[@]}"
