#!/bin/sh
# Muhammad Ayaan Naveed | Student ID: 23i-0041 | CS-D | Assignment 2
# Reproducible Linux build + benchmarks. Creates NO binaries inside submission.
# Requires gcc/make/OpenMP/AVX2/FMA and the C libgraphblas-dev package.
# On Debian/Ubuntu: sudo apt-get update && sudo apt-get install build-essential libgraphblas-dev git
# Optional report regeneration: python3 -m pip install reportlab matplotlib pillow
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD=${BUILD:-/tmp/23i-0041-CS-D-build}
DATA=${DATA:-/tmp/23i-0041-CS-D-data}
RESULTS=${RESULTS:-/tmp/23i-0041-CS-D-results}
mkdir -p "$DATA" "$RESULTS"
if [ ! -f "$DATA/web-Google.txt" ]; then
    # Public full-dataset mirror, pinned to the exact inspected commit.
    # Stanford original: https://snap.stanford.edu/data/web-Google.txt.gz
    git clone https://github.com/bhargavikesarkar/parallel-Webgraph-analysis "$DATA/mirror"
    git -C "$DATA/mirror" checkout --detach f7ac0ebb93c25e42ef99da7b8685259f74a7598b
    cp "$DATA/mirror/data/web-Google.txt" "$DATA/web-Google.txt"
fi
# Verify canonical LF text SHA256; do not silently accept a partial dataset.
python3 - "$DATA/web-Google.txt" <<'PY'
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for line in f:h.update(line.rstrip(b'\r\n')+b'\n')
assert h.hexdigest()=='543b17f820d78af6616618f3ece2da836e917f089140d68113045313a3523d45','Wrong dataset'
PY
make -C "$ROOT" BUILD="$BUILD" all
make -C "$ROOT" BUILD="$BUILD" check
python3 "$ROOT/23i-0041-CS-D-TASK0-V1.py" --build "$BUILD" --graph "$DATA/web-Google.txt" --output "$RESULTS"
# See the report for optional report regeneration and local GraphBLAS prefix flags.
echo "Complete. Source folder unchanged; results: $RESULTS; binaries: $BUILD"
