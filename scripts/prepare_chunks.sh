#!/bin/bash
# Pre-split the catalog into 8 chunks for the SGE array job.
# Run this once before `qsub run_palfitology_array.sh`.
#
# Usage:
#   ./prepare_chunks.sh <catalog.csv>           # 8 chunks
#   ./prepare_chunks.sh <catalog.csv> 12        # 12 chunks
#
# The chunks land in the cwd as cat_part_01.csv ... cat_part_NN.csv. The
# header row from the original catalog is preserved in every chunk.

set -euo pipefail

CATALOG="${1:-}"
N_CHUNKS="${2:-8}"

if [ -z "$CATALOG" ] || [ ! -f "$CATALOG" ]; then
    echo "Usage: $0 <catalog.csv> [n_chunks=8]" >&2
    exit 1
fi

# Strip prior chunk files to avoid a half-stale split if the row count changed.
rm -f cat_part_*.csv

python - "$CATALOG" "$N_CHUNKS" <<'PY'
import math
import sys
from pathlib import Path
import pandas as pd

catalog_path = Path(sys.argv[1])
n_chunks = int(sys.argv[2])

# comment="#" matches palfitology.catalog.load_catalog -- handles raw ADQL
# exports with a "#" SQL preamble.
df = pd.read_csv(catalog_path, comment="#")

n_rows = len(df)
if n_rows == 0:
    print(f"Catalog {catalog_path} is empty after stripping comments.", file=sys.stderr)
    sys.exit(2)

chunk_size = math.ceil(n_rows / n_chunks)
print(f"Splitting {n_rows} rows into {n_chunks} chunks of <= {chunk_size} rows")

for i in range(n_chunks):
    start = i * chunk_size
    end = min(start + chunk_size, n_rows)
    if start >= end:
        # Fewer rows than chunks -- write an empty chunk with just the header
        # so the array task still has a file to point at.
        sub = df.iloc[0:0]
    else:
        sub = df.iloc[start:end]
    out = Path(f"cat_part_{i + 1:02d}.csv")
    sub.to_csv(out, index=False)
    print(f"  {out}: {len(sub)} rows")
PY

echo "Done. Now submit with:  qsub scripts/run_palfitology_array.sh"
