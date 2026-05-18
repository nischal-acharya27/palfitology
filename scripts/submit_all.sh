#!/bin/bash
# One-shot driver: splits the catalog, submits the 8-task array job, then
# submits a held reconcile job that releases when the array finishes.
#
# Usage:  ./submit_all.sh <catalog.csv>

set -euo pipefail

CATALOG="${1:-}"
if [ -z "$CATALOG" ] || [ ! -f "$CATALOG" ]; then
    echo "Usage: $0 <catalog.csv>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Split catalog
"$SCRIPT_DIR/prepare_chunks.sh" "$CATALOG" 8

# 2. Submit the array job. Capture the job id from qsub's stdout, which on SGE
#    reads:  Your job-array 12345.1-8:1 ("palfit_v0.2") has been submitted
ARRAY_OUT=$(qsub "$SCRIPT_DIR/run_palfitology_array.sh")
echo "$ARRAY_OUT"
ARRAY_JOB_ID=$(echo "$ARRAY_OUT" | grep -oE '[0-9]+' | head -1)

if [ -z "$ARRAY_JOB_ID" ]; then
    echo "ERROR: could not parse array job id from qsub output." >&2
    exit 2
fi

# 3. Submit reconcile, held until the entire array finishes.
qsub -hold_jid "$ARRAY_JOB_ID" "$SCRIPT_DIR/merge_and_reconcile.sh"

echo ""
echo "Submitted:"
echo "  array job     -> $ARRAY_JOB_ID (8 tasks, up to 8 nodes concurrent)"
echo "  reconcile job -> held on $ARRAY_JOB_ID"
echo ""
echo "Monitor with:  qstat -u \$USER"
