#!/bin/bash
# Build the final retriever M2' = 0.5 * M1 + 0.5 * M2 (weight-space average). CPU only.
#
#   scripts/retriever/merge.sh <dataset> <setting>
_c="$(dirname "${BASH_SOURCE[0]}")/../common.sh"; [ -f "$_c" ] || _c="${SLURM_SUBMIT_DIR:-.}/scripts/common.sh"
source "$_c"     # under sbatch the script runs from a spool copy, so fall back to the submit dir

RETRIEVER=${RETRIEVER:-$(ls -1t "$OUT"/retriever/retriever_*.pt 2>/dev/null | head -1 || true)}
[ -n "$RETRIEVER" ] && [ -f "$RETRIEVER" ] || { echo "FATAL: no retriever checkpoint under $OUT/retriever" >&2; exit 1; }
"$PYTHON" "$SRC/retriever/merge_encoders.py" --stage1 "$ENCODER" --stage2 "$RETRIEVER" --out "$MERGED" --alpha 0.5
