#!/bin/bash
#SBATCH --job-name=assign_docids
#SBATCH --gpus=1
#SBATCH -t 2:00:00
# Compute the DocID of EVERY document (seen and unseen) with the trained DocID encoder.
# The result, docids.npz, is the fixed DocID table used by the retriever and by evaluation.
#
#   sbatch scripts/docid_encoder/assign_docids.sh <dataset> <setting>
_c="$(dirname "${BASH_SOURCE[0]}")/../common.sh"; [ -f "$_c" ] || _c="${SLURM_SUBMIT_DIR:-.}/scripts/common.sh"
source "$_c"     # under sbatch the script runs from a spool copy, so fall back to the submit dir

[ -f "$ENCODER" ] || { echo "FATAL: no DocID encoder at $ENCODER" >&2; exit 1; }
[ -e "$DOCIDS" ] && { echo "FATAL: refusing to overwrite $DOCIDS" >&2; exit 1; }
"$PYTHON" "$SRC/docid_encoder/assign_docids.py" --ckpt "$ENCODER" --tag "$RUN" --dataset "$DATASET" \
    --query_split "$ENC_QSPLIT" --batch_size "$ENC_BS" --max_seq_length "$MAXLEN" --out "$DOCIDS" \
    2>&1 | tee "$OUT/logs/assign_docids_$(date +%Y%m%d_%H%M%S).log"
