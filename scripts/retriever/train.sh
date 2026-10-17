#!/bin/bash
#SBATCH --job-name=retriever
#SBATCH --gpus=1
#SBATCH -t 24:00:00
# Train the retriever M2: initialised from the DocID encoder, DocIDs fixed, real queries plus
# QG generated queries per seen document (sampled 1:1), InfoNCE over all seen DocIDs.
#
#   sbatch scripts/retriever/train.sh <dataset> <setting>
#   T5:    SBERT=t5-base scripts/retriever/train.sh ...   (the optimizer follows the backbone)
#
# Hyperparameters (learning rates, batch, logit scale, patience, ...) are fixed in
# src/retriever/config.py. Assigns the DocIDs first if docids.npz is not there yet.
_c="$(dirname "${BASH_SOURCE[0]}")/../common.sh"; [ -f "$_c" ] || _c="${SLURM_SUBMIT_DIR:-.}/scripts/common.sh"
source "$_c"     # under sbatch the script runs from a spool copy, so fall back to the submit dir

[ -f "$ENCODER" ] || { echo "FATAL: no DocID encoder at $ENCODER" >&2; exit 1; }
[ -f "$ENCODER.arch.json" ] || { echo "FATAL: no sidecar $ENCODER.arch.json" >&2; exit 1; }
if [ ! -f "$DOCIDS" ]; then
  echo "=== assigning DocIDs ($DATASET) ==="
  "$PYTHON" "$SRC/docid_encoder/assign_docids.py" --ckpt "$ENCODER" --tag "$RUN" --dataset "$DATASET" \
      --query_split "$ENC_QSPLIT" --batch_size "$ENC_BS" --max_seq_length "$MAXLEN" --out "$DOCIDS"
fi
mkdir -p "$OUT/retriever"
LOG="$OUT/logs/retriever_$(date +%Y%m%d_%H%M%S).log"
echo "=== retriever | $DATASET $SETTING | real + $QG generated queries per document ==="
cd "$OUT/retriever"
"$PYTHON" "$SRC/retriever/train.py" \
    --dataset "$DATASET" \
    --ckpt "$ENCODER" \
    --enc "$DOCIDS" \
    "${SPLIT_ARG[@]}" \
    --qg_per_doc "$QG" \
    --max_seq_length "$MAXLEN" \
    --seed 42 \
    --save_dir "$OUT/retriever" 2>&1 | tee "$LOG"
