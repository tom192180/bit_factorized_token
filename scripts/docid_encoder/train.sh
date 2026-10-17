#!/bin/bash
#SBATCH --job-name=docid_encoder
#SBATCH --gpus=2
#SBATCH -t 24:00:00
# Train the DocID encoder M1 on real queries (contrastive loss, in-batch negatives).
#
#   sbatch scripts/docid_encoder/train.sh <dataset> <setting>
#   e.g.   scripts/docid_encoder/train.sh nq320k seen60_unseen40
#   T5:    SBERT=t5-base scripts/docid_encoder/train.sh nq320k seen60_unseen40
#
# Hyperparameters (learning rate, temperature, patience, ...) are fixed in
# src/docid_encoder/config.py. 2 GPUs x batch 512 = 1,024 documents per query: the loss gathers
# documents across GPUs, so the objective depends on that product. To use 4 GPUs set GPUS=4
# BATCH=256.
_c="$(dirname "${BASH_SOURCE[0]}")/../common.sh"; [ -f "$_c" ] || _c="${SLURM_SUBMIT_DIR:-.}/scripts/common.sh"
source "$_c"     # under sbatch the script runs from a spool copy, so fall back to the submit dir

GPUS=${GPUS:-2}
BATCH=${BATCH:-512}
[ $((GPUS * BATCH)) -eq 1024 ] || { echo "FATAL: GPUS x BATCH = $((GPUS * BATCH)), not 1024" >&2; exit 1; }
[ -e "$ENCODER" ] && { echo "FATAL: refusing to overwrite $ENCODER" >&2; exit 1; }
MPORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))
LOG="$OUT/logs/docid_encoder_$(date +%Y%m%d_%H%M%S).log"
echo "=== DocID encoder | $DATASET $SETTING | $SBERT | $BITS bits = $((BITS/NLEV)) tokens x $((2**NLEV)) ==="
echo "    checkpoint -> $ENCODER"

"$TORCHRUN" --master_port="$MPORT" --nproc_per_node="$GPUS" "$SRC/docid_encoder/train.py" \
    --dataset "$DATASET" \
    --sbert_model "$SBERT" \
    --n_levels "$NLEV" \
    --proj_dim "$BITS" \
    --max_seq_length "$MAXLEN" \
    --batch_size "$BATCH" \
    --seed 42 \
    "${SPLIT_ARG[@]}" \
    --save_dir "$OUT" 2>&1 | tee "$LOG"
