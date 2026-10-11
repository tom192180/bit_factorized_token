#!/bin/bash
#SBATCH --job-name=docid_encoder
#SBATCH --gpus=2
#SBATCH -t 24:00:00
# Train the DocID encoder M1 on real queries (contrastive loss, in-batch negatives).
#
#   sbatch scripts/docid_encoder/train.sh <dataset> <setting>
#   e.g.   scripts/docid_encoder/train.sh nq320k seen60_unseen40
#   T5:    SBERT=t5-base OPT=adafactor LR=1e-3 scripts/docid_encoder/train.sh nq320k seen60_unseen40
#
# 2 GPUs x batch 512 = 1,024 documents per query: the loss gathers documents across GPUs, so
# the objective depends on that product. To use 4 GPUs set GPUS=4 BATCH=256.
_c="$(dirname "${BASH_SOURCE[0]}")/../common.sh"; [ -f "$_c" ] || _c="${SLURM_SUBMIT_DIR:-.}/scripts/common.sh"
source "$_c"     # under sbatch the script runs from a spool copy, so fall back to the submit dir

GPUS=${GPUS:-2}
BATCH=${BATCH:-512}
[ $((GPUS * BATCH)) -eq 1024 ] || { echo "FATAL: GPUS x BATCH = $((GPUS * BATCH)), not 1024" >&2; exit 1; }
MINIBS=${MINIBS:-256}       # GradCache chunk; changes memory use only, not the result
[ -e "$ENCODER" ] && { echo "FATAL: refusing to overwrite $ENCODER" >&2; exit 1; }
PFX=$("$PYTHON" -c "
b=$BITS
w=[x for x in (8,12,16,24,32,36,40,48,56,60,64,72,80,84,88,96,104,108,112,120) if x<b]
print(','.join(str(x) for x in w+[b]))")
MPORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))
LOG="$OUT/logs/docid_encoder_$(date +%Y%m%d_%H%M%S).log"
echo "=== DocID encoder | $DATASET $SETTING | $SBERT | $BITS bits = $((BITS/NLEV)) tokens x $((2**NLEV)) | $OPT lr=$LR ==="
echo "    checkpoint -> $ENCODER"

"$TORCHRUN" --master_port="$MPORT" --nproc_per_node="$GPUS" "$SRC/docid_encoder/train.py" \
    --dataset "$DATASET" \
    --sbert_model "$SBERT" \
    --n_levels "$NLEV" \
    --proj_dim "$BITS" \
    --cosine_temp 0.07 \
    --lr "$LR" \
    --optimizer "$OPT" \
    --max_seq_length "$MAXLEN" \
    --batch_size "$BATCH" \
    --mini_batch_size "$MINIBS" \
    --epochs 200 \
    --eval_queries 3000 \
    --prefix_bits "$PFX" \
    --select_on pdim \
    --qg_per_doc 0 \
    --real_weight 1.0 \
    --patience 5 \
    --seed 42 \
    --no_wandb \
    "${SPLIT_ARG[@]}" \
    --run_name docid_encoder \
    --save_dir "$OUT" 2>&1 | tee "$LOG"
