#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --gpus=1
#SBATCH -t 4:00:00
# Evaluate with constrained beam search over 8 permuted prefix trees (beam 100), tie-aware.
#
#   sbatch scripts/evaluation/eval.sh <dataset> <setting> [encoder|retriever|merged ...]
#
# Default: the merged retriever M2' (the reported system). Name other models to evaluate them
# as well: encoder = M1 used directly, retriever = M2 before merging.
#   seen/unseen settings  two rows per model: test_seen and test_unseen
#   full setting          one row per model on the dataset's reported set (NQ320K dev, TriviaQA test)
# For msmarco9010 the seen queries are scored against the seen documents only (the index before
# new documents arrive), following the protocol of the split's source; elsewhere against all.
_c="$(dirname "${BASH_SOURCE[0]}")/../common.sh"; [ -f "$_c" ] || _c="${SLURM_SUBMIT_DIR:-.}/scripts/common.sh"
source "$_c"     # under sbatch the script runs from a spool copy, so fall back to the submit dir
shift 2
MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || MODELS=(merged)
[ -f "$DOCIDS" ] || { echo "FATAL: no DocID table at $DOCIDS" >&2; exit 1; }
RETRIEVER=${RETRIEVER:-$(ls -1t "$OUT"/retriever/retriever_*.pt 2>/dev/null | head -1 || true)}
mkdir -p "$OUT/eval"

run () {   # $1 model name   $2 checkpoint   $3.. extra flags
  local name=$1 ck=$2; shift 2
  [ -f "$ck" ] || { echo "FATAL: no $name checkpoint at $ck" >&2; exit 1; }
  local sets=(full); [ "$SETTING" = full ] || sets=(test_seen test_unseen)
  for s in "${sets[@]}"; do
    local extra=()
    if [ "$SETTING" != full ]; then
      extra=(--split_dir "$SPLIT_DIR" --split_file "$s")
      [ "$DATASET" = msmarco9010 ] && [ "$s" = test_seen ] && extra+=(--restrict "$SPLIT_DIR/d0_docs.txt")
    fi
    echo; echo "########## $DATASET $SETTING | $name | $s ##########"
    "$PYTHON" "$SRC/evaluation/eval_retrieval.py" --dataset "$DATASET" --ckpt "$ck" --enc "$DOCIDS" \
        --split "$TEST" "${extra[@]}" --max_seq_length "$MAXLEN" \
        --beam 100 --multi 8:100 "$@" \
        2>&1 | tee "$OUT/eval/${name}_${s}_$(date +%Y%m%d_%H%M%S).log"
  done
}
for m in "${MODELS[@]}"; do
  case "$m" in
    encoder)   run encoder   "$ENCODER"   --stage1_only ;;
    retriever) run retriever "$RETRIEVER" --stage1 "$ENCODER" ;;
    merged)    run merged    "$MERGED"    --stage1 "$ENCODER" ;;
    *) echo "FATAL: unknown model $m (encoder | retriever | merged)" >&2; exit 1 ;;
  esac
done
