# Sourced by every launcher. Resolves the repository, the run folder and per-dataset defaults.
#
#   DATASET  nq320k | trivia | msmarco9010
#   SETTING  full | seen60_unseen40 | seen90_unseen10      (msmarco9010: seen90_unseen10 only)
#
# Optional environment:
#   BITS=120  NLEV=12        DocID length in bits, and bits per token
#   SBERT=bert-base-uncased  backbone, or t5-base (both trainers then use Adafactor, see the
#                            config.py of each stage)
#   MAXLEN=128               input length in tokens
#   OUT=<dir>                run folder; default outputs/<dataset>/<setting>/<run name>
#   ENCODER=<.pt> DOCIDS=<.npz>   use an existing DocID encoder and DocID table (read only)
#   PYTHON=python            interpreter (torchrun is taken from the same environment)
set -eo pipefail
DATASET=${1:?usage: $0 <nq320k|trivia|msmarco9010> <full|seen60_unseen40|seen90_unseen10>}
SETTING=${2:?missing setting: full | seen60_unseen40 | seen90_unseen10}

_here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || true)
if   [ -f "$_here/src/corpus.py" ];                 then REPO="$_here"
elif [ -f "${SLURM_SUBMIT_DIR:-}/src/corpus.py" ];  then REPO="$SLURM_SUBMIT_DIR"
else echo "FATAL: cannot locate the repository; submit from its root or set REPO" >&2; exit 1; fi
SRC="$REPO/src"
DATA=${BFD_DATA:-$REPO/data}
export BFD_DATA="$DATA"
export TOKENIZERS_PARALLELISM=false
PYTHON=${PYTHON:-python}
TORCHRUN=$(dirname "$(command -v "$PYTHON")")/torchrun
[ -x "$TORCHRUN" ] || TORCHRUN=torchrun

case "$DATASET" in nq320k|trivia|msmarco9010) ;; *) echo "FATAL: unknown dataset $DATASET" >&2; exit 1;; esac
SPLIT_ARG=()
if [ "$SETTING" != full ]; then
  SPLIT_DIR="$DATA/$DATASET/$SETTING"
  [ -f "$SPLIT_DIR/manifest.json" ] || { echo "FATAL: no split at $SPLIT_DIR" >&2; exit 1; }
  SPLIT_ARG=(--split_dir "$SPLIT_DIR")
elif [ "$DATASET" = msmarco9010 ]; then
  echo "FATAL: msmarco9010 has no full setting; use seen90_unseen10" >&2; exit 1
fi

BITS=${BITS:-120}
NLEV=${NLEV:-12}
[ $((BITS % NLEV)) -eq 0 ] || { echo "FATAL: BITS $BITS is not a multiple of NLEV $NLEV" >&2; exit 1; }
SBERT=${SBERT:-bert-base-uncased}
MAXLEN=${MAXLEN:-128}

# Per-dataset defaults = the paper's settings.
#   QG       generated queries per document used to train the retriever
#   ENC_BS   batch size when assigning DocIDs (kept per dataset so DocIDs reproduce exactly)
#   TEST     the full-setting test set (NQ320K has no test set; its dev set is reported)
case "$DATASET" in
  nq320k)      QG=${QG:-10}; ENC_BS=512; TEST=dev;  ENC_QSPLIT=dev  ;;
  trivia)      QG=${QG:-10}; ENC_BS=512; TEST=test; ENC_QSPLIT=dev  ;;
  msmarco9010) QG=${QG:-3};  ENC_BS=256; TEST=dev;  ENC_QSPLIT=none ;;
esac

RUN=${RUN:-$(basename "$SBERT")_b${BITS}t${NLEV}}
OUT=${OUT:-$REPO/outputs/$DATASET/$SETTING/$RUN}
ENCODER=${ENCODER:-$OUT/docid_encoder_best.pt}   # written by docid_encoder/train.sh
DOCIDS=${DOCIDS:-$OUT/docids.npz}                # written by docid_encoder/assign_docids.sh
MERGED="$OUT/retriever_merged.pt"          # written by retriever/merge.sh
mkdir -p "$OUT/logs"
