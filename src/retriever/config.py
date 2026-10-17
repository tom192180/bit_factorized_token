"""Training settings of the retriever: the paper's values, fixed here rather than passed as
flags. The launcher only chooses the dataset, the DocID encoder, the DocID table and the
number of generated queries per document."""

# Steps and batch. One step is ACCUM micro batches of BATCH queries (512 queries per step).
STEPS            = 100_000
BATCH            = 256
ACCUM            = 2

# Optimizer, chosen by the encoder's backbone (from the checkpoint sidecar): AdamW for BERT,
# Adafactor for T5. LR_HEAD is for the per-bit weights w.
LR_ENC_ADAMW     = 2e-5
LR_ENC_ADAFACTOR = 1e-3
LR_HEAD          = 1e-3
MAX_GRAD_NORM    = 1.0

# InfoNCE inverse temperature (Eq. 6: tau_2 = 0.05).
LOGIT_SCALE      = 20.0       # 1 / tau_2

# Validation: constrained beam search every VAL_CHECK_INTERVAL epochs; the checkpoint with the
# best recall@1 is kept; stop after PATIENCE validations without improvement.
VAL_CHECK_INTERVAL = 0.5
EVAL_BEAM        = 100
PATIENCE         = 10
