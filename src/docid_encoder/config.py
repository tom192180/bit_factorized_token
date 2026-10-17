"""Training settings of the DocID encoder: the paper's values, fixed here rather than passed
as flags. The launcher only chooses the dataset, the backbone, the DocID shape and the batch."""

# One bit per dimension, N_LEVELS bits per token.
LEVEL           = 2
N_LEVELS        = 12

# Contrastive loss.
COSINE_TEMP     = 0.07

# Batch. BATCH_SIZE is per GPU: with 2 GPUs the loss scores each query against 1,024 documents.
# MINI_BATCH_SIZE is the GradCache chunk; it changes memory use only, not the result.
BATCH_SIZE      = 512
MINI_BATCH_SIZE = 256
MAX_SEQ_LENGTH  = 128

# Optimizer, chosen by backbone: AdamW for BERT, Adafactor for T5.
LR_ADAMW        = 2e-5
LR_ADAFACTOR    = 1e-3
MAX_GRAD_NORM   = 1.0

# Epochs, and early stopping after PATIENCE epochs without a better PDIM.
EPOCHS          = 200
PATIENCE        = 5

# Validation after every epoch.
EVAL_QUERIES    = 3000      # validation queries scored (a fixed seeded sample)
EVAL_BATCH_SIZE = 256       # texts per forward pass when encoding for validation
DOC_CHUNK       = 4096      # documents per block when ranking
NCCL_TIMEOUT_S  = 7200      # multi-GPU communication timeout, seconds
