"""Default hyperparameters of the DocID encoder trainer. The launchers override most of them."""
import os

import torch

MODEL_SAVE_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "outputs", "docid_encoder")

# One bit per dimension.
LEVEL           = 2

# Bits per token, and the resulting number of values per token.
N_LEVELS        = 12
CODEBOOK_SIZE   = LEVEL ** N_LEVELS

# Temperature of the contrastive loss.
COSINE_TEMP     = 0.07

# Training. BATCH_SIZE is per GPU; MINI_BATCH_SIZE is the GradCache chunk.
BATCH_SIZE      = 512
MINI_BATCH_SIZE = 256
MAX_SEQ_LENGTH  = 128
LR              = 1e-4
EPOCHS          = 200
PATIENCE        = 10
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Size of the --code_test smoke run.
CODE_TEST_PAIRS  = int(os.environ.get("STAGE1_CODE_TEST_PAIRS", 2000))
CODE_TEST_EPOCHS = int(os.environ.get("STAGE1_CODE_TEST_EPOCHS", 2))

# Validation during training.
EVAL_QUERIES    = 3000      # validation queries scored per epoch
DOC_CHUNK       = 4096      # documents per block when ranking
EVAL_BATCH_SIZE = 256       # texts per forward pass when encoding for validation
NCCL_TIMEOUT_S  = 7200      # multi-GPU communication timeout, seconds
