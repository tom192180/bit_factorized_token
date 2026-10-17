"""Load a trained DocID encoder from its checkpoint.

A checkpoint is two files: <name>.pt (the weights) and <name>.pt.arch.json (a sidecar with the
backbone, the number of bits and the bits per token).
"""
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from model import Encoder_bit                                    # noqa: E402


def load_encoder(ckpt, sbert_model, max_seq_length):
    """Rebuild the encoder from the sidecar and load its weights.

    sbert_model=None takes the backbone from the sidecar. Returns
    (model, n_tokens, values_per_token, sbert_model, bits_per_token)."""
    side = ckpt.rstrip("/") + ".arch.json"
    if not os.path.exists(side):
        raise SystemExit(f"no sidecar {side}; refusing to guess the architecture")
    arch = json.load(open(side))
    if arch.get("head", "linear") != "linear":
        raise SystemExit(f"{side} says head={arch['head']!r}, but only the linear head exists "
                         f"since 2026-09-04 (the mlp head went with the grid hinge). That "
                         f"checkpoint is retired; use a dense40 run.")
    if sbert_model is None:
        sbert_model = arch.get("sbert_model")
        if not sbert_model:
            raise SystemExit(f"{side} has no sbert_model; pass --sbert_model explicitly")
    elif arch.get("sbert_model") and arch["sbert_model"] != sbert_model:
        raise SystemExit(f"--sbert_model {sbert_model} contradicts the sidecar's "
                         f"{arch['sbert_model']}; drop the flag and let the sidecar decide")
    print(f"  arch: level={arch['level']} n_levels={arch.get('n_levels')} "
          f"proj_dim={arch['proj_dim']} sbert={sbert_model}")
    if arch["level"] != 2:
        raise SystemExit(f"{side} says level={arch['level']}; the encoder emits ONE BIT per "
                         f"dimension, so only level 2 exists. That checkpoint is retired.")
    model = Encoder_bit(proj_dim=arch["proj_dim"], sbert_model=sbert_model,
                        max_seq_length=max_seq_length)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    n_tok = arch["proj_dim"] // arch["n_levels"]
    kary = arch["level"] ** arch["n_levels"]
    print(f"  -> {n_tok} tokens x {kary} values")
    return model, n_tok, kary, sbert_model, arch["n_levels"]
