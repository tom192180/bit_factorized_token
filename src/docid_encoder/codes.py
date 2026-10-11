"""Group DocID bits into tokens, and back.

The encoder emits k bits. Reading them n_levels at a time gives k / n_levels tokens, each with
2**n_levels possible values (120 bits at 12 bits per token = 10 tokens of 4,096 values).
"""
import torch


def pack_bits(bits, n_levels):
    """(b, k) bits -> (b, k // n_levels) token values."""
    b, d = bits.shape
    assert d % n_levels == 0, f"code_dim ({d}) must be divisible by n_levels ({n_levels})"
    basis = (2 ** torch.arange(n_levels, device=bits.device)).long()
    return (bits.reshape(b, d // n_levels, n_levels).long() * basis).sum(-1)


def unpack_bits(idx, n_levels):
    """(b, n_tokens) token values -> (b, n_tokens * n_levels) bits."""
    b, n_part = idx.shape
    out = torch.empty(b, n_part * n_levels, dtype=torch.int8, device=idx.device)
    for k in range(n_levels):
        out[:, k::n_levels] = ((idx // (2 ** k)) % 2).to(torch.int8)
    return out
