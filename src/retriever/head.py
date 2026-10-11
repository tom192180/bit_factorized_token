"""The retriever's own parameters: one weight per DocID bit.

    u_q = w * z_q        z_q = tanh(LN(W x_q)) from the encoder, w initialised to 1

A query scores a document by s(q, d) = b_d . u_q, where b_d is the DocID as +-1 bits.
"""
import torch
import torch.nn as nn


class BitWeights(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.zs = nn.Parameter(torch.ones(n_bits))  # w; "zs" is its name in the checkpoints

    def forward(self, z):
        """z (..., n_bits) -> u = w * z."""
        return z * self.zs
