"""The prefix tree over DocIDs, stored as a sorted array.

Documents are sorted by their token sequence, so a tree node is a row range [lo, hi): going to
child v at depth t narrows the range to the rows whose t-th token is v. Beam search is repeated
narrowing. root_children precomputes depth 0, which is the same for every query.
"""
import numpy as np


def root_children(Cs):
    """(values, starts, ends) of the first-token blocks of a sorted (n, T) token array."""
    c0 = Cs[:, 0]
    i0 = np.flatnonzero(np.concatenate(([True], c0[1:] != c0[:-1])))
    starts = i0.astype(np.int64)
    return (c0[i0].astype(np.int64),
            starts,
            np.concatenate((starts[1:], [len(Cs)])).astype(np.int64))
