"""Evaluation settings: the paper's values, fixed here rather than passed as flags."""

# Constrained beam search. Two rows are printed: one tree with beam BEAM, and TREES permuted
# trees with beam BEAM each, whose union is ranked by the full score (the reported system).
# --beam and --trees override these, e.g. for the beam x trees sweep of Figure 3.
BEAM       = 100
TREES      = 8
PERM_SEED  = 0          # seed of the bit permutations of trees 2..TREES

# Hit@n cutoffs; MRR is computed up to the largest.
CUTOFFS    = (1, 5, 10, 20, 100)

# Queries per forward pass when encoding (memory only).
ENCODE_BATCH = 256
