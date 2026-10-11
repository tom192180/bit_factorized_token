# Bit-Factorized DocIDs

Code for generative retrieval with document identifiers that are **computed from the document
text**, so a document added after training gets its identifier the same way as every other
document, with no retraining.

## The method in three steps

1. **DocID encoder (M1).** A text encoder trained with a contrastive loss. A document's DocID
   is the sign of each of its `k` output dimensions: a flat string of `k` bits (120 by default).
2. **Retriever (M2).** Initialised from M1 and trained with the DocIDs fixed, so that a query
   scores its relevant document's DocID above all other seen DocIDs. The final retriever
   **M2'** is the average of the weights of M1 and M2.
3. **Inference.** The `k` bits are read as `T` tokens of `m` bits and decoded by constrained
   beam search. Because the score is a sum over bits, any bit order gives a valid prefix tree;
   8 permuted trees are searched and their candidates ranked together.

## Layout

```
data/                    processed data, not in git -- see docs/DATA.md
scripts/
  common.sh              shared setup and per-dataset defaults
  docid_encoder/         train.sh, assign_docids.sh
  retriever/             train.sh, merge.sh
  evaluation/            eval.sh
src/
  corpus.py              the only code that reads data/
  docid_encoder/         model, loss, training, DocID assignment
  retriever/             per-bit weights, prefix tree, training, weight merging
  evaluation/            beam-search evaluation, for any of the models
outputs/                 created by the launchers
```

## Setup

```
pip install -r requirements.txt
```

Place the processed data under `data/` (or point `BFD_DATA` at it). See `docs/DATA.md`.

## Running

Every launcher takes a dataset and a setting:

| dataset | settings |
|---|---|
| `nq320k` | `full`, `seen60_unseen40`, `seen90_unseen10` |
| `trivia` | `full`, `seen60_unseen40`, `seen90_unseen10` |
| `msmarco9010` | `seen90_unseen10` |

In a seen/unseen setting, training uses the seen documents only; the unseen documents receive
their DocIDs from the trained encoder and are evaluated without any further training.

```bash
D=nq320k; S=seen60_unseen40

scripts/docid_encoder/train.sh       $D $S    # 1. DocID encoder        (2 GPUs)
scripts/docid_encoder/assign_docids.sh $D $S  # 2. DocIDs for all documents
scripts/retriever/train.sh           $D $S    # 3. retriever            (1 GPU)
scripts/retriever/merge.sh           $D $S    # 4. M2' = average of M1 and M2
scripts/evaluation/eval.sh           $D $S    # 5. evaluate M2'
```

The scripts carry `#SBATCH` headers and can be submitted with `sbatch` from the repository
root. Results are written to `outputs/<dataset>/<setting>/<backbone>_b<bits>t<bits per token>/`.

T5-base instead of BERT-base:

```bash
export SBERT=t5-base OPT=adafactor LR=1e-3 S2_OPT=adafactor LR_ENC=1e-3
```

Other DocID lengths: `BITS=84` or `BITS=252` (with `NLEV=12` bits per token).

To evaluate the DocID encoder or the unmerged retriever as well:
`scripts/evaluation/eval.sh $D $S encoder retriever merged`.

## Settings used in the paper

| | DocID encoder | retriever |
|---|---|---|
| backbone | BERT-base / T5-base encoder | initialised from the DocID encoder |
| input length | 128 tokens | 128 tokens |
| loss | InfoNCE, cosine, temperature 0.07 | InfoNCE over all seen DocIDs, temperature 0.05 |
| negatives | in batch, 2 GPUs x 512 = 1,024 documents | every seen DocID |
| optimizer | AdamW 2e-5 (BERT) / Adafactor 1e-3 (T5) | AdamW 2e-5 encoder, 1e-3 per-bit weights (BERT) / Adafactor 1e-3 (T5) |
| training queries | real | real + generated: 10 per document (NQ320K, TriviaQA), 3 (MS MARCO) |
| budget | up to 200 epochs, patience 5 | up to 100,000 steps, patience 10 |
| model selection | MRR under exhaustive ranking by bit agreement, on the validation queries | Recall@1 with beam 100 on the validation queries |

Evaluation: 8 trees, beam 100. Documents that share a DocID are scored by the expected value
of the metric over a random order within the group.
