"""Training data for the DocID encoder, plus the pseudo-query selection used by the retriever.

Everything is read through corpus.py from the repository's data/ folder. The DocID encoder
trains on real queries only; select_pseudo_queries is here because the retriever imports it.

    select_pseudo_queries(dataset, n, seed)   the seeded n generated queries per document
    load_query_doc_pairs()                    NQ320K, full-corpus setting
    load_trivia_pairs(...)                    TriviaQA, full-corpus setting

The seen/unseen split runs do not use the two full-corpus loaders: the trainers read a split
through corpus.load_dynamic_split.
"""
import os
import sys

import torch
from torch.utils.data import Dataset

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from corpus import doc_ids_texts, load_queries, load_pseudo_queries        # noqa: E402


def select_pseudo_queries(dataset, n_per_doc, seed=42):
    """docid -> the n generated queries chosen for it. THE SELECTION IS THE CONTRACT.

    The DocID encoder and the retriever must be able to draw the same queries, so the choice
    depends on (n_per_doc, seed) alone: document ids are iterated in SORTED order and drawn
    from one seeded generator, never from dict or file order. A document with n or fewer
    generated queries keeps all of them (TriviaQA has between 11 and 50 per document; NQ320K
    has 20; msmarco9010 has 10 for seen documents and none for new ones).
    """
    by_doc = load_pseudo_queries(dataset)
    if not by_doc:
        raise SystemExit(f"no generated queries found for {dataset}")
    g = torch.Generator().manual_seed(seed)
    out = {}
    for did in sorted(by_doc):                  # SORTED: order must not depend on the file
        qs = by_doc[did]
        if n_per_doc >= len(qs):
            out[did] = list(qs)
        else:
            idx = torch.randperm(len(qs), generator=g)[:n_per_doc].tolist()
            out[did] = [qs[i] for i in sorted(idx)]
    return out


def load_trivia_pairs(splits=("train", "dev")):
    """TriviaQA, full-corpus setting -> (pairs_by_split, n_unique_docs, extra_docs).

    A pair is (query_text, doc_text, doc_id); doc_id is the document's position in the corpus,
    shared by all splits. One (query, gold) becomes one pair: train has one gold per row with
    the query repeated, dev and test list several golds per query.

    extra_docs is the TEXT of every corpus document that no returned split points at. The
    trainer adds them to the validation pool and nowhere else, so a validation query competes
    against the whole corpus and not only the documents these splits happen to mention.
    """
    if isinstance(splits, str):
        splits = (splits,)
    ids, texts = doc_ids_texts("trivia")
    out = {}
    for sp in splits:
        out[sp] = [(q, texts[i], i) for q, i in load_queries("trivia", sp, ids)]
        print(f"  trivia {sp}: {len(out[sp]):,} pairs")
    seen = {d for _, _, d in out.get("train", [])} | {d for _, _, d in out.get("dev", [])}
    extra_docs = [texts[i] for i in range(len(ids)) if i not in seen]
    print(f"  trivia corpus: {len(ids):,} docs "
          f"({len(seen):,} judged by these splits + {len(extra_docs):,} pool-only)")
    return out, len(ids), extra_docs


def load_query_doc_pairs():
    """NQ320K, full-corpus setting -> (pairs, n_docs, extra_docs, dev_rows).

    pairs      one (query_text, doc_text, doc_id) per train query, in train-file order. doc_id
               is a running index in order of first appearance (NOT the corpus position).
    n_docs     all 109,739 documents: those without a train query get a doc_id at the end.
    extra_docs always empty: every document is reached through a train or dev query.
    dev_rows   the dev queries in the same triple form; the trainer validates on them.
    """
    ids, texts = doc_ids_texts("nq320k")
    key_to_id, id_to_text = {}, {}
    pairs = []
    for q_text, i in load_queries("nq320k", "train", ids):
        key = ids[i]
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
            id_to_text[key_to_id[key]] = texts[i]
        doc_id = key_to_id[key]
        pairs.append((q_text, id_to_text[doc_id], doc_id))

    # Every corpus document gets a doc_id, including the 1,713 that only dev queries point to.
    # They are numbered in order of first appearance in the dev file.
    dev_pairs = load_queries("nq320k", "dev", ids)
    for _, i in dev_pairs:
        if ids[i] not in key_to_id:
            key_to_id[ids[i]] = len(key_to_id)
            id_to_text[key_to_id[ids[i]]] = texts[i]

    dev_rows = []
    for q_text, i in dev_pairs:
        if ids[i] in key_to_id:
            di = key_to_id[ids[i]]
            dev_rows.append((q_text, id_to_text[di], di))
    return pairs, len(key_to_id), [], dev_rows


class QueryDocPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs  # list[(query_text, doc_text, doc_id)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    queries, docs, doc_ids = zip(*batch)
    return list(queries), list(docs), torch.tensor(doc_ids)
