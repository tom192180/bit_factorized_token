"""Training data for the DocID encoder (and the pseudo-query selection the retriever reuses).

Everything is read through corpus.py from the repository's data/ folder.

    select_pseudo_queries(dataset, n, seed)   the seeded n generated queries per document
    load_query_doc_pairs(...)                 NQ320K, full-corpus setting
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


def load_trivia_pairs(splits=("train", "dev"), qg_per_doc=0, qg_seed=42):
    """TriviaQA, full-corpus setting -> (pairs_by_split, n_unique_docs, extra_docs).

    A pair is (query_text, doc_text, doc_id); doc_id is the document's position in the corpus,
    shared by all splits. One (query, gold) becomes one pair: train has one gold per row with
    the query repeated, dev and test list several golds per query.

    qg_per_doc > 0 appends that many generated queries per document to TRAIN only.

    extra_docs is the TEXT of every corpus document that no returned split points at. The
    trainer adds them to the validation pool and nowhere else, so a validation query competes
    against the whole corpus and not only the documents these splits happen to mention.
    """
    if isinstance(splits, str):
        splits = (splits,)
    ids, texts = doc_ids_texts("trivia")
    doc_id_of = {d: i for i, d in enumerate(ids)}
    out = {}
    for sp in splits:
        out[sp] = [(q, texts[i], i) for q, i in load_queries("trivia", sp, ids)]
        print(f"  trivia {sp}: {len(out[sp]):,} pairs")
    n_real_train = len(out.get("train", []))   # everything after this is GENERATED
    if qg_per_doc and "train" in out:
        chosen = select_pseudo_queries("trivia", qg_per_doc, qg_seed)
        n_before = len(out["train"])
        for did in sorted(chosen):              # sorted: reproducible independent of dict order
            if did not in doc_id_of:
                continue
            i = doc_id_of[did]
            for q in chosen[did]:
                out["train"].append((q, texts[i], i))
        n_aug = len({d for _, _, d in out["train"][n_before:]})
        print(f"  trivia qg augmentation: +{len(out['train'])-n_before:,} generated queries "
              f"({qg_per_doc}/doc, seed {qg_seed}) over {n_aug:,} docs "
              f"-> {len(out['train']):,} train pairs")
    seen = {d for _, _, d in out.get("train", [])} | {d for _, _, d in out.get("dev", [])}
    extra_docs = [texts[i] for i in range(len(ids)) if i not in seen]
    print(f"  trivia corpus: {len(ids):,} docs "
          f"({len(seen):,} judged by these splits + {len(extra_docs):,} pool-only)")
    out["_n_real_train"] = n_real_train
    return out, len(ids), extra_docs


def load_query_doc_pairs(qg_per_doc=0, qg_seed=42):
    """NQ320K, full-corpus setting -> (pairs, n_docs, pool_only, val_rows, n_real, dev_rows).

    pairs     one (query_text, doc_text, doc_id) per train query, in train-file order. doc_id
              is a running index in order of first appearance (NOT the corpus position).
    n_real    pairs[:n_real] are real queries; the rest are generated.
    n_docs    all 109,739 documents: those without a train query get a doc_id at the end.
    dev_rows  the dev queries in the same triple form; the trainer validates on them.
    pool_only, val_rows   always empty; kept so the trainer's unpacking is unchanged.

    qg_per_doc > 0 appends that many generated queries per document for ALL 109,739 documents,
    including the 1,713 with no real train query, which receive a doc_id at that point.
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
    n_real = len(pairs)     # everything after this index is GENERATED

    if qg_per_doc:
        text_of = dict(zip(ids, texts))
        chosen = select_pseudo_queries("nq320k", qg_per_doc, qg_seed)
        n_before, n_new_docs = len(pairs), 0
        for key in sorted(chosen):              # ALL documents -- see the docstring
            body = id_to_text.get(key_to_id[key]) if key in key_to_id else text_of.get(key)
            if not body:
                continue
            if key not in key_to_id:            # a doc with no real query: give it an id now
                key_to_id[key] = len(key_to_id)
                id_to_text[key_to_id[key]] = body
                n_new_docs += 1
            doc_id = key_to_id[key]
            for q in chosen[key]:
                pairs.append((q, id_to_text[doc_id], doc_id))
        print(f"  qg augmentation: +{len(pairs)-n_before:,} generated queries "
              f"({qg_per_doc}/doc, seed {qg_seed}) -> {len(pairs):,} pairs, "
              f"{len(key_to_id):,} docs (+{n_new_docs:,} that had no real query)")

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
    return pairs, len(key_to_id), [], [], n_real, dev_rows


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
