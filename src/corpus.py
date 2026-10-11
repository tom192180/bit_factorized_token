"""The one place data is read from disk. Everything is under the repository's data/ folder.

    data/<dataset>/corpus.jsonl             {"id", "text"} per line; line i is document i
    data/<dataset>/pseudo_queries.tsv       query \t docid   (generated queries)
    data/<dataset>/full/<name>_queries.tsv  query \t gold id[,gold id...]   (full-corpus setting)
    data/<dataset>/<split>/                 a seen/unseen split:
        d0_docs.txt  d1_docs.txt            seen / unseen document ids
        train_queries.tsv valid_queries.tsv test_seen_queries.tsv test_unseen_queries.tsv
        manifest.json                       dataset name, counts, digests

Set the environment variable BFD_DATA to use a data folder somewhere else.

    doc_ids_texts(dataset)             -> (ids, texts)   THE document order
    load_queries(dataset, name, ids)   -> [(query_text, doc_index), ...]     full-corpus sets
    load_pseudo_queries(dataset)       -> {docid: [query, ...]}
    load_dynamic_split(split_dir, ...) -> (parts, D0_idx, D1_idx, ids, texts)

WHY THE ORDER IS THE CONTRACT. A document's index everywhere downstream is its position in the
list doc_ids_texts returns: row i of the saved DocID table belongs to ids[i], and the prefix
trees are built over those rows. corpus.jsonl is stored in that order (ids sorted as strings),
and the loader checks it.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BFD_DATA", os.path.join(REPO, "data"))

DATASETS = ("nq320k", "trivia", "msmarco9010")
FULL_SETS = {"nq320k": ("train", "dev"), "trivia": ("train", "dev", "test"), "msmarco9010": ()}


def data_path(dataset, *parts):
    return os.path.join(DATA, dataset, *parts)


def doc_ids_texts(dataset="nq320k"):
    """(ids, texts) in THE canonical order. Position i is document i everywhere downstream."""
    if dataset not in DATASETS:
        raise SystemExit(f"unknown --dataset {dataset!r}; known: {DATASETS}")
    path = data_path(dataset, "corpus.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no corpus at {path} -- see data/README.md")
    ids, texts = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            ids.append(d["id"])
            texts.append(d["text"])
    if any(ids[i] >= ids[i + 1] for i in range(len(ids) - 1)):
        raise SystemExit(f"{path} is not sorted by id; the document order would be wrong")
    return ids, texts


def _read_query_tsv(path, id2i):
    """query \t gold[,gold...]  ->  one (query, doc_index) pair per gold found in the corpus."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2 or not f[0]:
                continue
            for g in f[1].split(","):
                g = g.strip()
                if g in id2i:
                    out.append((f[0], id2i[g]))
    return out


def load_queries(dataset, split, ids):
    """[(query_text, doc_index)] for a FULL-CORPUS query set; doc_index indexes `ids`.

        nq320k    train | dev | qg      (qg = every generated query, unselected)
        trivia    train | dev | test    (generated queries go through --qg_per_doc)
        msmarco9010  none -- its queries live only in its split folder
    """
    id2i = {d: i for i, d in enumerate(ids)}
    if dataset == "msmarco9010":
        raise SystemExit("msmarco9010 queries live only in its split dir; pass --split_dir")
    if dataset == "nq320k" and split == "qg":
        return _read_query_tsv(data_path(dataset, "pseudo_queries.tsv"), id2i)
    if dataset == "trivia" and split == "qg":
        # The seeded per-document choice is made by data.select_pseudo_queries. Returning all
        # 1.1M generated queries here would train the retriever on queries the DocID encoder
        # never saw.
        raise SystemExit("trivia generated queries go through --qg_per_doc, not a raw split")
    if dataset not in DATASETS:
        raise SystemExit(f"unknown --dataset {dataset!r}; known: {DATASETS}")
    if split not in FULL_SETS[dataset]:
        raise SystemExit(f"{dataset} has no split {split!r}; known: {sorted(FULL_SETS[dataset])}")
    return _read_query_tsv(data_path(dataset, "full", f"{split}_queries.tsv"), id2i)


def load_pseudo_queries(dataset):
    """{docid: [generated query, ...]} with the queries of a document in file order. The seeded
    selection in data.select_pseudo_queries depends on that order, so it must not change."""
    by_doc = {}
    with open(data_path(dataset, "pseudo_queries.tsv"), encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0]:
                by_doc.setdefault(p[1], []).append(p[0])
    return by_doc


def load_dynamic_split(split_dir, dataset=None, want=("train", "valid", "test")):
    """Read a seen/unseen split folder.

    Returns (parts, D0_idx, D1_idx, ids, texts) where parts maps a name to a list of
    (query_text, doc_index) pairs and the indices are positions in doc_ids_texts(dataset).

    THE INDEX IS D0 DURING TRAINING AND VALIDATION, D0 + D1 AT TEST. The corpus GROWS after
    the model is fixed. Callers building a retrieval pool for training or model selection must
    restrict it to D0_idx; only the final evaluation sees D1.
    """
    with open(os.path.join(split_dir, "manifest.json")) as fh:
        man = json.load(fh)
    dataset = dataset or man["dataset"]
    if dataset != man["dataset"]:
        raise SystemExit(f"{split_dir} is a {man['dataset']} split, not {dataset}")
    ids, texts = doc_ids_texts(dataset)
    i2 = {d: i for i, d in enumerate(ids)}

    def rd(f):                      # one id per line; strip, or the newline joins the id
        with open(os.path.join(split_dir, f), encoding="utf-8") as fh:
            return {ln.split("\t")[0].strip() for ln in fh if ln.strip()}
    D0 = {i2[d] for d in rd("d0_docs.txt") if d in i2}
    D1 = {i2[d] for d in rd("d1_docs.txt") if d in i2}
    if len(D0) + len(D1) != len(ids):
        raise SystemExit(f"{split_dir}: D0 {len(D0):,} + D1 {len(D1):,} != corpus {len(ids):,}")
    parts = {}
    for nm in want:
        rows = []
        with open(os.path.join(split_dir, f"{nm}_queries.tsv"), encoding="utf-8") as fh:
            for ln in fh:
                f = ln.rstrip("\n").split("\t")
                if len(f) < 2 or not f[0]:
                    continue
                for d in f[1].split(","):
                    if d in i2:
                        rows.append((f[0], i2[d]))
        parts[nm] = rows
    return parts, D0, D1, ids, texts
