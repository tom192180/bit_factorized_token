# Data

The processed data is not stored in git (about 3 GB). Put it in `data/` at the repository
root, or set `BFD_DATA` to its location. `data/CHECKSUMS.sha256` lists the SHA-256 of every file.

```
data/<dataset>/
    corpus.jsonl              one document per line: {"id": ..., "text": ...}
                              sorted by id (as a string); line i is document i everywhere
    pseudo_queries.tsv        generated queries:  query <tab> document id
    full/                     full-corpus setting (NQ320K, TriviaQA)
        train_queries.tsv  dev_queries.tsv  [test_queries.tsv]
    seen60_unseen40/          60% of the documents seen, 40% unseen   (NQ320K, TriviaQA)
    seen90_unseen10/          90% seen, 10% unseen                    (all three datasets)
        d0_docs.txt           ids of the seen documents
        d1_docs.txt           ids of the unseen documents
        train_queries.tsv     relevant documents all seen; used for training
        valid_queries.tsv     relevant documents all seen; used only to select a checkpoint
        test_seen_queries.tsv     evaluation, relevant documents seen
        test_unseen_queries.tsv   evaluation, relevant documents unseen
        manifest.json         counts and digests
```

Query files are `query <tab> id[,id,...]`: one row per query, relevant document ids comma-joined.

| dataset | documents | generated queries | notes |
|---|---|---|---|
| `nq320k` | 109,739 | 2,194,778 (20 per document) | document = title + abstract + content |
| `trivia` | 73,970 | 1,105,302 (11 to 50 per document) | several relevant documents per query |
| `msmarco9010` | 324,297 | 2,918,690 (10 per seen document) | MS MARCO document ranking; test queries are the official dev set |

## Sources and processing

Nothing is read from raw Wikipedia or raw web pages: every dataset starts from a preprocessed
release of earlier work.

### NQ320K

Source: the preprocessed NQ320K files released with NCI (Neural Corpus Indexer).

| our file | built from | processing |
|---|---|---|
| `corpus.jsonl` | `nq_title_abs.tsv` (title + abstract) and `NQ_doc_content.tsv` (body) | The two are joined on NCI's document id and concatenated as `title+abstract`, a space, `body`. A document is kept if a training or dev query refers to it: 109,739 documents. |
| `full/train_queries.tsv`, `full/dev_queries.tsv` | `nq_train_doc_newid.tsv` (307,373 rows), `nq_dev_doc_newid.tsv` (7,830 rows) | Query and relevant document id, unchanged. |
| `pseudo_queries.tsv` | `NQ_512_qg.tsv` | NCI's generated queries, 20 per document, unchanged. |

### TriviaQA

Source: the preprocessed TriviaQA files released with NCI.

| our file | built from | processing |
|---|---|---|
| `corpus.jsonl` | `trivia_title_cont.tsv` | The text ("title. content") is used as it is: 73,970 documents. |
| `full/{train,dev,test}_queries.tsv` | `train.tsv`, `dev.tsv`, `test.tsv` | Query and relevant document ids, unchanged. A query can have several relevant documents. |
| `pseudo_queries.tsv` | `trivia_512_qg.tsv` | NCI's generated queries, unchanged. |

### Seen/unseen splits of NQ320K and TriviaQA

These splits are ours.

1. Documents are ranked by `md5("42:<docid>")`. The first 60% (or 90%) are the seen set
   `d0_docs.txt`, the rest the unseen set `d1_docs.txt`.
2. All query files of the dataset are pooled (NQ320K: train and dev; TriviaQA: train, dev and
   test). Each query is placed by where its relevant documents fall:
   - all seen: `test_seen` (10,000 drawn at random), then `valid` (10% of the rest), then `train`;
   - all unseen: `test_unseen` (10,000 drawn at random, or all of them if there are fewer);
   - some seen and some unseen: dropped.

| split | documents | seen | unseen | train | valid | test_seen | test_unseen |
|---|---|---|---|---|---|---|---|
| NQ320K 60/40 | 109,739 | 65,843 | 43,896 | 162,077 | 18,009 | 10,000 | 10,000 |
| NQ320K 90/10 | 109,739 | 98,765 | 10,974 | 246,751 | 27,417 | 10,000 | 10,000 |
| TriviaQA 60/40 | 73,970 | 44,382 | 29,588 | 22,088 | 2,454 | 10,000 | 10,000 |
| TriviaQA 90/10 | 73,970 | 66,573 | 7,397 | 49,271 | 5,474 | 10,000 | 4,195 |

Train and valid counts are query-document pairs.

### MS MARCO (`msmarco9010`)

Source: the 90/10 split of the MS MARCO document ranking collection used by the DOME paper, and
the generated queries (docTTTTTquery) released with DDRO.

| our file | built from | processing |
|---|---|---|
| `corpus.jsonl` | the split's seen documents (291,880) and new documents (32,431) | 14 documents with empty text are dropped: 291,869 seen + 32,428 unseen = 324,297. |
| `seen90_unseen10/d0_docs.txt`, `d1_docs.txt` | the same two document sets | The split's seen/new assignment, unchanged. |
| `train_queries.tsv`, `valid_queries.tsv` | the split's training queries (330,452, official MS MARCO training queries with a seen relevant document) | Divided 90/10 by query id (seed 42): 297,397 train, 33,044 valid. The split's document-to-identifier training rows are not used. |
| `test_seen_queries.tsv`, `test_unseen_queries.tsv` | the split's two test sets | Unchanged: 4,675 queries with a seen relevant document, 518 with an unseen one. Together they are the official MS MARCO document dev set. |
| `pseudo_queries.tsv` | DDRO's generated queries | Seen documents only, the first 10 per document: 2,918,690 rows. |

Generated queries are selected at load time: a seeded draw of `n` per document, with documents
visited in sorted order, so the choice depends on `(n, seed)` only. Unseen documents never
receive generated queries.
