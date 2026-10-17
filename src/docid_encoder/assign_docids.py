"""Compute the DocID of every document with a trained DocID encoder, and save them.

Writes one .npz file:
    doc_z      (n_docs, k)   encoder output after tanh, in (-1, 1)
    doc_codes  (n_docs, T)   the DocID, as T token values (k bits read m at a time)
    qry_z, qry_codes, gold   the same for one query set, with each query's document index
    doc_x, qry_x             the embeddings before projection (only with --with_x / nq320k)

Row i belongs to document i of corpus.doc_ids_texts(dataset).

    python assign_docids.py --ckpt <encoder.pt> --tag <name> --dataset nq320k --out docids.npz
"""
import argparse, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
from checkpoint import load_encoder                                     # noqa: E402
from codes import pack_bits                                             # noqa: E402
from corpus import doc_ids_texts, load_queries, DATASETS                # noqa: E402


@torch.no_grad()
def encode(model, texts, bs, device, n_levels, want_x=True):
    """(X, Z, C) for `texts`; X is None when want_x is False."""
    X, Z, C = [], [], []
    for i in range(0, len(texts), bs):
        x, z, bits = model(texts[i:i + bs])
        idx = pack_bits(bits, n_levels)
        if want_x:
            X.append(x.float().cpu().numpy())
        Z.append(z.float().cpu().numpy())
        C.append(idx.cpu().numpy())
        if (i // bs) % 100 == 0 and i:
            print(f"    {i:,}/{len(texts):,}", flush=True)
    return (np.concatenate(X) if want_x else None,
            np.concatenate(Z), np.concatenate(C))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="nq320k", choices=list(DATASETS),
                    help="Which corpus to encode.")
    ap.add_argument("--query_split", default=None,
                    help="Query split to encode alongside the docs.")
    ap.add_argument("--with_x", action="store_true",
                    help="Save doc_x/qry_x (hidden_dim fp32).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, n_tok, kary, _, n_levels = load_encoder(a.ckpt, None, a.max_seq_length)
    model = model.to(device).eval()

    split = a.query_split or ("dev" if a.dataset == "nq320k" else "train")
    with_x = a.with_x or a.dataset == "nq320k"

    # DOCUMENTS.
    ids, texts = doc_ids_texts(a.dataset)
    print(f"DOCS [{a.dataset}]: {len(ids):,} documents")
    dX, dZ, dC = encode(model, texts, a.batch_size, device, n_levels, with_x)

    # QUERIES.
    if split == "none":
        # msmarco300k ships a corpus and NOTHING else.
        print("QUERIES: skipped (--query_split none)")
        qZ = np.zeros((0,) + dZ.shape[1:], dtype=np.float32)
        qC = np.zeros((0,) + dC.shape[1:], dtype=np.int32)
        qX = np.zeros((0,) + dX.shape[1:], dtype=np.float32) if with_x else None
        gold = np.zeros((0,), dtype=np.int64)
    else:
        id2i = {d: i for i, d in enumerate(ids)}
        pairs = load_queries(a.dataset, split, ids)
        print(f"QUERIES [{split}]: {len(pairs):,} with a gold document in the corpus")
        qX, qZ, qC = encode(model, [t for t, _ in pairs], a.batch_size, device,
                            n_levels, with_x)
        gold = np.array([g for _, g in pairs], dtype=np.int64)

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"enc_{a.tag}.npz")
    blob = dict(doc_z=dZ.astype(np.float32), doc_codes=dC.astype(np.int32),
                qry_z=qZ.astype(np.float32), qry_codes=qC.astype(np.int32),
                gold=gold, n_tok=np.int64(n_tok), kary=np.int64(kary))
    if with_x:
        blob["doc_x"] = dX.astype(np.float32)
        blob["qry_x"] = qX.astype(np.float32)
    else:
        hid = getattr(model, "code_dim", 768)
        print(f"  doc_x/qry_x NOT COMPUTED ({a.dataset}): would be ~"
              f"{len(ids)*768*4/2**30:.1f} GB of host RAM. eval_gen.py needs them; nothing "
              f"else does. Pass --with_x to keep them.")
    np.savez(out, **blob)
    print(f"  doc_z {dZ.shape}  codes {dC.shape}")
    print(f"  qry_z {qZ.shape}  gold {gold.shape}")
    print(f"  -> {out}  ({os.path.getsize(out)/2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
