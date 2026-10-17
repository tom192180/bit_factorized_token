"""Train the retriever M2: the query encoder plus one weight per DocID bit.

The encoder starts from the DocID encoder M1 and the weights w start at 1, so before training
the retriever equals M1. DocIDs are fixed: only the query side is trained.

    u_q = w * z_q                    s(q, d) = b_d . u_q
    loss = InfoNCE over the DocIDs of all seen documents (u_q normalised, scale LOGIT_SCALE)

Training queries are the real queries plus --qg_per_doc generated queries per seen document,
sampled so that real and generated queries each make up half of a batch. Validation runs
constrained beam search over the prefix tree of the seen documents; the checkpoint with the
best recall@1 is kept. Hyperparameters live in config.py.

    python train.py --dataset nq320k --ckpt <docid_encoder.pt> --enc <docids.npz> \\
        --split_dir <split> --qg_per_doc 10 --save_dir <dir>
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(HERE)
for _p in (_SRC, os.path.join(_SRC, "docid_encoder"), HERE):  # HERE ends up first
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
from head import BitWeights                                             # noqa: E402
from config import (STEPS, BATCH, ACCUM, LR_ENC_ADAMW, LR_ENC_ADAFACTOR, LR_HEAD,  # noqa: E402
                    MAX_GRAD_NORM, LOGIT_SCALE, VAL_CHECK_INTERVAL, EVAL_BEAM, PATIENCE)
from corpus import doc_ids_texts, load_queries, load_dynamic_split, DATASETS  # noqa: E402
from trie import root_children                                          # noqa: E402
from checkpoint import load_encoder                                     # noqa: E402
from data import select_pseudo_queries                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train the retriever.")
    ap.add_argument("--dataset", default="nq320k", choices=list(DATASETS))
    ap.add_argument("--ckpt", required=True, help="the trained DocID encoder (.pt with its sidecar)")
    ap.add_argument("--enc", required=True, help="the DocID table (.npz) from assign_docids.py")
    ap.add_argument("--split_dir", default=None,
                    help="seen/unseen split, the one the DocID encoder used; omit for the full corpus")
    ap.add_argument("--qg_per_doc", type=int, default=0,
                    help="generated queries per seen document (0 = real queries only)")
    ap.add_argument("--max_seq_length", type=int, default=128)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # The split must be the one the DocID encoder was trained on (recorded in its sidecar).
    side = {}
    if os.path.exists(a.ckpt + ".arch.json"):
        with open(a.ckpt + ".arch.json") as fh:
            side = json.load(fh)
    enc_d1 = side.get("d1_sha256")
    if a.split_dir:
        with open(os.path.join(a.split_dir, "manifest.json")) as fh:
            d1 = json.load(fh)["sha256"]["d1_docs.txt"]
        if enc_d1 and enc_d1 != d1:
            raise SystemExit(f"SPLIT MISMATCH: the DocID encoder was trained with unseen-document "
                             f"digest {enc_d1}, but {a.split_dir} has {d1}")
        if not enc_d1:
            print("[retriever] WARNING: the encoder sidecar records no split; not verified",
                  flush=True)
    elif enc_d1:
        raise SystemExit(f"the DocID encoder at {a.ckpt} was trained on a seen/unseen split; "
                         f"pass --split_dir {side.get('split_dir', '<the same directory>')}")

    # DocIDs, in the paper's notation: T tokens of m bits, k = T*m bits, V = 2^m token values.
    #   C      (n_docs, T)   row d = the tokens c_1..c_T of document d, each in [0, 2^m)
    #   B_all  (n_docs, k)   row d = b_d, the DocID as +-1 bits (all documents)
    d = np.load(a.enc)
    T, V = int(d["n_tok"]), int(d["kary"]); m = int(round(math.log2(V)))
    k = T * m
    C = d["doc_codes"].astype(np.int64)
    B_all = (((C[:, :, None] >> np.arange(m)) & 1) * 2 - 1).reshape(len(C), k)
    #   P      (2^m, m)      row v = p_v, the bits of token value v as +-1; (P @ u_t)[v] = p_v . u_t
    P = (((torch.arange(V)[:, None] >> torch.arange(m)) & 1) * 2 - 1).float().to(dev)

    ids, _ = doc_ids_texts(a.dataset)
    assert len(ids) == len(C), (
        f"{len(ids):,} documents in the corpus but {len(C):,} rows in {a.enc}")
    id2i = {x: i for i, x in enumerate(ids)}

    # Real queries. D0 = indices of the seen documents; empty means every document is seen.
    D0 = set()
    if a.split_dir:
        parts, D0, _, _, _ = load_dynamic_split(a.split_dir, a.dataset, want=("train", "valid"))
        Q, dvQ = list(parts["train"]), list(parts["valid"])
    else:
        Q, dvQ = load_queries(a.dataset, "train", ids), load_queries(a.dataset, "dev", ids)
    n_real = len(Q)  # real pairs are a prefix of Q
    if a.qg_per_doc:
        chosen = select_pseudo_queries(a.dataset, a.qg_per_doc, a.seed)
        Q += [(t, id2i[r]) for r in sorted(chosen)
              if r in id2i and (not D0 or id2i[r] in D0) for t in chosen[r]]
    qtext = [t for t, _ in Q]
    qgold = torch.tensor([g for _, g in Q])  # document index
    N = len(qtext)
    # Validation queries, each with the list of its relevant documents.
    by_query = {}
    for t, g in dvQ:
        by_query.setdefault(t, []).append(g)
    dtext = list(by_query)
    dgolds = [by_query[t] for t in dtext]
    print(f"[retriever] train {n_real:,} real + {N - n_real:,} generated pairs | "
          f"validation {len(dtext):,} queries", flush=True)

    # Classes of the loss: the distinct DocIDs of the seen documents (D0). Documents sharing a
    # DocID are one class. key only groups equal rows (it wraps around in int64).
    key = np.zeros(len(C), dtype=np.int64)
    for t in range(T):
        key = key * V + C[:, t]
    uniq, first_row, inv = np.unique(key, return_index=True, return_inverse=True)
    keep_cls = np.ones(len(uniq), dtype=bool)
    doc2class = torch.from_numpy(inv)
    if D0:
        in_D0 = np.zeros(len(C), dtype=bool)
        in_D0[np.fromiter(sorted(D0), dtype=np.int64, count=len(D0))] = True
        keep_cls = np.zeros(len(uniq), dtype=bool)
        keep_cls[inv[in_D0]] = True
        remap = -np.ones(len(uniq), dtype=np.int64)
        remap[keep_cls] = np.arange(int(keep_cls.sum()))
        doc2class = torch.from_numpy(remap[inv])
    B = torch.from_numpy(B_all[first_row[keep_cls]].astype(np.float32)).to(dev)
    B = B / math.sqrt(k)  # B: one row b_d / sqrt(k) per class, the paper's B scaled to unit rows
    print(f"[retriever] {len(C):,} documents, InfoNCE over {len(B):,} seen DocIDs",
          flush=True)

    # Prefix tree over D0: the rows of C sorted by token sequence (C_tree); a node is a row range.
    keep = (np.fromiter(sorted(D0), dtype=np.int64, count=len(D0))
            if D0 else np.arange(len(C), dtype=np.int64))
    order = np.lexsort([C[keep][:, t] for t in range(T - 1, -1, -1)])
    C_tree = C[keep][order]
    back = np.full(len(C), -1, dtype=np.int64)  # document index -> tree row
    back[keep[order]] = np.arange(len(keep))
    ROOT_VALS, ROOT_ST, ROOT_EN = root_children(C_tree)

    # The two trainable parts. The optimizer follows the backbone: AdamW (BERT), Adafactor (T5).
    model, n_tok_ck, kary_ck, sbert_model, _ = load_encoder(a.ckpt, None, a.max_seq_length)
    assert n_tok_ck == T and kary_ck == V, \
        f"checkpoint is {n_tok_ck}x{kary_ck}, {a.enc} is {T}x{V}"
    model = model.to(dev); model.train()
    head = BitWeights(k).to(dev)  # w, initialised to 1
    adafactor = "t5" in sbert_model.lower()
    lr_enc = LR_ENC_ADAFACTOR if adafactor else LR_ENC_ADAMW
    groups = [{"params": model.parameters(), "lr": lr_enc},
              {"params": head.parameters(), "lr": LR_HEAD}]
    opt = torch.optim.Adafactor(groups) if adafactor else torch.optim.AdamW(groups)
    logit_scale = torch.tensor(LOGIT_SCALE, device=dev)

    steps_per_epoch = max(1, N // (BATCH * ACCUM))
    eval_every = max(1, int(round(steps_per_epoch * VAL_CHECK_INTERVAL)))
    jid = os.environ.get("SLURM_JOB_ID", "")
    run = time.strftime("%Y%m%d-%H%M%S") + (f"_j{jid}" if jid else "")
    os.makedirs(a.save_dir, exist_ok=True)
    ck = os.path.join(a.save_dir, f"retriever_{run}.pt")
    print(f"[retriever] {'adafactor' if adafactor else 'adamw'} lr {lr_enc:g} (encoder) "
          f"{LR_HEAD:g} (bit weights) | batch {BATCH}x{ACCUM} | {steps_per_epoch:,} steps/epoch, "
          f"validation every {eval_every:,} steps | {os.path.basename(ck)}", flush=True)

    @torch.no_grad()
    def encode(texts, bs=256):
        return torch.cat([model(texts[i:i + bs])[1].float()
                          for i in range(0, len(texts), bs)])

    @torch.no_grad()
    def evaluate(beam):
        """Constrained beam search over the prefix tree, for every validation query."""
        model.eval(); head.eval()
        Zq = encode(dtext)
        nq = len(Zq)
        KS = tuple(n for n in (1, 5, 10, 20, 50, 100) if n <= beam)
        ranks = np.zeros(nq, dtype=np.int64)  # rank of the first relevant document, 0 = missed
        for qi in range(nq):
            # logp[t, v] = log p_t(v | q) = p_v . u_t - R_t(q), for every position t and value v
            # (Eq. 8). Rows of head(z_q).view(T, m) are u_1..u_T. No term depends on the prefix,
            # so one pass per query is enough.
            logp = torch.log_softmax(head(Zq[qi]).view(T, m) @ P.T, dim=-1).cpu().numpy()
            # A beam item is a tree node, the row range [lo, hi), with its score sc.
            sc = np.zeros(1, dtype=np.float32)
            lo = np.zeros(1, dtype=np.int64)
            hi = np.full(1, len(C_tree), dtype=np.int64)
            for t in range(T):
                if t == 0:
                    vals, st, en = ROOT_VALS, ROOT_ST, ROOT_EN
                    par = np.zeros(len(vals), dtype=np.int64)
                else:
                    # Children of every beam item: the distinct t-th tokens in its range.
                    vals_l, st_l, en_l, par_l = [], [], [], []
                    for j in range(len(lo)):
                        col = C_tree[lo[j]:hi[j], t]
                        idx = np.flatnonzero(np.concatenate(([True], col[1:] != col[:-1])))
                        st = lo[j] + idx
                        vals_l.append(col[idx]); st_l.append(st)
                        en_l.append(np.concatenate((st[1:], [hi[j]])))
                        par_l.append(np.full(len(idx), j, dtype=np.int64))
                    vals = np.concatenate(vals_l).astype(np.int64)
                    st = np.concatenate(st_l); en = np.concatenate(en_l)
                    par = np.concatenate(par_l)
                sc = sc[par] + logp[t, vals]
                lo, hi = st, en
                if len(sc) > beam:
                    top = np.argpartition(-sc, beam)[:beam]
                    sc, lo, hi = sc[top], lo[top], hi[top]
            ranked = lo[np.argsort(-sc, kind="stable")]
            hit = np.flatnonzero(np.isin(ranked, back[np.array(dgolds[qi])]))
            ranks[qi] = hit[0] + 1 if len(hit) else 0
        # "recall@n" is the paper's Hit@n: the share of queries with at least one relevant
        # document in the top n (from the first relevant rank). The key keeps its old name.
        res = {f"recall@{n}": float(((ranks > 0) & (ranks <= n)).sum()) / nq for n in KS}
        res[f"MRR@{beam}"] = float(np.where((ranks > 0) & (ranks <= beam),
                                          1.0 / np.maximum(ranks, 1), 0).sum()) / nq
        model.train(); head.train()
        return res

    def save(step, res, tag):
        torch.save({"enc": model.state_dict(), "head": head.state_dict(),
                    "args": vars(a), "step": step, "dev": res}, ck)
        print(f"    saved {tag} -> {os.path.basename(ck)} (step {step}, "
              f"recall@1 {res['recall@1']:.4f})", flush=True)

    t0 = time.time()
    best = evaluate(EVAL_BEAM)
    print("  [dev] step 0 (retriever == DocID encoder)  " +
          "  ".join(f"{name} {val:.4f}" for name, val in best.items()), flush=True)
    save(0, best, "step-0 baseline")
    evals_no_improve = 0

    # Sampling weights: real queries are drawn more often, so that real and generated queries
    # each make up half of a batch.
    qw, n_gen = None, N - n_real
    if n_real and n_gen:
        w_real = n_gen / n_real
        qw = torch.ones(N)
        qw[:n_real] = w_real
        print(f"[retriever] real queries sampled with weight {w_real:.2f}: "
              f"{w_real * n_real / (w_real * n_real + n_gen):.0%} of each batch", flush=True)

    for step in range(1, STEPS + 1):
        opt.zero_grad()
        tot = 0.0
        for _ in range(ACCUM):
            i = (torch.randint(0, N, (BATCH,)) if qw is None else
                 torch.multinomial(qw, BATCH, replacement=True))
            z_q = model([qtext[j] for j in i])[1].float()
            u_q = head(z_q)                                            # u_q = w * z_q (Eq. 4)
            u_q = u_q / u_q.norm(dim=1, keepdim=True).clamp(min=1e-9)
            # logits = s(q, d) / (tau_2 sqrt(k) ||u_q||) for every seen DocID (Eq. 6), 1/tau_2 = 20
            l = F.cross_entropy(logit_scale * (u_q @ B.T), doc2class[qgold[i]].to(dev))
            (l / ACCUM).backward(); tot += float(l.detach()) / ACCUM
        torch.nn.utils.clip_grad_norm_(
            [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad],
            MAX_GRAD_NORM)
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:7d}  loss {tot:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if step % eval_every == 0 or step == STEPS:
            res = evaluate(EVAL_BEAM)
            print(f"  [dev] step {step}  " +
                  "  ".join(f"{name} {val:.4f}" for name, val in res.items()), flush=True)
            if res["recall@1"] > best["recall@1"]:
                evals_no_improve = 0
                best = res; save(step, res, "new best")
            else:
                evals_no_improve += 1
                print(f"    not better than {best['recall@1']:.4f}, keeping the saved one "
                      f"[{evals_no_improve}/{PATIENCE}]", flush=True)
                if evals_no_improve >= PATIENCE:
                    print(f"[retriever] early stop: no recall@1 improvement for "
                          f"{PATIENCE} evaluations", flush=True)
                    break
    print(f"[retriever] best validation recall@1 {best['recall@1']:.4f} -> {os.path.basename(ck)}")


if __name__ == "__main__":
    main()
