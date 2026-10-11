"""Train the retriever M2: the query encoder plus one weight per DocID bit.

The encoder starts from the DocID encoder M1 and the weights w start at 1, so before training
the retriever equals M1. DocIDs are fixed: only the query side is trained.

    u_q = w * z_q                    s(q, d) = b_d . u_q
    loss = InfoNCE over the DocIDs of all seen documents (u_q normalised, scale --logit_scale)

Training queries are the real queries plus --qg_per_doc generated queries per seen document.
Validation runs constrained beam search over the prefix tree of the seen documents; the
checkpoint with the best recall@1 is kept.

    python train.py --dataset nq320k --ckpt <docid_encoder.pt> --enc <docids.npz> \\
        --split_dir <split> --qg_per_doc 10 --real_weight 0 --save_dir <dir>
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
from corpus import doc_ids_texts, load_queries, load_dynamic_split, DATASETS  # noqa: E402
from trie import root_children                                          # noqa: E402
from checkpoint import load_encoder                                     # noqa: E402
from data import select_pseudo_queries                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq320k", choices=list(DATASETS))
    ap.add_argument("--ckpt", required=True, help="the trained DocID encoder (.pt with its sidecar)")
    ap.add_argument("--enc", required=True, help="the DocID table (.npz) from assign_docids.py")
    ap.add_argument("--split_dir", default=None,
                    help="seen/unseen split, the one the DocID encoder used; omit for the full corpus")
    ap.add_argument("--qg_per_doc", type=int, default=0,
                    help="generated queries per seen document, the same selection the DocID encoder uses")
    ap.add_argument("--real_weight", type=float, default=1.0,
                    help="sampling weight of a real query relative to a generated one; "
                         "0 balances the two groups 1:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=256, help="micro batch")
    ap.add_argument("--accum", type=int, default=2, help="micro batches per step")
    ap.add_argument("--lr_enc", type=float, default=2e-5, help="encoder")
    ap.add_argument("--lr_head", type=float, default=1e-3, help="per-bit weights")
    ap.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw")
    ap.add_argument("--logit_scale", type=float, default=20.0,
                    help="inverse temperature of the InfoNCE loss")
    ap.add_argument("--val_check_interval", type=float, default=0.5,
                    help="fraction of an epoch between validations")
    ap.add_argument("--eval_n", type=int, default=-1, help="validation pairs to use (-1 = all)")
    ap.add_argument("--eval_beam", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many validations without a better recall@1 (0 = never)")
    ap.add_argument("--max_seq_length", type=int, default=128)
    ap.add_argument("--save_dir", required=True)
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

    # DocIDs: codes (n_docs, T) tokens, bits (n_docs, n_bits) in +-1.
    d = np.load(a.enc)
    T, kary = int(d["n_tok"]), int(d["kary"]); B = int(round(math.log2(kary)))
    n_bits = T * B
    codes = d["doc_codes"].astype(np.int64)
    bits = (((codes[:, :, None] >> np.arange(B)) & 1) * 2 - 1).reshape(len(codes), n_bits)
    # S[v] = token v as +-1 bits, so (u_t @ S.T)[v] is the logit of token v.
    S = (((torch.arange(kary)[:, None] >> torch.arange(B)) & 1) * 2 - 1).float().to(dev)

    ids, _ = doc_ids_texts(a.dataset)
    assert len(ids) == len(codes), (
        f"{len(ids):,} documents in the corpus but {len(codes):,} rows in {a.enc}")
    id2i = {x: i for i, x in enumerate(ids)}

    # Real queries. seen = document indices of D0; empty means every document is seen.
    seen = set()
    if a.split_dir:
        parts, seen, _, _, _ = load_dynamic_split(a.split_dir, a.dataset, want=("train", "valid"))
        Q, dvQ = list(parts["train"]), list(parts["valid"])
    else:
        Q, dvQ = load_queries(a.dataset, "train", ids), load_queries(a.dataset, "dev", ids)
    n_real = len(Q)  # real pairs are a prefix of Q
    if a.qg_per_doc:
        chosen = select_pseudo_queries(a.dataset, a.qg_per_doc, a.seed)
        Q += [(t, id2i[r]) for r in sorted(chosen)
              if r in id2i and (not seen or id2i[r] in seen) for t in chosen[r]]
    if a.eval_n > 0:
        dvQ = dvQ[:a.eval_n]
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

    # Classes of the loss: the distinct DocIDs of the seen documents. Documents sharing a
    # DocID are one class. key only groups equal rows (it wraps around in int64).
    key = np.zeros(len(codes), dtype=np.int64)
    for t in range(T):
        key = key * kary + codes[:, t]
    uq, first_row, inv = np.unique(key, return_index=True, return_inverse=True)
    keep_cls = np.ones(len(uq), dtype=bool)
    doc2class = torch.from_numpy(inv)
    if seen:
        in_seen = np.zeros(len(codes), dtype=bool)
        in_seen[np.fromiter(sorted(seen), dtype=np.int64, count=len(seen))] = True
        keep_cls = np.zeros(len(uq), dtype=bool)
        keep_cls[inv[in_seen]] = True
        remap = -np.ones(len(uq), dtype=np.int64)
        remap[keep_cls] = np.arange(int(keep_cls.sum()))
        doc2class = torch.from_numpy(remap[inv])
    Sd = torch.from_numpy(bits[first_row[keep_cls]].astype(np.float32)).to(dev)
    Sd = Sd / math.sqrt(n_bits)  # unit-norm b_d
    print(f"[retriever] {len(codes):,} documents, InfoNCE over {len(Sd):,} seen DocIDs",
          flush=True)

    # Prefix tree over the seen documents: rows sorted by token sequence.
    keep = (np.fromiter(sorted(seen), dtype=np.int64, count=len(seen))
            if seen else np.arange(len(codes), dtype=np.int64))
    order = np.lexsort([codes[keep][:, t] for t in range(T - 1, -1, -1)])
    Cs_np = codes[keep][order]
    back = np.full(len(codes), -1, dtype=np.int64)  # document index -> tree row
    back[keep[order]] = np.arange(len(keep))
    ROOT_VALS, ROOT_ST, ROOT_EN = root_children(Cs_np)

    # The two trainable parts.
    model, n_tok_ck, kary_ck, _, _ = load_encoder(a.ckpt, None, a.max_seq_length)
    assert n_tok_ck == T and kary_ck == kary, \
        f"checkpoint is {n_tok_ck}x{kary_ck}, {a.enc} is {T}x{kary}"
    model = model.to(dev); model.train()
    head = BitWeights(n_bits).to(dev)
    groups = [{"params": model.parameters(), "lr": a.lr_enc},
              {"params": head.parameters(), "lr": a.lr_head}]
    opt = torch.optim.Adafactor(groups) if a.optimizer == "adafactor" else torch.optim.AdamW(groups)
    logit_scale = torch.tensor(float(a.logit_scale), device=dev)

    steps_per_epoch = max(1, N // (a.batch * a.accum))
    eval_every = max(1, int(round(steps_per_epoch * a.val_check_interval)))
    jid = os.environ.get("SLURM_JOB_ID", "")
    run = time.strftime("%Y%m%d-%H%M%S") + (f"_j{jid}" if jid else "")
    os.makedirs(a.save_dir, exist_ok=True)
    ck = os.path.join(a.save_dir, f"retriever_{run}.pt")
    print(f"[retriever] {a.optimizer} lr {a.lr_enc:g} (encoder) {a.lr_head:g} (bit weights) | "
          f"batch {a.batch}x{a.accum} | {steps_per_epoch:,} steps/epoch, validation every "
          f"{eval_every:,} steps | {os.path.basename(ck)}", flush=True)

    @torch.no_grad()
    def encode(texts, bs=256):
        return torch.cat([model(texts[i:i + bs], prompt_name="query")[1].float()
                          for i in range(0, len(texts), bs)])

    @torch.no_grad()
    def evaluate(beam):
        """Constrained beam search over the prefix tree, for every validation query."""
        model.eval(); head.eval()
        Z = encode(dtext)
        nq = len(Z)
        KS = tuple(k for k in (1, 5, 10, 20, 50, 100) if k <= beam)
        ranks = np.zeros(nq, dtype=np.int64)  # rank of the first relevant document, 0 = missed
        for qi in range(nq):
            # Token log-probabilities at every position: (T, kary). They do not depend on
            # the prefix, so one pass per query is enough.
            tok = torch.log_softmax(head(Z[qi]).view(T, B) @ S.T, dim=-1).cpu().numpy()
            # A beam item is a tree node, the row range [lo, hi), with its score sc.
            sc = np.zeros(1, dtype=np.float32)
            lo = np.zeros(1, dtype=np.int64)
            hi = np.full(1, len(Cs_np), dtype=np.int64)
            for t in range(T):
                if t == 0:
                    vals, st, en = ROOT_VALS, ROOT_ST, ROOT_EN
                    par = np.zeros(len(vals), dtype=np.int64)
                else:
                    # Children of every beam item: the distinct t-th tokens in its range.
                    vals_l, st_l, en_l, par_l = [], [], [], []
                    for j in range(len(lo)):
                        col = Cs_np[lo[j]:hi[j], t]
                        idx = np.flatnonzero(np.concatenate(([True], col[1:] != col[:-1])))
                        st = lo[j] + idx
                        vals_l.append(col[idx]); st_l.append(st)
                        en_l.append(np.concatenate((st[1:], [hi[j]])))
                        par_l.append(np.full(len(idx), j, dtype=np.int64))
                    vals = np.concatenate(vals_l).astype(np.int64)
                    st = np.concatenate(st_l); en = np.concatenate(en_l)
                    par = np.concatenate(par_l)
                sc = sc[par] + tok[t, vals]
                lo, hi = st, en
                if len(sc) > beam:
                    top = np.argpartition(-sc, beam)[:beam]
                    sc, lo, hi = sc[top], lo[top], hi[top]
            ranked = lo[np.argsort(-sc, kind="stable")]
            hit = np.flatnonzero(np.isin(ranked, back[np.array(dgolds[qi])]))
            ranks[qi] = hit[0] + 1 if len(hit) else 0
        m = {f"recall@{k}": float(((ranks > 0) & (ranks <= k)).sum()) / nq for k in KS}
        m[f"MRR@{beam}"] = float(np.where((ranks > 0) & (ranks <= beam),
                                          1.0 / np.maximum(ranks, 1), 0).sum()) / nq
        model.train(); head.train()
        return m

    def save(step, m, tag):
        torch.save({"enc": model.state_dict(), "head": head.state_dict(),
                    "args": vars(a), "step": step, "dev": m, "run_name": run}, ck)
        print(f"    saved {tag} -> {os.path.basename(ck)} (step {step}, "
              f"recall@1 {m['recall@1']:.4f})", flush=True)

    t0 = time.time()
    best = evaluate(a.eval_beam)
    print("  [dev] step 0 (retriever == DocID encoder)  " +
          "  ".join(f"{k} {v:.4f}" for k, v in best.items()), flush=True)
    save(0, best, "step-0 baseline")
    evals_no_improve = 0

    # Sampling weights: real queries are drawn more often than generated ones.
    qw, n_gen = None, N - n_real
    if a.real_weight != 1.0 and n_real and n_gen:
        w = a.real_weight if a.real_weight != 0 else n_gen / n_real  # 0 = balance 1:1
        qw = torch.ones(N)
        qw[:n_real] = w
        print(f"[retriever] real queries sampled with weight {w:.2f}: "
              f"{w * n_real / (w * n_real + n_gen):.0%} of each batch", flush=True)

    for step in range(1, a.steps + 1):
        opt.zero_grad()
        tot = 0.0
        for _ in range(a.accum):
            i = (torch.randint(0, N, (a.batch,)) if qw is None else
                 torch.multinomial(qw, a.batch, replacement=True))
            z = model([qtext[j] for j in i], prompt_name="query")[1].float()
            u = head(z)
            u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
            l = F.cross_entropy(logit_scale * (u @ Sd.T), doc2class[qgold[i]].to(dev))
            (l / a.accum).backward(); tot += float(l.detach()) / a.accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad],
            1.0)
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:7d}  loss {tot:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if step % eval_every == 0 or step == a.steps:
            m = evaluate(a.eval_beam)
            print(f"  [dev] step {step}  " +
                  "  ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
            if m["recall@1"] > best["recall@1"]:
                evals_no_improve = 0
                best = m; save(step, m, "new best")
            else:
                evals_no_improve += 1
                print(f"    not better than {best['recall@1']:.4f}, keeping the saved one"
                      + (f" [{evals_no_improve}/{a.patience}]" if a.patience else ""),
                      flush=True)
                if a.patience and evals_no_improve >= a.patience:
                    print(f"[retriever] early stop: no recall@1 improvement for "
                          f"{a.patience} evaluations", flush=True)
                    break
    print(f"[retriever] best validation recall@1 {best['recall@1']:.4f} -> {os.path.basename(ck)}")


if __name__ == "__main__":
    main()
