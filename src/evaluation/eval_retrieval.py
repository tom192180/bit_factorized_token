"""Evaluate a DocID encoder or a retriever with constrained beam search, tie-aware.

Two kinds of rows are printed:
    beam <b>              one prefix tree, beam width b
    MULTI k=<k> beam <b>  k prefix trees over permuted bit orders each propose candidates,
                          which are then ranked by the full score s(q, d) = b_d . u_q

Documents that share a DocID get the same score. They are scored tie-aware: a metric is its
expected value over all orderings of the tied documents.

    python eval_retrieval.py --dataset nq320k --ckpt <retriever.pt> --stage1 <docid_encoder.pt> \\
        --enc <docids.npz> --split_dir <split> --split_file test_unseen
    python eval_retrieval.py --dataset nq320k --ckpt <docid_encoder.pt> --stage1_only ...
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(HERE)
for _p in (_SRC, os.path.join(_SRC, "docid_encoder"), os.path.join(_SRC, "retriever"), HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
from head import BitWeights                               # noqa: E402
from trie import root_children                            # noqa: E402
from corpus import doc_ids_texts, load_queries, DATASETS  # noqa: E402
from checkpoint import load_encoder                       # noqa: E402

KS = (1, 5, 10, 20, 100)


def _spans(lo, hi):
    """Every row in the half-open tree ranges [lo, hi)."""
    n = hi - lo
    if len(n) == 0 or n.sum() == 0:
        return np.zeros(0, dtype=np.int64)
    return np.repeat(lo - np.concatenate(([0], np.cumsum(n)[:-1])), n) + np.arange(n.sum())


def _first_relevant(scs, rows, gold, gid, shared):
    """Tie-aware rank of the first relevant document among the scored `rows`.

    Returns (ranks, probabilities): the possible ranks and how likely each is when tied
    documents are ordered at random. None if no relevant document was retrieved."""
    if not len(rows):
        return None
    scs = np.asarray(scs, dtype=np.float64)
    sh = shared[rows]
    if sh.any():
        # Documents with the same DocID must tie exactly.
        scs = scs.copy()
        g_r = gid[rows]
        for gg in np.unique(g_r[sh]):
            m = g_r == gg
            scs[m] = scs[m].max()
    o = np.argsort(-scs, kind="stable")
    s, isg = scs[o], np.isin(rows[o], list(gold))
    hit = np.flatnonzero(isg)
    if not len(hit):
        return None
    a = int(np.searchsorted(-s, -s[hit[0]], side="left"))  # strictly better documents
    b = int(np.searchsorted(-s, -s[hit[0]], side="right"))  # end of the tied block
    k, g = b - a, int(isg[a:b].sum())
    j = np.arange(1, k - g + 2)
    p = np.array([math.comb(k - x, g - 1) for x in j], dtype=np.float64) / math.comb(k, g)
    return a + j, p


def _metrics(ranks, cands):
    """Hit@k, MRR@100 and MRR@10 as expectations over the tie-aware ranks."""
    n = len(ranks)
    hits = [d for d in ranks if d is not None]
    m = {f"@{k}": float(sum(p[r <= k].sum() for r, p in hits)) / n for k in KS}
    m["MRR"] = float(sum((p * np.where(r <= max(KS), 1.0 / r, 0.0)).sum() for r, p in hits)) / n
    m["MRR@10"] = float(sum((p * np.where(r <= 10, 1.0 / r, 0.0)).sum() for r, p in hits)) / n
    m["cands"] = float(np.mean(cands))  # documents scored per query
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq320k", choices=list(DATASETS))
    ap.add_argument("--ckpt", required=True, help="retriever checkpoint, or a DocID encoder with --stage1_only")
    ap.add_argument("--stage1", default=None, help="the DocID encoder checkpoint (gives the architecture)")
    ap.add_argument("--stage1_only", action="store_true",
                    help="--ckpt is a DocID encoder: score it directly, with all bit weights 1")
    ap.add_argument("--enc", required=True, help="the DocID table (.npz) from assign_docids.py")
    ap.add_argument("--split", default="dev", help="full setting: query set to score (dev, test)")
    ap.add_argument("--split_dir", default=None, help="seen/unseen split folder")
    ap.add_argument("--split_file", default="test_unseen",
                    choices=["test_unseen", "test_seen", "valid", "train"],
                    help="which <name>_queries.tsv of the split to score")
    ap.add_argument("--restrict", default="",
                    help="file of docids, one per line: index only these documents")
    ap.add_argument("--beam", type=int, default=100, help="beam width of the single-tree row")
    ap.add_argument("--multi", default="8:100", help="multi-tree rows as k:beam pairs, e.g. '8:100,4:100'")
    ap.add_argument("--eval_n", type=int, default=-1, help="query-document pairs to use (-1 = all)")
    ap.add_argument("--max_seq_length", type=int, default=128)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="seed of the bit permutations")
    a = ap.parse_args()
    if not a.stage1_only and not a.stage1:
        raise SystemExit("--stage1 <DocID encoder checkpoint> is required unless --stage1_only")
    multi = [tuple(int(x) for x in p.split(":")) for p in a.multi.split(",")] if a.multi else []
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # DocIDs: codes (n_docs, T) tokens of B bits.
    d = np.load(a.enc)
    T, kary = int(d["n_tok"]), int(d["kary"]); B = int(round(math.log2(kary)))
    n_bits = T * B
    codes = d["doc_codes"].astype(np.int64)
    # S[v] = token v as +-1 bits, so (u_t @ S.T)[v] is the logit of token v.
    S = (((torch.arange(kary)[:, None] >> torch.arange(B)) & 1) * 2 - 1).float().to(dev)
    Snp = S.cpu().numpy()

    print(f"[eval] {os.path.basename(a.ckpt)}  dataset={a.dataset}  split={a.split}")
    head = BitWeights(n_bits).to(dev)  # all ones for a DocID encoder
    if a.stage1_only:
        model, nt, kk, _, _ = load_encoder(a.ckpt, None, a.max_seq_length)
    else:
        blob = torch.load(a.ckpt, map_location="cpu")
        print(f"[eval] step {blob['step']:,} | validation " +
              "  ".join(f"{k} {v:.4f}" for k, v in blob["dev"].items()), flush=True)
        model, nt, kk, _, _ = load_encoder(a.stage1, None, a.max_seq_length)
        model.load_state_dict(blob["enc"])
        if set(blob["head"]) != {"zs"}:
            raise SystemExit(f"{a.ckpt} does not hold per-bit weights only")
        head.load_state_dict(blob["head"])
    assert nt == T and kk == kary, f"checkpoint is {nt}x{kk}, {a.enc} is {T}x{kary}"
    model = model.to(dev).eval(); head.eval()

    ids, _ = doc_ids_texts(a.dataset)
    assert len(ids) == len(codes), (
        f"{len(ids):,} documents in the corpus but {len(codes):,} rows in {a.enc}")
    if a.restrict:
        with open(a.restrict, encoding="utf-8") as fh:
            keep = {l.split("\t")[0].strip() for l in fh if l.strip()}
        sel = [i for i, x in enumerate(ids) if x in keep]
        if len(sel) != len(keep):
            raise SystemExit(f"--restrict lists {len(keep)-len(sel):,} docids not in the corpus")
        ids = [ids[i] for i in sel]
        codes = codes[sel]
        print(f"[eval] index restricted to {len(ids):,} documents "
              f"({os.path.basename(a.restrict)})", flush=True)

    # Queries: [(query_text, doc_index)].
    if a.split_dir:
        with open(os.path.join(a.split_dir, "manifest.json")) as fh:
            man = json.load(fh)
        if man["dataset"] != a.dataset:
            raise SystemExit(f"{a.split_dir} is a {man['dataset']} split, not {a.dataset}")
        # The split must be the one the checkpoint was trained on (recorded in its sidecar).
        want = None
        if os.path.exists(a.ckpt + ".arch.json"):
            with open(a.ckpt + ".arch.json") as fh:
                want = json.load(fh).get("d1_sha256")
        if want and want != man["sha256"]["d1_docs.txt"]:
            raise SystemExit(f"SPLIT MISMATCH: the checkpoint was trained with unseen-document "
                             f"digest {want}, but {a.split_dir} has {man['sha256']['d1_docs.txt']}")
        i2 = {x: i for i, x in enumerate(ids)}
        dv = []
        with open(os.path.join(a.split_dir, f"{a.split_file}_queries.tsv")) as fh:
            for ln in fh:
                f = ln.rstrip("\n").split("\t")
                if len(f) < 2 or not f[0]:
                    continue
                dv += [(f[0], i2[x]) for x in f[1].split(",") if x in i2]
        if not dv:
            raise SystemExit(f"no query in {a.split_file}_queries.tsv has a relevant document "
                             f"inside the index")
        # Relevant documents must be unseen for test_unseen and seen otherwise.
        side = "d1" if a.split_file == "test_unseen" else "d0"
        with open(os.path.join(a.split_dir, f"{side}_docs.txt")) as fh:
            need = {i2[x] for x in (l.split("\t")[0].strip() for l in fh if l.strip()) if x in i2}
        bad = {g for _, g in dv} - need
        if bad:
            raise SystemExit(f"{len(bad):,} relevant documents of {a.split_file} are not in "
                             f"{side}_docs.txt: the split folder is inconsistent")
        print(f"[eval] {a.split_dir} | {a.split_file}"
              + ("" if want else "  [checkpoint records no split: not verified]"), flush=True)
    else:
        dv = load_queries(a.dataset, a.split, ids)
    if a.eval_n > 0:
        dv = dv[:a.eval_n]
    by_query = {}
    for t, g in dv:
        by_query.setdefault(t, []).append(g)
    texts = list(by_query)
    dgolds = [by_query[t] for t in texts]
    print(f"[eval] {len(ids):,} documents | {len(dv):,} pairs over {len(texts):,} queries",
          flush=True)

    # Prefix trees. Tree 0 uses the bits in their original order; each other tree permutes the
    # bits and regroups them into tokens. A tree is (permutation, sorted tokens, row -> document
    # index, root children).
    dbits = ((codes[:, :, None] >> np.arange(B)) & 1).reshape(len(codes), n_bits)
    rng = np.random.default_rng(a.seed)
    n_trees = max([k for k, _ in multi], default=1)
    PERMS = [np.arange(n_bits)] + [rng.permutation(n_bits) for _ in range(n_trees - 1)]
    TREES = []
    for pi in PERMS:
        cb = dbits[:, pi]
        cc = np.zeros((len(cb), T), dtype=np.int64)
        for t in range(T):
            cc[:, t] = (cb[:, t * B:(t + 1) * B] * (1 << np.arange(B))).sum(1)
        o = np.lexsort([cc[:, t] for t in range(T - 1, -1, -1)])
        TREES.append((pi, cc[o], o) + root_children(cc[o]))

    # Rows of tree 0 are the reference order: `back` maps a document index to its row.
    order = np.lexsort([codes[:, t] for t in range(T - 1, -1, -1)])
    Cs_np = codes[order]; back = np.argsort(order)
    ROOT_VALS, ROOT_ST, ROOT_EN = root_children(Cs_np)
    gold_rows = [set(back[np.array(g)].tolist()) for g in dgolds]

    # Documents sharing a DocID: GID = group of each row, SHARED = the group has several rows.
    _, gid, gcnt = np.unique(Cs_np, axis=0, return_inverse=True, return_counts=True)
    GID = np.asarray(gid).ravel()
    SHARED = gcnt[GID] > 1
    print(f"[eval] shared DocIDs: {int(SHARED.sum()):,} documents in "
          f"{int((gcnt > 1).sum()):,} groups (max {int(gcnt.max())})", flush=True)

    with torch.no_grad():
        Z = torch.cat([model(texts[i:i + a.bs])[1].float()
                       for i in range(0, len(texts), a.bs)])

    @torch.no_grad()
    def token_logprobs(z):
        """(T, kary): log-probability of every token at every position, for one query."""
        return torch.log_softmax(head(z).view(T, B) @ S.T, dim=-1)

    @torch.no_grad()
    def score(z, rows):
        """Full score of the documents at `rows`: their summed token log-probabilities."""
        if len(rows) == 0:
            return np.zeros(0, dtype=np.float32)
        tgt = torch.from_numpy(Cs_np[rows]).to(dev)  # (n, T) tokens
        return token_logprobs(z)[torch.arange(T, device=dev)[None, :], tgt].sum(1).cpu().numpy()

    def expand(Cs, lo, hi, t):
        """Children at depth t of the tree nodes [lo, hi): (token, start, end, parent)."""
        vl, sl, el, pl = [], [], [], []
        for j in range(len(lo)):
            col = Cs[lo[j]:hi[j], t]
            idx = np.flatnonzero(np.concatenate(([True], col[1:] != col[:-1])))
            st = lo[j] + idx
            vl.append(col[idx]); sl.append(st)
            el.append(np.concatenate((st[1:], [hi[j]])))
            pl.append(np.full(len(idx), j, dtype=np.int64))
        return (np.concatenate(vl).astype(np.int64), np.concatenate(sl),
                np.concatenate(el), np.concatenate(pl))

    def run_single(beam):
        """Beam search over tree 0; the retrieved documents keep their beam scores."""
        ranks = [None] * len(Z)
        ncand = np.zeros(len(Z), dtype=np.float64)
        for qi in range(len(Z)):
            tok = token_logprobs(Z[qi]).cpu().numpy()
            sc = np.zeros(1, dtype=np.float32)
            lo = np.zeros(1, dtype=np.int64)
            hi = np.full(1, len(Cs_np), dtype=np.int64)
            for t in range(T):
                if len(lo) == 0:
                    break
                if t == 0:
                    vals, st, en = ROOT_VALS, ROOT_ST, ROOT_EN
                    par = np.zeros(len(vals), dtype=np.int64)
                else:
                    vals, st, en, par = expand(Cs_np, lo, hi, t)
                sc = sc[par] + tok[t, vals]
                lo, hi = st, en
                if len(sc) > beam:
                    keep = np.argpartition(-sc, beam)[:beam]
                    sc, lo, hi = sc[keep], lo[keep], hi[keep]
            # A leaf holding several rows is a shared DocID: all its documents are retrieved.
            rows = _spans(lo, hi)
            ranks[qi] = _first_relevant(np.repeat(sc, hi - lo), rows, gold_rows[qi], GID, SHARED)
            ncand[qi] = len(rows)
        return _metrics(ranks, ncand)

    def run_multi(beam, k):
        """Beam search over k trees; the union of their documents is ranked by the full score."""
        zs_np = head.zs.detach().cpu().numpy()
        ranks = [None] * len(Z)
        ncand = np.zeros(len(Z), dtype=np.float64)
        for qi in range(len(Z)):
            zq = Z[qi].cpu().numpy()
            cand = set()
            for pi, Cs_p, row2doc, rv_p, rs_p, re_p in TREES[:k]:
                g = (zq * zs_np)[pi]  # u_q with its bits in this tree's order
                lp = []  # token log-probabilities per position
                for t in range(T):
                    lo_ = Snp @ g[t * B:(t + 1) * B]
                    m = lo_.max()
                    lp.append(lo_ - (m + np.log(np.exp(lo_ - m).sum())))
                sc = np.zeros(1, dtype=np.float32)
                lo = np.zeros(1, dtype=np.int64)
                hi = np.full(1, len(Cs_p), dtype=np.int64)
                for t in range(T):
                    if len(lo) == 0:
                        break
                    if t == 0:
                        vals, st, en = rv_p, rs_p, re_p
                        par = np.zeros(len(vals), dtype=np.int64)
                    else:
                        vals, st, en, par = expand(Cs_p, lo, hi, t)
                    sc = sc[par] + lp[t][vals]
                    lo, hi = st, en
                    if len(sc) > beam:
                        keep = np.argpartition(-sc, beam)[:beam]
                        sc, lo, hi = sc[keep], lo[keep], hi[keep]
                    # A node with one document left is finished: take it out of the beam.
                    done = (hi - lo) == 1
                    if done.any():
                        cand.update(row2doc[lo[done]].tolist())
                        keep = ~done
                        sc, lo, hi = sc[keep], lo[keep], hi[keep]
                cand.update(row2doc[_spans(lo, hi)].tolist())
            rows = back[np.fromiter(cand, dtype=np.int64, count=len(cand))]
            ranks[qi] = _first_relevant(score(Z[qi], rows), rows, gold_rows[qi], GID, SHARED)
            ncand[qi] = len(rows)
        return _metrics(ranks, ncand)

    hdr = "  {:<22}".format("config") + "".join(f"{'@'+str(k):>7}" for k in KS) + \
          f"{'MRR':>8}{'cands':>8}{'sec':>7}{'ms/q':>8}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    def row(name, m, dt):
        cells = "".join(f"{m[f'@{k}']*100:7.2f}" for k in KS)
        print(f"  {name:<22}{cells}{m['MRR']*100:8.2f}{m['cands']:8.0f}{dt:7.0f}"
              f"{dt / max(1, len(texts)) * 1000:8.1f}   | MRR@10 {m['MRR@10']*100:.2f}",
              flush=True)

    t0 = time.time(); m = run_single(a.beam)
    row(f"beam {a.beam}", m, time.time() - t0)
    for k, b in multi:
        t0 = time.time(); m = run_multi(b, k)
        row(f"MULTI k={k} beam {b}", m, time.time() - t0)


if __name__ == "__main__":
    main()
