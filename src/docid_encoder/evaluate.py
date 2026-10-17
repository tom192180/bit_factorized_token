"""Validation during DocID-encoder training: rank every document for each query, no tree.

Every metric starts from the rank of the query's own document, obtained by counting the
documents that score higher. Three scores are reported:
    pdim_*    number of agreeing bits (all k)          -- "PDIM", used to select the checkpoint
    (plain)   number of agreeing tokens
    dense_*   cosine of the embeddings before projection, as a reference
Ties are placed at their expected rank (+ tied / 2); the *_opt keys place them first.
"""
from contextlib import nullcontext
import torch
import torch.nn.functional as F

from codes import pack_bits
from config import DOC_CHUNK


def _shard(seq, rank, world):
    """This GPU's contiguous share of seq, padded so that every GPU gets the same length."""
    n = len(seq)
    per = (n + world - 1) // world
    padded = list(seq) + [seq[-1]] * (per * world - n)
    return padded[rank * per : (rank + 1) * per], per


def compute_mrr(model, queries, true_doc_idx, doc_texts, batch_size, device,
                doc_chunk=DOC_CHUNK, accelerator=None, level=None, n_levels=None):
    """Rank all doc_texts for each query and return a dict of metrics.

    true_doc_idx[i] is the index in doc_texts of query i's document."""
    model.eval()
    # Use bf16 autocast, as training does.
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
               else nullcontext())
    world = accelerator.num_processes if accelerator is not None else 1
    proc_rank = accelerator.process_index if accelerator is not None else 0
    # Embeddings are kept in half precision on GPU to save memory.
    emb_dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32

    with torch.no_grad(), amp_ctx:
        # Encode the documents: each GPU encodes its share, then the results are gathered.
        n_docs_true = len(doc_texts)
        my_docs, _ = _shard(doc_texts, proc_rank, world)
        doc_indices, doc_bits, doc_emb = [], [], []
        for i in range(0, len(my_docs), batch_size):
            x, _, bits = model(my_docs[i : i + batch_size])
            doc_bits.append(bits)
            doc_indices.append(pack_bits(bits, n_levels).to(torch.int32)
                               if n_levels else torch.zeros(len(bits), 0, dtype=torch.int32,
                                                            device=bits.device))
            doc_emb.append(F.normalize(x.float(), dim=-1).to(emb_dtype))
        doc_indices = torch.cat(doc_indices, dim=0)
        doc_bits = torch.cat(doc_bits, dim=0)
        doc_emb = torch.cat(doc_emb, dim=0)
        if world > 1:
            doc_indices = accelerator.gather(doc_indices)
            doc_bits = accelerator.gather(doc_bits)
            doc_emb = accelerator.gather(doc_emb)
        doc_indices = doc_indices[:n_docs_true]  # drop the padding
        doc_bits = doc_bits[:n_docs_true]
        doc_emb = doc_emb[:n_docs_true]

        n_docs, n_part = doc_indices.shape
        # Bit-level scoring (PDIM).
        doc_levels = doc_bits if (level and n_levels) else None
        # How many distinct values each token position uses, and their perplexity.
        usage_sum, perplexity_sum = 0, 0.0
        for p in range(n_part):
            counts = torch.unique(doc_indices[:, p], return_counts=True)[1].double()
            usage_sum += counts.numel()
            probs = counts / counts.sum()
            perplexity_sum += float(1.0 / (probs * probs).sum())
        code_usage = usage_sum / n_part
        code_perplexity = perplexity_sum / n_part

        reciprocal_ranks, dense_reciprocal_ranks = [], []
        mrr10, dense_mrr10 = [], []
        recall1, recall10, dense_recall1, dense_recall10 = [], [], [], []
        true_overlaps, rand_overlaps = [], []
        rr_exp, mrr10_exp, recall1_exp, recall10_exp, tied_counts = [], [], [], [], []
        pd_rr, pd_mrr10, pd_recall1, pd_recall10, pd_tied = [], [], [], [], []
        # Encode and score the queries, each GPU its share.
        n_queries_true = len(queries)
        my_queries, _ = _shard(queries, proc_rank, world)
        my_true_idx, _ = _shard(true_doc_idx, proc_rank, world)
        for i in range(0, len(my_queries), batch_size):
            q_batch = my_queries[i : i + batch_size]
            true_idx = torch.tensor(my_true_idx[i : i + batch_size], device=device)
            q_x, _, q_bits = model(q_batch)
            q_idx = (pack_bits(q_bits, n_levels).to(torch.int32) if n_levels
                     else torch.zeros(len(q_bits), 0, dtype=torch.int32, device=q_bits.device))
            q_emb = F.normalize(q_x.float(), dim=-1).to(emb_dtype)
            b = q_idx.size(0)

            # Token agreement with the query's own document and with one random document.
            true_overlap = (q_idx == doc_indices[true_idx]).sum(dim=-1)
            rand_idx = torch.randint(0, n_docs, (b,), device=device)
            rand_overlap = (q_idx == doc_indices[rand_idx]).sum(dim=-1)

            # Count, block by block, the documents that score higher than (or equal to) the
            # query's own document. rank = 1 + number of better documents.
            true_score = (q_emb * doc_emb[true_idx]).sum(dim=-1)

            better = torch.zeros(b, dtype=torch.long, device=device)
            tied = torch.zeros(b, dtype=torch.long, device=device)
            better_dense = torch.zeros(b, dtype=torch.long, device=device)
            if doc_levels is not None:
                q_lv = q_bits
                true_pd = (q_lv == doc_levels[true_idx]).sum(dim=-1)
                better_pd = torch.zeros(b, dtype=torch.long, device=device)
                tied_pd = torch.zeros(b, dtype=torch.long, device=device)
            for s in range(0, n_docs, doc_chunk):
                block = doc_indices[s : s + doc_chunk]  # (C, P)
                # A document is never counted against itself.
                cols = torch.arange(s, s + block.size(0), device=device)
                is_self = cols.unsqueeze(0) == true_idx.unsqueeze(1)  # (b, C)

                ov = (q_idx.unsqueeze(1) == block.unsqueeze(0)).sum(dim=-1)  # (b, C)
                better += ((ov > true_overlap.unsqueeze(1)) & ~is_self).sum(dim=1)
                tied += ((ov == true_overlap.unsqueeze(1)) & ~is_self).sum(dim=1)

                sim = q_emb @ doc_emb[s : s + doc_chunk].T  # (b, C)
                better_dense += ((sim > true_score.unsqueeze(1)) & ~is_self).sum(dim=1)

                if doc_levels is not None:
                    eq_pd = (q_lv.unsqueeze(1) == doc_levels[s : s + doc_chunk].unsqueeze(0))
                    ov_pd = eq_pd.sum(dim=-1)
                    better_pd += ((ov_pd > true_pd.unsqueeze(1)) & ~is_self).sum(dim=1)
                    tied_pd += ((ov_pd == true_pd.unsqueeze(1)) & ~is_self).sum(dim=1)
            # rank: ties placed first. rank_exp: ties at their expected position.
            rank = (better + 1).float()
            rank_exp = (better + 1 + tied / 2.0).float()
            rank_dense = (better_dense + 1).float()

            reciprocal_ranks.extend((1.0 / rank).cpu().tolist())
            dense_reciprocal_ranks.extend((1.0 / rank_dense).cpu().tolist())
            mrr10.extend(torch.where(rank <= 10, 1.0 / rank, torch.zeros_like(rank)).cpu().tolist())
            dense_mrr10.extend(torch.where(rank_dense <= 10, 1.0 / rank_dense,
                                           torch.zeros_like(rank_dense)).cpu().tolist())
            recall1.extend((rank <= 1).float().cpu().tolist())
            recall10.extend((rank <= 10).float().cpu().tolist())
            dense_recall1.extend((rank_dense <= 1).float().cpu().tolist())
            dense_recall10.extend((rank_dense <= 10).float().cpu().tolist())
            true_overlaps.extend(true_overlap.float().cpu().tolist())
            rand_overlaps.extend(rand_overlap.float().cpu().tolist())
            rr_exp.extend((1.0 / rank_exp).cpu().tolist())
            mrr10_exp.extend(torch.where(rank_exp <= 10, 1.0 / rank_exp,
                                         torch.zeros_like(rank_exp)).cpu().tolist())
            recall1_exp.extend((rank_exp <= 1).float().cpu().tolist())
            recall10_exp.extend((rank_exp <= 10).float().cpu().tolist())
            tied_counts.extend(tied.float().cpu().tolist())
            if doc_levels is not None:
                rk = (better_pd + 1 + tied_pd / 2.0).float()  # ties at their expected position
                pd_rr.extend((1.0 / rk).cpu().tolist())
                pd_mrr10.extend(torch.where(rk <= 10, 1.0 / rk,
                                            torch.zeros_like(rk)).cpu().tolist())
                pd_recall1.extend((rk <= 1).float().cpu().tolist())
                pd_recall10.extend((rk <= 10).float().cpu().tolist())
                pd_tied.extend(tied_pd.float().cpu().tolist())

    # Mean over all queries, gathered from every GPU, without the padding.
    def mean(v):
        t = torch.tensor(v, dtype=torch.float32, device=device)
        if world > 1:
            t = accelerator.gather(t)
        return t[:n_queries_true].mean().item()

    out = {
        # ranking by agreeing tokens
        "mrr":            mean(rr_exp),
        "mrr@10":         mean(mrr10_exp),
        "recall@1":       mean(recall1_exp),
        "recall@10":      mean(recall10_exp),
        "mrr_opt":        mean(reciprocal_ranks),
        "mrr@10_opt":     mean(mrr10),
        "recall@1_opt":   mean(recall1),
        "recall@10_opt":  mean(recall10),
        "tied@true":      mean(tied_counts),
        # ranking by agreeing bits (PDIM)
        "pdim_mrr":       mean(pd_rr) if pd_rr else float("nan"),
        "pdim_mrr@10":    mean(pd_mrr10) if pd_mrr10 else float("nan"),
        "pdim_recall@1":  mean(pd_recall1) if pd_recall1 else float("nan"),
        "pdim_recall@10": mean(pd_recall10) if pd_recall10 else float("nan"),
        "pdim_tied@true": mean(pd_tied) if pd_tied else float("nan"),
        # ranking by cosine of the embeddings
        "dense_mrr":      mean(dense_reciprocal_ranks),
        "dense_mrr@10":   mean(dense_mrr10),
        "dense_recall@1": mean(dense_recall1),
        "dense_recall@10": mean(dense_recall10),
        # diagnostics
        "overlap_true":   mean(true_overlaps),
        "overlap_rand":   mean(rand_overlaps),
        "code_usage":     code_usage,
        "code_ppl":       code_perplexity,
    }
    return out
