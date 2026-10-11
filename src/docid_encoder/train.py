"""Train the DocID encoder.

    1. load (query, document) training pairs and a validation set
    2. each step: InfoNCE on the encoder outputs, in-batch negatives shared across GPUs
    3. each epoch: rank all documents for a sample of validation queries; keep the checkpoint
       with the best PDIM (MRR by number of agreeing bits); stop after `patience` epochs
       without improvement

Run through torchrun on 2 GPUs; see scripts/docid_encoder/train.sh.
"""
import os
# wandb logs offline unless WANDB_MODE says otherwise.
os.environ.setdefault("WANDB_MODE", "offline")

import json
import math
import time
import argparse
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast

import sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SRC, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import Encoder_bit
from config import (
    MODEL_SAVE_DIR, LEVEL, N_LEVELS, CODEBOOK_SIZE,
    COSINE_TEMP, BATCH_SIZE, MINI_BATCH_SIZE, MAX_SEQ_LENGTH,
    LR, EPOCHS, PATIENCE,
    CODE_TEST_PAIRS, CODE_TEST_EPOCHS, EVAL_QUERIES, DOC_CHUNK, EVAL_BATCH_SIZE, NCCL_TIMEOUT_S,
)
from data import (load_query_doc_pairs, load_trivia_pairs, select_pseudo_queries,
                  QueryDocPairDataset, collate_fn)
from loss import cached_compute_loss
from evaluate import compute_mrr


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbert_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--query_prompt", type=str, default=None,
                        help="Text prefix for queries (default: none for BERT/T5).")
    parser.add_argument("--doc_prompt", type=str, default=None,
                        help="Text prefix for documents (default: none for BERT/T5).")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory for the best checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Query-document pairs per GPU per step.")
    parser.add_argument("--mini_batch_size", type=int, default=MINI_BATCH_SIZE,
                        help="GradCache chunk size; affects memory only.")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--n_levels", type=int, default=None,
                        help="Bits per token (default: config.N_LEVELS).")
    parser.add_argument("--proj_dim", type=int, default=None,
                        help="DocID length k in bits.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate.")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw",
                        help="adamw (BERT) or adafactor (T5).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient-norm clip (0 disables).")
    parser.add_argument("--cosine_temp", type=float, default=COSINE_TEMP,
                        help="Temperature of the contrastive loss.")
    parser.add_argument("--prefix_bits", type=str, default="24,32",
                        help="Comma-separated bit widths at which validation also ranks by a DocID prefix.")
    parser.add_argument("--select_on", choices=["pdim", "prefix"], default="prefix",
                        help="Checkpoint selection metric: pdim (all bits) or prefix (first --select_width bits).")
    parser.add_argument("--select_width", type=int, default=None,
                        help="Prefix width used when --select_on prefix.")
    parser.add_argument("--dataset", type=str, default="nq320k",
                        choices=["nq320k", "trivia", "msmarco9010"],
                        help="Dataset under data/; msmarco9010 needs --split_dir.")
    parser.add_argument("--qg_per_doc", type=int, default=0,
                        help="Generated queries per document added to training (0 = real queries only).")
    parser.add_argument("--real_weight", type=float, default=1.0,
                        help="Sampling weight of real vs generated queries; 0 balances them 1:1.")
    parser.add_argument("--split_dir", default=None,
                        help="A seen/unseen split folder, e.g. data/nq320k/seen60_unseen40.")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Stop after this many epochs without improvement.")
    parser.add_argument("--eval_pool", type=int, default=None,
                        help="Subsample the validation pool to this many documents (default: all).")
    parser.add_argument("--eval_queries", type=int, default=EVAL_QUERIES,
                        help="Number of validation queries scored each epoch.")
    parser.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE,
                        help="Batch size when encoding for validation.")
    parser.add_argument("--doc_chunk", type=int, default=DOC_CHUNK,
                        help="Documents per block when ranking; lower it if validation runs out of memory.")
    parser.add_argument("--grad_checkpointing", action="store_true",
                        help="Gradient checkpointing in the backbone: less memory, slower.")
    parser.add_argument("--code_test", action="store_true",
                        help=f"Smoke test on {CODE_TEST_PAIRS} pairs for {CODE_TEST_EPOCHS} epochs.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="bit-factorized-docid")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    if args.code_test:
        args.epochs = CODE_TEST_EPOCHS
    eval_bs = args.eval_batch_size

    # One bit per dimension.
    n_levels = args.n_levels if args.n_levels is not None else N_LEVELS
    n_tokens = (args.proj_dim or 0) // n_levels

    # Multi-GPU setup.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_wandb = not args.no_wandb
    # Long communication timeout: validation and saving run on the main process only.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=NCCL_TIMEOUT_S))
    accelerator = Accelerator(mixed_precision="bf16" if use_bf16 else "no",
                              log_with="wandb" if use_wandb else None,
                              kwargs_handlers=[pg_kwargs])
    device = accelerator.device

    set_seed(args.seed)

    # ---- Load the training and validation pairs.
    held_out = None
    if args.split_dir:
        # Seen/unseen split: train and validation queries from the split folder.
        from corpus import load_dynamic_split
        _parts, _D0, _D1, _ids, _texts = load_dynamic_split(
            args.split_dir, args.dataset, want=("train", "valid"))
        pairs = [(q, _texts[d], d) for q, d in _parts["train"]]
        held_out = [(q, _texts[d], d) for q, d in _parts["valid"]]
        _n_real = len(pairs)  # split TSVs hold real queries only
        if args.qg_per_doc:
            # Generated queries, for seen documents only.
            _i2 = {d: i for i, d in enumerate(_ids)}
            _chosen = select_pseudo_queries(args.dataset, args.qg_per_doc, args.seed)
            _n0 = len(pairs)
            for _rid in sorted(_chosen):
                _di = _i2.get(_rid)
                if _di is None or _di not in _D0:
                    continue
                for _qt in _chosen[_rid]:
                    pairs.append((_qt, _texts[_di], _di))
            accelerator.print(f"  qg: +{len(pairs)-_n0:,} generated queries "
                              f"({args.qg_per_doc}/doc, seed {args.seed}) over D0 only")
        # Validation pool = the seen documents; add those no query points to.
        _seen_now = {d for _, _, d in pairs} | {d for _, _, d in held_out}
        extra_docs = [_texts[d] for d in sorted(_D0 - _seen_now)]
        n_unique_docs = len(_D0)
        accelerator.print(
            f"DYNAMIC SPLIT {args.split_dir}\n"
            f"  D0 {len(_D0):,} docs (trained on, and the ONLY documents in the pool)\n"
            f"  D1 {len(_D1):,} docs -- absent until test time\n"
            f"  train {len(pairs):,} pairs | valid {len(held_out):,} pairs")
    elif args.dataset == "trivia":
        # TriviaQA, full corpus: train on train, validate on dev.
        by_split, n_unique_docs, extra_docs = load_trivia_pairs(
            splits=("train", "dev"), qg_per_doc=args.qg_per_doc, qg_seed=args.seed,
            )
        _n_real = by_split.pop("_n_real_train", len(by_split["train"]))
        pairs = by_split["train"]
        held_out = by_split.get("val") or by_split["dev"]
        # extra_docs: documents no train or dev query points to; validation pool only.
    elif args.dataset == "nq320k":
        pairs, n_unique_docs, extra_docs, _val_rows, _n_real, _dev_rows = \
            load_query_doc_pairs(qg_per_doc=args.qg_per_doc, qg_seed=args.seed)
        held_out = _dev_rows or None
        if _val_rows:
            held_out = _val_rows
    else:
        raise SystemExit(f"--dataset {args.dataset} has no full-corpus setting; pass --split_dir")
    # Corpus size before the optional --code_test subsample.
    corpus_size = n_unique_docs
    if args.code_test:
        rng = torch.Generator().manual_seed(args.seed)
        keep = torch.randperm(len(pairs), generator=rng)[:CODE_TEST_PAIRS].tolist()
        # Keep real pairs before generated ones.
        keep = sorted(keep, key=lambda i: (i >= _n_real, i))
        _n_real = sum(1 for i in keep if i < _n_real)
        pairs = [pairs[i] for i in keep]
        if held_out is not None:
            held_out = held_out[:max(1, CODE_TEST_PAIRS // 10)]
        n_unique_docs = len({d for _, _, d in pairs} | {d for _, _, d in (held_out or [])})
        accelerator.print(f"[code_test] subsampled to {len(pairs)} pairs, {args.epochs} epochs")
    accelerator.print(f"Pairs: {len(pairs)} | unique docs (canonical): {n_unique_docs}"
                      + (f" | val = official dev ({len(held_out):,} pairs)"
                         if args.dataset == "trivia" else ""))
    full_dataset = QueryDocPairDataset(pairs)
    if args.split_dir:
        # Seen/unseen split: validate on the split's validation queries.
        _train_src = list(range(len(pairs)))
        train_dataset = full_dataset
        val_source = held_out
        val_dataset = QueryDocPairDataset(val_source)
        accelerator.print(f"Val = D0 QUERY SLICE: {len(val_source):,} pairs / "
                          f"{len({q for q, _, _ in val_source}):,} queries, all golds in D0. "
                          f"D1 is not scored until test.", flush=True)
    elif args.dataset == "trivia":
        _train_src = list(range(len(pairs)))
        train_dataset = full_dataset
        val_source = held_out
        val_dataset = QueryDocPairDataset(val_source)
        accelerator.print(f"Val = OFFICIAL DEV: {len(val_source):,} pairs / "
                          f"{len({q for q, _, _ in val_source}):,} queries. test.tsv is NOT "
                          f"read here -- report it once at the end.", flush=True)
    else:
        # NQ320K, full corpus: validate on dev.
        _train_src = list(range(len(pairs)))
        train_dataset = full_dataset
        val_source = held_out
        val_dataset = QueryDocPairDataset(val_source)
        accelerator.print(
            f"Val = OFFICIAL DEV: {len(val_source):,} pairs / "
            f"{len({q for q, _, _ in val_source}):,} queries, all REAL. Selecting on dev as "
            f"NCI/DSI/GenRet do -- optimistic in the same way theirs are, and comparable.")
    accelerator.print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Validation pool: every document of the train and validation pairs.
    unique_docs, doc_id_to_row = [], {}
    for _, doc_text, doc_id in list(full_dataset.pairs) + list(held_out or []):
        if doc_id not in doc_id_to_row:
            doc_id_to_row[doc_id] = len(unique_docs)
            unique_docs.append(doc_text)
    # Plus the documents no query points to.
    if extra_docs:
        n_judged = len(unique_docs)
        unique_docs.extend(extra_docs)
        tail = f"= the full corpus ({corpus_size:,})"
        accelerator.print(f"MRR pool: {n_judged:,} judged + {len(extra_docs):,} pool-only "
                          f"= {len(unique_docs):,} docs ({tail})")

    val_pairs = val_source
    # A fixed, seeded sample of validation queries, the same every epoch.
    if len(val_pairs) > args.eval_queries:
        g = torch.Generator().manual_seed(args.seed)
        pick = torch.randperm(len(val_pairs), generator=g)[: args.eval_queries].tolist()
        val_pairs = [val_pairs[i] for i in pick]

    # Optional: subsample the validation pool (--eval_pool).
    if args.eval_pool and args.eval_pool < len(unique_docs):
        gold_ids = {d for _, _, d in val_pairs}
        gold_rows = {doc_id_to_row[d] for d in gold_ids}
        g = torch.Generator().manual_seed(args.seed)
        others = [r for r in range(len(unique_docs)) if r not in gold_rows]
        need = max(0, args.eval_pool - len(gold_rows))
        take = torch.randperm(len(others), generator=g)[:need].tolist()
        keep = sorted(gold_rows | {others[i] for i in take})
        row_of = {old: new for new, old in enumerate(keep)}
        unique_docs = [unique_docs[r] for r in keep]
        doc_id_to_row = {d: row_of[r] for d, r in doc_id_to_row.items() if r in row_of}
        accelerator.print(f"MRR pool subsampled: {len(gold_rows):,} val gold + "
                          f"{len(keep)-len(gold_rows):,} sampled judged = {len(keep):,} docs "
                          f"(SELECTION metric -- not a full-corpus number)")

    val_queries = [q for q, _, _ in val_pairs]
    val_true_doc_idx = [doc_id_to_row[doc_id] for _, _, doc_id in val_pairs]
    accelerator.print(f"MRR: {len(val_queries)} eval queries vs pool of "
                      f"{len(unique_docs)} docs")

    # A fixed batch used for per-epoch diagnostics.
    n_probe = min(len(train_dataset), args.batch_size * max(accelerator.num_processes, 1))
    # Indices of real and generated pairs in the training set.
    probe = [train_dataset[i] for i in range(n_probe)]
    probe_q = [q for q, _, _ in probe]
    probe_d = [d for _, d, _ in probe]
    probe_ids = torch.tensor([i for _, _, i in probe])

    # Optional: sample real queries more often than generated ones (--real_weight).
    _real_pos = [j for j, o in enumerate(_train_src) if o < _n_real]
    _n_real_tr = len(_real_pos)
    _sampler, _shuffle = None, True
    _n_gen = len(train_dataset) - _n_real_tr
    if args.real_weight != 1.0 and (_n_gen == 0 or _n_real_tr == 0):
        accelerator.print(f"--real_weight ignored: {_n_real_tr:,} real / {_n_gen:,} generated "
                          f"-- nothing to rebalance")
    elif args.real_weight != 1.0:
        _w = args.real_weight
        if _w == 0:  # 0 = balance to 1:1
            _w = _n_gen / _n_real_tr
        _wt = torch.ones(len(train_dataset), dtype=torch.double)
        _wt[_real_pos] = _w
        _sampler = torch.utils.data.WeightedRandomSampler(
            _wt, num_samples=len(train_dataset), replacement=True)
        _shuffle = False
        _share = _w * _n_real_tr / (_w * _n_real_tr + _n_gen)
        accelerator.print(
            f"Real upsampling: weight {_w:.2f} on {_n_real_tr:,} real pairs vs {_n_gen:,} "
            f"generated -> {_share:.0%} of each batch is a REAL query "
            f"(was {_n_real_tr/len(train_dataset):.0%}); epoch length unchanged")
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=_sampler,
        shuffle=_shuffle if _sampler is None else False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


    # ---- Model and optimizer.
    model = Encoder_bit(
        proj_dim=args.proj_dim,  # linear head -> proj_dim == bits per document
        sbert_model=args.sbert_model,
        max_seq_length=args.max_seq_length,
        query_prompt=args.query_prompt,  # None -> auto per backbone (E5
        doc_prompt=args.doc_prompt,
    )
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    accelerator.print(f"prompts: query={model.query_prompt!r} doc={model.doc_prompt!r}")
    trainable, total = model.trainable_parameter_summary()
    accelerator.print(
        f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%) | full fine-tune\n"
        f"proj_dim={args.proj_dim} bits -> {n_tokens} tokens x {LEVEL**n_levels} values "
        f"(read-time grouping) | loss=cosine InfoNCE on z_cont "
        f"temp={args.cosine_temp} lr={args.lr} | batch={args.batch_size} "
        f"mini_batch={args.mini_batch_size} max_seq_len={args.max_seq_length} bf16={use_bf16} "
        f"grad_ckpt={args.grad_checkpointing}")
    _params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adafactor":
        # Adafactor, used for T5.
        optimizer = torch.optim.Adafactor(_params, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(_params, lr=args.lr)
    accelerator.print(f"[optim] {args.optimizer} lr={args.lr}")

    # Let accelerate place the model, optimizer and loaders on the GPUs.
    model, optimizer, dataloader, val_loader = accelerator.prepare(
        model, optimizer, dataloader, val_loader
    )
    # Bit widths at which validation also ranks by a prefix of the DocID.
    prefix_widths = [int(v) for v in args.prefix_bits.split(",") if v.strip()]
    full_dim = args.proj_dim or accelerator.unwrap_model(model).code_dim
    for pb in prefix_widths:
        assert 0 < pb <= full_dim, f"--prefix_bits {pb} outside code width {full_dim}"
    if args.select_on == "prefix":
        assert prefix_widths, "--select_on prefix needs at least one --prefix_bits width"
        sel_w = args.select_width or prefix_widths[-1]
        assert sel_w in prefix_widths, \
            f"--select_width {sel_w} not among --prefix_bits {prefix_widths}"
        sel_key, sel_name = f"prefix_mrr@{sel_w}", f"PFX{sel_w}"
    else:
        sel_key, sel_name = "pdim_mrr", "PDIM"
    accelerator.print(f"PFX probe widths {prefix_widths} | selecting best checkpoint and "
                      f"early stopping on {sel_name} ({sel_key})")

    model_save_dir = args.save_dir if args.save_dir is not None else MODEL_SAVE_DIR
    # The checkpoint is named after the run.
    run_tag = (args.run_name or args.sbert_model.replace("/", "_")).replace("/", "_")
    ckpt_path = os.path.join(model_save_dir, f"{run_tag}_best.pt")
    if accelerator.is_main_process:
        os.makedirs(model_save_dir, exist_ok=True)
    best_mrr = -float("inf")  # selection metric (see --select_on), higher is better
    epochs_no_improve = 0
    n_batches = len(dataloader)
    train_start = time.time()

    if use_wandb:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config={**vars(args),
                    "level": LEVEL,
                    "n_tokens": n_tokens,
                    "codes_per_token": LEVEL ** n_levels,
                    "cosine_temp": args.cosine_temp,
                    "lr": LR, "bf16": use_bf16, "world_size": accelerator.num_processes},
            init_kwargs={"wandb": {"name": args.run_name}},
        )

    def run_eval():
        """Val loss (sharded + reduced) plus MRR/diagnostics (also sharded internally over the
        full corpus)."""
        model.eval()
        val_sum = torch.zeros(1, device=device)
        val_cnt = torch.zeros(1, device=device)
        with torch.no_grad():
            for query_batch, doc_batch, doc_ids in val_loader:
                val_sum += cached_compute_loss(model, accelerator, query_batch, doc_batch,
                                               doc_ids, args.mini_batch_size, device,
                                               train=False, cosine_temp=args.cosine_temp)
                val_cnt += 1
        val_sum = accelerator.reduce(val_sum, reduction="sum")
        val_cnt = accelerator.reduce(val_cnt, reduction="sum")
        avg = (val_sum / val_cnt).item()

        # Validation metrics; computed across all GPUs.
        metrics = compute_mrr(accelerator.unwrap_model(model), val_queries,
                              val_true_doc_idx, unique_docs, eval_bs, device,
                              doc_chunk=args.doc_chunk, accelerator=accelerator,
                              level=LEVEL, n_levels=n_levels,
                              prefix_bits=prefix_widths)
        return avg, metrics

    def log_eval(tag, step, avg_val_loss, metrics, train_loss=None):
        if not accelerator.is_main_process:
            return
        m = metrics
        code_ppl = m["code_ppl"]
        accelerator.print(
            f"{tag} | train={'n/a' if train_loss is None else f'{train_loss:.4f}'} "
            f"| val={avg_val_loss:.6f} "
            + "".join(f"| PFX{pb}={m[f'prefix_mrr@{pb}']:.4f} tied={m[f'prefix_tied@{pb}']:.0f} "
                      for pb in prefix_widths)
            + f"| PDIM={m['pdim_mrr']:.4f} @10={m['pdim_mrr@10']:.4f} R@1={m['pdim_recall@1']:.4f} "
            f"tied={m['pdim_tied@true']:.0f} "  # <- SELECTION METRIC, listed first
            f"| MRR={m['mrr']:.4f} (opt {m['mrr_opt']:.4f}) @10={m['mrr@10']:.4f} "
            f"R@1={m['recall@1']:.4f} R@10={m['recall@10']:.4f} "
            f"| dense MRR={m['dense_mrr']:.4f} @10={m['dense_mrr@10']:.4f} R@1={m['dense_recall@1']:.4f} "
            f"| overlap true/rand={m['overlap_true']:.1f}/{m['overlap_rand']:.2f} tied={m['tied@true']:.0f} "
            f"| codes/part={m['code_usage']:.1f} ppl={code_ppl:.1f} "
            f"postings={len(unique_docs)/max(code_ppl, 1e-9):.0f}")
        payload = {f"val/{k}": v for k, v in m.items()}
        payload.update({
            "val/loss": avg_val_loss,
            "val/overlap_gap": m["overlap_true"] - m["overlap_rand"],  # matters most
            "val/quantization_gap": m["dense_mrr"] - m["mrr"],  # cost of quantizing
            "val/quantization_gap_pdim": m["dense_mrr"] - m["pdim_mrr"],
            "val/tie_cost": m["mrr_opt"] - m["mrr"],  # MRR lost to ties alone
            "val/expected_postings_len": len(unique_docs) / max(code_ppl, 1e-9),
        })
        if train_loss is not None:
            payload["train/loss"] = train_loss
        accelerator.log(payload, step=step)

    # ---- Validation before any training, as a reference.
    accelerator.print("Evaluating pretrained baseline (no fine-tuning) ...")
    base_val_loss, base_metrics = run_eval()
    log_eval("Baseline (pretrained)", 0, base_val_loss, base_metrics)
    baseline_mrr = base_metrics["mrr"] if base_metrics is not None else float("nan")
    accelerator.wait_for_everyone()

    for epoch in range(args.epochs):
        # ---- One epoch of training.
        model.train()
        total_loss = 0.0
        gnorm_sum, gnorm_max = 0.0, 0.0
        # With clipping off, the gradient norm is still measured.
        clip_val = args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
        for query_batch, doc_batch, doc_ids in dataloader:
            optimizer.zero_grad()
            # Forward and backward both happen inside cached_compute_loss.
            loss = cached_compute_loss(model, accelerator, query_batch, doc_batch, doc_ids,
                                       args.mini_batch_size, device, train=True,
                                       cosine_temp=args.cosine_temp)
            # Clip the gradient norm; the value before clipping is logged.
            gnorm = accelerator.clip_grad_norm_(model.parameters(), clip_val).item()
            optimizer.step()

            total_loss += loss.item()
            gnorm_sum += gnorm
            gnorm_max = max(gnorm_max, gnorm)

        avg_val_loss, metrics = run_eval()
        log_eval(f"Epoch {epoch + 1}/{args.epochs}", epoch + 1, avg_val_loss, metrics,
                 train_loss=total_loss / n_batches)
        if accelerator.is_main_process:
            accelerator.log({"train/grad_norm": gnorm_sum / n_batches,
                             "train/grad_norm_max": gnorm_max}, step=epoch + 1)

        # ---- Keep the checkpoint if the selection metric improved; else count toward patience.
        sel = (metrics[sel_key] if metrics is not None else 0.0)
        if sel != sel:  # nan -> level/n_levels not passed
            sel = metrics["mrr"]
        mrr_t = torch.tensor([sel if metrics is not None else 0.0],
                             device=device, dtype=torch.float32)
        current_mrr = broadcast(mrr_t, from_process=0).item()

        if current_mrr > best_mrr:
            best_mrr = current_mrr
            epochs_no_improve = 0
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                # Sidecar file describing the architecture (and the split, if any).
                with open(ckpt_path + ".arch.json", "w") as fh:
                    _arch = {"level": LEVEL, "n_levels": n_levels,
                             "proj_dim": args.proj_dim,
                             "sbert_model": args.sbert_model,
                             "run_name": args.run_name}
                    if args.split_dir:
                        import json as _j
                        with open(os.path.join(args.split_dir, "manifest.json")) as _fh:
                            _man = _j.load(_fh)
                        _arch["split_dir"] = args.split_dir
                        _arch["d1_sha256"] = _man["sha256"]["d1_docs.txt"]
                        _arch["n_d0"] = _man["n"]["d0_docs"]
                        _arch["n_d1"] = _man["n"]["d1_docs"]
                    json.dump(_arch, fh, indent=2)
                accelerator.print(f"  -> New best {sel_name}={best_mrr:.4f}, saved {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                accelerator.print(f"Early stopping triggered (no {sel_name} improvement for {args.patience} epochs).")
                break

        accelerator.wait_for_everyone()  # resync before the next epoch (main did MRR/save alone)

    total_time = time.time() - train_start
    # NOTE.
    if accelerator.is_main_process:
        verdict = "BEATS" if best_mrr > baseline_mrr else "DOES NOT BEAT"
        accelerator.print(f"Best {sel_name}={best_mrr:.4f} vs pretrained baseline={baseline_mrr:.4f} "
                          f"-> fine-tuning {verdict} the baseline.")
        accelerator.log({"final/best_mrr": best_mrr, "final/baseline_mrr": baseline_mrr,
                         "final/mrr_gain": best_mrr - baseline_mrr}, step=args.epochs)
    accelerator.print(f"Total training time: {total_time / 60:.1f} min ({total_time:.1f} s)")
    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
