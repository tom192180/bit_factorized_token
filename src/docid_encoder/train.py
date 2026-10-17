"""Train the DocID encoder.

    1. load (query, document) training pairs and a validation set
    2. each step: InfoNCE on the encoder outputs, in-batch negatives shared across GPUs
    3. each epoch: rank all documents for a sample of validation queries; keep the checkpoint
       with the best PDIM (MRR by number of agreeing bits); stop after PATIENCE epochs
       without improvement

Hyperparameters live in config.py. Run through torchrun on 2 GPUs; see
scripts/docid_encoder/train.sh.
"""
import os
import sys
import json
import time
import argparse
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SRC, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import Encoder_bit
from config import (
    LEVEL, N_LEVELS, COSINE_TEMP, BATCH_SIZE, MINI_BATCH_SIZE, MAX_SEQ_LENGTH,
    LR_ADAMW, LR_ADAFACTOR, MAX_GRAD_NORM, EPOCHS, PATIENCE,
    EVAL_QUERIES, EVAL_BATCH_SIZE, DOC_CHUNK, NCCL_TIMEOUT_S,
)
from data import load_query_doc_pairs, load_trivia_pairs, QueryDocPairDataset, collate_fn
from loss import cached_compute_loss
from evaluate import compute_mrr

CKPT_NAME = "docid_encoder_best.pt"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train the DocID encoder.")
    parser.add_argument("--dataset", type=str, default="nq320k",
                        choices=["nq320k", "trivia", "msmarco9010"],
                        help="Dataset under data/; msmarco9010 needs --split_dir.")
    parser.add_argument("--split_dir", default=None,
                        help="A seen/unseen split folder, e.g. data/nq320k/seen60_unseen40.")
    parser.add_argument("--sbert_model", type=str, default="bert-base-uncased",
                        help="Backbone: bert-base-uncased (AdamW) or t5-base (Adafactor).")
    parser.add_argument("--proj_dim", type=int, default=120,
                        help="DocID length k in bits.")
    parser.add_argument("--n_levels", type=int, default=N_LEVELS,
                        help="Bits per token m; k must be a multiple of m.")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH,
                        help="Input length in tokens, for queries and documents.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Query-document pairs per GPU per step; the loss sees batch_size x GPUs documents.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help=f"Directory for the best checkpoint ({CKPT_NAME} and its .arch.json sidecar).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_levels = args.n_levels
    if args.proj_dim % n_levels:
        raise SystemExit(f"--proj_dim {args.proj_dim} is not a multiple of --n_levels {n_levels}")
    n_tokens = args.proj_dim // n_levels
    ckpt_path = os.path.join(args.save_dir, CKPT_NAME)
    if os.path.exists(ckpt_path):
        raise SystemExit(f"refusing to overwrite {ckpt_path}")

    # Multi-GPU setup.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # Long communication timeout: validation and saving run on the main process only.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=NCCL_TIMEOUT_S))
    accelerator = Accelerator(mixed_precision="bf16" if use_bf16 else "no",
                              kwargs_handlers=[pg_kwargs])
    device = accelerator.device

    set_seed(args.seed)

    # ---- Load the training and validation pairs: (query_text, doc_text, doc_id).
    if args.split_dir:
        # Seen/unseen split: train and validation queries from the split folder.
        from corpus import load_dynamic_split
        _parts, _D0, _D1, _ids, _texts = load_dynamic_split(
            args.split_dir, args.dataset, want=("train", "valid"))
        pairs = [(q, _texts[d], d) for q, d in _parts["train"]]
        held_out = [(q, _texts[d], d) for q, d in _parts["valid"]]
        # Validation pool = the seen documents; add those no query points to.
        _seen_now = {d for _, _, d in pairs} | {d for _, _, d in held_out}
        extra_docs = [_texts[d] for d in sorted(_D0 - _seen_now)]
        n_unique_docs = len(_D0)
        val_desc = "the split's validation queries, all golds in D0"
        accelerator.print(
            f"DYNAMIC SPLIT {args.split_dir}\n"
            f"  D0 {len(_D0):,} docs (trained on, and the ONLY documents in the pool)\n"
            f"  D1 {len(_D1):,} docs -- absent until test time")
    elif args.dataset == "trivia":
        # TriviaQA, full corpus: train on train, validate on dev; test.tsv is not read here.
        by_split, n_unique_docs, extra_docs = load_trivia_pairs(splits=("train", "dev"))
        pairs, held_out = by_split["train"], by_split["dev"]
        val_desc = "official dev"
    elif args.dataset == "nq320k":
        # NQ320K, full corpus: validate on dev, as NCI/DSI/GenRet do (NQ320K has no test set).
        pairs, n_unique_docs, extra_docs, held_out = load_query_doc_pairs()
        val_desc = "official dev"
    else:
        raise SystemExit(f"--dataset {args.dataset} has no full-corpus setting; pass --split_dir")
    accelerator.print(f"Train: {len(pairs):,} pairs | Val: {len(held_out):,} pairs ({val_desc}) "
                      f"| unique docs: {n_unique_docs:,}")
    train_dataset = QueryDocPairDataset(pairs)
    val_dataset = QueryDocPairDataset(held_out)

    # Validation pool: every document of the train and validation pairs ...
    unique_docs, doc_id_to_row = [], {}
    for _, doc_text, doc_id in pairs + held_out:
        if doc_id not in doc_id_to_row:
            doc_id_to_row[doc_id] = len(unique_docs)
            unique_docs.append(doc_text)
    # ... plus the documents no query points to.
    if extra_docs:
        n_judged = len(unique_docs)
        unique_docs.extend(extra_docs)
        accelerator.print(f"MRR pool: {n_judged:,} judged + {len(extra_docs):,} pool-only "
                          f"= {len(unique_docs):,} docs (= the full corpus, {n_unique_docs:,})")

    # A fixed, seeded sample of validation queries, the same every epoch.
    val_pairs = held_out
    if len(val_pairs) > EVAL_QUERIES:
        g = torch.Generator().manual_seed(args.seed)
        pick = torch.randperm(len(val_pairs), generator=g)[:EVAL_QUERIES].tolist()
        val_pairs = [val_pairs[i] for i in pick]
    val_queries = [q for q, _, _ in val_pairs]
    val_true_doc_idx = [doc_id_to_row[doc_id] for _, _, doc_id in val_pairs]
    accelerator.print(f"MRR: {len(val_queries)} eval queries vs pool of {len(unique_docs)} docs")

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # ---- Model and optimizer.
    model = Encoder_bit(proj_dim=args.proj_dim, sbert_model=args.sbert_model,
                        max_seq_length=args.max_seq_length)
    params = [p for p in model.parameters() if p.requires_grad]
    if "t5" in args.sbert_model.lower():
        opt_name, lr = "adafactor", LR_ADAFACTOR
        optimizer = torch.optim.Adafactor(params, lr=lr)
    else:
        opt_name, lr = "adamw", LR_ADAMW
        optimizer = torch.optim.AdamW(params, lr=lr)
    trainable, total = model.trainable_parameter_summary()
    accelerator.print(
        f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%) | full fine-tune\n"
        f"proj_dim={args.proj_dim} bits -> {n_tokens} tokens x {LEVEL**n_levels} values "
        f"(read-time grouping) | loss=cosine InfoNCE on z_cont temp={COSINE_TEMP} "
        f"| {opt_name} lr={lr} clip={MAX_GRAD_NORM} | batch={args.batch_size} x "
        f"{accelerator.num_processes} GPUs, mini_batch={MINI_BATCH_SIZE} "
        f"max_seq_len={args.max_seq_length} bf16={use_bf16}")

    # Let accelerate place the model, optimizer and loaders on the GPUs.
    model, optimizer, dataloader, val_loader = accelerator.prepare(
        model, optimizer, dataloader, val_loader)
    accelerator.print(f"Selecting the checkpoint and early stopping on PDIM (pdim_mrr); "
                      f"checkpoint -> {ckpt_path}")
    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)

    def run_eval():
        """(validation loss, metrics dict); both computed across all GPUs."""
        model.eval()
        val_sum = torch.zeros(1, device=device)
        val_cnt = torch.zeros(1, device=device)
        with torch.no_grad():
            for query_batch, doc_batch, doc_ids in val_loader:
                val_sum += cached_compute_loss(model, accelerator, query_batch, doc_batch,
                                               doc_ids, MINI_BATCH_SIZE, device, train=False,
                                               cosine_temp=COSINE_TEMP)
                val_cnt += 1
        val_sum = accelerator.reduce(val_sum, reduction="sum")
        val_cnt = accelerator.reduce(val_cnt, reduction="sum")
        metrics = compute_mrr(accelerator.unwrap_model(model), val_queries, val_true_doc_idx,
                              unique_docs, EVAL_BATCH_SIZE, device, doc_chunk=DOC_CHUNK,
                              accelerator=accelerator, level=LEVEL, n_levels=n_levels)
        return (val_sum / val_cnt).item(), metrics

    def log_eval(tag, avg_val_loss, m, train_loss=None):
        code_ppl = m["code_ppl"]
        accelerator.print(
            f"{tag} | train={'n/a' if train_loss is None else f'{train_loss:.4f}'} "
            f"| val={avg_val_loss:.6f} "
            f"| PDIM={m['pdim_mrr']:.4f} @10={m['pdim_mrr@10']:.4f} R@1={m['pdim_recall@1']:.4f} "
            f"tied={m['pdim_tied@true']:.0f} "  # <- selection metric
            f"| MRR={m['mrr']:.4f} (opt {m['mrr_opt']:.4f}) @10={m['mrr@10']:.4f} "
            f"R@1={m['recall@1']:.4f} R@10={m['recall@10']:.4f} "
            f"| dense MRR={m['dense_mrr']:.4f} @10={m['dense_mrr@10']:.4f} R@1={m['dense_recall@1']:.4f} "
            f"| overlap true/rand={m['overlap_true']:.1f}/{m['overlap_rand']:.2f} tied={m['tied@true']:.0f} "
            f"| codes/part={m['code_usage']:.1f} ppl={code_ppl:.1f} "
            f"postings={len(unique_docs)/max(code_ppl, 1e-9):.0f}")

    # ---- Validation before any training, as a reference.
    accelerator.print("Evaluating pretrained baseline (no fine-tuning) ...")
    base_val_loss, base_metrics = run_eval()
    log_eval("Baseline (pretrained)", base_val_loss, base_metrics)
    baseline_pdim = base_metrics["pdim_mrr"]
    accelerator.wait_for_everyone()

    best_pdim = -float("inf")
    epochs_no_improve = 0
    n_batches = len(dataloader)
    train_start = time.time()
    for epoch in range(EPOCHS):
        # ---- One epoch of training.
        model.train()
        total_loss = 0.0
        for query_batch, doc_batch, doc_ids in dataloader:
            optimizer.zero_grad()
            # Forward and backward both happen inside cached_compute_loss.
            loss = cached_compute_loss(model, accelerator, query_batch, doc_batch, doc_ids,
                                       MINI_BATCH_SIZE, device, train=True,
                                       cosine_temp=COSINE_TEMP)
            accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item()

        avg_val_loss, metrics = run_eval()
        log_eval(f"Epoch {epoch + 1}/{EPOCHS}", avg_val_loss, metrics,
                 train_loss=total_loss / n_batches)

        # ---- Keep the checkpoint if PDIM improved; else count toward patience.
        pdim = broadcast(torch.tensor([metrics["pdim_mrr"]], device=device, dtype=torch.float32),
                         from_process=0).item()
        if pdim > best_pdim:
            best_pdim = pdim
            epochs_no_improve = 0
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                # Sidecar file describing the architecture (and the split, if any).
                arch = {"level": LEVEL, "n_levels": n_levels, "proj_dim": args.proj_dim,
                        "sbert_model": args.sbert_model}
                if args.split_dir:
                    with open(os.path.join(args.split_dir, "manifest.json")) as fh:
                        man = json.load(fh)
                    arch.update({"split_dir": args.split_dir,
                                 "d1_sha256": man["sha256"]["d1_docs.txt"],
                                 "n_d0": man["n"]["d0_docs"], "n_d1": man["n"]["d1_docs"]})
                with open(ckpt_path + ".arch.json", "w") as fh:
                    json.dump(arch, fh, indent=2)
                accelerator.print(f"  -> New best PDIM={best_pdim:.4f}, saved {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                accelerator.print(f"Early stopping triggered (no PDIM improvement for {PATIENCE} epochs).")
                break

        accelerator.wait_for_everyone()  # resync before the next epoch (main saved alone)

    total_time = time.time() - train_start
    accelerator.print(f"Best PDIM={best_pdim:.4f} vs pretrained baseline={baseline_pdim:.4f}")
    accelerator.print(f"Total training time: {total_time / 60:.1f} min ({total_time:.1f} s)")


if __name__ == "__main__":
    main()
