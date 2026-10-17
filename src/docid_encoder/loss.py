"""The DocID encoder's loss: InfoNCE on cosine similarity, with in-batch negatives.

    cosine_infonce        the loss
    false_negative_mask   ignores other copies of a query's own document in the batch
    GatherLayer           shares document embeddings across GPUs, with gradients
    cached_compute_loss   computes the full-batch loss in memory-sized chunks (GradCache)

The loss is computed on z_cont, the encoder's output after tanh.
"""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.distributed as dist

from config import COSINE_TEMP


def _dist_ready():
    """True when training runs on more than one process (GPU)."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class GatherLayer(torch.autograd.Function):
    """Collect a tensor from every GPU, keeping gradients.

    forward concatenates every GPU's tensor. backward sums the gradients computed on all GPUs
    and returns the slice that belongs to this GPU."""

    @staticmethod
    def forward(ctx, x):
        if not _dist_ready():
            return x
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx, grad):
        if not _dist_ready():
            return grad
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        b = grad.size(0) // dist.get_world_size()
        r = dist.get_rank()
        return grad[r * b : (r + 1) * b]


def all_gather_nograd(x):
    """Collect a tensor from every GPU, without gradients (used for document ids)."""
    if not _dist_ready():
        return x
    out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


def false_negative_mask(doc_ids, global_doc_ids=None, positive_offset=0):
    """(n_queries, n_documents) mask: True where a column holds the query's own document but is
    not its positive column. positive_offset is where this GPU's documents start."""
    if global_doc_ids is None:
        global_doc_ids = doc_ids
    same = doc_ids.unsqueeze(1) == global_doc_ids.unsqueeze(0)  # (B, G)
    pos_cols = torch.arange(doc_ids.size(0), device=doc_ids.device) + positive_offset
    cols = torch.arange(global_doc_ids.size(0), device=doc_ids.device)
    is_positive = cols.unsqueeze(0) == pos_cols.unsqueeze(1)  # (B, G)
    return same & ~is_positive


def cosine_infonce(x1, x2, mask=None, labels=None, temperature=COSINE_TEMP):
    """Cross-entropy over cosine similarities divided by the temperature.

    x1: query vectors, x2: document vectors, labels: each query's positive column (default: the
    diagonal). Masked entries are excluded from the softmax."""
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    logits = (x1 @ x2.T) / temperature
    if labels is None:
        labels = torch.arange(x1.size(0), device=x1.device)
    if mask is not None:
        logits = logits.masked_fill(mask, float("-inf"))
    return F.cross_entropy(logits, labels)


def _chunks(seq, size):
    """Yield (start_index, slice) pairs."""
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def _get_rng_state(device):
    cuda = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda


def _set_rng_state(state, device):
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda, device)


def cached_compute_loss(model, accelerator, query_batch, doc_batch, doc_ids,
                        mini_batch_size, device, train=True, cosine_temp=COSINE_TEMP):
    """Full-batch contrastive loss with memory bounded by mini_batch_size (GradCache).

    Returns the loss value; the weight gradients are already accumulated when it returns.
    With train=False it only computes the loss."""
    # Phase 1: embed every chunk without gradients, documents first, then queries. The random
    # state is saved per chunk so phase 3 repeats the same dropout.
    rng_states = []
    zc2_parts, zc1_parts = [], []
    with torch.no_grad():
        for _, dc in _chunks(doc_batch, mini_batch_size):
            rng_states.append(_get_rng_state(device))
            zc2_parts.append(model(dc)[1])
        for _, qc in _chunks(query_batch, mini_batch_size):
            rng_states.append(_get_rng_state(device))
            zc1_parts.append(model(qc)[1])

    zc1 = torch.cat(zc1_parts, dim=0)
    zc2 = torch.cat(zc2_parts, dim=0)

    # Phase 2: the loss on all embeddings, giving a gradient per embedding.
    local_ids = doc_ids.to(device)
    if not train:
        with torch.no_grad():
            return cosine_infonce(zc1, zc2, false_negative_mask(local_ids), None,
                                  cosine_temp).detach()

    zc1 = zc1.detach().requires_grad_(True); zc2 = zc2.detach().requires_grad_(True)

    if _dist_ready():
        zc2_all = GatherLayer.apply(zc2)  # documents from all GPUs
        ids_all = all_gather_nograd(local_ids)
        offset = dist.get_rank() * zc2.size(0)  # where this GPU's documents start
    else:
        zc2_all, ids_all, offset = zc2, local_ids, 0

    mask = false_negative_mask(local_ids, ids_all, offset)
    labels = torch.arange(zc1.size(0), device=device) + offset
    loss = cosine_infonce(zc1, zc2_all, mask, labels, cosine_temp)
    loss.backward()  # stops at the embeddings
    gzc1, gzc2 = zc1.grad, zc2.grad

    # Phase 3: encode each chunk again with gradients and backpropagate its embedding
    # gradients. GPUs synchronise on the last chunk only.
    steps = ([("d", i, c) for i, c in _chunks(doc_batch, mini_batch_size)]
             + [("q", i, c) for i, c in _chunks(query_batch, mini_batch_size)])
    for k, (kind, start, chunk) in enumerate(steps):
        _set_rng_state(rng_states[k], device)
        ctx = nullcontext() if k == len(steps) - 1 else accelerator.no_sync(model)
        with ctx:
            if kind == "q":
                zc_c = model(chunk)[1]
                gzc = gzc1
            else:
                zc_c = model(chunk)[1]
                gzc = gzc2
            n = len(chunk)
            if gzc is not None:
                torch.autograd.backward([zc_c], [gzc[start : start + n]])

    return loss.detach()
