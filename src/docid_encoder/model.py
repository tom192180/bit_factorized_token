"""The DocID encoder: text -> k values in (-1, 1) -> k bits.

    x      = backbone(text), mean-pooled and L2-normalised          (hidden size)
    z_cont = tanh(LayerNorm(W x))                                   (k values; the loss trains these)
    bits   = (LayerNorm(W x) >= 0)                                  (k bits; the DocID)
"""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize


def ensure_normalize(st: SentenceTransformer) -> SentenceTransformer:
    """Append an L2 Normalize module if the backbone does not already end in one."""
    if not any(isinstance(m, Normalize) for m in st):
        st.append(Normalize())
    return st


class Encoder_bit(nn.Module):
    def __init__(
        self,
        sbert_model: str = "Qwen/Qwen3-Embedding-0.6B",
        max_seq_length: int = 256,
        proj_dim: int | None = None,
        query_prompt: str | None = None,
        doc_prompt: str | None = None,
    ):
        """proj_dim is the DocID length k in bits; None uses the backbone's hidden size."""
        super().__init__()
        # Load the backbone in float32.
        self.sbert = SentenceTransformer(sbert_model,
                                         model_kwargs={"torch_dtype": torch.float32})
        ensure_normalize(self.sbert)

        # Text prefixes for queries and documents (empty for BERT and T5).
        self.query_prompt, self.doc_prompt = self._resolve_prompts(
            sbert_model, query_prompt, doc_prompt)
        # Maximum input length in tokens, for queries and documents.
        self.sbert.max_seq_length = max_seq_length

        # Freeze BERT's unused [CLS] pooler head: mean pooling never calls it.
        auto = self.sbert._first_module().auto_model
        pooler = getattr(auto, "pooler", None)
        if pooler is not None:
            for p in pooler.parameters():
                p.requires_grad_(False)
        hidden_dim = self.sbert.get_sentence_embedding_dimension()

        self.proj_dim = proj_dim
        if proj_dim is not None:
            # Projection W: hidden size -> k, no bias.
            self.code_dim = proj_dim
            self.proj = nn.Linear(hidden_dim, proj_dim, bias=False)
            if proj_dim == hidden_dim:
                # Same size in and out: start as the identity.
                with torch.no_grad():
                    self.proj.weight.copy_(torch.eye(hidden_dim))
        else:
            # No projection.
            self.code_dim = hidden_dim
            self.proj = nn.Identity()

        # LayerNorm over the k projected values (skipped when there is no projection).
        self.ln = nn.Identity() if self.proj_dim is None else nn.LayerNorm(self.code_dim)

    def _resolve_prompts(self, sbert_model, query_prompt, doc_prompt):
        """(query_prompt, doc_prompt) prefix strings for this backbone."""
        if query_prompt is not None or doc_prompt is not None:
            return query_prompt or "", doc_prompt or ""
        name = sbert_model.lower()
        if "e5" in name:  # E5 was trained with these exact prefixes
            return "query: ", "passage: "
        # Otherwise use the model's own prompts, if it has any.
        prompts = getattr(self.sbert, "prompts", {}) or {}
        return prompts.get("query", ""), prompts.get("document", "")

    def trainable_parameter_summary(self):
        """(trainable, total) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def gradient_checkpointing_enable(self):
        """Trade recompute for activation memory in the backbone."""
        self.sbert._first_module().auto_model.gradient_checkpointing_enable()

    def forward(self, sentences, prompt_name=None):
        """sentences: list of str. prompt_name="query" marks queries.

        Returns (x, z_cont, bits): the embedding, the k values in (-1, 1), and the k bits."""
        prefix = self.query_prompt if prompt_name == "query" else self.doc_prompt
        if prefix:
            sentences = [prefix + s for s in sentences]

        features = self.sbert.tokenize(sentences)
        # Move tensor features to the model's device; leave non-tensor entries as they are.
        device = next(self.sbert.parameters()).device
        features = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in features.items()}
        x = self.sbert(features)["sentence_embedding"]

        x_norm = self.ln(self.proj(x))
        z_cont = torch.tanh(x_norm)
        # >= 0 so that an exact zero is still a valid bit.
        bits = (x_norm >= 0).to(torch.int8)
        return x, z_cont, bits
