"""Common head interface for the cluster-aware Orthrus output heads.

All heads consume the K diffusion hidden states {h_1..h_K} per block plus
a pooled context summary and produce:

  - log_q_joint(c_1..c_K)  used in the NLL loss against AR-teacher tokens
  - log_q_cluster_marginal(c)  per position, used to compute the model's
                               token marginal (for marginal-KL loss / sampling)
  - log_p_token_given_cluster(y | c)  per position, restricted softmax

The composite joint over y_{1:K} factors as
    q_theta(y_{1:K}) = q_CP(kappa(y_1)..kappa(y_K)) * prod_k p(y_k | kappa(y_k), h_k).
For the factorized heads (cluster_ff, cluster_ff_gain), q_CP itself
factorizes — the same interface accommodates both.

All log-probabilities are returned in natural log.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..clustering import ClusterSystem


@dataclass
class HeadOutput:
    """Output bundle for one batch of anchor blocks.

    All tensors share leading dims [B_anchors, K] except joint and pool.
        log_q_cluster_marginal: [B, K, C]
        log_p_token_in_cluster: [B, K, max_tpc]   (restricted softmax)
        log_q_joint_clusters:   scalar function evaluated on target clusters
                                returned as [B] via .joint_log_prob(c_target).
        log_w_post:             [B, r] mixture posterior after observing prefix
                                (None for factorized heads, used at inference)
    """

    log_q_cluster_marginal: torch.Tensor
    log_p_token_in_cluster: torch.Tensor
    joint_log_prob: callable  # (cluster_targets [B, K]) -> [B] log q_CP(c_1..c_K)
    log_w_pre: torch.Tensor | None = None  # [B, r] for CP head, else None
    aux: dict | None = None  # extra metrics e.g. mixture entropy


class ClusterHead(nn.Module):
    """Abstract head. Subclasses implement forward()."""

    def __init__(
        self,
        hidden_size: int,
        clusters: ClusterSystem,
        K: int,
        *,
        tie_token_embeddings: bool = True,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.K = K
        self.C = clusters.num_clusters
        self.V = clusters.vocab_size
        self.tpc = clusters.max_tokens_per_cluster
        self.tie_token_embeddings = tie_token_embeddings
        self.init_scale = init_scale

        # buffers — moved with the module
        self.register_buffer("token_to_cluster", clusters.token_to_cluster.long(), persistent=False)
        self.register_buffer("token_ordering", clusters.token_ordering.long(), persistent=False)
        self.register_buffer("cluster_sizes", clusters.cluster_sizes.long(), persistent=False)

        # Frozen token embeddings (tied to backbone). The trainable per-position
        # token head is a diagonal-plus-bias correction on top of the embedding
        # dot products — design-doc §3.3.
        self.token_gain = nn.Parameter(torch.ones(self.tpc))                # [max_tpc]
        self.token_bias = nn.Parameter(torch.zeros(self.tpc))               # [max_tpc]
        # placeholder buffer; filled by attach_token_embeddings()
        self.register_buffer("_token_embed", torch.empty(0), persistent=False)

    def attach_token_embeddings(self, E: torch.Tensor) -> None:
        """Attach the [V, d] backbone token embedding table (always frozen)."""
        assert self.tie_token_embeddings, "non-tied embeddings not supported in first cut"
        self._token_embed = E.detach()

    # ---- subclass API ----
    def forward(
        self,
        h_diff: torch.Tensor,        # [B, K, d]   diffusion hidden states
        h_pool: torch.Tensor,        # [B, d]      pooled context summary
    ) -> HeadOutput:
        raise NotImplementedError

    # ---- shared utilities ----
    def restricted_token_log_probs(
        self,
        h_diff: torch.Tensor,        # [B, K, d]
        cluster_ids: torch.Tensor,   # [B, K]  cluster id per position
    ) -> torch.Tensor:
        """log p_theta(y_k | c_k=cluster_ids[b,k], h_k) for tokens in c_k.

        Returns [B, K, max_tpc] log-softmax over (padded) tokens-in-cluster.
        Pad positions are -inf and get zero softmax mass.
        """
        if self._token_embed.numel() == 0:
            raise RuntimeError("attach_token_embeddings(E) must be called once")
        B, K, d = h_diff.shape
        # gather token ids for each (b, k) cluster: [B, K, tpc]
        ids = self.token_ordering[cluster_ids]                              # [B, K, tpc] long, -1 pad
        valid = ids >= 0                                                     # [B, K, tpc]
        ids_clamped = ids.clamp_min(0)
        # E[ids]: [B, K, tpc, d]
        E_sel = self._token_embed[ids_clamped]                              # [B, K, tpc, d]
        # logits = h * E^T (dot product over d)
        logits = torch.einsum("bkd,bktd->bkt", h_diff, E_sel)               # [B, K, tpc]
        # gain + bias correction
        logits = logits * self.token_gain.view(1, 1, -1) + self.token_bias.view(1, 1, -1)
        logits = logits.masked_fill(~valid, float("-inf"))
        return logits.log_softmax(dim=-1)                                   # [B, K, tpc]

    def target_token_log_probs(
        self,
        h_diff: torch.Tensor,        # [B, K, d]
        target_tokens: torch.Tensor, # [B, K]
        target_clusters: torch.Tensor, # [B, K]
    ) -> torch.Tensor:
        """Lookup log p_theta(y_k* | c(y_k*), h_k) without dense cluster padding."""
        if self._token_embed.numel() == 0:
            raise RuntimeError("attach_token_embeddings(E) must be called once")

        B, K, d = h_diff.shape
        h_flat = h_diff.reshape(B * K, d)
        token_flat = target_tokens.reshape(B * K)
        cluster_flat = target_clusters.reshape(B * K)
        out = torch.full(
            (B * K,),
            float("-inf"),
            dtype=h_diff.dtype,
            device=h_diff.device,
        )

        for c in cluster_flat.unique():
            c_int = int(c.item())
            row_mask = cluster_flat == c_int
            token_ids = self.token_ordering[c_int]
            token_ids = token_ids[token_ids >= 0]
            if token_ids.numel() == 0:
                continue

            E_c = self._token_embed[token_ids].to(dtype=h_diff.dtype)        # [tpc_c, d]
            logits = h_flat[row_mask] @ E_c.t()                              # [n_c, tpc_c]
            logits = logits * self.token_gain[:token_ids.numel()].to(logits.dtype).view(1, -1)
            logits = logits + self.token_bias[:token_ids.numel()].to(logits.dtype).view(1, -1)

            targets = token_flat[row_mask]
            matches = token_ids.view(1, -1) == targets.view(-1, 1)
            any_match = matches.any(dim=-1)
            pos = matches.float().argmax(dim=-1)
            picked = logits.log_softmax(dim=-1).gather(-1, pos.unsqueeze(-1)).squeeze(-1)
            out[row_mask] = torch.where(any_match, picked, torch.full_like(picked, float("-inf")))

        return out.view(B, K)

    def gather_token_log_probs(
        self,
        log_p_token_in_cluster: torch.Tensor,   # [B, K, tpc] — already log-softmax
        target_tokens: torch.Tensor,            # [B, K] token ids
        target_clusters: torch.Tensor,          # [B, K] cluster ids of target tokens
    ) -> torch.Tensor:
        """Lookup log p_theta(y_k* | c(y_k*), h_k) for each (b, k).

        Returns [B, K] log-probabilities (-inf if target outside cluster ordering).
        """
        B, K, tpc = log_p_token_in_cluster.shape
        # find the position of target_tokens within their cluster ordering row
        ids = self.token_ordering[target_clusters]                          # [B, K, tpc]
        match = ids == target_tokens.unsqueeze(-1)                          # [B, K, tpc]
        # exactly one True per row (hard partition); allow no-match returning -inf
        pos = match.float().argmax(dim=-1)                                  # [B, K]
        any_match = match.any(dim=-1)                                       # [B, K]
        out = log_p_token_in_cluster.gather(-1, pos.unsqueeze(-1)).squeeze(-1)
        out = torch.where(any_match, out, torch.full_like(out, float("-inf")))
        return out
