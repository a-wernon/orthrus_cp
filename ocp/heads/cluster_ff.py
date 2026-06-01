"""Honest baseline: per-position cluster scoring head, no joint structure.

    q(c_1..c_K | h_{1:K}) = prod_k softmax(W_c h_k)

A direct lift of Orthrus's factorized output, projected through the
cluster partition. This is the weakest cluster-aware baseline — it
matches everything the CP head does except the joint mixture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ClusterHead, HeadOutput


class ClusterFFHead(ClusterHead):
    def __init__(self, hidden_size: int, clusters, K: int, **kwargs):
        super().__init__(hidden_size, clusters, K, **kwargs)
        self.cluster_proj = nn.Linear(hidden_size, self.C, bias=True)
        nn.init.normal_(self.cluster_proj.weight, std=self.init_scale)
        nn.init.zeros_(self.cluster_proj.bias)

    def forward(self, h_diff: torch.Tensor, h_pool: torch.Tensor) -> HeadOutput:
        B, K, d = h_diff.shape
        cluster_logits = self.cluster_proj(h_diff)                          # [B, K, C]
        log_q_c_marg = cluster_logits.log_softmax(dim=-1)                   # [B, K, C]

        # joint over cluster trajectory factorizes
        def joint_log_prob(c_target: torch.Tensor) -> torch.Tensor:         # [B, K] -> [B]
            picked = log_q_c_marg.gather(-1, c_target.unsqueeze(-1)).squeeze(-1)
            return picked.sum(dim=-1)

        # within-cluster head is computed lazily per call site (it depends on
        # which cluster we're conditioning on). At training time the caller
        # passes the target cluster ids; here we expose a placeholder that
        # the training loop will fill in.
        log_p_token = None  # caller fills via restricted_token_log_probs

        return HeadOutput(
            log_q_cluster_marginal=log_q_c_marg,
            log_p_token_in_cluster=log_p_token,
            joint_log_prob=joint_log_prob,
            log_w_pre=None,
            aux={"mixture_entropy": None},
        )
