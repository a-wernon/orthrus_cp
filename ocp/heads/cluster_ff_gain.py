"""Stronger honest baseline: 2-layer MLP + gain modulation per position.

Matches the parameter count of cp_cluster_rN within ~50%, while staying
strictly factorized. This is the head that the Qwen3-CP NO-GO identified
as the right comparison target; CP must beat it by >=2% to pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ClusterHead, HeadOutput


class ClusterFFGainHead(ClusterHead):
    def __init__(
        self,
        hidden_size: int,
        clusters,
        K: int,
        *,
        mlp_intermediate: int = 4096,
        **kwargs,
    ):
        super().__init__(hidden_size, clusters, K, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_intermediate),
            nn.GELU(),
            nn.Linear(mlp_intermediate, hidden_size),
        )
        self.gain = nn.Parameter(torch.ones(K, hidden_size))                # per-position scaling
        self.cluster_proj = nn.Linear(hidden_size, self.C, bias=True)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.init_scale)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cluster_proj.weight, std=self.init_scale)
        nn.init.zeros_(self.cluster_proj.bias)

    def forward(self, h_diff: torch.Tensor, h_pool: torch.Tensor) -> HeadOutput:
        B, K, d = h_diff.shape
        h = self.mlp(h_diff) + h_diff                                       # residual MLP
        h = h * self.gain.unsqueeze(0)                                      # per-position gain
        cluster_logits = self.cluster_proj(h)                               # [B, K, C]
        log_q_c_marg = cluster_logits.log_softmax(dim=-1)

        def joint_log_prob(c_target: torch.Tensor) -> torch.Tensor:
            picked = log_q_c_marg.gather(-1, c_target.unsqueeze(-1)).squeeze(-1)
            return picked.sum(dim=-1)

        return HeadOutput(
            log_q_cluster_marginal=log_q_c_marg,
            log_p_token_in_cluster=None,
            joint_log_prob=joint_log_prob,
            log_w_pre=None,
            aux={"mixture_entropy": None},
        )
