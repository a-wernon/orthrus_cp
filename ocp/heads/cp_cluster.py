"""CP-cluster joint head.

    q_CP(c_1..c_K | h_{1:K}, h_pool) = sum_alpha w_alpha(h_pool) *
                                       prod_k q_{k, alpha}(c_k | h_k)

Shared-trunk parameterisation per the design doc:
    W_{k, alpha} = U_alpha + P_k        (low-rank component + per-position bias)

We expose:
  - log_q_cluster_marginal[b, k, c] = log sum_alpha w_alpha q_{k,alpha}(c)
  - joint_log_prob(c_target)        = log sum_alpha [log w_alpha + sum_k log q_{k,alpha}(c_target[b,k])]
  - log_w_pre                       = log mixture weights pre-prefix update

Per-position conditional q_CP(c_k | c_<k) is computable in O(r) by
mixture-of-products posterior update — implemented as helper on the
HeadOutput.aux for inference-time consensus.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ClusterHead, HeadOutput


class CPClusterHead(ClusterHead):
    def __init__(
        self,
        hidden_size: int,
        clusters,
        K: int,
        *,
        rank: int = 16,
        pool: str = "mean",
        **kwargs,
    ):
        super().__init__(hidden_size, clusters, K, **kwargs)
        self.rank = rank
        self.pool = pool

        # mixture-weight head: h_pool -> r logits
        self.weight_proj = nn.Linear(hidden_size, rank, bias=True)
        # per-component projection: applied to every position's hidden state
        # to produce r per-cluster logits, then position bias adds a (k, alpha, c)
        # term. Shape: U: [r, hidden, C]; P: [K, r, C].
        # Implementation: a single Linear(hidden, r * C) shared across positions
        # + an additive [K, r, C] position bias.
        self.component_proj = nn.Linear(hidden_size, rank * self.C, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(K, rank, self.C))

        nn.init.normal_(self.weight_proj.weight, std=self.init_scale)
        nn.init.zeros_(self.weight_proj.bias)
        nn.init.normal_(self.component_proj.weight, std=self.init_scale)

    def _pool(self, h_diff: torch.Tensor, h_pool_override: torch.Tensor | None) -> torch.Tensor:
        if h_pool_override is not None:
            return h_pool_override
        if self.pool == "mean":
            return h_diff.mean(dim=1)
        if self.pool == "first":
            return h_diff[:, 0, :]
        if self.pool == "last":
            return h_diff[:, -1, :]
        raise ValueError(f"unknown pool {self.pool!r}")

    def forward(self, h_diff: torch.Tensor, h_pool: torch.Tensor | None = None) -> HeadOutput:
        B, K, d = h_diff.shape
        assert K == self.K, f"head built for K={self.K} but got K={K}"
        r, C = self.rank, self.C

        h_p = self._pool(h_diff, h_pool)                                    # [B, d]
        log_w = self.weight_proj(h_p).log_softmax(dim=-1)                   # [B, r]

        # per-component, per-position cluster log-probs:
        # comp[b, k, alpha, c] = log_softmax_c( (component_proj(h_k) + position_bias[k])_alpha )
        comp = self.component_proj(h_diff)                                  # [B, K, r*C]
        comp = comp.view(B, K, r, C) + self.position_bias.view(1, K, r, C)  # [B, K, r, C]
        log_q_per_comp = comp.log_softmax(dim=-1)                           # [B, K, r, C]

        # cluster marginal at each position:
        # log q(c_k) = logsumexp_alpha (log w_alpha + log q_{k,alpha}(c_k))
        log_q_c_marg = torch.logsumexp(
            log_w.view(B, 1, r, 1) + log_q_per_comp,
            dim=2,
        )                                                                    # [B, K, C]

        # joint log-prob factory — captured closures over the per-component grid
        def joint_log_prob(c_target: torch.Tensor) -> torch.Tensor:
            """c_target: [B, K] long.

            Returns [B] log q_CP(c_{1:K}).

            log q_CP(c_{1:K}) = logsumexp_alpha [log w_alpha + sum_k log q_{k,alpha}(c_k)]
            """
            # pick log q_{k,alpha}(c_target[b,k]) -> [B, K, r]
            idx = c_target.view(B, K, 1, 1).expand(B, K, r, 1)
            picked = log_q_per_comp.gather(-1, idx).squeeze(-1)              # [B, K, r]
            comp_sum = picked.sum(dim=1)                                     # [B, r]
            return torch.logsumexp(log_w + comp_sum, dim=-1)                 # [B]

        # mixture entropy diagnostic (per-batch scalar avg)
        mix_ent = -(log_w.exp() * log_w).sum(dim=-1).mean()

        return HeadOutput(
            log_q_cluster_marginal=log_q_c_marg,
            log_p_token_in_cluster=None,
            joint_log_prob=joint_log_prob,
            log_w_pre=log_w,
            aux={
                "mixture_entropy": mix_ent.detach(),
                "log_q_per_component": log_q_per_comp,                       # [B, K, r, C] - for inference
            },
        )

    @staticmethod
    def per_position_conditional(
        log_q_per_comp: torch.Tensor,    # [B, K, r, C]
        log_w_pre: torch.Tensor,         # [B, r]
        c_prefix: torch.Tensor,          # [B, k]  cluster ids already observed (k may be 0)
        k_target: int,                   # position to condition for
    ) -> torch.Tensor:
        """Return log q_CP(c_{k_target} | c_{<k_target}, ctx) — shape [B, C].

        Mixture posterior update:
            log w_alpha^(k) = log w_alpha + sum_{k'<k} log q_{k', alpha}(c_{k'})  (renormalised)
            log q(c_k | c_<k) = logsumexp_alpha (log w_alpha^(k) + log q_{k, alpha}(c_k))

        Cost: O(r * C + r * k).
        """
        B, K, r, C = log_q_per_comp.shape
        if c_prefix.shape[1] == 0:
            log_w_post = log_w_pre
        else:
            k_obs = c_prefix.shape[1]
            idx = c_prefix.view(B, k_obs, 1, 1).expand(B, k_obs, r, 1)
            picked = log_q_per_comp[:, :k_obs].gather(-1, idx).squeeze(-1)   # [B, k_obs, r]
            log_w_post = (log_w_pre + picked.sum(dim=1))                     # [B, r]
            log_w_post = log_w_post - torch.logsumexp(log_w_post, dim=-1, keepdim=True)
        return torch.logsumexp(
            log_w_post.view(B, r, 1) + log_q_per_comp[:, k_target],          # [B, r, C]
            dim=1,
        )                                                                    # [B, C]
