"""Frozen Orthrus backbone wrapper.

Responsibilities:
  - load the Orthrus checkpoint via `AutoModelForCausalLM.from_pretrained(...,
    trust_remote_code=True)` — matches the model card's recipe, and pulls
    the modeling code straight from the Hub repo so we don't have to track
    the local `orthrus/` clone.
  - freeze all parameters, expose .forward_features() which returns
    (h_diff, h_pool, teacher_logits, target_tokens) for B anchor blocks.

We do NOT reimplement the dual-pass attention — we call the released model's
forward with `is_diffusion_pass=True`, `ar_seq_len`, `causal_limit`. We DO
insert a thin extraction layer that pulls diffusion hidden states at the
anchor block positions and AR teacher logits at the clean-context positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


@dataclass
class FrozenFeatures:
    """Output of forward_features().

    h_diff:           [B, K, d]   diffusion hidden states at the K positions
                                  of each anchor block
    h_pool:           [B, d]      pooled context (mean of h_diff across K)
    teacher_logits:   [B, K, V]   AR teacher distribution at clean positions
                                  (a_b + k - 1) — for marginal-KL loss
    target_tokens:    [B, K]      clean AR-greedy tokens y_k* (== input ids
                                  at the corresponding clean positions)
    """

    h_diff: torch.Tensor
    h_pool: torch.Tensor
    teacher_logits: torch.Tensor
    target_tokens: torch.Tensor


class FrozenOrthrus(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()
        logger.info(f"loading Orthrus checkpoint: {checkpoint} ({attn_implementation})")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                dtype=dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        except TypeError:
            # Older transformers used `torch_dtype` instead of `dtype`.
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.dtype = dtype
        self.K = int(self.model.config.block_size)
        self.mask_token_id = int(self.model.config.mask_token_id)
        self.hidden_size = int(self.model.config.hidden_size)
        self.vocab_size = int(self.model.config.vocab_size)

    @property
    def input_embeddings(self) -> torch.Tensor:
        """[V, d] frozen input embedding matrix — used for k-means and head tying."""
        return self.model.model.embed_tokens.weight.detach()

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """[V, d] AR LM head weight — tied to embeddings in Orthrus checkpoints."""
        return self.model.lm_head.weight.detach()

    @torch.no_grad()
    def forward_features(
        self,
        input_ids: torch.Tensor,        # [B, L]  clean sequence
        anchor_positions: torch.Tensor, # [B, A]  per-sequence anchor indices (L-K+1 max)
    ) -> FrozenFeatures:
        """Run one dual-pass forward and extract diffusion features + AR logits.

        Layout follows the released Orthrus model: first run the clean sequence
        through the AR path to populate the KV cache, then run corrupted blocks
        `[x_a, <mask>, ..., <mask>]` through the diffusion path against that
        cache. The dual-pass attention mask routes draft tokens through causal
        AR cache tokens plus bidirectional attention within each draft block.

        For each (sequence, anchor) we extract:
          h_diff[b_block, k] = diffusion hidden state of the k-th position
                               in the corrupted block for that anchor
          teacher_logits[b_block, k] = AR LM head applied to clean hidden
                                       state at position (a + k - 1)
          target_tokens[b_block, k] = input_ids[b, a + k - 1]
        """
        B, L = input_ids.shape
        A = anchor_positions.shape[1]
        K = self.K
        device = input_ids.device

        # 1) build the corrupted block stream — total of B*A blocks
        anchor_tokens = input_ids.gather(1, anchor_positions)               # [B, A]
        anchor_tokens = anchor_tokens.unsqueeze(-1)                         # [B, A, 1]
        mask_tail = torch.full(
            (B, A, K - 1), self.mask_token_id, dtype=input_ids.dtype, device=device,
        )
        blocks = torch.cat([anchor_tokens, mask_tail], dim=-1)              # [B, A, K]
        blocks_flat = blocks.view(B, A * K)                                 # [B, A*K]

        # 2) build causal_limit: each diffusion query at position (a, k) attends
        # to clean tokens up to index a (the anchor index). For positions other
        # than the anchor, the limit is the anchor index too (they only see
        # clean context up to the anchor, plus bidirectional within their block).
        causal_limit = (
            anchor_positions.unsqueeze(-1).expand(B, A, K).reshape(B, A * K)
        ).long()                                                            # [B, A*K]

        # 3) populate the AR KV cache, then run the draft blocks as diffusion
        # queries. Use an explicit additive mask with eager attention to avoid
        # Orthrus' training-only compiled flex-attention path.
        past_key_values = DynamicCache(config=self.model.config)
        ar_position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        Q = A * K
        ar_idx = torch.arange(L, device=device).view(1, 1, L)
        valid_ar = ar_idx <= causal_limit.unsqueeze(-1)                     # [B, Q, L]
        q_block = torch.arange(Q, device=device).view(Q, 1) // K
        kv_block = torch.arange(Q, device=device).view(1, Q) // K
        valid_diff = (q_block == kv_block).unsqueeze(0).expand(B, Q, Q)      # [B, Q, Q]
        valid_kv = torch.cat([valid_ar, valid_diff], dim=-1).unsqueeze(1)    # [B, 1, Q, L+Q]
        diff_attention_mask = torch.zeros(
            valid_kv.shape, dtype=self.dtype, device=device,
        ).masked_fill(~valid_kv, torch.finfo(self.dtype).min)

        old_attn_implementation = self.model.config._attn_implementation
        try:
            ar_out = self.model(
                input_ids=input_ids,
                position_ids=ar_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            diff_position_ids = (
                anchor_positions.unsqueeze(-1) + torch.arange(K, device=device).view(1, 1, K)
            ).reshape(B, A * K)
            self.model.config._attn_implementation = "eager"
            diff_out = self.model(
                input_ids=blocks_flat,
                attention_mask=diff_attention_mask,
                position_ids=diff_position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                is_diffusion_pass=True,
                ar_seq_len=L,
                causal_limit=causal_limit,
            )
        finally:
            self.model.config._attn_implementation = old_attn_implementation

        h_clean = ar_out.hidden_states[0]                                    # [B, L, d]
        h_blocks = diff_out.hidden_states[0].view(B, A, K, -1)               # [B, A, K, d]

        # diffusion hidden states for the head: flatten (B, A) -> B_anchors
        h_diff = h_blocks.reshape(B * A, K, -1).to(self.dtype)              # [B*A, K, d]
        h_pool = h_diff.mean(dim=1)                                         # [B*A, d]

        # AR teacher logits at positions (a + k - 1) on the CLEAN side
        # NOTE on indexing: position k=1 in our block aligns with prediction of
        # token x_a (since the block's k=1 slot is the anchor); position k>1
        # aligns with prediction of x_{a + k - 1}. So the teacher distribution
        # we want is p_AR(. | x_{<a+k-1}) — i.e., applied at hidden state
        # h_clean[a + k - 2] (predict-next semantics). The target token is
        # input_ids[a + k - 1].
        # We index safely (clamp to L-1; downstream loss masks invalid).
        idx_h = (anchor_positions.unsqueeze(-1) + torch.arange(K, device=device)
                 .view(1, 1, K) - 1).clamp(0, L - 1)                        # [B, A, K]
        idx_h_flat = idx_h.view(B, A * K).unsqueeze(-1).expand(B, A * K, h_clean.shape[-1])
        h_teacher = h_clean.gather(1, idx_h_flat).view(B, A, K, -1)         # [B, A, K, d]
        teacher_logits = self.model.lm_head(h_teacher.to(self.dtype))        # [B, A, K, V]
        teacher_logits = teacher_logits.float().view(B * A, K, -1)

        target_idx = (anchor_positions.unsqueeze(-1) + torch.arange(K, device=device)
                      .view(1, 1, K)).clamp(0, L - 1)                       # [B, A, K]
        target_tokens = input_ids.gather(1, target_idx.view(B, A * K)).view(B * A, K)

        return FrozenFeatures(
            h_diff=h_diff,
            h_pool=h_pool,
            teacher_logits=teacher_logits,
            target_tokens=target_tokens,
        )
