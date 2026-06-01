"""Frozen Orthrus backbone wrapper.

Responsibilities:
  - locate / install the orthrus package (added to sys.path if not installed)
  - load the Orthrus checkpoint, freeze all parameters
  - expose .forward_features(input_ids, ar_seq_len, anchor_positions) which
    returns (h_diff, h_pool, teacher_logits) for B anchor blocks.

We do NOT reimplement the dual-pass attention or diffusion masking — we
call into the released Orthrus model code. We DO insert a thin extraction
layer that pulls diffusion hidden states at the anchor block positions
and AR teacher logits at the same clean-context positions.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger


def _ensure_orthrus_on_path() -> None:
    """Add ../orthrus to sys.path if the package isn't already importable."""
    try:
        import orthrus  # noqa: F401
        return
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    candidate = here.parents[1] / "orthrus"  # repo-sibling layout
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        logger.info(f"added {candidate} to sys.path for orthrus import")


_ensure_orthrus_on_path()

try:
    from src.model import OrthrusLM
    from src.configuration import OrthrusConfig
except ImportError:
    # When orthrus is installed as a package, layout may differ — try
    # `orthrus.src.model` as a fallback.
    from orthrus.src.model import OrthrusLM  # type: ignore
    from orthrus.src.configuration import OrthrusConfig  # type: ignore


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
        logger.info(f"loading Orthrus checkpoint: {checkpoint}")
        self.model = OrthrusLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
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

        Layout follows Orthrus training: the input to the model is the
        concatenation `[clean_seq, corrupted_blocks]` where each corrupted
        block is `[x_{a}, <mask>, ..., <mask>]` (K positions). The dual-pass
        attention mask routes clean tokens through causal AR attention and
        corrupted blocks through bidirectional-within-block + causal-up-to-anchor.

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

        # 3) concat clean + corrupted along sequence, set is_diffusion_pass
        full_ids = torch.cat([input_ids, blocks_flat], dim=1)               # [B, L + A*K]

        out = self.model(
            input_ids=full_ids,
            is_diffusion_pass=True,
            ar_seq_len=L,
            causal_limit=causal_limit,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        # `out.hidden_states[0]` would hold full hidden states — but the
        # OrthrusLM.forward we read returns hidden_states as a single-tuple of
        # the last hidden state. For robustness we re-run through .model
        # below to recover it explicitly.

        # The OrthrusLM forward computes hidden_states then applies lm_head
        # only to a slice. We need both the diffusion part of hidden_states
        # (for our head) AND lm_head logits on the AR part. The cleanest path
        # is to call self.model.model directly to get full hidden states.
        base_out = self.model.model(
            input_ids=full_ids,
            is_diffusion_pass=True,
            ar_seq_len=L,
            causal_limit=causal_limit,
            use_cache=False,
        )
        h_all = base_out.last_hidden_state                                  # [B, L + A*K, d]

        h_clean = h_all[:, :L, :]                                           # [B, L, d]
        h_blocks = h_all[:, L:, :].view(B, A, K, -1)                        # [B, A, K, d]

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
