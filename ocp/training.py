"""Training loop, losses, LR schedule, checkpointing.

Three loss components, with weights controlled by head config:
  L_NLL        = -E [ log q_theta(y_{1:K}^* | ctx) ]
                 (joint NLL of AR-teacher tokens under our composite head)
  L_marg       = sum_k KL( p_AR(. | ctx) || q_theta^marg(. | h_k) )
                 (per-position marginal KL to AR teacher distribution)
  L_ent        = -lambda * H[w(h_pool)]
                 (anti-collapse entropy regulariser on the CP mixture)

Total: L = L_NLL + marginal_kl_weight * L_marg + entropy_reg * L_ent.

The marginal q_theta^marg(y | h_k) under our composite head is
   q^marg(y | h_k) = q^marg_cluster(kappa(y) | h_k) * p_theta(y | kappa(y), h_k)
where q^marg_cluster is the head's log_q_cluster_marginal at position k.
Hard-partition collapse makes this tractable per token.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from .clustering import ClusterSystem
from .heads.base import ClusterHead, HeadOutput
from .model import FrozenOrthrus, FrozenFeatures
from .utils import is_main_process


@dataclass
class StepMetrics:
    loss: float
    nll: float
    marg_kl: float | None
    entropy: float | None
    grad_norm: float
    lr: float


def cosine_with_warmup(step: int, *, max_steps: int, warmup_ratio: float, peak_lr: float, min_lr_ratio: float) -> float:
    warmup_steps = int(max_steps * warmup_ratio)
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)
    coef = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coef)


def compute_loss(
    head_out: HeadOutput,
    feats: FrozenFeatures,
    head: ClusterHead,
    clusters: ClusterSystem,
    *,
    marginal_kl_weight: float,
    entropy_reg: float,
) -> tuple[torch.Tensor, dict]:
    B, K, _ = head_out.log_q_cluster_marginal.shape
    target_tokens = feats.target_tokens.to(head_out.log_q_cluster_marginal.device)
    target_clusters = clusters.token_to_cluster.to(target_tokens.device)[target_tokens]   # [B, K]

    # --- L_NLL: joint over cluster trajectory + per-position token-in-cluster ---
    log_q_clusters = head_out.joint_log_prob(target_clusters)                              # [B]
    log_p_token = head.restricted_token_log_probs(feats.h_diff, target_clusters)           # [B, K, tpc]
    log_p_y_given_c = head.gather_token_log_probs(
        log_p_token, target_tokens, target_clusters,
    )                                                                                      # [B, K]
    # mask out positions with invalid targets (target outside cluster ordering)
    valid = torch.isfinite(log_p_y_given_c).all(dim=-1)                                    # [B]
    nll_joint = -(log_q_clusters + log_p_y_given_c.sum(dim=-1))                            # [B]
    nll_joint = nll_joint[valid].mean()

    loss = nll_joint
    metrics = {"nll": nll_joint.detach()}

    # --- L_marg: per-position CLUSTER-level marginal KL to AR teacher ---
    # We distil at the cluster level only — within-cluster supervision comes from
    # L_NLL above. This is what makes the loss tractable: no V-wide enumeration.
    if marginal_kl_weight > 0.0:
        loss_marg, kl_est = _cluster_marginal_kl(
            head_out=head_out,
            teacher_logits=feats.teacher_logits,
            kappa=clusters.token_to_cluster.to(feats.h_diff.device),
            num_clusters=clusters.num_clusters,
        )
        loss = loss + marginal_kl_weight * loss_marg
        metrics["marg_kl"] = kl_est.detach()

    # --- L_ent: entropy regulariser ---
    if entropy_reg > 0.0 and head_out.aux is not None and head_out.aux.get("mixture_entropy") is not None:
        H_w = head_out.aux["mixture_entropy"]
        loss = loss - entropy_reg * H_w
        metrics["entropy"] = H_w.detach()

    return loss, metrics


def _cluster_marginal_kl(
    *,
    head_out: HeadOutput,
    teacher_logits: torch.Tensor,      # [B, K, V]
    kappa: torch.Tensor,               # [V]
    num_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL between AR teacher's cluster-aggregated marginal and ours.

    p_AR^c(c) = sum_{v : kappa(v) = c} p_AR(v)   — scatter_add over softmax
    q^c(c)    = exp(head_out.log_q_cluster_marginal[b, k, c])

    Returns:
        loss term used for backward (cross-entropy form),
        and the KL estimate as a scalar diagnostic.
    """
    B, K, V = teacher_logits.shape
    C = num_clusters
    device = teacher_logits.device

    p_ar = teacher_logits.float().softmax(dim=-1)                          # [B, K, V]
    cluster_mass = torch.zeros(B, K, C, device=device, dtype=p_ar.dtype)   # [B, K, C]
    kappa_exp = kappa.view(1, 1, V).expand(B, K, V)
    cluster_mass.scatter_add_(2, kappa_exp, p_ar)

    log_p_ar_c = (cluster_mass.clamp_min(1e-30)).log()                     # [B, K, C]
    log_q_c = head_out.log_q_cluster_marginal.float()                      # [B, K, C]

    # forward KL: sum_c p (log p - log q)
    kl = (cluster_mass * (log_p_ar_c - log_q_c)).sum(dim=-1)               # [B, K]
    kl_est = kl.mean()

    # backward signal: cross-entropy form -E_{c~p}[log q_c]; matches the
    # standard "soft-label" distillation Orthrus reports +10% TPF for.
    loss = -(cluster_mass.detach() * log_q_c).sum(dim=-1).mean()
    return loss, kl_est


def clip_grad_norm(params, max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def save_checkpoint(head: ClusterHead, save_dir: Path, step: int, keep_last: int = 3) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    p = save_dir / f"head_step{step:06d}.pt"
    torch.save({"step": step, "state_dict": head.state_dict()}, p)
    # rotate
    existing = sorted(save_dir.glob("head_step*.pt"))
    for old in existing[:-keep_last]:
        old.unlink(missing_ok=True)
    logger.info(f"checkpoint -> {p}")


def train_loop(
    *,
    cfg,
    head: ClusterHead,
    frozen: FrozenOrthrus,
    clusters: ClusterSystem,
    dataloader,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=cfg.train.peak_lr,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )
    head_loss_cfg = cfg.head.loss
    log_dir = Path(cfg.logging.log_dir)
    save_dir = Path(cfg.checkpointing.save_dir)
    writer = SummaryWriter(log_dir=str(log_dir)) if is_main_process() else None

    step = 0
    t0 = time.time()
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    loss_ema = None
    nll_ema = None

    head.train()
    frozen.eval()

    for batch in dataloader:
        if step >= cfg.train.max_steps:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        anchors = batch["anchor_positions"].to(device, non_blocking=True)

        with torch.no_grad():
            feats = frozen.forward_features(input_ids, anchors)

        head_out = head(feats.h_diff, feats.h_pool)
        loss, metrics = compute_loss(
            head_out, feats, head, clusters,
            marginal_kl_weight=head_loss_cfg.marginal_kl_weight,
            entropy_reg=head_loss_cfg.entropy_reg,
        )
        (loss / cfg.train.grad_accum_steps).backward()
        accum += 1

        if accum < cfg.train.grad_accum_steps:
            continue

        # --- optimizer step ---
        grad_norm = clip_grad_norm(head.parameters(), cfg.train.grad_clip)
        lr = cosine_with_warmup(
            step,
            max_steps=cfg.train.max_steps,
            warmup_ratio=cfg.train.warmup_ratio,
            peak_lr=cfg.train.peak_lr,
            min_lr_ratio=cfg.train.min_lr_ratio,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        # --- logging ---
        nll = float(metrics["nll"].item())
        a = cfg.logging.ema_alpha
        nll_ema = nll if nll_ema is None else a * nll_ema + (1 - a) * nll
        loss_v = float(loss.item())
        loss_ema = loss_v if loss_ema is None else a * loss_ema + (1 - a) * loss_v

        if writer is not None and step % 10 == 0:
            writer.add_scalar("train/loss", loss_v, step)
            writer.add_scalar("train/loss_ema", loss_ema, step)
            writer.add_scalar("train/nll", nll, step)
            writer.add_scalar("train/nll_ema", nll_ema, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", grad_norm, step)
            if "marg_kl" in metrics and metrics["marg_kl"] is not None:
                writer.add_scalar("train/marg_kl", float(metrics["marg_kl"].item()), step)
            if "entropy" in metrics and metrics["entropy"] is not None:
                writer.add_scalar("train/mixture_entropy", float(metrics["entropy"].item()), step)

        if step % 50 == 0 and is_main_process():
            tput = (step + 1) * cfg.train.micro_batch_size * cfg.train.grad_accum_steps * world_size / max(1, time.time() - t0)
            logger.info(
                f"step={step:>6d}  loss={loss_v:.4f}  nll={nll:.4f}  "
                f"lr={lr:.2e}  gn={grad_norm:.2f}  tput={tput:.1f} seq/s"
            )

        if step > 0 and step % cfg.checkpointing.save_every_steps == 0 and is_main_process():
            save_checkpoint(head, save_dir, step, keep_last=cfg.checkpointing.keep_last)

        step += 1

    if is_main_process():
        save_checkpoint(head, save_dir, step, keep_last=cfg.checkpointing.keep_last)
    if writer is not None:
        writer.close()
