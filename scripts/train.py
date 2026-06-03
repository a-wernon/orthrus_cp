"""Stage 2: head-only training on a frozen Orthrus backbone.

Loads the cached ClusterSystem from stage 1, builds one of the heads
(cluster_ff, cluster_ff_gain, cp_cluster), then trains it via the
composite NLL + marginal-KL + entropy-reg loss.

Usage (1xH200):
    python scripts/train.py model.checkpoint=<orthrus-1.7b-path>

Usage (8xH200 DDP):
    torchrun --nproc_per_node=8 scripts/train.py \
        model.checkpoint=<orthrus-1.7b-path> train=train_h8
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocp.clustering import ClusterSystem  # noqa: E402
from ocp.data import build_dataloader  # noqa: E402
from ocp.heads import build_head  # noqa: E402
from ocp.model import FrozenOrthrus  # noqa: E402
from ocp.training import train_loop  # noqa: E402
from ocp.utils import (  # noqa: E402
    DiskCache,
    cleanup_distributed,
    is_main_process,
    register_resolvers,
    setup_distributed,
    world_size,
)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    register_resolvers()
    rank, ws, device = setup_distributed()
    torch.manual_seed(cfg.train.seed + rank)
    if is_main_process():
        logger.info("training config:\n" + OmegaConf.to_yaml(cfg))

    # 1. load cached clusters
    cache = DiskCache(cfg.clustering.cache_dir)
    if not cache.has(cfg.clustering.cache_key):
        raise FileNotFoundError(
            f"cluster cache {cache.path(cfg.clustering.cache_key)} not found. "
            "Run `python scripts/build_clusters.py model.checkpoint=...` first."
        )
    payload = cache.load(cfg.clustering.cache_key)
    clusters = ClusterSystem.from_state(payload["clusters"])
    if is_main_process():
        logger.info(
            f"loaded clusters from {cache.path(cfg.clustering.cache_key)} "
            f"(C={clusters.num_clusters}, V={clusters.vocab_size}, "
            f"source={clusters.source})"
        )

    # 2. load frozen backbone, attach to device
    frozen = FrozenOrthrus(
        cfg.model.checkpoint,
        dtype=getattr(torch, cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    frozen.model.to(device)

    # 3. build head
    head = build_head(
        cfg.head.kind,
        hidden_size=frozen.hidden_size,
        clusters=clusters,
        K=frozen.K,
        tie_token_embeddings=cfg.head.tie_token_embeddings,
        init_scale=cfg.head.init_scale,
        **{k: v for k, v in cfg.head.items()
           if k not in {"name", "kind", "tie_token_embeddings", "init_scale", "loss"}
           and not k.startswith("_")},
    ).to(device)
    head_dtype = next(p.dtype for p in head.parameters() if p.is_floating_point())
    head.attach_token_embeddings(frozen.input_embeddings.to(device=device, dtype=head_dtype))

    if is_main_process():
        n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        logger.info(f"head {cfg.head.name} trainable params: {n_params:,}")

    if ws > 1:
        head = DDP(head, device_ids=[device.index] if device.type == "cuda" else None,
                   find_unused_parameters=False)
        head_module = head.module
    else:
        head_module = head

    # 4. data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    loader = build_dataloader(
        cfg, tokenizer, K=frozen.K,
        batch_size=cfg.train.micro_batch_size,
        num_workers=cfg.data.num_workers,
    )

    # 5. go
    try:
        train_loop(
            cfg=cfg,
            head=head_module,
            frozen=frozen,
            clusters=clusters,
            dataloader=loader,
            device=device,
            rank=rank,
            world_size=ws,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
