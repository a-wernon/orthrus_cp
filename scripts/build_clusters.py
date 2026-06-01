"""Stage 1: build the token -> cluster partition kappa.

Runs once per (backbone, num_clusters) pair. Caches the ClusterSystem to
disk; subsequent runs read from cache unless --force-recompute is set.

Also runs a small cluster-recall diagnostic against the AR teacher so we
know whether to bother with stage 2 at all.

Usage:
    python scripts/build_clusters.py model.checkpoint=<orthrus-1.7b-path>
    python scripts/build_clusters.py model.checkpoint=... clustering.force_recompute=true
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocp.clustering import (  # noqa: E402
    ClusterSystem,
    build_kmeans_clusters,
    cluster_recall_at_k,
)
from ocp.model import FrozenOrthrus  # noqa: E402
from ocp.utils import DiskCache, register_resolvers, is_main_process  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="clustering")
def main(cfg: DictConfig) -> None:
    register_resolvers()
    if is_main_process():
        logger.info("clustering config:\n" + OmegaConf.to_yaml(cfg))

    cache = DiskCache(cfg.clustering.cache_dir)
    key = cfg.clustering.cache_key

    if not cfg.clustering.force_recompute and cache.has(key):
        logger.info(f"cache hit: {cache.path(key)} — loading; pass clustering.force_recompute=true to rebuild")
        payload = cache.load(key)
        clusters = ClusterSystem.from_state(payload["clusters"])
    else:
        # Load the frozen backbone for its input embeddings only.
        frozen = FrozenOrthrus(cfg.model.checkpoint, dtype=getattr(torch, cfg.model.dtype))
        E = frozen.input_embeddings.detach().to(torch.float32).cpu()
        if cfg.clustering.source == "kmeans":
            clusters = build_kmeans_clusters(
                E,
                num_clusters=cfg.clustering.num_clusters,
                batch_size=cfg.clustering.kmeans.batch_size,
                max_iter=cfg.clustering.kmeans.max_iter,
                n_init=cfg.clustering.kmeans.n_init,
                seed=cfg.clustering.kmeans.seed,
            )
        else:
            raise ValueError(f"clustering.source={cfg.clustering.source!r} not implemented")
        cache.save(key, {"clusters": clusters.state_dict(), "cfg": OmegaConf.to_container(cfg)})
        logger.info(f"saved cluster system to {cache.path(key)}")

        # Free model memory before recall
        del frozen
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # cluster-recall diagnostic
    if cfg.clustering.eval_recall.enable:
        _run_recall_diagnostic(cfg, clusters)


def _run_recall_diagnostic(cfg, clusters: ClusterSystem) -> None:
    """Sample N prompts, run AR teacher one step, measure cluster recall at top-k."""
    import gc
    from datasets import load_dataset
    from transformers import AutoTokenizer

    logger.info("running cluster-recall diagnostic")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = FrozenOrthrus(cfg.model.checkpoint, dtype=getattr(torch, cfg.model.dtype))
    frozen.model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)

    ds = load_dataset(cfg.clustering.eval_recall.dataset, split=cfg.clustering.eval_recall.split, streaming=True)
    seq_len = int(cfg.clustering.eval_recall.seq_len)
    n = int(cfg.clustering.eval_recall.n_prompts)
    top_k = int(cfg.clustering.eval_recall.top_k)

    all_logits = []
    count = 0
    ds_iter = iter(ds)
    progress = tqdm(ds_iter)
    try:
        for example in progress:
            text = example.get("text") or "\n".join(
                f"{m.get('role','')}: {m.get('content','')}" for m in (example.get("messages") or [])
            )
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)[:seq_len]
            if len(ids) < 16:
                continue
            ids_t = torch.tensor(ids, device=device).unsqueeze(0)
            # run AR teacher (not the diffusion path) to get a next-token distribution
            # at every position in the sequence
            with torch.no_grad():
                base_out = frozen.model.model(
                    input_ids=ids_t,
                    is_diffusion_pass=False,
                    use_cache=False,
                )
                logits = frozen.model.lm_head(base_out.last_hidden_state).float()  # [1, L, V]
            all_logits.append(logits.squeeze(0))
            count += 1
            if count >= n:
                break
    finally:
        progress.close()
        close = getattr(ds_iter, "close", None)
        if close is not None:
            close()
        del ds_iter
        del ds

    teacher_logits = torch.cat(all_logits, dim=0)                              # [N*L, V]
    metrics = cluster_recall_at_k(teacher_logits, clusters.to(device), top_k=top_k)
    logger.info(
        f"cluster recall @ top-{top_k}: greedy={metrics['greedy_recall']:.4f} "
        f"sampled={metrics['sampled_recall']:.4f}"
    )

    cache = DiskCache(cfg.clustering.cache_dir)
    payload = cache.load(cfg.clustering.cache_key)
    payload["recall"] = metrics
    cache.save(cfg.clustering.cache_key, payload)

    del teacher_logits, all_logits, frozen
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
