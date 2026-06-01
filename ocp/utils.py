"""Caching, hashing, distributed helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        ws = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        return rank, ws, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def stable_hash(obj: Any) -> str:
    if is_dataclass(obj):
        obj = asdict(obj)
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def checkpoint_slug(path: str) -> str:
    """Hash a checkpoint path/name into a short filesystem-safe slug."""
    base = Path(str(path)).name or str(path)
    base = base.replace("/", "_").replace(":", "_")[:48]
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


# Hydra config resolver registration — call from entry points
def register_resolvers() -> None:
    from omegaconf import OmegaConf

    try:
        OmegaConf.register_new_resolver("checkpoint_slug", checkpoint_slug, replace=True)
    except Exception:  # noqa: BLE001
        pass


class DiskCache:
    """Tiny file-backed cache keyed by a string. Stores torch tensors / dicts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.pt"

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def save(self, key: str, payload: dict) -> Path:
        p = self.path(key)
        tmp = p.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.rename(p)
        return p

    def load(self, key: str, map_location: str | torch.device = "cpu") -> dict:
        return torch.load(self.path(key), map_location=map_location, weights_only=False)
