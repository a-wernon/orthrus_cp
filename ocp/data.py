"""Streaming Nemotron-style data pipeline + anchor-block collator.

We mirror the Orthrus paper's recipe:
  - source: Nemotron-Post-Training-Dataset-v2 (Math / Code / Chat 1:1:1)
  - pack sequences up to L=2048 tokens
  - per sample, draw A random anchor positions (paper: 256 per L=2048; we
    expose this as cfg.data.anchors_per_seq for memory tuning)

Each yielded batch is a dict of tensors ready for FrozenOrthrus.forward_features.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from datasets import load_dataset, interleave_datasets
from loguru import logger
from torch.utils.data import IterableDataset, DataLoader


def _resolve_text(example: dict) -> str:
    """Concatenate message-style records into a single string."""
    if "messages" in example and isinstance(example["messages"], list):
        # Nemotron post-training dataset has role/content message lists
        return "\n".join(
            f"{m.get('role','')}: {m.get('content','')}" for m in example["messages"]
        )
    for key in ("text", "content", "completion", "response"):
        if isinstance(example.get(key), str) and example[key]:
            return example[key]
    return ""


class NemotronStream(IterableDataset):
    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        tokenizer,
        seq_len: int,
        anchors_per_seq: int,
        K: int,
        seed: int = 42,
        shuffle_buffer: int = 1000,
    ):
        self.dataset_name = dataset
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.anchors_per_seq = anchors_per_seq
        self.K = K
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _stream(self):
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        if self.shuffle_buffer:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        return ds

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))
        buf: list[int] = []
        total = self.seq_len

        for example in self._stream():
            text = _resolve_text(example)
            if not text:
                continue
            toks = self.tokenizer.encode(text, add_special_tokens=False)
            if not toks:
                continue
            buf.extend(toks)
            buf.append(self.tokenizer.eos_token_id or 0)

            while len(buf) >= total:
                chunk = buf[:total]
                buf = buf[total:]

                input_ids = torch.tensor(chunk, dtype=torch.long)
                # anchor positions must allow a full K-block, so range is [0, L-K]
                max_anchor = self.seq_len - self.K
                if max_anchor <= 0:
                    continue
                anchors = torch.tensor(
                    sorted(rng.sample(range(max_anchor + 1),
                                      min(self.anchors_per_seq, max_anchor + 1))),
                    dtype=torch.long,
                )

                yield {
                    "input_ids": input_ids,            # [L]
                    "anchor_positions": anchors,       # [A] <= anchors_per_seq
                }


def collate_anchor_batch(samples: list[dict]) -> dict:
    """Pad anchor positions to the same A across samples (simplest scheme)."""
    input_ids = torch.stack([s["input_ids"] for s in samples], dim=0)        # [B, L]
    A_max = max(s["anchor_positions"].numel() for s in samples)
    anchors = torch.zeros(len(samples), A_max, dtype=torch.long)
    anchor_mask = torch.zeros(len(samples), A_max, dtype=torch.bool)
    for i, s in enumerate(samples):
        a = s["anchor_positions"]
        anchors[i, : a.numel()] = a
        anchor_mask[i, : a.numel()] = True
    return {
        "input_ids": input_ids,
        "anchor_positions": anchors,
        "anchor_mask": anchor_mask,
    }


def build_dataloader(
    cfg,
    tokenizer,
    K: int,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    stream = NemotronStream(
        dataset=cfg.data.dataset,
        split=cfg.data.split,
        tokenizer=tokenizer,
        seq_len=cfg.data.seq_len,
        anchors_per_seq=cfg.data.anchors_per_seq,
        K=K,
        seed=cfg.data.seed,
        shuffle_buffer=cfg.data.shuffle_buffer,
    )
    return DataLoader(
        stream,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_anchor_batch,
        drop_last=True,
    )
