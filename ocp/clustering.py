"""Stage 1: cluster system + builder + cluster-recall diagnostic.

ClusterSystem is the interface consumed by the heads. It is the same shape
as gemma_mtp.gmtp.clusters.ClusterSystem so head code that already works on
Gemma 4 can be ported across.

For an Orthrus backbone on Qwen3 (no native clusters), we build kappa by
mini-batch k-means over the backbone's input embedding matrix. This is
cheap (a few minutes on CPU for V=152K, d=2048).

Output: kappa (token_to_cluster, [V]) plus a derived token_ordering
[C, max_tokens_per_cluster] (padded with -1 for variable cluster sizes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from sklearn.cluster import MiniBatchKMeans


@dataclass
class ClusterSystem:
    """Hard partition kappa : V -> {1..C} plus the inverse map.

    token_to_cluster[v] = c          shape [V]
    token_ordering[c, j] = token_id  shape [C, max_per_c], padded with -1
    cluster_sizes[c] = #tokens in c  shape [C]
    """

    num_clusters: int
    vocab_size: int
    token_to_cluster: torch.Tensor   # [V] long
    token_ordering: torch.Tensor     # [C, max_per_c] long, -1 padded
    cluster_sizes: torch.Tensor      # [C] long
    source: str                      # "kmeans" / "gemma_token_ordering" / etc.

    @property
    def max_tokens_per_cluster(self) -> int:
        return int(self.token_ordering.shape[1])

    def cluster_of(self, token_ids: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(token_ids, fill_value=-1, dtype=torch.long)
        valid = (token_ids >= 0) & (token_ids < self.vocab_size)
        out[valid] = self.token_to_cluster.to(token_ids.device)[token_ids[valid]]
        return out

    def to(self, device: str | torch.device) -> "ClusterSystem":
        return ClusterSystem(
            num_clusters=self.num_clusters,
            vocab_size=self.vocab_size,
            token_to_cluster=self.token_to_cluster.to(device),
            token_ordering=self.token_ordering.to(device),
            cluster_sizes=self.cluster_sizes.to(device),
            source=self.source,
        )

    def state_dict(self) -> dict:
        return {
            "num_clusters": self.num_clusters,
            "vocab_size": self.vocab_size,
            "token_to_cluster": self.token_to_cluster,
            "token_ordering": self.token_ordering,
            "cluster_sizes": self.cluster_sizes,
            "source": self.source,
        }

    @classmethod
    def from_state(cls, s: dict) -> "ClusterSystem":
        return cls(**s)


def _build_token_ordering(token_to_cluster: torch.Tensor, num_clusters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [C, max_per_c] padded ordering and cluster_sizes from V-vector kappa."""
    V = int(token_to_cluster.numel())
    sizes = torch.zeros(num_clusters, dtype=torch.long)
    # bincount handles missing clusters as zero
    bc = torch.bincount(token_to_cluster.clamp_min(0), minlength=num_clusters)
    sizes[: bc.numel()] = bc
    max_per_c = int(sizes.max().item())
    ordering = torch.full((num_clusters, max_per_c), -1, dtype=torch.long)
    # for each cluster, collect token ids
    fill = torch.zeros(num_clusters, dtype=torch.long)
    for v in range(V):
        c = int(token_to_cluster[v].item())
        j = int(fill[c].item())
        if j < max_per_c:
            ordering[c, j] = v
            fill[c] = j + 1
    return ordering, sizes


def build_kmeans_clusters(
    embeddings: torch.Tensor,
    num_clusters: int,
    *,
    batch_size: int = 8192,
    max_iter: int = 100,
    n_init: int = 4,
    seed: int = 42,
) -> ClusterSystem:
    """Mini-batch k-means over token embeddings -> ClusterSystem.

    embeddings: [V, d] float, lives on CPU (we move to fp32 for sklearn).
    """
    V, d = embeddings.shape
    emb_np = embeddings.detach().to(torch.float32).cpu().numpy()
    logger.info(f"running MiniBatchKMeans on [{V}, {d}] -> C={num_clusters} (this can take a few minutes)")

    km = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=seed,
        verbose=0,
    )
    km.fit(emb_np)
    labels = torch.from_numpy(km.labels_).long()

    ordering, sizes = _build_token_ordering(labels, num_clusters)

    logger.info(
        f"cluster sizes — min={int(sizes.min())} max={int(sizes.max())} "
        f"mean={float(sizes.float().mean()):.1f} std={float(sizes.float().std()):.1f}"
    )
    empty = int((sizes == 0).sum())
    if empty:
        logger.warning(f"{empty} of {num_clusters} clusters are EMPTY after k-means — bump n_init or rerun.")

    return ClusterSystem(
        num_clusters=num_clusters,
        vocab_size=V,
        token_to_cluster=labels,
        token_ordering=ordering,
        cluster_sizes=sizes,
        source="kmeans",
    )


def from_gemma_token_ordering(
    token_ordering: torch.Tensor, vocab_size: int | None = None,
) -> ClusterSystem:
    """Adapt a pretrained [C, tpc] ordering (Gemma 4) to ClusterSystem."""
    num_c, tpc = token_ordering.shape
    V = vocab_size or (num_c * tpc)
    flat = token_ordering.reshape(-1).long().cpu()
    cluster_for_row = (
        torch.arange(num_c).unsqueeze(1).expand(num_c, tpc).reshape(-1).long()
    )
    inv = torch.full((V,), -1, dtype=torch.long)
    valid = (flat >= 0) & (flat < V)
    inv[flat[valid]] = cluster_for_row[valid]
    sizes = torch.bincount(inv.clamp_min(0), minlength=num_c)
    return ClusterSystem(
        num_clusters=num_c,
        vocab_size=V,
        token_to_cluster=inv,
        token_ordering=token_ordering.long().cpu(),
        cluster_sizes=sizes,
        source="gemma_token_ordering",
    )


@torch.no_grad()
def cluster_recall_at_k(
    teacher_logits: torch.Tensor,
    clusters: ClusterSystem,
    *,
    top_k: int = 32,
) -> dict[str, float]:
    """Cluster recall at top-k against AR teacher's argmax token.

    teacher_logits: [N, V]   — AR teacher next-token distribution.

    Returns: greedy_recall, sampled_recall (T=1), mean_active_clusters.

    Greedy: AR teacher's argmax token lies in one of the top-k clusters
    scored by sum of teacher prob mass.
    """
    N, V = teacher_logits.shape
    device = teacher_logits.device
    kappa = clusters.token_to_cluster.to(device)
    C = clusters.num_clusters

    probs = teacher_logits.softmax(dim=-1)                              # [N, V]
    # cluster_mass[n, c] = sum over tokens in cluster c of probs[n, v]
    cluster_mass = torch.zeros(N, C, device=device, dtype=probs.dtype)
    cluster_mass.scatter_add_(1, kappa.unsqueeze(0).expand(N, V), probs)

    topk_clusters = cluster_mass.topk(top_k, dim=-1).indices            # [N, top_k]

    # Greedy: argmax token's cluster
    argmax_tok = teacher_logits.argmax(dim=-1)                           # [N]
    argmax_clu = kappa[argmax_tok]                                       # [N]
    greedy_hits = (topk_clusters == argmax_clu.unsqueeze(-1)).any(dim=-1)
    greedy_recall = greedy_hits.float().mean().item()

    # Sampled (T=1): one sample per row
    sampled_tok = torch.multinomial(probs, 1).squeeze(-1)                # [N]
    sampled_clu = kappa[sampled_tok]
    sampled_hits = (topk_clusters == sampled_clu.unsqueeze(-1)).any(dim=-1)
    sampled_recall = sampled_hits.float().mean().item()

    return {
        "greedy_recall": float(greedy_recall),
        "sampled_recall": float(sampled_recall),
        "top_k": int(top_k),
    }
