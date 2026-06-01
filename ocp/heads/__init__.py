from .base import ClusterHead, HeadOutput
from .cluster_ff import ClusterFFHead
from .cluster_ff_gain import ClusterFFGainHead
from .cp_cluster import CPClusterHead

HEAD_REGISTRY = {
    "cluster_ff": ClusterFFHead,
    "cluster_ff_gain": ClusterFFGainHead,
    "cp_cluster": CPClusterHead,
}


def build_head(kind: str, **kwargs) -> ClusterHead:
    if kind not in HEAD_REGISTRY:
        raise KeyError(f"unknown head kind {kind!r}; have {list(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[kind](**kwargs)


__all__ = ["build_head", "ClusterHead", "HeadOutput", "HEAD_REGISTRY"]
