# orthrus_cp

Head-only training of a CP-cluster output head on top of a frozen Orthrus
backbone. See `journal/reports/orthrus_cp_cluster/orthrus_cp_cluster.pdf`
for the design.

Two-stage pipeline:

1. **Clustering** (`scripts/build_clusters.py`). Compute the hard
   partition `kappa : V -> {1..C}` from the backbone's input embeddings
   via mini-batch k-means. Cached to disk. Cheap, deterministic.

2. **Head training** (`scripts/train.py`). Frozen Orthrus backbone +
   trainable cluster-aware head. Loss is forward NLL of AR teacher
   continuations under the composite joint, optionally augmented with
   marginal forward-KL and an entropy regulariser on the mixture
   weights.

```bash
# stage 1: build (and cache) clusters
python scripts/build_clusters.py model.checkpoint=<orthrus-1.7b-path>

# stage 1 again, recomputing from scratch
python scripts/build_clusters.py model.checkpoint=... clustering.force_recompute=true

# stage 2: train CP head (default config, 1xH200)
python scripts/train.py head=head_cp model.checkpoint=<orthrus-1.7b-path>

# stage 2: train honest baseline for comparison
python scripts/train.py head=head_cluster_ff_gain model.checkpoint=...

# scale to 8xH200
torchrun --nproc_per_node=8 scripts/train.py head=head_cp train=train_h8 model.checkpoint=...
```

Logs land in TensorBoard at `runs/<run_name>/`. Checkpoints at
`checkpoints/<run_name>/`.

## Layout

```
configs/
  base.yaml                       defaults, overridden by sub-configs
  clustering.yaml                 stage 1 settings
  head_cp.yaml                    CP-cluster head, rank r
  head_cluster_ff.yaml            honest baseline
  head_cluster_ff_gain.yaml       stronger honest baseline (2-layer MLP + gain)
  train_h1.yaml                   single-GPU defaults
  train_h8.yaml                   8x scaling overrides
ocp/
  clustering.py                   ClusterSystem + balanced k-means + recall
  model.py                        frozen Orthrus loader, diffusion forward, AR teacher logits
  data.py                         Nemotron streaming + anchor-block collator
  heads/                          ClusterFF, ClusterFFGain, CPClusterHead
  training.py                     loop, losses, schedule, checkpointing
  utils.py                        cache + DDP helpers
scripts/
  build_clusters.py               stage 1 entrypoint
  train.py                        stage 2 entrypoint
```
