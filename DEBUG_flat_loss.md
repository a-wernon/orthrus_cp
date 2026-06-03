# Debugging: flat loss (~220–230 nats) on cp_cluster head

Symptom: `python scripts/train.py model.checkpoint=chiennv/Orthrus-Qwen3-1.7B`
sits at `loss≈220–230, nll≈226` from step ~0 to step 7950, `gn≈26`, never moving.

This file is a debugging plan you run on the GPU box. It is ordered: each step
is a cheap check that localizes the failure before we change architecture. Do
them in order and paste the printed numbers back.

---

## What we already ruled out (by reading `orthrus/src/model.py`)

- **Not** an embedding-vs-final-layer bug. `OrthrusLM.forward` returns
  `hidden_states=(final_post_norm_hidden,)`, so `diff_out.hidden_states[0]` in
  `ocp/model.py:177` *is* the final hidden state. Correct.
- **Not** a diffusion-projection fallback. `OrthrusDiffusionAttention.forward`
  gates on `is_diffusion_pass` (line 121), not `self.training`, so
  `q_proj_diff/k_proj_diff` are used even under `eval()`. In eval it takes the
  `else` branch (line 161) and uses our additive `attention_mask` with eager
  attention — which is exactly what `forward_features` sets up. Correct.

So the backbone path is plausibly fine. The failure is in the loss scale /
gradient, and the suspects are (B) below.

---

## Sanity numbers (memorize these)

K=32, C=2048, mean cluster size ~128 (V≈151936 / 2048).

- Uniform cluster joint:     `-Σ_k log(1/C)  = 32·log(2048)  ≈ 244 nats`
- Uniform token-in-cluster:  `-Σ_k log(1/128)= 32·log(128)   ≈ 155 nats`
- Untrained total NLL floor: **≈ 400 nats**
- Observed: **~226 nats** → 226/32 ≈ **7.06 nats/pos** ≈ log(1160).

Reading: the **cluster term alone is ~226** and the **token-in-cluster term is
contributing ≈ 0** (it would *add* ~155 if it were uniform). That means
`log_p_y_given_c` is ≈ 0 per position — i.e. each target token's cluster
contains essentially **one** candidate (log p ≈ log 1 = 0), OR the token term is
being masked out / saturated. And the cluster term is frozen near uniform-ish.

This points at two concrete bugs, A and B.

---

## STEP 1 — instrument the loss components (no code change to train loop)

Add a one-off debug script `scripts/debug_one_batch.py`:

```python
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hydra
from omegaconf import DictConfig
from ocp.clustering import ClusterSystem
from ocp.data import build_dataloader
from ocp.heads import build_head
from ocp.model import FrozenOrthrus
from ocp.utils import DiskCache, register_resolvers
from transformers import AutoTokenizer

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    register_resolvers()
    dev = torch.device("cuda")
    cache = DiskCache(cfg.clustering.cache_dir)
    clusters = ClusterSystem.from_state(cache.load(cfg.clustering.cache_key)["clusters"])
    frozen = FrozenOrthrus(cfg.model.checkpoint, dtype=torch.bfloat16,
                           attn_implementation=cfg.model.attn_implementation)
    frozen.model.to(dev)
    head = build_head(cfg.head.kind, hidden_size=frozen.hidden_size, clusters=clusters,
                      K=frozen.K, tie_token_embeddings=True, init_scale=cfg.head.init_scale,
                      rank=cfg.head.get("rank", 16), pool=cfg.head.get("pool", "mean")).to(dev)
    head.attach_token_embeddings(frozen.input_embeddings.to(dev, next(head.parameters()).dtype))
    tok = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    loader = build_dataloader(cfg, tok, K=frozen.K, batch_size=2, num_workers=0)

    batch = next(iter(loader))
    ids = batch["input_ids"].to(dev); anc = batch["anchor_positions"].to(dev)
    with torch.no_grad():
        feats = frozen.forward_features(ids, anc)

    # --- cluster validity of targets ---
    k2c = clusters.token_to_cluster.to(dev)
    tgt = feats.target_tokens.to(dev)                # [B,K]
    tc  = k2c[tgt]                                   # [B,K]
    # how many candidate tokens does each target's cluster hold?
    sizes = clusters.cluster_sizes.to(dev)[tc]       # [B,K]
    print("cluster size of target clusters: min/median/max =",
          int(sizes.min()), int(sizes.median()), int(sizes.max()))

    # --- token-in-cluster term ---
    lp_tok = head.target_token_log_probs(feats.h_diff.to(next(head.parameters()).dtype), tgt, tc)
    print("log p(y|c): finite frac =", torch.isfinite(lp_tok).float().mean().item(),
          " mean(finite) =", lp_tok[torch.isfinite(lp_tok)].mean().item())

    # --- teacher logits sanity (are they real AR logits or garbage?) ---
    tl = feats.teacher_logits                        # [B,K,V] float
    print("teacher_logits: argmax==target frac =",
          (tl.argmax(-1) == tgt).float().mean().item(),
          " logit std =", tl.std().item())

    # --- head cluster marginal vs teacher cluster ---
    out = head(feats.h_diff.to(next(head.parameters()).dtype), feats.h_pool.to(next(head.parameters()).dtype))
    lqcm = out.log_q_cluster_marginal               # [B,K,C]
    pred_c = lqcm.argmax(-1)
    print("head cluster-marginal argmax==teacher-token cluster frac =",
          (pred_c == tc).float().mean().item())
    print("joint_log_prob(target_clusters) mean =", out.joint_log_prob(tc).mean().item())
    print("log_w (mixture) entropy =", (-(out.log_w_pre.exp()*out.log_w_pre).sum(-1)).mean().item(),
          " (max possible =", torch.log(torch.tensor(float(cfg.head.get('rank',16)))).item(), ")")

if __name__ == "__main__":
    main()
```

Run:
```
python scripts/debug_one_batch.py model.checkpoint=chiennv/Orthrus-Qwen3-1.7B
```

### How to read STEP 1 output

1. **`teacher_logits: argmax==target frac`** — THIS IS THE LINCHPIN.
   - If this is **near 1.0**: the teacher path is correct, targets are the AR
     greedy tokens, good. Proceed.
   - If this is **low (<0.3)**: `forward_features` teacher indexing is wrong.
     The off-by-one in `ocp/model.py:191` (`anchor + k - 1`) means
     `teacher_logits[b,k]` is `p_AR(.|x_{<a+k-1})` but `target_tokens[b,k]` is
     `x_{a+k}` (line 198, no `-1`). **These are misaligned by one position.**
     See FIX 1.

2. **`log p(y|c): finite frac`** — if `< 1.0`, some targets fall outside their
   own cluster ordering → `-inf` → those whole anchors are dropped by `valid`
   in `compute_loss`. If finite frac is low, the loss is computed on a tiny
   biased subset. (Shouldn't happen with a hard partition, but the k-means
   build may have a token→cluster vs cluster→ordering mismatch. See FIX 3.)

3. **`mean(finite) log p(y|c)`** — if this is ≈ 0, clusters are near-singleton
   for the *observed* tokens, so the token term carries no signal and all the
   work is on the cluster joint. Expected given high recall, but confirms the
   ~226 = cluster-only reading.

4. **`head cluster-marginal argmax==teacher cluster frac`** at init will be ~0;
   after training it should climb. If after training it's still ~1/C, the head
   gets no gradient → bug B.

5. **`log_w entropy` vs `max`** — if entropy ≈ `log(rank)` (i.e. uniform over
   components) and *stays* there during training, the rank-16 mixture has
   collapsed to rank-1. This is the Qwen3-CP NO-GO degeneracy. See FIX 2.

---

## FIX 1 — teacher/target off-by-one (check STEP 1.1 first)

In `ocp/model.py`:
- `idx_h` (teacher hidden) uses `anchor + k - 1` → predicts token at `anchor+k`.
- `target_idx` uses `anchor + k` (no −1) → token at `anchor+k`.

If `argmax==target frac` is low, align them. The intended semantics: position
`k` of the block predicts `x_{a+k}` from context `x_{<=a+k-1}`. Then teacher
hidden is `h_clean[a+k-1]` (predict-next) and target is `x_{a+k}`. That is what
the code does — so if frac is low, the bug is instead that **the block's k=0
slot is the anchor token itself** (it's `x_a`, given, not predicted). In the
Orthrus block `[x_a, mask, mask, ...]`, position 0 is the *anchor input*, and
the K predicted tokens are `x_{a+1..a+K}`. So h_diff position 0 should predict
`x_{a+1}`, not `x_a`. Re-derive target_idx as `anchor + k + 1` and teacher
hidden as `h_clean[a+k]`. **Confirm against `orthrus/src/model.py` generate()
which shows how blocks map to outputs** before committing.

## FIX 2 — CP mixture collapse (the architecturally important one)

Per the design doc and the Qwen3-CP NO-GO memory, at large C a rank-r CP head
degenerates to rank-1 + gain unless the components are *forced* apart. Note:
- `head.loss.entropy_reg` is currently **inert, not helpful**. `mixture_entropy`
  is `.detach()`-ed in cp_cluster.py:106, so the `loss - entropy_reg*H_w` line
  (training.py:100) shifts the *logged* loss by a near-constant but
  back-propagates **zero** gradient. Set `entropy_reg: 0.0` so the logged number
  is honest. (If we later want a real anti-collapse term it must be a
  non-detached penalty on *component similarity*, not on the mixture entropy.)
- The real init issue: `weight_proj`, `component_proj` all at std=0.02 and
  `position_bias=0` → all r components see near-identical logits at init →
  rank-16 mixture behaves as rank-1. Break symmetry: init `position_bias` with
  small noise (e.g. `nn.init.normal_(self.position_bias, std=0.02)`) and/or give
  `weight_proj` a larger init so `log_w` isn't flat.

For the first GPU run: **set `entropy_reg: 0.0`** and compare cp_cluster vs
`cluster_ff_gain`. If cp can't beat ff_gain, that's the NO-GO repeating and we
default to the simpler head — exactly the honest-baseline discipline.

## FIX 3 — token outside its cluster ordering

If STEP 1.2 finetie frac < 1.0: in `ocp/clustering.py`, verify
`token_to_cluster[v]` and `token_ordering[token_to_cluster[v]]` round-trip
(every token appears in its own cluster's ordering row). A mismatch silently
drops anchors. Add an assertion in `build_kmeans_clusters`.

---

## STEP 2 — overfit one batch (the decisive test)

Before any long run, prove the head *can* learn:

```
python scripts/train.py model.checkpoint=chiennv/Orthrus-Qwen3-1.7B \
    train.max_steps=300 train.peak_lr=1e-3 train.warmup_ratio=0.0 \
    head.loss.entropy_reg=0.0 head.loss.marginal_kl_weight=0.0 \
    data.anchors_per_seq=8 train.grad_accum_steps=1
```

…but point the dataloader at a SINGLE repeated batch (add a `--overfit` flag, or
just `break` after first batch and loop on it). Expected: NLL should fall from
~226 toward <50 within 200 steps. If it does → the architecture learns and the
flat run was a data/lr/loss-weight issue. If it does NOT move even on one
batch → gradient is not reaching the parameters; check `requires_grad`,
`find_unused_parameters`, and that `joint_log_prob`'s closure tensors are in the
graph (they are captured from `forward`, so they carry grad — verify with
`out.joint_log_prob(tc).requires_grad`).

---

## STEP 3 — only after STEP 2 passes

Re-enable marginal_kl_weight=0.5, run 1000 steps, watch `train/marg_kl` in
TensorBoard: it should decrease. Then the full 8000-step run. The metric that
decides the project is **acceptance length of cp_cluster vs cluster_ff_gain**,
not NLL — but NLL must move first.

---

## Most likely root cause (my bet, ranked)

1. **`entropy_reg` has the wrong sign** (training.py:100 subtracts it) → mixture
   driven to uniform → rank collapse → cluster joint stuck near uniform → ~226
   nats. Cheapest to test: set `entropy_reg=0.0` and rerun 500 steps.
2. **teacher/target off-by-one** → marginal_kl distills toward a shifted
   distribution, fighting the NLL. Test via STEP 1.1.
3. **lr too low after warmup** (you were at 2e-5 by step 7950 with gn=26) —
   the schedule decayed while grad norm stayed huge, classic sign the loss
   surface never had a basin to descend (consequence of 1/2, not a root cause).
