# tuplet examples

Each example is runnable from the repo root. Output excerpts below are real,
captured from release runs for the quick examples. `fashion_metric_learning` is
the heavier data-gated eval harness.

## Which example should I run?

| I want to... | Example | Notes |
|---|---|---|
| Compare classical metric learning and contrastive losses | `metric_learning` | Pure Rust |
| Train embeddings without a tensor framework | `embedding_training` | Pure Rust |
| Connect miners to losses | `mining_pipeline` | Pure Rust |
| Check Burn losses forward and backward | `burn_training` | Requires `burn-ndarray` |
| Train supervised embeddings on Fashion-MNIST | `fashion_metric_learning` | Data-gated eval |

## Pure Rust

### `metric_learning`: how do classical methods and tuple losses compare?

Generates three overlapping 2D clusters, applies Mahalanobis/NCA/LMNN-style
metric learning, then prints several contrastive loss values on the same data.

```bash
cargo run --release --example metric_learning
```

```text
=== Metric Learning Lineage ===

Baseline (Euclidean):       100.0%
Mahalanobis learning:      100.0%
NCA:                       100.0%
LMNN:                      100.0%
KISSME:                    (M computed, 4 non-zero entries)
ITML:                      (M computed, trace=3.4069)

=== Loss Functions ===

Triplet loss:              0.0000
Contrastive loss:          0.0352
InfoNCE loss:              4.4751
SupCon loss:               4.9857
Circle loss:               439.0785
```

### `embedding_training`: can embeddings be fine-tuned without Burn?

Learns an NCA projection, then manually applies InfoNCE gradients to the
projected embeddings.

```bash
cargo run --release --example embedding_training
```

```text
=== Phase 1: NCA projection ===

NCA 1-NN accuracy: 100.0%

=== Phase 2: InfoNCE fine-tuning ===

epoch   0: loss=2.8760, 1-NN acc=100.0%
epoch  10: loss=1.8639, 1-NN acc=100.0%
epoch  20: loss=1.6208, 1-NN acc=100.0%
epoch  40: loss=1.5491, 1-NN acc=100.0%

Final 1-NN accuracy: 100.0%
```

### `mining_pipeline`: which negatives do the miners select?

Runs hardest, semi-hard, multi-similarity, and distance-weighted miners on a
small embedding pool, then feeds selected tuples into loss functions.

```bash
cargo run --release --example mining_pipeline
```

```text
=== Hardest Negative Mining + Triplet Loss ===

Anchor: idx 0 (class 0)
Positive: idx 1 (class 0)
Hardest negative: idx [3] (class 1)
Triplet loss: 0.0000

=== Multi-Similarity Mining + Multi-Similarity Loss ===

Anchor: idx 6 (boundary class-0 point)
Informative positives: []
Informative negatives: []
Multi-similarity loss (full batch): 0.4711
Mean grad norm: 0.1793
```

## Burn Backend

### `burn_training`: do Burn losses compute forward and backward?

Computes triplet, InfoNCE, and contrastive losses on the `NdArray` autodiff
backend and calls `.backward()` on each.

```bash
cargo run --release --features burn-ndarray --example burn_training
```

```text
triplet_loss = 0.0000
infonce_loss = 0.0400
contrastive_loss = 0.0025

All three losses computed forward + backward on Autodiff<NdArray>.
```

## Data-Gated Eval

### `fashion_metric_learning`: does SupCon improve Fashion-MNIST kNN?

Trains a two-layer MLP embedding on 6000 Fashion-MNIST train images using
SupCon gradients, then compares kNN accuracy on 2000 test images against raw
pixels.

```bash
./scripts/fetch_fashion_mnist.sh
cargo run --release --example fashion_metric_learning
```

If the dataset is absent, the example exits 0 and prints the fetch command. The
full run is heavier than the smoke examples because it trains for 60 epochs and
evaluates kNN against the training gallery.
