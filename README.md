# mahala

Metric learning losses with analytical gradients.

Contrastive, proxy-based, and self-supervised losses for embedding training.
Two backends: pure-Rust `f32` slices with hand-derived gradients, and
[candle](https://github.com/huggingface/candle) tensors with autograd.

## Install

```toml
[dependencies]
mahala = "0.1"

# Optional: SIMD-accelerated similarity via innr
mahala = { version = "0.1", features = ["simd"] }

# Optional: candle tensor losses with autograd
mahala = { version = "0.1", features = ["candle"] }
```

## Losses

| Loss | f32 | candle | Reference |
|------|-----|--------|-----------|
| Triplet | `triplet_loss` | `candle_losses::triplet_loss` | Schroff et al. 2015 |
| InfoNCE / NT-Xent | `infonce_loss` | `candle_losses::infonce_loss` | Oord et al. 2018 |
| MNRL | `mnrl_loss` | `candle_losses::mnrl_loss` | Henderson et al. 2017 |
| Contrastive | `contrastive_loss` | `candle_losses::contrastive_loss` | Hadsell et al. 2006 |
| Cosine embedding | `cosine_embedding_loss` | `candle_losses::cosine_embedding_loss` | |
| Multi-Similarity | `multi_similarity_loss` | `candle_losses::multi_similarity_loss` | Wang et al. 2019 |
| SupCon | `supcon_loss` | `candle_losses::supcon_loss` | Khosla et al. 2020 |
| Circle | `circle_loss` | `candle_losses::circle_loss` | Sun et al. 2020 |
| Lifted Structured | `lifted_structured_loss` | `candle_losses::lifted_structured_loss` | Song et al. 2016 |
| N-Pairs | `n_pairs_loss` | `candle_losses::n_pairs_loss` | Sohn 2016 |
| ArcFace | `arcface_loss` | `candle_losses::arcface_loss` | Deng et al. 2019 |
| ProxyAnchor | `proxy_anchor_loss` | `candle_losses::proxy_anchor_loss` | Kim et al. 2020 |
| VICReg | `vicreg_loss` | `candle_losses::vicreg_loss` | Bardes et al. 2022 |
| Matryoshka (wrapper) | `matryoshka_loss` | `candle_losses::matryoshka_loss` | Kusupati et al. 2022 |

Plus: `CrossBatchMemory` for expanding the negative pool across batches.

## Distance learning

Classical Mahalanobis distance learning: `learn_transform`, `nca`, `lmnn`, `itml`, `kissme`.

## Negative miners

`HardestMiner`, `SemiHardMiner`, `RandomMiner`, `InBatchMiner`, `MultiSimilarityMiner`, `DistanceWeightedMiner`.

## Usage

### Pure Rust (no framework)

```rust
use mahala::*;

let anchors: Vec<&[f32]> = vec![&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]];
let positives: Vec<&[f32]> = vec![&[0.9, 0.1, 0.0], &[0.1, 0.9, 0.0]];

let out = infonce_loss(&anchors, &positives, 0.07);
println!("loss: {}", out.loss);

// Apply gradients manually
for (emb, grad) in anchors.iter().zip(&out.grad_anchors) {
    // emb[d] -= lr * grad[d]
}
```

### Candle (autograd)

```rust
use mahala::candle_losses;
use candle_core::{Tensor, Device};

let a = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu)?;
let b = Tensor::new(&[[0.9f32, 0.1], [0.1, 0.9]], &Device::Cpu)?;

let loss = candle_losses::infonce_loss(&a, &b, 0.07)?;
let grads = loss.backward()?;
```

### Cross-batch memory

```rust
use mahala::*;

let mut memory = CrossBatchMemory::new(1024, 128); // capacity, dim

// Each training step:
memory.enqueue(&batch_embeddings, &batch_labels);
let negatives = memory.embeddings();
let out = mnrl_loss(&anchors, &positives, &negatives, 0.07);
```

## Features

| Feature | What it adds |
|---------|-------------|
| `mining` (default) | Negative miners (`HardestMiner`, etc.) |
| `simd` | SIMD-accelerated similarity via [innr](https://crates.io/crates/innr) |
| `candle` | Tensor-based losses with autograd via [candle-core](https://crates.io/crates/candle-core) |
| `serde` | Serialize/deserialize config structs |

## Examples

```sh
cargo run --example metric_learning      # all methods overview
cargo run --example embedding_training   # f32 training loop (NCA + InfoNCE)
cargo run --example mining_pipeline      # miners + losses end-to-end
cargo run --features candle --example candle_training  # candle autograd training
```

License: MIT OR Apache-2.0
