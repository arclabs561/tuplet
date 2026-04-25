# tuplet

Tuple-based metric learning on Burn tensors.

Contrastive, proxy, and self-supervised losses operate on pairs, triplets, and
n-tuples of embeddings. Hard-negative mining and classical Mahalanobis distance
learning (NCA, LMNN, ITML, KISSME) are pure Rust and need no autograd backend.

## Install

```toml
[dependencies]
tuplet = "0.1"

# CPU training (default Burn backend):
tuplet = { version = "0.1", features = ["burn-ndarray"] }

# WGPU / Metal / Vulkan / WebGPU:
tuplet = { version = "0.1", default-features = false, features = ["mining", "burn-wgpu"] }

# libtorch:
tuplet = { version = "0.1", default-features = false, features = ["mining", "burn-tch"] }

# CUDA (extends burn-ndarray):
tuplet = { version = "0.1", features = ["burn-cuda"] }
```

Pick exactly one `burn-*` backend per binary. The autograd backend is wired
through the chosen Burn backend; you don't need to enable `burn-autodiff`
yourself.

## Losses

All losses operate on `Tensor<B: AutodiffBackend, _>` and return a scalar
loss tensor you can call `.backward()` on.

| Loss | Function | Reference |
|------|----------|-----------|
| Triplet | `triplet_loss` | Schroff et al. 2015 |
| Contrastive | `contrastive_loss` | Hadsell et al. 2006 |
| InfoNCE / NT-Xent | `infonce_loss` | Oord et al. 2018 |
| MNRL | `mnrl_loss` | Henderson et al. 2017 |
| N-Pairs | `n_pairs_loss` | Sohn 2016 |
| Cosine embedding | `cosine_embedding_loss` | |
| SupCon | `supcon_loss` | Khosla et al. 2020 |
| Multi-Similarity | `multi_similarity_loss` | Wang et al. 2019 |
| Circle | `circle_loss` | Sun et al. 2020 |
| Lifted Structured | `lifted_structured_loss` | Song et al. 2016 |
| ArcFace | `arcface_loss` | Deng et al. 2019 |
| ProxyAnchor | `proxy_anchor_loss` | Kim et al. 2020 |
| VICReg | `vicreg_loss` | Bardes et al. 2022 |
| Matryoshka (wrapper) | `matryoshka_loss` | Kusupati et al. 2022 |

`CrossBatchMemory` expands the negative pool across batches.

## Distance learning

`learn_transform`, `nca`, `lmnn`, `itml`, `kissme` — closed-form / first-order
solvers on `&[f32]` slices. No autograd, no Burn dependency.

## Negative miners

`HardestMiner`, `SemiHardMiner`, `RandomMiner`, `InBatchMiner`,
`MultiSimilarityMiner`, `DistanceWeightedMiner`. Operate on similarity matrices.

## Axiom utilities

`axioms::triangle_violation` measures the worst triangle-inequality gap of a
distance function over a sample. `bounds::ConvexProjector` clamps a distance
matrix to the metric cone via Floyd-Warshall-style triangle propagation.
Useful for sanity-checking learned metrics.

## Usage

```rust
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use tuplet::burn_losses;

type B = Autodiff<NdArray>;
let device = Default::default();

let anchors:   Tensor<B, 2> = Tensor::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
let positives: Tensor<B, 2> = Tensor::from_floats([[0.9, 0.1], [0.1, 0.9]], &device);
let negatives: Tensor<B, 2> = Tensor::from_floats([[0.0, 1.0], [1.0, 0.0]], &device);

let loss = burn_losses::triplet_loss(anchors, positives, negatives, 0.2);
let grads = loss.backward();
```

## Features

| Feature | What it adds |
|---------|-------------|
| `mining` (default) | Negative miners |
| `burn-ndarray` | Burn losses with `NdArray` autodiff backend (multi-core CPU) |
| `burn-wgpu` | Burn losses on Metal / Vulkan / WebGPU |
| `burn-tch` | Burn losses on libtorch |
| `burn-cuda` | Burn losses on CUDA (extends `burn-ndarray`) |
| `simd` | SIMD-accelerated similarity via [innr](https://crates.io/crates/innr) |
| `serde` | Serialize / deserialize config structs |

## Examples

```sh
cargo run --example metric_learning      # classical methods overview
cargo run --example mining_pipeline      # miners + losses end-to-end
cargo run --features burn-ndarray --example burn_training  # Burn autograd training
```

License: MIT OR Apache-2.0
