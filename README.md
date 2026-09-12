# tuplet

Tuple-based metric learning utilities, with selected losses on Burn tensors.

Contrastive, proxy, and self-supervised losses operate on pairs, triplets, and
n-tuples of embeddings. Hard-negative mining and classical Mahalanobis distance
learning (NCA, LMNN, ITML, KISSME) are pure Rust and need no autograd backend.

## Install

```toml
[dependencies]
# Unreleased development API, pure-Rust features only:
tuplet = { git = "https://github.com/arclabs561/tuplet", branch = "main" }
```

For an application that uses Burn tensors, enable any required Tuplet runtime
features. This CPU configuration also pins the direct Burn dependency needed by
the Rust code below:

```toml
[dependencies]
tuplet = { git = "https://github.com/arclabs561/tuplet", branch = "main", features = ["burn-flex"] }
burn = { git = "https://github.com/tracel-ai/burn", rev = "1414c8a14e5169ef5e5fc67f9b8ab01a25d6352d", default-features = false, features = ["std", "autodiff", "flex"] }
```

For Metal, replace `burn-flex` / `flex` with `burn-metal` / `metal`, and add
`fusion` to Burn's features:

```toml
[dependencies]
tuplet = { git = "https://github.com/arclabs561/tuplet", branch = "main", features = ["burn-metal"] }
burn = { git = "https://github.com/tracel-ai/burn", rev = "1414c8a14e5169ef5e5fc67f9b8ab01a25d6352d", default-features = false, features = ["std", "autodiff", "metal", "fusion"] }
```

The default pure-Rust feature set supports Rust 1.85. Any `burn-*` feature on
the development branch uses the pinned Burn 0.22 revision and requires Rust
1.95. Multiple `burn-*` features may be enabled together; `burn-ndarray`,
`burn-wgpu`, `burn-tch`, and `burn-cuda` remain available alongside the new
`burn-flex` and `burn-metal` features.

The latest crates.io release, Tuplet 0.3, remains on Burn 0.21.

### Burn 0.22 development migration

Burn 0.22 dispatches `Tensor<D>` through its runtime device. Replace generic
`Tensor<B, D>` APIs and `Backend`/`BackendTypes` device plumbing with
`Tensor<D>` and `Device`. For CPU training use `Device::flex().autodiff()`;
for Metal use `Device::metal(Default::default()).autodiff()`.

## Losses

The Burn implementations of triplet, contrastive, and InfoNCE losses operate on
runtime-dispatched `Tensor<_>` values and return a scalar tensor for
`.backward()`.
The other losses in this table use the pure-Rust slice API.

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

`learn_transform`, `nca`, `lmnn`, `itml`, `kissme`: closed-form / first-order
solvers on `&[f32]` slices. No autograd, no Burn dependency.

## Negative miners

`HardestMiner`, `SemiHardMiner`, `RandomMiner`, `InBatchMiner`,
`MultiSimilarityMiner`, `DistanceWeightedMiner`. Operate on similarity matrices.

## Axiom utilities

`axioms::triangle_violation` measures the worst triangle-inequality gap of a
distance function over a sample. It is useful for sanity-checking learned
metrics.

## Usage

```rust
use burn::tensor::{DType, Device, Tensor};
use tuplet::burn_losses;

let device = Device::flex().autodiff();

let anchors = Tensor::<2>::from_data(
    [[1.0, 0.0], [0.0, 1.0]], (&device, DType::F32),
).require_grad();
let positives = Tensor::<2>::from_data(
    [[0.9, 0.1], [0.1, 0.9]], (&device, DType::F32),
);
let negatives = Tensor::<2>::from_data(
    [[0.0, 1.0], [1.0, 0.0]], (&device, DType::F32),
);

let loss = burn_losses::triplet_loss(anchors, positives, negatives, 0.2);
let grads = loss.backward();
```

## Features

| Feature | What it adds |
|---------|-------------|
| `mining` (default) | Negative miners |
| `burn-flex` | Burn losses with the Flex CPU runtime |
| `burn-ndarray` | Burn losses with Burn's ndarray runtime |
| `burn-wgpu` | Burn losses on Metal / Vulkan / WebGPU |
| `burn-metal` | Burn losses on Burn's Metal runtime with fusion |
| `burn-tch` | Burn losses on libtorch |
| `burn-cuda` | Burn losses on CUDA |
| `simd` | SIMD-accelerated similarity via [innr](https://crates.io/crates/innr) |
| `serde` | Serialize / deserialize config structs |

## Examples

See [examples/README.md](examples/README.md) for runnable examples with
captured output.

License: MIT OR Apache-2.0
