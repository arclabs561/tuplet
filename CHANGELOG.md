# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- **Breaking:** optional Burn losses now use the pinned Burn 0.22 development
  revision and runtime-dispatched `Tensor<D>` values. Callers no longer supply
  a `Backend` type parameter or `BackendTypes` device associated type.
- Added `burn-flex` for the CPU runtime and `burn-metal` for the Metal runtime.
  Existing `burn-ndarray`, `burn-wgpu`, `burn-tch`, and `burn-cuda` feature
  names remain available; multiple backend features may coexist.
- The default pure-Rust feature set remains Rust 1.85. Any `burn-*` feature on
  the development branch requires Rust 1.95.

## [0.3.0] - 2026-09-10

### Changed

- Updated optional Burn integrations to 0.21. `burn-*` features require Rust 1.92.

## [0.2.0] - 2026-04-25

### Added

- `new()` constructors for Mahalanobis configurations.

### Changed

- Marked Mahalanobis configuration types as `#[non_exhaustive]` (breaking).

## [0.1.0] - 2026-04-25

### Added

- Contrastive, multi-similarity, SupCon, circle, lifted-structured, and N-pairs losses.
- ArcFace, VICReg, and MNRL/matryoshka losses on Burn tensors.
- Mahalanobis distance learning: NCA, LMNN, ITML, KISSME.
- Hard-negative mining (`MSMiner`) and a `metric_learning` example.
- Analytical gradients for multi-similarity, circle, and lifted-structured losses.
- Candle and SIMD feature integrations with cross-backend tests.

### Fixed

- Candle softplus overflow.
- `partial_cmp` panic.

[0.3.0]: https://github.com/arclabs561/tuplet/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/arclabs561/tuplet/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/arclabs561/tuplet/releases/tag/v0.1.0
