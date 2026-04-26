//! Burn-tensor metric-learning losses with autograd.
//!
//! All functions take `Tensor<B, _>` where `B: AutodiffBackend` and return a
//! scalar loss tensor on which `.backward()` can be called. Embeddings are
//! `[batch, dim]`; labels are `[batch]` of `Int`.
//!
//! Backend selection is per-binary: pick exactly one of `burn-ndarray`,
//! `burn-wgpu`, `burn-tch`, or `burn-cuda` in your `Cargo.toml` features.
//! The same loss code runs on all of them.
//!
//! ## Status
//!
//! Implemented: `triplet_loss`, `contrastive_loss`, `infonce_loss`.
//! The remaining 13 losses (multi_similarity, supcon, circle, lifted_structured,
//! n_pairs, cosine_embedding, mnrl, arcface, vicreg, proxy_anchor, matryoshka,
//! and CrossBatchMemory) are being ported from the previous candle backend in
//! follow-up commits.

use burn::tensor::{Int, Tensor, backend::AutodiffBackend};

// Re-export the Burn types that appear in this module's public API. Per
// Effective Rust Item 24, a major-version bump of `burn` is a breaking change
// of this crate's public surface; re-exporting lets downstream code import the
// matching version through `tuplet::burn_losses::{Tensor, AutodiffBackend, Int}`
// instead of pinning `burn` independently and risking a type-mismatch error
// with no obvious cause.
pub use burn::tensor::{Int as BurnInt, Tensor as BurnTensor, backend::AutodiffBackend as Backend};

/// L2-normalize each row to unit length. Used by similarity-based losses.
fn l2_normalize<B: AutodiffBackend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt();
    let safe = norm.clamp_min(1e-12);
    x / safe
}

/// Pairwise euclidean distance between two equally-shaped batches `[B, D]`.
/// Returns `[B]` (rank-1).
fn pairwise_euclidean<B: AutodiffBackend>(a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 1> {
    let n = a.dims()[0];
    (a - b)
        .powf_scalar(2.0)
        .sum_dim(1)
        .clamp_min(1e-12)
        .sqrt()
        .reshape([n])
}

/// Triplet margin loss: `mean(max(0, d(a, p) - d(a, n) + margin))`.
///
/// All three inputs share shape `[batch, dim]`. Reduction is mean over the
/// batch (matches PyTorch's `TripletMarginLoss` default).
pub fn triplet_loss<B: AutodiffBackend>(
    anchors: Tensor<B, 2>,
    positives: Tensor<B, 2>,
    negatives: Tensor<B, 2>,
    margin: f32,
) -> Tensor<B, 1> {
    let d_ap = pairwise_euclidean(anchors.clone(), positives);
    let d_an = pairwise_euclidean(anchors, negatives);
    (d_ap - d_an + margin).clamp_min(0.0).mean().unsqueeze()
}

/// Pairwise contrastive loss with a single label vector (1 = same class, 0 = different).
///
/// Same-class pairs minimize squared distance; different-class pairs are
/// pushed beyond `margin`. Reduction is mean over the batch.
pub fn contrastive_loss<B: AutodiffBackend>(
    a: Tensor<B, 2>,
    b: Tensor<B, 2>,
    same_class: Tensor<B, 1, Int>,
    margin: f32,
) -> Tensor<B, 1> {
    let d = pairwise_euclidean(a, b);
    let pos_mask = same_class.clone().float();
    let neg_mask = -pos_mask.clone() + 1.0;
    let pos_term = pos_mask * d.clone().powf_scalar(2.0);
    let neg_term = neg_mask * (-d + margin).clamp_min(0.0).powf_scalar(2.0);
    (pos_term + neg_term).mean().unsqueeze()
}

/// InfoNCE / NT-Xent over anchor / positive pairs in a batch.
///
/// Each anchor's positive is paired by row index; all other rows in
/// `positives` serve as negatives. Symmetric variant: also computes the
/// loss with anchors / positives swapped, then averages.
pub fn infonce_loss<B: AutodiffBackend>(
    anchors: Tensor<B, 2>,
    positives: Tensor<B, 2>,
    temperature: f32,
) -> Tensor<B, 1> {
    let device = anchors.device();
    let n = anchors.dims()[0];

    let a = l2_normalize(anchors);
    let p = l2_normalize(positives);

    // [n, n] similarity matrices.
    let logits_ap = a.clone().matmul(p.clone().transpose()) / temperature;
    let logits_pa = logits_ap.clone().transpose();

    // Targets: identity row indices (positive is on the diagonal).
    let targets = Tensor::<B::InnerBackend, 1, Int>::arange(0..n as i64, &device);
    let targets = Tensor::<B, 1, Int>::from_inner(targets);

    let loss_ap = cross_entropy(logits_ap, targets.clone());
    let loss_pa = cross_entropy(logits_pa, targets);
    (loss_ap + loss_pa) / 2.0
}

/// Numerically stable softmax cross-entropy for a `[N, C]` logits tensor and
/// `[N]` int targets. Returns scalar (mean over rows).
fn cross_entropy<B: AutodiffBackend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let n = logits.dims()[0];
    let max = logits.clone().max_dim(1);
    let shifted = logits - max.clone();
    let log_z = shifted.clone().exp().sum_dim(1).log().reshape([n]);
    // Pick out targets[i]-th column from each row.
    let target_logits = shifted.gather(1, targets.unsqueeze_dim(1)).reshape([n]);
    (log_z - target_logits).mean().unsqueeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type B = Autodiff<NdArray>;

    fn tensor2(data: &[&[f32]]) -> Tensor<B, 2> {
        let device = Default::default();
        let n = data.len();
        let d = data[0].len();
        let flat: Vec<f32> = data.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::<B, 1>::from_data(TensorData::new(flat, [n * d]), &device).reshape([n, d])
    }

    #[test]
    fn triplet_zero_when_satisfied() {
        // d(a,p)=0, d(a,n)=2, margin=0.5 => max(0, 0-2+0.5)=0
        let a = tensor2(&[&[0.0, 0.0]]);
        let p = tensor2(&[&[0.0, 0.0]]);
        let n = tensor2(&[&[2.0, 0.0]]);
        let loss = triplet_loss(a, p, n, 0.5);
        let v: f32 = loss.into_scalar();
        assert!(v.abs() < 1e-5, "expected zero loss, got {v}");
    }

    #[test]
    fn triplet_positive_when_violated() {
        // d(a,p)=2, d(a,n)=0, margin=0.5 => max(0, 2-0+0.5)=2.5
        let a = tensor2(&[&[0.0, 0.0]]);
        let p = tensor2(&[&[2.0, 0.0]]);
        let n = tensor2(&[&[0.0, 0.0]]);
        let loss = triplet_loss(a, p, n, 0.5);
        let v: f32 = loss.into_scalar();
        assert!((v - 2.5).abs() < 1e-4, "expected 2.5, got {v}");
    }

    #[test]
    fn triplet_backward_runs() {
        // Smoke test: gradient should flow without panicking.
        let a = tensor2(&[&[1.0, 0.0], &[0.0, 1.0]]).require_grad();
        let p = tensor2(&[&[0.9, 0.1], &[0.1, 0.9]]);
        let n = tensor2(&[&[0.0, 1.0], &[1.0, 0.0]]);
        let loss = triplet_loss(a.clone(), p, n, 0.2);
        let _grads = loss.backward();
    }
}
