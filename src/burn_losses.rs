//! Burn-tensor metric-learning losses with autograd.
//!
//! All functions take backend-independent `Tensor<_>` values and return a
//! scalar loss tensor on which `.backward()` can be called. Embeddings are
//! `[batch, dim]`; labels are `[batch]` of `Int`.
//!
//! Enable the `burn-*` feature for the backend used by the calling binary.
//! The same loss code is generic over all supported autodiff backends.

use burn::tensor::{Int, Tensor};

// Re-export aliases for the Burn types in this module's public API. Downstream
// callers can import `BurnTensor` and `BurnInt` from here to match tuplet's
// Burn version.
pub use burn::tensor::{Int as BurnInt, Tensor as BurnTensor};

/// L2-normalize each row to unit length. Used by similarity-based losses.
fn l2_normalize(x: Tensor<2>) -> Tensor<2> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt();
    let safe = norm.clamp_min(1e-12);
    x / safe
}

/// Pairwise euclidean distance between two equally-shaped batches `[B, D]`.
/// Returns `[B]` (rank-1).
fn pairwise_euclidean(a: Tensor<2>, b: Tensor<2>) -> Tensor<1> {
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
pub fn triplet_loss(
    anchors: Tensor<2>,
    positives: Tensor<2>,
    negatives: Tensor<2>,
    margin: f32,
) -> Tensor<1> {
    let d_ap = pairwise_euclidean(anchors.clone(), positives);
    let d_an = pairwise_euclidean(anchors, negatives);
    (d_ap - d_an + margin).clamp_min(0.0).mean().unsqueeze()
}

/// Pairwise contrastive loss with a single label vector (1 = same class, 0 = different).
///
/// Same-class pairs minimize squared distance; different-class pairs are
/// pushed beyond `margin`. Reduction is mean over the batch.
pub fn contrastive_loss(
    a: Tensor<2>,
    b: Tensor<2>,
    same_class: Tensor<1, Int>,
    margin: f32,
) -> Tensor<1> {
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
pub fn infonce_loss(anchors: Tensor<2>, positives: Tensor<2>, temperature: f32) -> Tensor<1> {
    let device = anchors.device();
    let n = anchors.dims()[0];

    let a = l2_normalize(anchors);
    let p = l2_normalize(positives);

    // [n, n] similarity matrices.
    let logits_ap = a.clone().matmul(p.clone().transpose()) / temperature;
    let logits_pa = logits_ap.clone().transpose();

    // Targets: identity row indices (positive is on the diagonal).
    let targets = Tensor::<1, Int>::arange(0..n as i64, &device);

    let loss_ap = cross_entropy(logits_ap, targets.clone());
    let loss_pa = cross_entropy(logits_pa, targets);
    (loss_ap + loss_pa) / 2.0
}

/// Numerically stable softmax cross-entropy for a `[N, C]` logits tensor and
/// `[N]` int targets. Returns scalar (mean over rows).
fn cross_entropy(logits: Tensor<2>, targets: Tensor<1, Int>) -> Tensor<1> {
    let n = logits.dims()[0];
    let max = logits.clone().max_dim(1);
    let shifted = logits - max.clone();
    let log_z = shifted.clone().exp().sum_dim(1).log().reshape([n]);
    // Pick out targets[i]-th column from each row.
    let target_logits = shifted.gather(1, targets.unsqueeze_dim(1)).reshape([n]);
    (log_z - target_logits).mean().unsqueeze()
}

#[cfg(all(test, feature = "burn-flex"))]
mod tests {
    use super::*;
    use burn::tensor::{Device, FloatDType, TensorData};

    fn tensor2(data: &[&[f32]], device: &Device) -> Tensor<2> {
        let n = data.len();
        let d = data[0].len();
        let flat: Vec<f32> = data.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::<1>::from_data(
            TensorData::new(flat, [n * d]),
            (device, FloatDType::F32.into()),
        )
        .reshape([n, d])
    }

    #[test]
    fn triplet_zero_when_satisfied() {
        // d(a,p)=0, d(a,n)=2, margin=0.5 => max(0, 0-2+0.5)=0
        let device = Device::flex().autodiff();
        let a = tensor2(&[&[0.0, 0.0]], &device);
        let p = tensor2(&[&[0.0, 0.0]], &device);
        let n = tensor2(&[&[2.0, 0.0]], &device);
        let loss = triplet_loss(a, p, n, 0.5);
        let value: f32 = loss.into_scalar();
        assert!(value.abs() < 1e-5, "expected zero loss, got {value}");
    }

    #[test]
    fn triplet_positive_when_violated() {
        // d(a,p)=2, d(a,n)=0, margin=0.5 => max(0, 2-0+0.5)=2.5
        let device = Device::flex().autodiff();
        let a = tensor2(&[&[0.0, 0.0]], &device);
        let p = tensor2(&[&[2.0, 0.0]], &device);
        let n = tensor2(&[&[0.0, 0.0]], &device);
        let loss = triplet_loss(a, p, n, 0.5);
        let value: f32 = loss.into_scalar();
        assert!((value - 2.5).abs() < 1e-4, "expected 2.5, got {value}");
    }

    #[test]
    fn triplet_backward_runs() {
        let device = Device::flex().autodiff();
        let a = tensor2(&[&[1.0, 0.0], &[0.0, 1.0]], &device).require_grad();
        let p = tensor2(&[&[0.9, 0.1], &[0.1, 0.9]], &device);
        let n = tensor2(&[&[0.0, 1.0], &[1.0, 0.0]], &device);
        let loss = triplet_loss(a.clone(), p, n, 0.2);
        let _ = loss.backward();
    }

    #[cfg(all(target_os = "macos", feature = "burn-metal"))]
    mod metal_parity {
        use super::*;
        use burn::tensor::Gradients;

        fn gradient(input: &Tensor<2>, gradients: &Gradients) -> Vec<f32> {
            input
                .grad(gradients)
                .expect("input requires gradients")
                .to_data()
                .as_slice::<f32>()
                .expect("f32 gradient data")
                .to_vec()
        }

        fn evaluate(device: &Device) -> [(f32, Vec<f32>); 3] {
            let anchors = tensor2(&[&[0.0, 0.0], &[0.0, 1.0]], device).require_grad();
            let positives = tensor2(&[&[0.3, 0.3], &[0.2, 0.8]], device).require_grad();
            let negatives = tensor2(&[&[0.0, 0.7], &[0.5, 0.2]], device).require_grad();
            let triplet = triplet_loss(anchors.clone(), positives.clone(), negatives.clone(), 0.8);
            let triplet_value = triplet.clone().into_scalar();
            let triplet_grads = triplet.backward();
            let mut triplet_input_grads = gradient(&anchors, &triplet_grads);
            triplet_input_grads.extend(gradient(&positives, &triplet_grads));
            triplet_input_grads.extend(gradient(&negatives, &triplet_grads));

            let a = tensor2(&[&[0.2, 0.4], &[0.8, -0.3]], device).require_grad();
            let b = tensor2(&[&[0.6, 0.1], &[0.1, 0.2]], device).require_grad();
            let labels = Tensor::<1, Int>::from_ints([1, 0], device);
            let contrastive = contrastive_loss(a.clone(), b.clone(), labels, 1.0);
            let contrastive_value = contrastive.clone().into_scalar();
            let contrastive_grads = contrastive.backward();
            let mut contrastive_input_grads = gradient(&a, &contrastive_grads);
            contrastive_input_grads.extend(gradient(&b, &contrastive_grads));

            let info_anchors = tensor2(&[&[1.0, 0.0], &[0.0, 1.0]], device).require_grad();
            let info_positives = tensor2(&[&[1.0, 0.0], &[0.0, 1.0]], device).require_grad();
            let infonce = infonce_loss(info_anchors.clone(), info_positives.clone(), 1.0);
            let infonce_value = infonce.clone().into_scalar();
            let infonce_grads = infonce.backward();
            let mut infonce_input_grads = gradient(&info_anchors, &infonce_grads);
            infonce_input_grads.extend(gradient(&info_positives, &infonce_grads));

            [
                (triplet_value, triplet_input_grads),
                (contrastive_value, contrastive_input_grads),
                (infonce_value, infonce_input_grads),
            ]
        }

        #[test]
        #[ignore = "requires a Metal GPU"]
        fn loss_values_and_input_gradients_match_flex() {
            let cpu = evaluate(&Device::flex().autodiff());
            for ((name, actual), expected) in [
                ("triplet", cpu[0].0),
                ("contrastive", cpu[1].0),
                ("infonce", cpu[2].0),
            ]
            .into_iter()
            .zip([0.331_854_34, 0.134_767_52, 0.313_261_7])
            {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "{name} Flex loss differs: expected {expected}, got {actual}"
                );
            }

            let metal = evaluate(&Device::metal(Default::default()).autodiff());
            for (name, (cpu_result, metal_result)) in ["triplet", "contrastive", "infonce"]
                .into_iter()
                .zip(cpu.iter().zip(metal.iter()))
            {
                assert!(
                    (cpu_result.0 - metal_result.0).abs() < 1e-4,
                    "{name} loss differs: Flex={} Metal={}",
                    cpu_result.0,
                    metal_result.0
                );
                assert_eq!(
                    cpu_result.1.len(),
                    metal_result.1.len(),
                    "{name} gradient length differs"
                );
                for (index, (cpu_grad, metal_grad)) in
                    cpu_result.1.iter().zip(metal_result.1.iter()).enumerate()
                {
                    assert!(
                        (cpu_grad - metal_grad).abs() < 1e-4,
                        "{name} gradient {index} differs: Flex={cpu_grad} Metal={metal_grad}"
                    );
                }
            }
        }
    }
}
