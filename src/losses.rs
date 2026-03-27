use crate::similarity::{cosine_similarity, dot, euclidean_distance};

/// Result of a loss computation.
#[derive(Debug, Clone)]
pub struct LossOutput {
    /// Scalar loss value.
    pub loss: f32,
    /// Gradients w.r.t. anchor embeddings. Shape: `[batch_size][dim]`.
    pub grad_anchors: Vec<Vec<f32>>,
    /// Gradients w.r.t. positive embeddings. Shape: `[batch_size][dim]`.
    pub grad_positives: Vec<Vec<f32>>,
    /// Gradients w.r.t. negative embeddings. Shape: `[n_negatives][dim]`.
    /// Empty for in-batch negative losses.
    pub grad_negatives: Vec<Vec<f32>>,
}

/// Distance function selector for triplet loss.
#[derive(Debug, Clone, Copy)]
pub enum DistanceFn {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Cosine distance: `1 - cosine_similarity`.
    Cosine,
}

/// Triplet loss: `max(0, d(a,p) - d(a,n) + margin)`, averaged over the batch.
pub fn triplet_loss(
    anchors: &[&[f32]],
    positives: &[&[f32]],
    negatives: &[&[f32]],
    margin: f32,
) -> LossOutput {
    triplet_loss_with_distance(anchors, positives, negatives, margin, DistanceFn::Euclidean)
}

/// Triplet loss with a configurable distance function.
pub fn triplet_loss_with_distance(
    anchors: &[&[f32]],
    positives: &[&[f32]],
    negatives: &[&[f32]],
    margin: f32,
    distance: DistanceFn,
) -> LossOutput {
    let batch = anchors.len();
    assert_eq!(batch, positives.len());
    assert_eq!(batch, negatives.len());
    assert!(batch > 0);

    let dim = anchors[0].len();
    let mut total_loss = 0.0f32;
    let mut grad_a = vec![vec![0.0f32; dim]; batch];
    let mut grad_p = vec![vec![0.0f32; dim]; batch];
    let mut grad_n = vec![vec![0.0f32; dim]; batch];

    for i in 0..batch {
        let (d_ap, d_an) = match distance {
            DistanceFn::Euclidean => (
                euclidean_distance(anchors[i], positives[i]),
                euclidean_distance(anchors[i], negatives[i]),
            ),
            DistanceFn::Cosine => (
                1.0 - cosine_similarity(anchors[i], positives[i]),
                1.0 - cosine_similarity(anchors[i], negatives[i]),
            ),
        };

        let loss_i = (d_ap - d_an + margin).max(0.0);
        total_loss += loss_i;

        if loss_i > 0.0 {
            match distance {
                DistanceFn::Euclidean => {
                    let inv_d_ap = if d_ap > 1e-8 { 1.0 / d_ap } else { 0.0 };
                    let inv_d_an = if d_an > 1e-8 { 1.0 / d_an } else { 0.0 };

                    for d in 0..dim {
                        let diff_ap = anchors[i][d] - positives[i][d];
                        let diff_an = anchors[i][d] - negatives[i][d];

                        // dL/d_anchor = (a-p)/d(a,p) - (a-n)/d(a,n)
                        grad_a[i][d] += diff_ap * inv_d_ap - diff_an * inv_d_an;
                        // dL/d_positive = -(a-p)/d(a,p)
                        grad_p[i][d] += -diff_ap * inv_d_ap;
                        // dL/d_negative = (a-n)/d(a,n)
                        grad_n[i][d] += diff_an * inv_d_an;
                    }
                }
                DistanceFn::Cosine => {
                    // d(cosine_distance)/d(a) = -d(cosine_similarity)/d(a)
                    // cosine_sim = dot(a,b) / (|a| * |b|)
                    let norm_a = dot(anchors[i], anchors[i]).sqrt();
                    let norm_p = dot(positives[i], positives[i]).sqrt();
                    let norm_n = dot(negatives[i], negatives[i]).sqrt();

                    if norm_a > 1e-8 && norm_p > 1e-8 && norm_n > 1e-8 {
                        let cos_ap = dot(anchors[i], positives[i]) / (norm_a * norm_p);
                        let cos_an = dot(anchors[i], negatives[i]) / (norm_a * norm_n);

                        for d in 0..dim {
                            // d(cos_sim(a,p))/d(a_d) = p_d/(|a|*|p|) - a_d*cos_ap/|a|^2
                            let dcos_ap_da = positives[i][d] / (norm_a * norm_p)
                                - anchors[i][d] * cos_ap / (norm_a * norm_a);
                            let dcos_ap_dp = anchors[i][d] / (norm_a * norm_p)
                                - positives[i][d] * cos_ap / (norm_p * norm_p);

                            let dcos_an_da = negatives[i][d] / (norm_a * norm_n)
                                - anchors[i][d] * cos_an / (norm_a * norm_a);
                            let dcos_an_dn = anchors[i][d] / (norm_a * norm_n)
                                - negatives[i][d] * cos_an / (norm_n * norm_n);

                            // loss = (1 - cos_ap) - (1 - cos_an) + margin = cos_an - cos_ap + margin
                            // dL/da = -dcos_ap/da + dcos_an/da
                            grad_a[i][d] += -dcos_ap_da + dcos_an_da;
                            grad_p[i][d] += -dcos_ap_dp;
                            grad_n[i][d] += dcos_an_dn;
                        }
                    }
                }
            }
        }
    }

    let scale = 1.0 / batch as f32;
    total_loss *= scale;
    for i in 0..batch {
        for d in 0..dim {
            grad_a[i][d] *= scale;
            grad_p[i][d] *= scale;
            grad_n[i][d] *= scale;
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_a,
        grad_positives: grad_p,
        grad_negatives: grad_n,
    }
}

/// InfoNCE (NT-Xent) loss with in-batch negatives.
///
/// For anchor i, the positive is `positives[i]` and negatives are all `positives[j]` where `j != i`.
pub fn infonce_loss(anchors: &[&[f32]], positives: &[&[f32]], temperature: f32) -> LossOutput {
    let batch = anchors.len();
    assert_eq!(batch, positives.len());
    assert!(batch > 0);
    assert!(temperature > 0.0);

    let dim = anchors[0].len();

    // Compute all anchor-positive similarities
    let mut sims = vec![vec![0.0f32; batch]; batch];
    for i in 0..batch {
        for j in 0..batch {
            sims[i][j] = cosine_similarity(anchors[i], positives[j]) / temperature;
        }
    }

    // Softmax per row and cross-entropy loss
    let mut total_loss = 0.0f32;
    let mut softmax = vec![vec![0.0f32; batch]; batch];

    for i in 0..batch {
        let max_sim = sims[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for j in 0..batch {
            softmax[i][j] = (sims[i][j] - max_sim).exp();
            sum_exp += softmax[i][j];
        }
        for val in softmax[i].iter_mut().take(batch) {
            *val /= sum_exp;
        }
        // Loss = -log(softmax[i][i])
        total_loss += -softmax[i][i].ln();
    }
    total_loss /= batch as f32;

    // Gradients
    // dL/d(sim[i][j]) = softmax[i][j] - indicator(j == i)
    // Then chain through cosine similarity to get gradients w.r.t. embeddings
    let mut grad_a = vec![vec![0.0f32; dim]; batch];
    let mut grad_p = vec![vec![0.0f32; dim]; batch];

    for i in 0..batch {
        let norm_a = dot(anchors[i], anchors[i]).sqrt();
        if norm_a < 1e-8 {
            continue;
        }

        for j in 0..batch {
            let norm_p = dot(positives[j], positives[j]).sqrt();
            if norm_p < 1e-8 {
                continue;
            }

            // dL/d(sim_ij) = (softmax_ij - delta_ij) / (batch * temperature)
            let delta = if i == j { 1.0 } else { 0.0 };
            let dsim = (softmax[i][j] - delta) / (batch as f32 * temperature);

            let cos_val = dot(anchors[i], positives[j]) / (norm_a * norm_p);

            for d in 0..dim {
                let dcos_da = positives[j][d] / (norm_a * norm_p)
                    - anchors[i][d] * cos_val / (norm_a * norm_a);
                let dcos_dp = anchors[i][d] / (norm_a * norm_p)
                    - positives[j][d] * cos_val / (norm_p * norm_p);

                grad_a[i][d] += dsim * dcos_da;
                grad_p[j][d] += dsim * dcos_dp;
            }
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_a,
        grad_positives: grad_p,
        grad_negatives: vec![],
    }
}

/// Multiple Negatives Ranking Loss (MNRL).
///
/// Like InfoNCE but with explicit negatives in addition to in-batch negatives.
/// The positive for anchor i should have the highest similarity among all candidates.
pub fn mnrl_loss(
    anchors: &[&[f32]],
    positives: &[&[f32]],
    negatives: &[&[f32]],
    temperature: f32,
) -> LossOutput {
    let batch = anchors.len();
    assert_eq!(batch, positives.len());
    assert!(batch > 0);
    assert!(temperature > 0.0);

    let dim = anchors[0].len();
    let n_neg = negatives.len();
    let n_candidates = batch + n_neg; // in-batch positives + explicit negatives

    // Compute similarities: for each anchor, similarity to all positives + all negatives
    let mut sims = vec![vec![0.0f32; n_candidates]; batch];
    for i in 0..batch {
        for j in 0..batch {
            sims[i][j] = cosine_similarity(anchors[i], positives[j]) / temperature;
        }
        for k in 0..n_neg {
            sims[i][batch + k] = cosine_similarity(anchors[i], negatives[k]) / temperature;
        }
    }

    // Softmax + cross-entropy (target is index i for anchor i)
    let mut total_loss = 0.0f32;
    let mut softmax = vec![vec![0.0f32; n_candidates]; batch];

    for i in 0..batch {
        let max_sim = sims[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for j in 0..n_candidates {
            softmax[i][j] = (sims[i][j] - max_sim).exp();
            sum_exp += softmax[i][j];
        }
        for val in softmax[i].iter_mut().take(n_candidates) {
            *val /= sum_exp;
        }
        total_loss += -softmax[i][i].ln();
    }
    total_loss /= batch as f32;

    // Gradients
    let mut grad_a = vec![vec![0.0f32; dim]; batch];
    let mut grad_p = vec![vec![0.0f32; dim]; batch];
    let mut grad_n = vec![vec![0.0f32; dim]; n_neg];

    for i in 0..batch {
        let norm_a = dot(anchors[i], anchors[i]).sqrt();
        if norm_a < 1e-8 {
            continue;
        }

        // Gradients through positive candidates
        for j in 0..batch {
            let norm_p = dot(positives[j], positives[j]).sqrt();
            if norm_p < 1e-8 {
                continue;
            }

            let delta = if i == j { 1.0 } else { 0.0 };
            let dsim = (softmax[i][j] - delta) / (batch as f32 * temperature);
            let cos_val = dot(anchors[i], positives[j]) / (norm_a * norm_p);

            for d in 0..dim {
                let dcos_da = positives[j][d] / (norm_a * norm_p)
                    - anchors[i][d] * cos_val / (norm_a * norm_a);
                let dcos_dp = anchors[i][d] / (norm_a * norm_p)
                    - positives[j][d] * cos_val / (norm_p * norm_p);

                grad_a[i][d] += dsim * dcos_da;
                grad_p[j][d] += dsim * dcos_dp;
            }
        }

        // Gradients through explicit negatives
        for k in 0..n_neg {
            let norm_n = dot(negatives[k], negatives[k]).sqrt();
            if norm_n < 1e-8 {
                continue;
            }

            let dsim = softmax[i][batch + k] / (batch as f32 * temperature);
            let cos_val = dot(anchors[i], negatives[k]) / (norm_a * norm_n);

            for d in 0..dim {
                let dcos_da = negatives[k][d] / (norm_a * norm_n)
                    - anchors[i][d] * cos_val / (norm_a * norm_a);
                let dcos_dn = anchors[i][d] / (norm_a * norm_n)
                    - negatives[k][d] * cos_val / (norm_n * norm_n);

                grad_a[i][d] += dsim * dcos_da;
                grad_n[k][d] += dsim * dcos_dn;
            }
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_a,
        grad_positives: grad_p,
        grad_negatives: grad_n,
    }
}

/// Cosine embedding loss for labeled pairs.
///
/// For positive pairs (`label = true`): `loss = 1 - cos(a, b)`.
/// For negative pairs (`label = false`): `loss = max(0, cos(a, b) - margin)`.
pub fn cosine_embedding_loss(
    pairs: &[(&[f32], &[f32])],
    labels: &[bool],
    margin: f32,
) -> LossOutput {
    assert_eq!(pairs.len(), labels.len());
    let n = pairs.len();
    assert!(n > 0);

    let dim = pairs[0].0.len();
    let mut total_loss = 0.0f32;
    let mut grad_a = vec![vec![0.0f32; dim]; n];
    let mut grad_p = vec![vec![0.0f32; dim]; n];

    for i in 0..n {
        let (a, b) = pairs[i];
        let cos = cosine_similarity(a, b);

        let norm_a = dot(a, a).sqrt();
        let norm_b = dot(b, b).sqrt();

        if labels[i] {
            // Positive: loss = 1 - cos
            total_loss += 1.0 - cos;

            if norm_a > 1e-8 && norm_b > 1e-8 {
                for d in 0..dim {
                    // dL/da = -d(cos)/da
                    let dcos_da = b[d] / (norm_a * norm_b) - a[d] * cos / (norm_a * norm_a);
                    let dcos_db = a[d] / (norm_a * norm_b) - b[d] * cos / (norm_b * norm_b);
                    grad_a[i][d] = -dcos_da;
                    grad_p[i][d] = -dcos_db;
                }
            }
        } else {
            // Negative: loss = max(0, cos - margin)
            let loss_i = (cos - margin).max(0.0);
            total_loss += loss_i;

            if loss_i > 0.0 && norm_a > 1e-8 && norm_b > 1e-8 {
                for d in 0..dim {
                    let dcos_da = b[d] / (norm_a * norm_b) - a[d] * cos / (norm_a * norm_a);
                    let dcos_db = a[d] / (norm_a * norm_b) - b[d] * cos / (norm_b * norm_b);
                    grad_a[i][d] = dcos_da;
                    grad_p[i][d] = dcos_db;
                }
            }
        }
    }

    let scale = 1.0 / n as f32;
    total_loss *= scale;
    for i in 0..n {
        for d in 0..dim {
            grad_a[i][d] *= scale;
            grad_p[i][d] *= scale;
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_a,
        grad_positives: grad_p,
        grad_negatives: vec![],
    }
}

/// Matryoshka loss: compute an inner loss at multiple truncated dimensions, then weighted sum.
///
/// `dims` specifies the truncation dimensions (e.g., `[64, 128, 256, 768]`).
/// `weights` must have the same length as `dims`.
/// The inner loss function receives truncated embedding slices.
pub fn matryoshka_loss<F>(
    loss_fn: F,
    anchors: &[&[f32]],
    positives: &[&[f32]],
    dims: &[usize],
    weights: &[f32],
) -> LossOutput
where
    F: Fn(&[&[f32]], &[&[f32]]) -> LossOutput,
{
    assert_eq!(dims.len(), weights.len());
    assert!(!dims.is_empty());

    let batch = anchors.len();
    let full_dim = anchors[0].len();

    let mut total_loss = 0.0f32;
    let mut grad_a = vec![vec![0.0f32; full_dim]; batch];
    let mut grad_p = vec![vec![0.0f32; full_dim]; batch];

    for (dim_idx, &trunc_dim) in dims.iter().enumerate() {
        assert!(
            trunc_dim <= full_dim,
            "truncation dim {trunc_dim} exceeds embedding dim {full_dim}"
        );
        let w = weights[dim_idx];

        // Truncate embeddings
        let trunc_anchors: Vec<&[f32]> = anchors.iter().map(|a| &a[..trunc_dim]).collect();
        let trunc_positives: Vec<&[f32]> = positives.iter().map(|p| &p[..trunc_dim]).collect();

        let output = loss_fn(&trunc_anchors, &trunc_positives);
        total_loss += w * output.loss;

        // Accumulate gradients into the first trunc_dim dimensions
        for i in 0..batch {
            for d in 0..trunc_dim {
                grad_a[i][d] += w * output.grad_anchors[i][d];
                grad_p[i][d] += w * output.grad_positives[i][d];
            }
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_a,
        grad_positives: grad_p,
        grad_negatives: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triplet_loss_zero() {
        // anchor close to positive, far from negative -> loss should be 0
        let anchor: &[f32] = &[1.0, 0.0, 0.0];
        let positive: &[f32] = &[1.0, 0.1, 0.0]; // close
        let negative: &[f32] = &[-1.0, -1.0, -1.0]; // far

        let out = triplet_loss(&[anchor], &[positive], &[negative], 0.1);
        assert!(
            out.loss < 1e-6,
            "loss should be ~0 when positive is much closer, got {}",
            out.loss
        );
    }

    #[test]
    fn test_triplet_loss_positive() {
        // anchor closer to negative than positive -> loss > 0
        let anchor: &[f32] = &[1.0, 0.0, 0.0];
        let positive: &[f32] = &[-1.0, -1.0, -1.0]; // far
        let negative: &[f32] = &[1.0, 0.1, 0.0]; // close

        let out = triplet_loss(&[anchor], &[positive], &[negative], 0.5);
        assert!(
            out.loss > 0.0,
            "loss should be > 0 when negative is closer, got {}",
            out.loss
        );
    }

    #[test]
    fn test_triplet_gradient_direction() {
        let anchor: &[f32] = &[1.0, 0.0];
        let positive: &[f32] = &[0.0, 1.0];
        let negative: &[f32] = &[0.5, 0.0];

        let out = triplet_loss(&[anchor], &[positive], &[negative], 1.0);

        // With loss > 0, gradient should push anchor toward positive
        // and away from negative.
        // The gradient w.r.t. anchor for the positive direction should have
        // a component pointing toward positive (positive[d] - anchor[d] direction).
        // We check that grad_anchor dot (positive - anchor) < 0 (descent direction reduces d(a,p))
        let ga = &out.grad_anchors[0];
        let diff_ap: Vec<f32> = anchor
            .iter()
            .zip(positive.iter())
            .map(|(a, p)| p - a)
            .collect();
        let align_pos: f32 = ga.iter().zip(diff_ap.iter()).map(|(g, d)| g * d).sum();

        // Taking a step in -gradient direction should move anchor toward positive
        // So gradient dot (p - a) should be negative (gradient points away from positive)
        // Actually, the gradient of d(a,p) w.r.t. a points away from p.
        // The full triplet gradient = grad(d_ap) - grad(d_an), so it's more complex.
        // Let's just verify gradients are non-zero.
        let grad_norm: f32 = ga.iter().map(|g| g * g).sum();
        assert!(
            grad_norm > 1e-8,
            "anchor gradient should be non-zero when loss > 0"
        );

        let gp_norm: f32 = out.grad_positives[0].iter().map(|g| g * g).sum();
        assert!(gp_norm > 1e-8, "positive gradient should be non-zero");

        let gn_norm: f32 = out.grad_negatives[0].iter().map(|g| g * g).sum();
        assert!(gn_norm > 1e-8, "negative gradient should be non-zero");

        // Verify direction: stepping anchor by -grad should decrease loss
        let eps = 0.01;
        let new_anchor: Vec<f32> = anchor
            .iter()
            .zip(ga.iter())
            .map(|(a, g)| a - eps * g)
            .collect();
        let new_out = triplet_loss(&[&new_anchor], &[positive], &[negative], 1.0);
        assert!(
            new_out.loss <= out.loss + 1e-6,
            "stepping in -gradient direction should not increase loss: {} -> {}",
            out.loss,
            new_out.loss
        );
        let _ = align_pos;
    }

    #[test]
    fn test_infonce_perfect() {
        // Anchors identical to their positives, dissimilar to others
        let a0: &[f32] = &[1.0, 0.0, 0.0];
        let a1: &[f32] = &[0.0, 1.0, 0.0];
        let a2: &[f32] = &[0.0, 0.0, 1.0];
        let p0: &[f32] = &[1.0, 0.01, 0.0];
        let p1: &[f32] = &[0.01, 1.0, 0.0];
        let p2: &[f32] = &[0.0, 0.01, 1.0];

        let out = infonce_loss(&[a0, a1, a2], &[p0, p1, p2], 0.05);

        // With temperature=0.05 and nearly orthogonal classes, loss should be low
        assert!(
            out.loss < 0.5,
            "loss should be low for well-separated embeddings, got {}",
            out.loss
        );
    }

    #[test]
    fn test_infonce_gradient_exists() {
        let a0: &[f32] = &[1.0, 0.5];
        let a1: &[f32] = &[0.5, 1.0];
        let p0: &[f32] = &[0.8, 0.3];
        let p1: &[f32] = &[0.3, 0.8];

        let out = infonce_loss(&[a0, a1], &[p0, p1], 0.1);

        let grad_norm: f32 = out
            .grad_anchors
            .iter()
            .flat_map(|g| g.iter())
            .map(|g| g * g)
            .sum();
        assert!(grad_norm > 1e-8, "gradients should be non-zero");
    }

    #[test]
    fn test_mnrl_with_negatives() {
        let a: &[f32] = &[1.0, 0.0];
        let p: &[f32] = &[0.9, 0.1];

        // Without explicit negatives
        let out_no_neg = infonce_loss(&[a], &[p], 0.1);

        // With an explicit negative that's similar to anchor
        let neg: &[f32] = &[0.95, 0.05];
        let out_neg = mnrl_loss(&[a], &[p], &[neg], 0.1);

        // Adding a confusing negative should increase loss
        assert!(
            out_neg.loss > out_no_neg.loss - 0.01,
            "explicit similar negatives should not decrease loss: mnrl={}, infonce={}",
            out_neg.loss,
            out_no_neg.loss
        );
    }

    #[test]
    fn test_cosine_embedding_positive_pair() {
        let a: &[f32] = &[1.0, 0.0];
        let b: &[f32] = &[0.0, 1.0]; // orthogonal -> cos = 0

        let out = cosine_embedding_loss(&[(a, b)], &[true], 0.0);
        // loss = 1 - cos(a,b) = 1 - 0 = 1
        assert!(
            (out.loss - 1.0).abs() < 1e-5,
            "positive orthogonal pair loss should be 1.0, got {}",
            out.loss
        );
    }

    #[test]
    fn test_cosine_embedding_negative_pair() {
        let a: &[f32] = &[1.0, 0.0];
        let b: &[f32] = &[-1.0, 0.0]; // opposite -> cos = -1

        let out = cosine_embedding_loss(&[(a, b)], &[false], 0.0);
        // loss = max(0, cos(a,b) - margin) = max(0, -1 - 0) = 0
        assert!(
            out.loss.abs() < 1e-5,
            "negative opposite pair loss should be 0.0, got {}",
            out.loss
        );
    }

    #[test]
    fn test_matryoshka_dimensions() {
        let a: &[f32] = &[1.0, 0.5, 0.3, 0.1];
        let p: &[f32] = &[0.8, 0.6, 0.2, 0.15];

        let dims = [2, 4];
        let weights = [1.0, 1.0];

        let out = matryoshka_loss(
            |anchors, positives| infonce_loss(anchors, positives, 0.1),
            &[a],
            &[p],
            &dims,
            &weights,
        );

        // Loss should be the sum of losses at dim=2 and dim=4
        let loss_2 = infonce_loss(&[&a[..2]], &[&p[..2]], 0.1).loss;
        let loss_4 = infonce_loss(&[&a[..4]], &[&p[..4]], 0.1).loss;

        assert!(
            (out.loss - (loss_2 + loss_4)).abs() < 1e-4,
            "matryoshka loss should be sum of sub-losses: got {}, expected {}",
            out.loss,
            loss_2 + loss_4
        );
    }

    #[test]
    fn test_gradient_numerical() {
        // Verify analytical gradients match numerical gradients for triplet loss
        let anchor: Vec<f32> = vec![0.7, 0.3, -0.2];
        let positive: Vec<f32> = vec![0.5, 0.8, 0.1];
        let negative: Vec<f32> = vec![-0.3, 0.1, 0.9];

        let out = triplet_loss(&[&anchor], &[&positive], &[&negative], 1.0);

        let eps = 1e-4;
        let tol = 1e-2;

        // Check anchor gradients via central differences
        for d in 0..3 {
            let mut a_plus = anchor.clone();
            let mut a_minus = anchor.clone();
            a_plus[d] += eps;
            a_minus[d] -= eps;

            let loss_plus = triplet_loss(&[&a_plus], &[&positive], &[&negative], 1.0).loss;
            let loss_minus = triplet_loss(&[&a_minus], &[&positive], &[&negative], 1.0).loss;
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (out.grad_anchors[0][d] - numerical).abs() < tol,
                "anchor grad[{d}]: analytical={}, numerical={}",
                out.grad_anchors[0][d],
                numerical
            );
        }

        // Check positive gradients
        for d in 0..3 {
            let mut p_plus = positive.clone();
            let mut p_minus = positive.clone();
            p_plus[d] += eps;
            p_minus[d] -= eps;

            let loss_plus = triplet_loss(&[&anchor], &[&p_plus], &[&negative], 1.0).loss;
            let loss_minus = triplet_loss(&[&anchor], &[&p_minus], &[&negative], 1.0).loss;
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (out.grad_positives[0][d] - numerical).abs() < tol,
                "positive grad[{d}]: analytical={}, numerical={}",
                out.grad_positives[0][d],
                numerical
            );
        }

        // Check negative gradients
        for d in 0..3 {
            let mut n_plus = negative.clone();
            let mut n_minus = negative.clone();
            n_plus[d] += eps;
            n_minus[d] -= eps;

            let loss_plus = triplet_loss(&[&anchor], &[&positive], &[&n_plus], 1.0).loss;
            let loss_minus = triplet_loss(&[&anchor], &[&positive], &[&n_minus], 1.0).loss;
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (out.grad_negatives[0][d] - numerical).abs() < tol,
                "negative grad[{d}]: analytical={}, numerical={}",
                out.grad_negatives[0][d],
                numerical
            );
        }
    }
}
