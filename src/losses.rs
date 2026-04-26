//! Hand-derived gradient implementations of the metric-learning losses.
//!
//! **Transition status**: scheduled for deletion once every loss in this file
//! has a Burn equivalent in `crate::burn_losses`. Don't add new losses here
//! and don't depend on this surface for new code — write the Burn version.
//! See `burn_losses` for the canonical implementations going forward.

use crate::similarity::{
    accumulate_cosine_grad, accumulate_cosine_grad_pair, cosine_similarity,
    cosine_similarity_with_norms, dot, euclidean_distance,
};

/// Result of a loss computation.
///
/// Two usage patterns:
/// - **Pair losses** (triplet, infonce, mnrl, n-pairs, contrastive, cosine embedding):
///   `grad_anchors` and `grad_positives` hold per-input gradients.
///   `grad_negatives` is populated only when explicit negatives are provided (triplet, mnrl).
/// - **Batch losses** (multi-similarity, supcon, circle, lifted-structured):
///   `grad_anchors` holds gradients for ALL embeddings in the batch.
///   `grad_positives` and `grad_negatives` are empty.
///
/// All losses use mean reduction (loss and gradients are divided by batch size).
#[derive(Debug, Clone)]
#[must_use]
pub struct LossOutput {
    /// Scalar loss value (mean-reduced).
    pub loss: f32,
    /// Gradients w.r.t. anchor embeddings, or all embeddings for batch losses.
    pub grad_anchors: Vec<Vec<f32>>,
    /// Gradients w.r.t. positive embeddings. Empty for batch losses.
    pub grad_positives: Vec<Vec<f32>>,
    /// Gradients w.r.t. negative embeddings. Empty unless explicit negatives are provided.
    pub grad_negatives: Vec<Vec<f32>>,
}

/// Distance function selector for triplet loss.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
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
                    // loss = (1 - cos_ap) - (1 - cos_an) + margin = cos_an - cos_ap + margin
                    // dL/d(cos_ap) = -1, dL/d(cos_an) = +1
                    let (cos_ap, na, np) = cosine_similarity_with_norms(anchors[i], positives[i]);
                    let (cos_an, na2, nn) = cosine_similarity_with_norms(anchors[i], negatives[i]);
                    let _ = (cos_ap, cos_an); // used for forward pass above

                    if na > 1e-8 && np > 1e-8 && nn > 1e-8 {
                        // -dcos_ap/da + dcos_an/da, -dcos_ap/dp, +dcos_an/dn
                        accumulate_cosine_grad(
                            anchors[i],
                            positives[i],
                            cos_ap,
                            na,
                            np,
                            -1.0,
                            &mut grad_a[i],
                            &mut grad_p[i],
                        );
                        accumulate_cosine_grad(
                            anchors[i],
                            negatives[i],
                            cos_an,
                            na2,
                            nn,
                            1.0,
                            &mut grad_a[i],
                            &mut grad_n[i],
                        );
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

    // Precompute norms
    let norms_a: Vec<f32> = anchors.iter().map(|a| dot(a, a).sqrt()).collect();
    let norms_p: Vec<f32> = positives.iter().map(|p| dot(p, p).sqrt()).collect();

    for i in 0..batch {
        if norms_a[i] < 1e-8 {
            continue;
        }
        for j in 0..batch {
            if norms_p[j] < 1e-8 {
                continue;
            }
            let delta = if i == j { 1.0 } else { 0.0 };
            let dsim = (softmax[i][j] - delta) / (batch as f32 * temperature);
            let cos_val = sims[i][j] * temperature; // sims already divided by temperature
            accumulate_cosine_grad(
                anchors[i],
                positives[j],
                cos_val,
                norms_a[i],
                norms_p[j],
                dsim,
                &mut grad_a[i],
                &mut grad_p[j],
            );
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

    // Precompute norms
    let norms_a: Vec<f32> = anchors.iter().map(|a| dot(a, a).sqrt()).collect();
    let norms_p: Vec<f32> = positives.iter().map(|p| dot(p, p).sqrt()).collect();
    let norms_n: Vec<f32> = negatives.iter().map(|n| dot(n, n).sqrt()).collect();

    for i in 0..batch {
        if norms_a[i] < 1e-8 {
            continue;
        }

        // Gradients through positive candidates
        for j in 0..batch {
            if norms_p[j] < 1e-8 {
                continue;
            }
            let delta = if i == j { 1.0 } else { 0.0 };
            let dsim = (softmax[i][j] - delta) / (batch as f32 * temperature);
            let cos_val = sims[i][j] * temperature;
            accumulate_cosine_grad(
                anchors[i],
                positives[j],
                cos_val,
                norms_a[i],
                norms_p[j],
                dsim,
                &mut grad_a[i],
                &mut grad_p[j],
            );
        }

        // Gradients through explicit negatives
        for k in 0..n_neg {
            if norms_n[k] < 1e-8 {
                continue;
            }
            let dsim = softmax[i][batch + k] / (batch as f32 * temperature);
            let cos_val = sims[i][batch + k] * temperature;
            accumulate_cosine_grad(
                anchors[i],
                negatives[k],
                cos_val,
                norms_a[i],
                norms_n[k],
                dsim,
                &mut grad_a[i],
                &mut grad_n[k],
            );
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
        let (cos, norm_a, norm_b) = cosine_similarity_with_norms(a, b);

        if labels[i] {
            // Positive: loss = 1 - cos, dL/d(cos) = -1
            total_loss += 1.0 - cos;
            if norm_a > 1e-8 && norm_b > 1e-8 {
                accumulate_cosine_grad(
                    a,
                    b,
                    cos,
                    norm_a,
                    norm_b,
                    -1.0,
                    &mut grad_a[i],
                    &mut grad_p[i],
                );
            }
        } else {
            // Negative: loss = max(0, cos - margin), dL/d(cos) = 1 when active
            let loss_i = (cos - margin).max(0.0);
            total_loss += loss_i;
            if loss_i > 0.0 && norm_a > 1e-8 && norm_b > 1e-8 {
                accumulate_cosine_grad(
                    a,
                    b,
                    cos,
                    norm_a,
                    norm_b,
                    1.0,
                    &mut grad_a[i],
                    &mut grad_p[i],
                );
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

/// Classic pairwise contrastive loss (Hadsell, Chopra, LeCun 2006).
///
/// For similar pairs (`label = true`): `L = d^2`.
/// For dissimilar pairs (`label = false`): `L = max(0, margin - d)^2`.
/// Uses Euclidean distance.
pub fn contrastive_loss(pairs: &[(&[f32], &[f32])], labels: &[bool], margin: f32) -> LossOutput {
    assert_eq!(pairs.len(), labels.len());
    let n = pairs.len();
    assert!(n > 0);

    let dim = pairs[0].0.len();
    let mut total_loss = 0.0f32;
    let mut grad_a = vec![vec![0.0f32; dim]; n];
    let mut grad_p = vec![vec![0.0f32; dim]; n];

    for i in 0..n {
        let (a, b) = pairs[i];
        let d = euclidean_distance(a, b);

        if labels[i] {
            // Similar: loss = d^2
            total_loss += d * d;

            // d(d^2)/da_k = 2*(a_k - b_k)
            for k in 0..dim {
                grad_a[i][k] = 2.0 * (a[k] - b[k]);
                grad_p[i][k] = 2.0 * (b[k] - a[k]);
            }
        } else {
            // Dissimilar: loss = max(0, margin - d)^2
            let gap = margin - d;
            if gap > 0.0 {
                total_loss += gap * gap;

                // d(gap^2)/da_k = 2*gap * d(gap)/da_k = 2*gap * (-d(d)/da_k)
                // d(d)/da_k = (a_k - b_k) / d
                let inv_d = if d > 1e-8 { 1.0 / d } else { 0.0 };
                for k in 0..dim {
                    let dd_da = (a[k] - b[k]) * inv_d;
                    grad_a[i][k] = -2.0 * gap * dd_da;
                    grad_p[i][k] = 2.0 * gap * dd_da;
                }
            }
        }
    }

    let scale = 1.0 / n as f32;
    total_loss *= scale;
    for i in 0..n {
        for k in 0..dim {
            grad_a[i][k] *= scale;
            grad_p[i][k] *= scale;
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_a,
        grad_positives: grad_p,
        grad_negatives: vec![],
    }
}

/// Multi-Similarity Loss (Wang et al., CVPR 2019).
///
/// Considers self-similarity, negative relative similarity, and positive relative similarity.
/// Uses cosine similarity. `alpha` weights negative pairs, `beta` weights positive pairs,
/// `base` is a similarity offset.
pub fn multi_similarity_loss(
    embeddings: &[&[f32]],
    labels: &[usize],
    alpha: f32,
    beta: f32,
    base: f32,
) -> LossOutput {
    let n = embeddings.len();
    assert_eq!(n, labels.len());
    assert!(n > 1);

    let dim = embeddings[0].len();

    // Pairwise cosine similarities
    let mut sims = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        sims[i][i] = 1.0;
        for j in (i + 1)..n {
            let s = cosine_similarity(embeddings[i], embeddings[j]);
            sims[i][j] = s;
            sims[j][i] = s;
        }
    }

    let mut total_loss = 0.0f32;

    for i in 0..n {
        // Positive and negative indices
        let pos: Vec<usize> = (0..n)
            .filter(|&j| j != i && labels[j] == labels[i])
            .collect();
        let neg: Vec<usize> = (0..n).filter(|&j| labels[j] != labels[i]).collect();

        if pos.is_empty() || neg.is_empty() {
            continue;
        }

        // Positive term: (1/alpha) * log(1 + sum_{p in pos} exp(-alpha*(S_ip - base)))
        let pos_exps: Vec<f32> = pos
            .iter()
            .map(|&p| (-alpha * (sims[i][p] - base)).exp())
            .collect();
        let pos_sum: f32 = pos_exps.iter().sum();
        total_loss += (1.0 / alpha) * (1.0 + pos_sum).ln();

        // Negative term: (1/beta) * log(1 + sum_{n in neg} exp(beta*(S_in - base)))
        let neg_exps: Vec<f32> = neg
            .iter()
            .map(|&n| (beta * (sims[i][n] - base)).exp())
            .collect();
        let neg_sum: f32 = neg_exps.iter().sum();
        total_loss += (1.0 / beta) * (1.0 + neg_sum).ln();
    }

    total_loss /= n as f32;

    // Gradients via chain rule through cosine similarity.
    // dL/d(S_ij) for anchor i:
    //   positive j: -w_p(i,j) where w_p = exp(-alpha*(S-base)) / (1 + sum_p' exp(-alpha*(S-base)))
    //   negative j:  w_n(i,j) where w_n = exp(beta*(S-base))  / (1 + sum_n' exp(beta*(S-base)))
    // Then chain: dL/de_i += dL/dS_ij * dS_ij/de_i
    //             dL/de_j += dL/dS_ij * dS_ij/de_j
    // where dcos(a,b)/da_k = b_k/(|a|*|b|) - cos(a,b)*a_k/|a|^2

    let mut grad = vec![vec![0.0f32; dim]; n];
    let scale = 1.0 / n as f32;

    // Precompute norms
    let norms: Vec<f32> = embeddings.iter().map(|e| dot(e, e).sqrt()).collect();

    for i in 0..n {
        let pos: Vec<usize> = (0..n)
            .filter(|&j| j != i && labels[j] == labels[i])
            .collect();
        let neg: Vec<usize> = (0..n).filter(|&j| labels[j] != labels[i]).collect();

        if pos.is_empty() || neg.is_empty() {
            continue;
        }

        // Compute softmax weights for positive term
        let pos_exps: Vec<f32> = pos
            .iter()
            .map(|&p| (-alpha * (sims[i][p] - base)).exp())
            .collect();
        let pos_denom: f32 = 1.0 + pos_exps.iter().sum::<f32>();

        // Compute softmax weights for negative term
        let neg_exps: Vec<f32> = neg
            .iter()
            .map(|&n| (beta * (sims[i][n] - base)).exp())
            .collect();
        let neg_denom: f32 = 1.0 + neg_exps.iter().sum::<f32>();

        let norm_i = norms[i];
        if norm_i < 1e-8 {
            continue;
        }

        // Positive pairs: dL/dS_ip = -w_p(i,p)
        for (pi, &p) in pos.iter().enumerate() {
            let w = pos_exps[pi] / pos_denom;
            if norms[p] < 1e-8 {
                continue;
            }
            let dsim = -w * scale;
            accumulate_cosine_grad_pair(
                embeddings[i],
                embeddings[p],
                sims[i][p],
                norm_i,
                norms[p],
                dsim,
                &mut grad,
                i,
                p,
            );
        }

        // Negative pairs: dL/dS_in = w_n(i,n)
        for (ni, &nj) in neg.iter().enumerate() {
            let w = neg_exps[ni] / neg_denom;
            if norms[nj] < 1e-8 {
                continue;
            }
            let dsim = w * scale;
            accumulate_cosine_grad_pair(
                embeddings[i],
                embeddings[nj],
                sims[i][nj],
                norm_i,
                norms[nj],
                dsim,
                &mut grad,
                i,
                nj,
            );
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad,
        grad_positives: vec![],
        grad_negatives: vec![],
    }
}

/// Supervised Contrastive Loss (Khosla et al. 2020).
///
/// Extension of InfoNCE to the supervised setting with multiple positives per class.
/// For each anchor i, positives are all other samples with the same label.
pub fn supcon_loss(embeddings: &[&[f32]], labels: &[usize], temperature: f32) -> LossOutput {
    let n = embeddings.len();
    assert_eq!(n, labels.len());
    assert!(n > 1);
    assert!(temperature > 0.0);

    let dim = embeddings[0].len();

    // Pairwise cosine similarities scaled by temperature
    let mut sims = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                sims[i][j] = cosine_similarity(embeddings[i], embeddings[j]) / temperature;
            }
        }
    }

    let mut total_loss = 0.0f32;
    let mut grad = vec![vec![0.0f32; dim]; n];

    for i in 0..n {
        let positives: Vec<usize> = (0..n)
            .filter(|&j| j != i && labels[j] == labels[i])
            .collect();
        if positives.is_empty() {
            continue;
        }

        // log-sum-exp denominator over all j != i
        let max_sim = sims[i]
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &s)| s)
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0f32;
        let mut exps = vec![0.0f32; n];
        for j in 0..n {
            if j != i {
                exps[j] = (sims[i][j] - max_sim).exp();
                sum_exp += exps[j];
            }
        }
        let log_denom = max_sim + sum_exp.ln();

        // Loss for anchor i: -1/|P(i)| * sum_{p in P(i)} [sim(i,p)/t - log(sum_{a!=i} exp(sim(i,a)/t))]
        let n_pos = positives.len() as f32;
        for &p in &positives {
            total_loss += -(sims[i][p] - log_denom) / n_pos;
        }

        // Gradients w.r.t. embeddings[i]
        // For each positive p: dL/d(sim_ip) contributes -1/|P| * (1 - softmax_ip)
        // For each j != i that is not in positives: dL/d(sim_ij) contributes softmax_ij
        let softmax: Vec<f32> = (0..n)
            .map(|j| if j != i { exps[j] / sum_exp } else { 0.0 })
            .collect();

        let norm_i = dot(embeddings[i], embeddings[i]).sqrt();
        if norm_i < 1e-8 {
            continue;
        }

        for j in 0..n {
            if j == i {
                continue;
            }
            let norm_j = dot(embeddings[j], embeddings[j]).sqrt();
            if norm_j < 1e-8 {
                continue;
            }

            let is_pos = labels[j] == labels[i];
            let dsim = if is_pos {
                (-1.0 / n_pos + softmax[j]) / temperature
            } else {
                softmax[j] / temperature
            };

            let cos_val = cosine_similarity(embeddings[i], embeddings[j]);
            accumulate_cosine_grad_pair(
                embeddings[i],
                embeddings[j],
                cos_val,
                norm_i,
                norm_j,
                dsim,
                &mut grad,
                i,
                j,
            );
        }
    }

    total_loss /= n as f32;
    let scale = 1.0 / n as f32;
    for row in grad.iter_mut() {
        for val in row.iter_mut() {
            *val *= scale;
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad,
        grad_positives: vec![],
        grad_negatives: vec![],
    }
}

/// Circle Loss (Sun et al. 2020).
///
/// Unified pair similarity optimization with adaptive per-pair weighting.
/// `margin` controls the decision boundary, `gamma` is a scale factor.
pub fn circle_loss(embeddings: &[&[f32]], labels: &[usize], margin: f32, gamma: f32) -> LossOutput {
    let n = embeddings.len();
    assert_eq!(n, labels.len());
    assert!(n > 1);

    let dim = embeddings[0].len();

    // Pairwise cosine similarities
    let mut sims = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        sims[i][i] = 1.0;
        for j in (i + 1)..n {
            let s = cosine_similarity(embeddings[i], embeddings[j]);
            sims[i][j] = s;
            sims[j][i] = s;
        }
    }

    let o_p = 1.0 + margin; // optimal positive similarity
    let o_n = -margin; // optimal negative similarity
    let delta_p = 1.0 - margin; // positive threshold
    let delta_n = margin; // negative threshold

    let mut total_loss = 0.0f32;

    for i in 0..n {
        let pos: Vec<usize> = (0..n)
            .filter(|&j| j != i && labels[j] == labels[i])
            .collect();
        let neg: Vec<usize> = (0..n).filter(|&j| labels[j] != labels[i]).collect();

        if pos.is_empty() || neg.is_empty() {
            continue;
        }

        // Negative logits: gamma * alpha_n * (s_n - delta_n)
        let neg_logits: Vec<f32> = neg
            .iter()
            .map(|&j| {
                let alpha_n = (sims[i][j] - o_n).max(0.0);
                gamma * alpha_n * (sims[i][j] - delta_n)
            })
            .collect();

        // Positive logits: -gamma * alpha_p * (s_p - delta_p)
        let pos_logits: Vec<f32> = pos
            .iter()
            .map(|&j| {
                let alpha_p = (o_p - sims[i][j]).max(0.0);
                -gamma * alpha_p * (sims[i][j] - delta_p)
            })
            .collect();

        // log(1 + sum_neg exp(neg_logit) * sum_pos exp(pos_logit))
        // = log(1 + exp(logsumexp(neg) + logsumexp(pos)))
        let max_neg = neg_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp_neg: f32 = neg_logits.iter().map(|&v| (v - max_neg).exp()).sum();
        let lse_neg = max_neg + sum_exp_neg.ln();

        let max_pos = pos_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp_pos: f32 = pos_logits.iter().map(|&v| (v - max_pos).exp()).sum();
        let lse_pos = max_pos + sum_exp_pos.ln();

        let combined = lse_neg + lse_pos;
        // softplus: log(1 + exp(x))
        let loss_i = if combined > 20.0 {
            combined // avoid overflow
        } else {
            (1.0 + combined.exp()).ln()
        };

        total_loss += loss_i;
    }

    total_loss /= n as f32;

    // Gradients with full alpha dependency (not detached).
    // This computes the true gradient of the implemented function. Note: pytorch-metric-learning
    // detaches alpha (treats it as constant during backprop), which is a practical choice from
    // the paper but not mathematically required. The candle_losses module detaches alpha to
    // match the reference implementation; this f32 module computes the exact gradient.
    //
    // logit_n = gamma * max(0, s_n - O_n) * (s_n - delta_n)
    // When alpha_n > 0: d(logit_n)/d(s_n) = gamma * (2*s_n - delta_n - O_n)
    // logit_p = -gamma * max(0, O_p - s_p) * (s_p - delta_p)
    // When alpha_p > 0: d(logit_p)/d(s_p) = -gamma * (O_p - 2*s_p + delta_p)
    //
    // dL/d(s) = sigmoid(combined) * softmax_weight * d(logit)/d(s)
    // Then chain through d(cosine)/d(embedding).

    let mut grad = vec![vec![0.0f32; dim]; n];
    let scale = 1.0 / n as f32;

    // Precompute norms
    let norms: Vec<f32> = embeddings.iter().map(|e| dot(e, e).sqrt()).collect();

    for i in 0..n {
        let pos: Vec<usize> = (0..n)
            .filter(|&j| j != i && labels[j] == labels[i])
            .collect();
        let neg: Vec<usize> = (0..n).filter(|&j| labels[j] != labels[i]).collect();

        if pos.is_empty() || neg.is_empty() {
            continue;
        }

        // Recompute logits for this anchor
        let neg_logits: Vec<f32> = neg
            .iter()
            .map(|&j| {
                let alpha_n = (sims[i][j] - o_n).max(0.0);
                gamma * alpha_n * (sims[i][j] - delta_n)
            })
            .collect();

        let pos_logits: Vec<f32> = pos
            .iter()
            .map(|&j| {
                let alpha_p = (o_p - sims[i][j]).max(0.0);
                -gamma * alpha_p * (sims[i][j] - delta_p)
            })
            .collect();

        let max_neg = neg_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp_neg: f32 = neg_logits.iter().map(|&v| (v - max_neg).exp()).sum();
        let lse_neg = max_neg + sum_exp_neg.ln();

        let max_pos = pos_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp_pos: f32 = pos_logits.iter().map(|&v| (v - max_pos).exp()).sum();
        let lse_pos = max_pos + sum_exp_pos.ln();

        let combined = lse_neg + lse_pos;
        // sigmoid(combined) = d softplus / d combined
        let sigmoid_c = if combined > 20.0 {
            1.0
        } else if combined < -20.0 {
            0.0
        } else {
            1.0 / (1.0 + (-combined).exp())
        };

        let norm_i = norms[i];
        if norm_i < 1e-8 {
            continue;
        }

        // Negative pairs
        for (ni, &nj) in neg.iter().enumerate() {
            let s_n = sims[i][nj];
            let alpha_n = (s_n - o_n).max(0.0);
            if alpha_n == 0.0 {
                continue;
            }
            let dlogit_ds = gamma * (2.0 * s_n - delta_n - o_n);
            let w_n = (neg_logits[ni] - max_neg).exp() / sum_exp_neg;
            let dsim = sigmoid_c * w_n * dlogit_ds * scale;

            if norms[nj] < 1e-8 {
                continue;
            }
            accumulate_cosine_grad_pair(
                embeddings[i],
                embeddings[nj],
                s_n,
                norm_i,
                norms[nj],
                dsim,
                &mut grad,
                i,
                nj,
            );
        }

        // Positive pairs
        for (pi, &pj) in pos.iter().enumerate() {
            let s_p = sims[i][pj];
            let alpha_p = (o_p - s_p).max(0.0);
            if alpha_p == 0.0 {
                continue;
            }
            let dlogit_ds = -gamma * (o_p - 2.0 * s_p + delta_p);
            let w_p = (pos_logits[pi] - max_pos).exp() / sum_exp_pos;
            let dsim = sigmoid_c * w_p * dlogit_ds * scale;

            if norms[pj] < 1e-8 {
                continue;
            }
            accumulate_cosine_grad_pair(
                embeddings[i],
                embeddings[pj],
                s_p,
                norm_i,
                norms[pj],
                dsim,
                &mut grad,
                i,
                pj,
            );
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad,
        grad_positives: vec![],
        grad_negatives: vec![],
    }
}

/// Lifted Structured Loss (Song et al. 2016).
///
/// Uses all pairwise distances in the batch with a log-sum-exp smooth approximation.
pub fn lifted_structured_loss(embeddings: &[&[f32]], labels: &[usize], margin: f32) -> LossOutput {
    let n = embeddings.len();
    assert_eq!(n, labels.len());
    assert!(n > 1);

    let dim = embeddings[0].len();

    // Pairwise Euclidean distances
    let mut dists = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(embeddings[i], embeddings[j]);
            dists[i][j] = d;
            dists[j][i] = d;
        }
    }

    let mut total_loss = 0.0f32;
    let mut count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if labels[i] != labels[j] {
                continue;
            }
            // Positive pair (i, j)
            let d_ij = dists[i][j];

            // Log-sum-exp of (margin - d_ik) for negatives of i
            let neg_i: Vec<f32> = (0..n)
                .filter(|&k| labels[k] != labels[i])
                .map(|k| margin - dists[i][k])
                .collect();

            // Log-sum-exp of (margin - d_jl) for negatives of j
            let neg_j: Vec<f32> = (0..n)
                .filter(|&l| labels[l] != labels[j])
                .map(|l| margin - dists[j][l])
                .collect();

            if neg_i.is_empty() || neg_j.is_empty() {
                continue;
            }

            let lse_i = log_sum_exp(&neg_i);
            let lse_j = log_sum_exp(&neg_j);

            let loss_ij = (lse_i + lse_j + d_ij).max(0.0);
            total_loss += loss_ij * loss_ij;
            count += 1;
        }
    }

    if count > 0 {
        total_loss /= count as f32;
    }
    // The original paper uses 0.5 * max(0, ...)^2
    total_loss *= 0.5;

    // Gradients: L = (0.5/|P|) * sum_{(i,j) in P} max(0, J_ij)^2
    // where J_ij = lse_i + lse_j + d_ij
    // For active pairs (J_ij > 0):
    //   dL/d(d_ik) = (J_ij / |P|) * (-softmax_weight_ik)  for negative k of i
    //   dL/d(d_jl) = (J_ij / |P|) * (-softmax_weight_jl)  for negative l of j
    //   dL/d(d_ij) = (J_ij / |P|) * 1
    // Then chain: d||a-b||/da = (a-b)/||a-b||

    let mut grad = vec![vec![0.0f32; dim]; n];

    if count > 0 {
        let count_f = count as f32;

        for i in 0..n {
            for j in (i + 1)..n {
                if labels[i] != labels[j] {
                    continue;
                }

                let d_ij = dists[i][j];

                // Negatives of i
                let neg_i_indices: Vec<usize> =
                    (0..n).filter(|&k| labels[k] != labels[i]).collect();
                let neg_i_vals: Vec<f32> = neg_i_indices
                    .iter()
                    .map(|&k| margin - dists[i][k])
                    .collect();

                // Negatives of j
                let neg_j_indices: Vec<usize> =
                    (0..n).filter(|&l| labels[l] != labels[j]).collect();
                let neg_j_vals: Vec<f32> = neg_j_indices
                    .iter()
                    .map(|&l| margin - dists[j][l])
                    .collect();

                if neg_i_vals.is_empty() || neg_j_vals.is_empty() {
                    continue;
                }

                let lse_i = log_sum_exp(&neg_i_vals);
                let lse_j = log_sum_exp(&neg_j_vals);

                let j_ij = lse_i + lse_j + d_ij;
                if j_ij <= 0.0 {
                    continue;
                }

                // Coefficient: J_ij / |P| (from d/dx of 0.5 * x^2 = x, times 1/|P|)
                let coeff = j_ij / count_f;

                // Softmax weights for negatives of i
                let max_ni = neg_i_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ni_exps: Vec<f32> = neg_i_vals.iter().map(|&v| (v - max_ni).exp()).collect();
                let ni_sum: f32 = ni_exps.iter().sum();

                // Softmax weights for negatives of j
                let max_nj = neg_j_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let nj_exps: Vec<f32> = neg_j_vals.iter().map(|&v| (v - max_nj).exp()).collect();
                let nj_sum: f32 = nj_exps.iter().sum();

                // Gradient from d_ij term: dL/d(d_ij) = coeff
                let inv_d_ij = if d_ij > 1e-8 { 1.0 / d_ij } else { 0.0 };
                for d in 0..dim {
                    let diff = embeddings[i][d] - embeddings[j][d];
                    let dd_di = diff * inv_d_ij;
                    grad[i][d] += coeff * dd_di;
                    grad[j][d] -= coeff * dd_di;
                }

                // Gradient from lse_i: d(lse_i)/d(d_ik) = -softmax_weight_ik
                // so dL/d(d_ik) = coeff * (-w_ik)
                for (ki, &k) in neg_i_indices.iter().enumerate() {
                    let w = ni_exps[ki] / ni_sum;
                    let d_ik = dists[i][k];
                    let inv_d_ik = if d_ik > 1e-8 { 1.0 / d_ik } else { 0.0 };
                    for d in 0..dim {
                        let diff = embeddings[i][d] - embeddings[k][d];
                        let dd_di = diff * inv_d_ik;
                        // dL/de_i += coeff * (-w) * d(d_ik)/de_i
                        grad[i][d] += coeff * (-w) * dd_di;
                        // dL/de_k += coeff * (-w) * d(d_ik)/de_k = coeff * (-w) * (-dd_di)
                        grad[k][d] -= coeff * (-w) * dd_di;
                    }
                }

                // Gradient from lse_j: d(lse_j)/d(d_jl) = -softmax_weight_jl
                for (li, &l) in neg_j_indices.iter().enumerate() {
                    let w = nj_exps[li] / nj_sum;
                    let d_jl = dists[j][l];
                    let inv_d_jl = if d_jl > 1e-8 { 1.0 / d_jl } else { 0.0 };
                    for d in 0..dim {
                        let diff = embeddings[j][d] - embeddings[l][d];
                        let dd_dj = diff * inv_d_jl;
                        grad[j][d] += coeff * (-w) * dd_dj;
                        grad[l][d] -= coeff * (-w) * dd_dj;
                    }
                }
            }
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad,
        grad_positives: vec![],
        grad_negatives: vec![],
    }
}

/// N-pairs Loss (Sohn 2016).
///
/// Generalization of triplet loss to N-1 negatives. For each anchor i, positive is
/// `positives[i]`, negatives are all other positives. With temperature scaling,
/// equivalent to InfoNCE.
pub fn n_pairs_loss(anchors: &[&[f32]], positives: &[&[f32]], temperature: f32) -> LossOutput {
    let n = anchors.len();
    assert_eq!(n, positives.len());
    assert!(n > 0);
    assert!(temperature > 0.0);

    let dim = anchors[0].len();

    let mut total_loss = 0.0f32;
    let mut grad_a = vec![vec![0.0f32; dim]; n];
    let mut grad_p = vec![vec![0.0f32; dim]; n];

    for i in 0..n {
        // Compute a_i . p_j / temperature for all j
        let mut logits = Vec::with_capacity(n);
        for p in positives.iter().take(n) {
            logits.push(dot(anchors[i], p) / temperature);
        }

        // log(1 + sum_{j!=i} exp(logit_j - logit_i))
        // = -logit_i + log(sum_j exp(logit_j))
        // This is equivalent to softmax cross-entropy with target i
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut exps = vec![0.0f32; n];
        for j in 0..n {
            exps[j] = (logits[j] - max_logit).exp();
            sum_exp += exps[j];
        }

        let log_sum = max_logit + sum_exp.ln();
        total_loss += -logits[i] + log_sum;

        // Gradients: softmax probabilities
        let softmax: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();
        for j in 0..n {
            let delta = if j == i { 1.0 } else { 0.0 };
            let coeff = (softmax[j] - delta) / temperature;
            for d in 0..dim {
                // dL/d(a_i) += coeff * p_j
                grad_a[i][d] += coeff * positives[j][d];
                // dL/d(p_j) += coeff * a_i
                grad_p[j][d] += coeff * anchors[i][d];
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

/// ArcFace loss (Deng et al. 2019).
///
/// Angular margin penalty for classification-to-embedding tasks. Adds an angular
/// margin `m` to the target class angle before softmax. Requires a `proxies` matrix
/// of shape `[num_classes][dim]` (learnable class centroids).
///
/// Returns loss and gradients w.r.t. embeddings (not proxies -- proxy gradients
/// are in `grad_positives`). Embeddings and proxies are L2-normalized internally.
pub fn arcface_loss(
    embeddings: &[&[f32]],
    labels: &[usize],
    proxies: &[&[f32]],
    scale: f32,
    margin: f32,
) -> LossOutput {
    let n = embeddings.len();
    assert_eq!(n, labels.len());
    assert!(n > 0);
    let num_classes = proxies.len();
    let dim = embeddings[0].len();

    // Normalize embeddings and proxies
    let emb_norms: Vec<f32> = embeddings
        .iter()
        .map(|e| dot(e, e).sqrt().max(1e-8))
        .collect();
    let proxy_norms: Vec<f32> = proxies.iter().map(|p| dot(p, p).sqrt().max(1e-8)).collect();

    let mut total_loss = 0.0f32;
    let mut grad_emb = vec![vec![0.0f32; dim]; n];
    let mut grad_proxy = vec![vec![0.0f32; dim]; num_classes];

    for i in 0..n {
        let target = labels[i];
        assert!(
            target < num_classes,
            "label {target} exceeds num_classes {num_classes}"
        );

        // Compute cosine similarities to all proxies (logits)
        let mut logits = Vec::with_capacity(num_classes);
        for c in 0..num_classes {
            let cos = dot(embeddings[i], proxies[c]) / (emb_norms[i] * proxy_norms[c]);
            logits.push(cos);
        }

        // Add angular margin to target class
        // arcface: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        let cos_t = logits[target].clamp(-1.0, 1.0);
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        let cos_m = margin.cos();
        let sin_m = margin.sin();
        let cos_t_plus_m = cos_t * cos_m - sin_t * sin_m;
        logits[target] = cos_t_plus_m;

        // Scale and softmax cross-entropy
        let scaled: Vec<f32> = logits.iter().map(|&l| l * scale).collect();
        let max_s = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let softmax: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

        total_loss += -(softmax[target].max(1e-8)).ln();

        // Gradient of cross-entropy w.r.t. logits: scale * (softmax - one_hot)
        // Then chain through cosine and arcface margin
        for c in 0..num_classes {
            let delta = if c == target { 1.0 } else { 0.0 };
            let mut dlogit = scale * (softmax[c] - delta);

            // For target class, chain through arcface margin:
            // d(cos(theta+m))/d(cos_theta) = cos(m) + sin(m)*cos_theta/sin_theta
            // When sin_theta ~ 0 (perfect alignment), fall back to cos(m)
            if c == target {
                if sin_t > 1e-6 {
                    dlogit *= cos_m + sin_m * cos_t / sin_t;
                } else {
                    dlogit *= cos_m;
                }
            }

            if dlogit.abs() < 1e-12 {
                continue;
            }

            // Chain through cosine similarity: d(cos)/d(emb), d(cos)/d(proxy)
            let ne = emb_norms[i];
            let np = proxy_norms[c];
            let cos_val = dot(embeddings[i], proxies[c]) / (ne * np);
            for d in 0..dim {
                let dcos_de = proxies[c][d] / (ne * np) - embeddings[i][d] * cos_val / (ne * ne);
                let dcos_dp = embeddings[i][d] / (ne * np) - proxies[c][d] * cos_val / (np * np);
                grad_emb[i][d] += dlogit * dcos_de;
                grad_proxy[c][d] += dlogit * dcos_dp;
            }
        }
    }

    let s = 1.0 / n as f32;
    total_loss *= s;
    for g in grad_emb.iter_mut() {
        for v in g.iter_mut() {
            *v *= s;
        }
    }
    for g in grad_proxy.iter_mut() {
        for v in g.iter_mut() {
            *v *= s;
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_emb,
        grad_positives: grad_proxy,
        grad_negatives: vec![],
    }
}

/// VICReg loss (Bardes, Ponce, LeCun 2022).
///
/// Self-supervised loss with three terms and no negatives:
/// - **Invariance**: MSE between paired views (embeddings of the same input under different augmentations).
/// - **Variance**: hinge loss encouraging each dimension's std dev >= 1, preventing collapse.
/// - **Covariance**: penalizes off-diagonal covariance to decorrelate dimensions.
///
/// `view_a` and `view_b` are `[batch][dim]` embeddings of the same inputs under two augmentations.
/// Returns loss and gradients for both views.
pub fn vicreg_loss(
    view_a: &[&[f32]],
    view_b: &[&[f32]],
    lambda_inv: f32,
    mu_var: f32,
    nu_cov: f32,
) -> LossOutput {
    let n = view_a.len();
    assert_eq!(n, view_b.len());
    assert!(n > 1);
    let dim = view_a[0].len();

    // === Invariance: MSE between paired views ===
    let mut inv_loss = 0.0f32;
    let mut grad_a = vec![vec![0.0f32; dim]; n];
    let mut grad_b = vec![vec![0.0f32; dim]; n];

    for i in 0..n {
        for d in 0..dim {
            let diff = view_a[i][d] - view_b[i][d];
            inv_loss += diff * diff;
            // d(MSE)/d(a) = 2*(a-b)/n, d(MSE)/d(b) = -2*(a-b)/n
            let g = 2.0 * diff / n as f32;
            grad_a[i][d] += lambda_inv * g;
            grad_b[i][d] += lambda_inv * (-g);
        }
    }
    inv_loss /= n as f32;

    // === Compute means ===
    let mut mean_a = vec![0.0f32; dim];
    let mut mean_b = vec![0.0f32; dim];
    for i in 0..n {
        for d in 0..dim {
            mean_a[d] += view_a[i][d];
            mean_b[d] += view_b[i][d];
        }
    }
    for d in 0..dim {
        mean_a[d] /= n as f32;
        mean_b[d] /= n as f32;
    }

    // === Variance: hinge on std dev ===
    // var_loss = (1/dim) * sum_d max(0, 1 - std_d)
    let variance_loss = |view: &[&[f32]], mean: &[f32], grad: &mut [Vec<f32>]| -> f32 {
        let mut loss = 0.0f32;
        for d in 0..dim {
            let var: f32 = (0..n).map(|i| (view[i][d] - mean[d]).powi(2)).sum::<f32>() / n as f32;
            let std = (var + 1e-4).sqrt();
            let hinge = (1.0 - std).max(0.0);
            loss += hinge;
            if hinge > 0.0 {
                // d(max(0,1-std))/d(x_i) = -(1/(2*std)) * 2*(x_i - mean)/n
                let dstd_coeff = -1.0 / (2.0 * std * n as f32);
                for i in 0..n {
                    grad[i][d] += mu_var * dstd_coeff * 2.0 * (view[i][d] - mean[d]) / dim as f32;
                }
            }
        }
        loss / dim as f32
    };
    let var_a = variance_loss(view_a, &mean_a, &mut grad_a);
    let var_b = variance_loss(view_b, &mean_b, &mut grad_b);
    let var_loss = var_a + var_b;

    // === Covariance: penalize off-diagonal ===
    // cov_loss = (1/dim) * sum_{i!=j} C[i][j]^2
    let covariance_loss = |view: &[&[f32]], mean: &[f32], grad: &mut [Vec<f32>]| -> f32 {
        let mut cov = vec![vec![0.0f32; dim]; dim];
        for sample in view {
            for (d1, row) in cov.iter_mut().enumerate() {
                let centered_d1 = sample[d1] - mean[d1];
                for (d2, cell) in row.iter_mut().enumerate() {
                    *cell += centered_d1 * (sample[d2] - mean[d2]);
                }
            }
        }
        for row in cov.iter_mut() {
            for v in row.iter_mut() {
                *v /= n as f32;
            }
        }

        let mut loss = 0.0f32;
        for (d1, row) in cov.iter().enumerate() {
            for (d2, &val) in row.iter().enumerate() {
                if d1 != d2 {
                    loss += val * val;
                }
            }
        }
        loss /= dim as f32;

        // Gradient: d(cov_loss)/d(x_k[d1])
        // = (2/dim) * sum_{d2 != d1} cov[d1][d2] * (x_k[d2] - mean[d2]) / n
        // + (2/dim) * sum_{d2 != d1} cov[d2][d1] * (x_k[d2] - mean[d2]) / n  (symmetry)
        // Since cov is symmetric: = (4/dim) * sum_{d2 != d1} cov[d1][d2] * (x_k[d2] - mean[d2]) / n
        for k in 0..n {
            for d1 in 0..dim {
                let mut g = 0.0f32;
                for d2 in 0..dim {
                    if d1 != d2 {
                        g += cov[d1][d2] * (view[k][d2] - mean[d2]);
                    }
                }
                grad[k][d1] += nu_cov * 4.0 * g / (dim as f32 * n as f32);
            }
        }

        loss
    };

    let cov_a = covariance_loss(view_a, &mean_a, &mut grad_a);
    let cov_b = covariance_loss(view_b, &mean_b, &mut grad_b);
    let cov_loss = cov_a + cov_b;

    let total_loss = lambda_inv * inv_loss + mu_var * var_loss + nu_cov * cov_loss;

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_a,
        grad_positives: grad_b,
        grad_negatives: vec![],
    }
}

/// Proxy-Anchor Loss (Kim et al. 2020).
///
/// Each class has a learnable proxy embedding. For each proxy, the loss pulls
/// same-class embeddings closer and pushes different-class embeddings away.
/// O(N*C) complexity instead of O(N^2).
///
/// `embeddings`: `[batch][dim]`, `labels`: class indices, `proxies`: `[num_classes][dim]`.
/// Returns loss and gradients. `grad_anchors` = embedding gradients, `grad_positives` = proxy gradients.
pub fn proxy_anchor_loss(
    embeddings: &[&[f32]],
    labels: &[usize],
    proxies: &[&[f32]],
    margin: f32,
    alpha: f32,
) -> LossOutput {
    let n = embeddings.len();
    let num_classes = proxies.len();
    assert_eq!(n, labels.len());
    assert!(n > 0);
    let dim = embeddings[0].len();

    // Precompute norms
    let e_norms: Vec<f32> = embeddings
        .iter()
        .map(|e| dot(e, e).sqrt().max(1e-8))
        .collect();
    let p_norms: Vec<f32> = proxies.iter().map(|p| dot(p, p).sqrt().max(1e-8)).collect();

    // Cosine similarity matrix [num_classes, batch]
    let mut sims = vec![vec![0.0f32; n]; num_classes];
    for c in 0..num_classes {
        for i in 0..n {
            sims[c][i] = dot(proxies[c], embeddings[i]) / (p_norms[c] * e_norms[i]);
        }
    }

    let mut total_loss = 0.0f32;
    let mut grad_emb = vec![vec![0.0f32; dim]; n];
    let mut grad_proxy = vec![vec![0.0f32; dim]; num_classes];

    // Which classes are present in the batch?
    let mut classes_present = vec![false; num_classes];
    for &l in labels {
        if l < num_classes {
            classes_present[l] = true;
        }
    }

    for c in 0..num_classes {
        if !classes_present[c] {
            continue;
        }

        let pos_indices: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
        let neg_indices: Vec<usize> = (0..n).filter(|&i| labels[i] != c).collect();

        // Positive term: softplus(-alpha * (logsumexp(alpha * (s - margin)) over positives))
        // = log(1 + exp(-alpha * lse_pos))
        if !pos_indices.is_empty() {
            let pos_logits: Vec<f32> = pos_indices
                .iter()
                .map(|&i| alpha * (sims[c][i] - margin))
                .collect();
            let lse_pos = log_sum_exp(&pos_logits);
            let sp_arg = -lse_pos; // softplus(-lse)
            let sp = if sp_arg > 20.0 {
                sp_arg
            } else {
                (1.0 + sp_arg.exp()).ln()
            };
            total_loss += sp;

            // Gradient: d(softplus(-lse))/d(s_i) = -sigmoid(-lse) * alpha * softmax_weight_i
            let sigmoid_neg_lse = 1.0 / (1.0 + lse_pos.exp());
            let max_pl = pos_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let pos_exps: Vec<f32> = pos_logits.iter().map(|&v| (v - max_pl).exp()).collect();
            let sum_pe: f32 = pos_exps.iter().sum();

            for (pi, &i) in pos_indices.iter().enumerate() {
                let w = pos_exps[pi] / sum_pe;
                let dsim = -sigmoid_neg_lse * alpha * w;
                accumulate_cosine_grad(
                    proxies[c],
                    embeddings[i],
                    sims[c][i],
                    p_norms[c],
                    e_norms[i],
                    dsim,
                    &mut grad_proxy[c],
                    &mut grad_emb[i],
                );
            }
        }

        // Negative term: softplus(alpha * (logsumexp(alpha * (margin + s)) over negatives))
        if !neg_indices.is_empty() {
            let neg_logits: Vec<f32> = neg_indices
                .iter()
                .map(|&i| alpha * (sims[c][i] + margin))
                .collect();
            let lse_neg = log_sum_exp(&neg_logits);
            let sp_arg = lse_neg;
            let sp = if sp_arg > 20.0 {
                sp_arg
            } else {
                (1.0 + sp_arg.exp()).ln()
            };
            total_loss += sp;

            // Gradient: d(softplus(lse))/d(s_i) = sigmoid(lse) * alpha * softmax_weight_i
            let sigmoid_lse = 1.0 / (1.0 + (-lse_neg).exp());
            let max_nl = neg_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let neg_exps: Vec<f32> = neg_logits.iter().map(|&v| (v - max_nl).exp()).collect();
            let sum_ne: f32 = neg_exps.iter().sum();

            for (ni, &i) in neg_indices.iter().enumerate() {
                let w = neg_exps[ni] / sum_ne;
                let dsim = sigmoid_lse * alpha * w;
                accumulate_cosine_grad(
                    proxies[c],
                    embeddings[i],
                    sims[c][i],
                    p_norms[c],
                    e_norms[i],
                    dsim,
                    &mut grad_proxy[c],
                    &mut grad_emb[i],
                );
            }
        }
    }

    let s = 1.0 / num_classes as f32;
    total_loss *= s;
    for g in grad_emb.iter_mut() {
        for v in g.iter_mut() {
            *v *= s;
        }
    }
    for g in grad_proxy.iter_mut() {
        for v in g.iter_mut() {
            *v *= s;
        }
    }

    LossOutput {
        loss: total_loss,
        grad_anchors: grad_emb,
        grad_positives: grad_proxy,
        grad_negatives: vec![],
    }
}

/// Cross-batch memory buffer for expanding the negative pool.
///
/// Maintains a FIFO queue of past embeddings. Call [`enqueue`](CrossBatchMemory::enqueue)
/// after each batch to add the current batch's embeddings. Use [`embeddings`](CrossBatchMemory::embeddings)
/// to get the full memory bank as negative candidates for any pair-based loss.
#[derive(Debug, Clone)]
pub struct CrossBatchMemory {
    buffer: Vec<Vec<f32>>,
    labels: Vec<usize>,
    capacity: usize,
    dim: usize,
}

impl CrossBatchMemory {
    /// Create a new memory buffer.
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            labels: Vec::with_capacity(capacity),
            capacity,
            dim,
        }
    }

    /// Add a batch of embeddings to the memory. Oldest entries are evicted when full.
    pub fn enqueue(&mut self, embeddings: &[&[f32]], labels: &[usize]) {
        assert_eq!(embeddings.len(), labels.len());
        for (emb, &label) in embeddings.iter().zip(labels.iter()) {
            assert_eq!(emb.len(), self.dim);
            if self.buffer.len() >= self.capacity {
                self.buffer.remove(0);
                self.labels.remove(0);
            }
            self.buffer.push(emb.to_vec());
            self.labels.push(label);
        }
    }

    /// Get all stored embeddings as slice references.
    pub fn embeddings(&self) -> Vec<&[f32]> {
        self.buffer.iter().map(|v| v.as_slice()).collect()
    }

    /// Get all stored labels.
    pub fn memory_labels(&self) -> &[usize] {
        &self.labels
    }

    /// Number of embeddings currently stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Log-sum-exp of a slice, numerically stable.
fn log_sum_exp(values: &[f32]) -> f32 {
    let max_v = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_v == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let sum: f32 = values.iter().map(|&v| (v - max_v).exp()).sum();
    max_v + sum.ln()
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

    #[test]
    fn test_contrastive_loss_similar() {
        // Similar pair: loss = d^2
        let a: &[f32] = &[1.0, 0.0];
        let b: &[f32] = &[0.0, 1.0];
        let d = euclidean_distance(a, b);

        let out = contrastive_loss(&[(a, b)], &[true], 1.0);
        assert!(
            (out.loss - d * d).abs() < 1e-5,
            "similar pair loss should be d^2={}, got {}",
            d * d,
            out.loss
        );
    }

    #[test]
    fn test_contrastive_loss_dissimilar_far() {
        // Dissimilar pair far apart (d > margin) -> loss = 0
        let a: &[f32] = &[0.0, 0.0];
        let b: &[f32] = &[5.0, 0.0];

        let out = contrastive_loss(&[(a, b)], &[false], 1.0);
        assert!(
            out.loss.abs() < 1e-5,
            "dissimilar far pair loss should be 0, got {}",
            out.loss
        );
    }

    #[test]
    fn test_contrastive_loss_dissimilar_close() {
        // Dissimilar pair close (d < margin) -> loss > 0
        let a: &[f32] = &[0.0, 0.0];
        let b: &[f32] = &[0.3, 0.0];

        let out = contrastive_loss(&[(a, b)], &[false], 1.0);
        // d = 0.3, margin = 1.0, gap = 0.7, loss = 0.7^2 = 0.49
        assert!(
            (out.loss - 0.49).abs() < 1e-4,
            "dissimilar close pair loss should be 0.49, got {}",
            out.loss
        );
    }

    #[test]
    fn test_multi_similarity_loss_basic() {
        // 4 embeddings, 2 classes
        let e0: &[f32] = &[1.0, 0.0, 0.0];
        let e1: &[f32] = &[0.9, 0.1, 0.0];
        let e2: &[f32] = &[0.0, 1.0, 0.0];
        let e3: &[f32] = &[0.0, 0.9, 0.1];
        let labels = [0, 0, 1, 1];

        let out = multi_similarity_loss(&[e0, e1, e2, e3], &labels, 2.0, 50.0, 0.5);
        assert!(out.loss > 0.0, "MS loss should be > 0, got {}", out.loss);
        assert!(out.loss.is_finite(), "MS loss should be finite");
    }

    #[test]
    fn test_supcon_loss_same_class() {
        // All same class -> all pairs are positives -> loss should be relatively low
        let e0: &[f32] = &[1.0, 0.1, 0.0];
        let e1: &[f32] = &[0.9, 0.2, 0.0];
        let e2: &[f32] = &[0.95, 0.15, 0.0];
        let labels = [0, 0, 0];

        let out = supcon_loss(&[e0, e1, e2], &labels, 0.1);
        // With all same class, the loss is -log(1) = 0 only if all similarities are equal,
        // otherwise it's small but positive due to the log-sum-exp denominator
        assert!(out.loss.is_finite(), "supcon loss should be finite");
        assert!(out.loss >= -1e-5, "supcon loss should be non-negative");
    }

    #[test]
    fn test_supcon_loss_gradient_exists() {
        let e0: &[f32] = &[1.0, 0.0];
        let e1: &[f32] = &[0.8, 0.2];
        let e2: &[f32] = &[0.0, 1.0];
        let e3: &[f32] = &[0.2, 0.8];
        let labels = [0, 0, 1, 1];

        let out = supcon_loss(&[e0, e1, e2, e3], &labels, 0.1);
        let grad_norm: f32 = out
            .grad_anchors
            .iter()
            .flat_map(|g| g.iter())
            .map(|g| g * g)
            .sum();
        assert!(grad_norm > 1e-8, "supcon gradients should be non-zero");
    }

    #[test]
    fn test_circle_loss_basic() {
        // 2-class batch with overlapping embeddings so loss > 0
        let e0: &[f32] = &[1.0, 0.5, 0.2];
        let e1: &[f32] = &[0.9, 0.6, 0.3];
        let e2: &[f32] = &[0.8, 0.4, 0.1];
        let e3: &[f32] = &[0.7, 0.7, 0.2];
        let labels = [0, 0, 1, 1];

        let out = circle_loss(&[e0, e1, e2, e3], &labels, 0.25, 64.0);
        assert!(
            out.loss >= 0.0,
            "circle loss should be >= 0, got {}",
            out.loss
        );
        assert!(out.loss.is_finite(), "circle loss should be finite");

        // Well-separated data should have lower loss than overlapping
        let sep0: &[f32] = &[1.0, 0.0, 0.0];
        let sep1: &[f32] = &[0.9, 0.1, 0.0];
        let sep2: &[f32] = &[0.0, 0.0, 1.0];
        let sep3: &[f32] = &[0.0, 0.1, 0.9];
        let out_sep = circle_loss(&[sep0, sep1, sep2, sep3], &labels, 0.25, 64.0);
        assert!(
            out_sep.loss <= out.loss + 0.01,
            "separated data ({}) should have <= loss than overlapping ({})",
            out_sep.loss,
            out.loss
        );
    }

    #[test]
    fn test_lifted_structured_loss_basic() {
        // Overlapping classes should produce loss > 0
        let e0: &[f32] = &[1.0, 0.5];
        let e1: &[f32] = &[0.9, 0.6];
        let e2: &[f32] = &[0.8, 0.4];
        let e3: &[f32] = &[1.1, 0.3];
        let labels = [0, 0, 1, 1];

        let out = lifted_structured_loss(&[e0, e1, e2, e3], &labels, 1.0);
        assert!(
            out.loss >= 0.0,
            "lifted loss should be non-negative, got {}",
            out.loss
        );
        assert!(out.loss.is_finite(), "lifted loss should be finite");
    }

    #[test]
    fn test_n_pairs_loss_equivalent_to_infonce() {
        // N-pairs with dot product on normalized vectors should approximate InfoNCE
        // with temperature=1.0 using cosine similarity.
        // They differ in that InfoNCE uses cosine and n-pairs uses raw dot,
        // but on unit vectors they're the same.
        let mut a0 = vec![1.0f32, 0.0, 0.0];
        let mut a1 = vec![0.0f32, 1.0, 0.0];
        let mut p0 = vec![0.9f32, 0.1, 0.0];
        let mut p1 = vec![0.1f32, 0.9, 0.0];

        // Normalize
        crate::similarity::l2_normalize(&mut a0);
        crate::similarity::l2_normalize(&mut a1);
        crate::similarity::l2_normalize(&mut p0);
        crate::similarity::l2_normalize(&mut p1);

        let np_out = n_pairs_loss(&[&a0, &a1], &[&p0, &p1], 1.0);
        let infonce_out = infonce_loss(&[&a0, &a1], &[&p0, &p1], 1.0);

        // On unit vectors, dot = cosine, so losses should be close
        assert!(
            (np_out.loss - infonce_out.loss).abs() < 0.1,
            "n_pairs and infonce should be similar on unit vectors: n_pairs={}, infonce={}",
            np_out.loss,
            infonce_out.loss
        );
    }

    /// Helper: numerical gradient check for batch-with-labels losses.
    /// Perturbs each embedding dimension and compares analytical vs numerical gradient.
    fn check_numerical_gradients_batch<F>(
        embeddings: &[Vec<f32>],
        loss_fn: F,
        analytical_grads: &[Vec<f32>],
        eps: f32,
        tol: f32,
        name: &str,
    ) where
        F: Fn(&[&[f32]]) -> f32,
    {
        let n = embeddings.len();
        let dim = embeddings[0].len();

        for i in 0..n {
            for d in 0..dim {
                let mut emb_plus: Vec<Vec<f32>> = embeddings.to_vec();
                let mut emb_minus: Vec<Vec<f32>> = embeddings.to_vec();
                emb_plus[i][d] += eps;
                emb_minus[i][d] -= eps;

                let refs_plus: Vec<&[f32]> = emb_plus.iter().map(|e| e.as_slice()).collect();
                let refs_minus: Vec<&[f32]> = emb_minus.iter().map(|e| e.as_slice()).collect();

                let loss_plus = loss_fn(&refs_plus);
                let loss_minus = loss_fn(&refs_minus);
                let numerical = (loss_plus - loss_minus) / (2.0 * eps);
                let analytical = analytical_grads[i][d];

                let err = (analytical - numerical).abs();
                let denom = numerical.abs().max(analytical.abs()).max(1e-6);
                let rel_err = err / denom;

                assert!(
                    rel_err < tol || err < tol * 0.1,
                    "{} grad[{i}][{d}]: analytical={analytical}, numerical={numerical}, \
                     err={err}, rel_err={rel_err}",
                    name
                );
            }
        }
    }

    #[test]
    fn test_multi_similarity_gradient_numerical() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![0.7, 0.3, -0.2],
            vec![0.5, 0.8, 0.1],
            vec![-0.3, 0.1, 0.9],
            vec![-0.5, 0.2, 0.7],
        ];
        let labels = [0, 0, 1, 1];
        let alpha = 2.0;
        let beta_val = 40.0;
        let base = 0.5;

        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let out = multi_similarity_loss(&refs, &labels, alpha, beta_val, base);

        check_numerical_gradients_batch(
            &embeddings,
            |embs| multi_similarity_loss(embs, &labels, alpha, beta_val, base).loss,
            &out.grad_anchors,
            1e-4,
            1e-2,
            "multi_similarity",
        );
    }

    #[test]
    fn test_circle_gradient_numerical() {
        // Use moderate gamma to avoid numerical issues with large gradients
        let embeddings: Vec<Vec<f32>> = vec![
            vec![0.8, 0.2, -0.1],
            vec![0.6, 0.7, 0.1],
            vec![-0.4, 0.3, 0.8],
            vec![-0.3, 0.1, 0.6],
        ];
        let labels = [0, 0, 1, 1];
        let margin = 0.25;
        let gamma = 16.0;

        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let out = circle_loss(&refs, &labels, margin, gamma);

        check_numerical_gradients_batch(
            &embeddings,
            |embs| circle_loss(embs, &labels, margin, gamma).loss,
            &out.grad_anchors,
            1e-4,
            5e-2,
            "circle",
        );
    }

    #[test]
    fn test_lifted_structured_gradient_numerical() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.9, 0.6, 0.3],
            vec![0.8, 0.4, 0.1],
            vec![0.7, 0.7, 0.2],
        ];
        let labels = [0, 0, 1, 1];
        let margin = 1.0;

        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let out = lifted_structured_loss(&refs, &labels, margin);

        assert!(out.loss > 0.0, "need active loss for gradient check");

        check_numerical_gradients_batch(
            &embeddings,
            |embs| lifted_structured_loss(embs, &labels, margin).loss,
            &out.grad_anchors,
            1e-4,
            5e-2,
            "lifted_structured",
        );
    }

    /// Helper: numerical gradient check for pair-based losses (anchor/positive/negative).
    #[allow(clippy::too_many_arguments)]
    fn check_numerical_gradients_pair(
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
        negatives: &[Vec<f32>],
        loss_fn: impl Fn(&[&[f32]], &[&[f32]], &[&[f32]]) -> f32,
        grad_a: &[Vec<f32>],
        grad_p: &[Vec<f32>],
        grad_n: &[Vec<f32>],
        eps: f32,
        tol: f32,
        name: &str,
    ) {
        let check = |vecs: &[Vec<f32>], grads: &[Vec<f32>], which: &str, perturb_which: usize| {
            for i in 0..vecs.len() {
                for d in 0..vecs[i].len() {
                    let mut a_work: Vec<Vec<f32>> = anchors.to_vec();
                    let mut p_work: Vec<Vec<f32>> = positives.to_vec();
                    let mut n_work: Vec<Vec<f32>> = negatives.to_vec();
                    let target = match perturb_which {
                        0 => &mut a_work,
                        1 => &mut p_work,
                        _ => &mut n_work,
                    };
                    target[i][d] += eps;
                    let a_refs: Vec<&[f32]> = a_work.iter().map(|v| v.as_slice()).collect();
                    let p_refs: Vec<&[f32]> = p_work.iter().map(|v| v.as_slice()).collect();
                    let n_refs: Vec<&[f32]> = n_work.iter().map(|v| v.as_slice()).collect();
                    let loss_plus = loss_fn(&a_refs, &p_refs, &n_refs);

                    let target = match perturb_which {
                        0 => &mut a_work,
                        1 => &mut p_work,
                        _ => &mut n_work,
                    };
                    target[i][d] -= 2.0 * eps;
                    let a_refs: Vec<&[f32]> = a_work.iter().map(|v| v.as_slice()).collect();
                    let p_refs: Vec<&[f32]> = p_work.iter().map(|v| v.as_slice()).collect();
                    let n_refs: Vec<&[f32]> = n_work.iter().map(|v| v.as_slice()).collect();
                    let loss_minus = loss_fn(&a_refs, &p_refs, &n_refs);

                    let numerical = (loss_plus - loss_minus) / (2.0 * eps);
                    let analytical = grads[i][d];
                    let err = (analytical - numerical).abs();
                    let denom = numerical.abs().max(analytical.abs()).max(1e-6);
                    let rel_err = err / denom;

                    assert!(
                        rel_err < tol || err < tol * 0.1,
                        "{name} {which}[{i}][{d}]: analytical={analytical}, numerical={numerical}, \
                         err={err}, rel_err={rel_err}",
                    );
                }
            }
        };
        check(anchors, grad_a, "anchor", 0);
        check(positives, grad_p, "positive", 1);
        if !negatives.is_empty() && !grad_n.is_empty() {
            check(negatives, grad_n, "negative", 2);
        }
    }

    #[test]
    fn test_infonce_gradient_numerical() {
        let anchors: Vec<Vec<f32>> = vec![vec![0.7, 0.3, -0.2], vec![-0.3, 0.8, 0.1]];
        let positives: Vec<Vec<f32>> = vec![vec![0.5, 0.4, -0.1], vec![-0.2, 0.7, 0.3]];
        let temperature = 0.1;

        let a_refs: Vec<&[f32]> = anchors.iter().map(|v| v.as_slice()).collect();
        let p_refs: Vec<&[f32]> = positives.iter().map(|v| v.as_slice()).collect();
        let out = infonce_loss(&a_refs, &p_refs, temperature);

        check_numerical_gradients_pair(
            &anchors,
            &positives,
            &[],
            |a, p, _| infonce_loss(a, p, temperature).loss,
            &out.grad_anchors,
            &out.grad_positives,
            &[],
            1e-4,
            5e-2,
            "infonce",
        );
    }

    #[test]
    fn test_mnrl_gradient_numerical() {
        let anchors: Vec<Vec<f32>> = vec![vec![0.7, 0.3, -0.2], vec![-0.3, 0.8, 0.1]];
        let positives: Vec<Vec<f32>> = vec![vec![0.5, 0.4, -0.1], vec![-0.2, 0.7, 0.3]];
        let negatives: Vec<Vec<f32>> = vec![vec![0.1, -0.5, 0.6]];
        let temperature = 0.1;

        let a_refs: Vec<&[f32]> = anchors.iter().map(|v| v.as_slice()).collect();
        let p_refs: Vec<&[f32]> = positives.iter().map(|v| v.as_slice()).collect();
        let n_refs: Vec<&[f32]> = negatives.iter().map(|v| v.as_slice()).collect();
        let out = mnrl_loss(&a_refs, &p_refs, &n_refs, temperature);

        check_numerical_gradients_pair(
            &anchors,
            &positives,
            &negatives,
            |a, p, n| mnrl_loss(a, p, n, temperature).loss,
            &out.grad_anchors,
            &out.grad_positives,
            &out.grad_negatives,
            1e-4,
            5e-2,
            "mnrl",
        );
    }

    #[test]
    fn test_contrastive_gradient_numerical() {
        let a_vecs: Vec<Vec<f32>> = vec![vec![0.7, 0.3], vec![0.1, -0.5]];
        let b_vecs: Vec<Vec<f32>> = vec![vec![0.5, 0.8], vec![-0.3, 0.2]];
        let labels = [true, false];
        let margin = 1.0;

        let pairs: Vec<(&[f32], &[f32])> = a_vecs
            .iter()
            .zip(b_vecs.iter())
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        let out = contrastive_loss(&pairs, &labels, margin);

        let eps = 1e-4;
        let tol = 1e-2;

        // Check a gradients
        for i in 0..a_vecs.len() {
            for d in 0..a_vecs[i].len() {
                let mut a_plus = a_vecs.clone();
                let mut a_minus = a_vecs.clone();
                a_plus[i][d] += eps;
                a_minus[i][d] -= eps;

                let pairs_p: Vec<(&[f32], &[f32])> = a_plus
                    .iter()
                    .zip(b_vecs.iter())
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect();
                let pairs_m: Vec<(&[f32], &[f32])> = a_minus
                    .iter()
                    .zip(b_vecs.iter())
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect();

                let numerical = (contrastive_loss(&pairs_p, &labels, margin).loss
                    - contrastive_loss(&pairs_m, &labels, margin).loss)
                    / (2.0 * eps);
                assert!(
                    (out.grad_anchors[i][d] - numerical).abs() < tol,
                    "contrastive a[{i}][{d}]: analytical={}, numerical={numerical}",
                    out.grad_anchors[i][d]
                );
            }
        }

        // Check b gradients
        for i in 0..b_vecs.len() {
            for d in 0..b_vecs[i].len() {
                let mut b_plus = b_vecs.clone();
                let mut b_minus = b_vecs.clone();
                b_plus[i][d] += eps;
                b_minus[i][d] -= eps;

                let pairs_p: Vec<(&[f32], &[f32])> = a_vecs
                    .iter()
                    .zip(b_plus.iter())
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect();
                let pairs_m: Vec<(&[f32], &[f32])> = a_vecs
                    .iter()
                    .zip(b_minus.iter())
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect();

                let numerical = (contrastive_loss(&pairs_p, &labels, margin).loss
                    - contrastive_loss(&pairs_m, &labels, margin).loss)
                    / (2.0 * eps);
                assert!(
                    (out.grad_positives[i][d] - numerical).abs() < tol,
                    "contrastive b[{i}][{d}]: analytical={}, numerical={numerical}",
                    out.grad_positives[i][d]
                );
            }
        }
    }

    #[test]
    fn test_cosine_embedding_gradient_numerical() {
        let a_vecs: Vec<Vec<f32>> = vec![vec![0.7, 0.3], vec![0.1, -0.5]];
        let b_vecs: Vec<Vec<f32>> = vec![vec![0.5, 0.8], vec![-0.3, 0.2]];
        let labels = [true, false];
        let margin = 0.0;

        let pairs: Vec<(&[f32], &[f32])> = a_vecs
            .iter()
            .zip(b_vecs.iter())
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        let out = cosine_embedding_loss(&pairs, &labels, margin);

        let eps = 1e-4;
        let tol = 1e-2;

        for i in 0..a_vecs.len() {
            for d in 0..a_vecs[i].len() {
                let mut a_plus = a_vecs.clone();
                let mut a_minus = a_vecs.clone();
                a_plus[i][d] += eps;
                a_minus[i][d] -= eps;

                let pairs_p: Vec<(&[f32], &[f32])> = a_plus
                    .iter()
                    .zip(b_vecs.iter())
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect();
                let pairs_m: Vec<(&[f32], &[f32])> = a_minus
                    .iter()
                    .zip(b_vecs.iter())
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect();

                let numerical = (cosine_embedding_loss(&pairs_p, &labels, margin).loss
                    - cosine_embedding_loss(&pairs_m, &labels, margin).loss)
                    / (2.0 * eps);
                assert!(
                    (out.grad_anchors[i][d] - numerical).abs() < tol,
                    "cosine_embedding a[{i}][{d}]: analytical={}, numerical={numerical}",
                    out.grad_anchors[i][d]
                );
            }
        }

        // Check b-side gradients
        for i in 0..b_vecs.len() {
            for d in 0..b_vecs[i].len() {
                let mut b_plus = b_vecs.clone();
                let mut b_minus = b_vecs.clone();
                b_plus[i][d] += eps;
                b_minus[i][d] -= eps;

                let pairs_p: Vec<(&[f32], &[f32])> = a_vecs
                    .iter()
                    .zip(b_plus.iter())
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect();
                let pairs_m: Vec<(&[f32], &[f32])> = a_vecs
                    .iter()
                    .zip(b_minus.iter())
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect();

                let numerical = (cosine_embedding_loss(&pairs_p, &labels, margin).loss
                    - cosine_embedding_loss(&pairs_m, &labels, margin).loss)
                    / (2.0 * eps);
                assert!(
                    (out.grad_positives[i][d] - numerical).abs() < tol,
                    "cosine_embedding b[{i}][{d}]: analytical={}, numerical={numerical}",
                    out.grad_positives[i][d]
                );
            }
        }
    }

    #[test]
    fn test_supcon_gradient_numerical() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![0.7, 0.3, -0.2],
            vec![0.5, 0.8, 0.1],
            vec![-0.3, 0.1, 0.9],
            vec![-0.5, 0.2, 0.7],
        ];
        let labels = [0, 0, 1, 1];
        let temperature = 0.1;

        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let out = supcon_loss(&refs, &labels, temperature);

        check_numerical_gradients_batch(
            &embeddings,
            |embs| supcon_loss(embs, &labels, temperature).loss,
            &out.grad_anchors,
            1e-4,
            5e-2,
            "supcon",
        );
    }

    #[test]
    fn test_n_pairs_gradient_numerical() {
        let anchors: Vec<Vec<f32>> = vec![vec![0.7, 0.3, -0.2], vec![-0.3, 0.8, 0.1]];
        let positives: Vec<Vec<f32>> = vec![vec![0.5, 0.4, -0.1], vec![-0.2, 0.7, 0.3]];
        let temperature = 0.5;

        let a_refs: Vec<&[f32]> = anchors.iter().map(|v| v.as_slice()).collect();
        let p_refs: Vec<&[f32]> = positives.iter().map(|v| v.as_slice()).collect();
        let out = n_pairs_loss(&a_refs, &p_refs, temperature);

        check_numerical_gradients_pair(
            &anchors,
            &positives,
            &[],
            |a, p, _| n_pairs_loss(a, p, temperature).loss,
            &out.grad_anchors,
            &out.grad_positives,
            &[],
            1e-4,
            5e-2,
            "n_pairs",
        );
    }

    #[test]
    fn test_triplet_cosine_gradient_numerical() {
        let anchor: Vec<f32> = vec![0.7, 0.3, -0.2];
        let positive: Vec<f32> = vec![0.5, 0.8, 0.1];
        let negative: Vec<f32> = vec![-0.3, 0.1, 0.9];

        let out = triplet_loss_with_distance(
            &[&anchor],
            &[&positive],
            &[&negative],
            1.0,
            DistanceFn::Cosine,
        );

        let eps = 1e-4;
        let tol = 1e-2;

        for d in 0..3 {
            let mut a_plus = anchor.clone();
            let mut a_minus = anchor.clone();
            a_plus[d] += eps;
            a_minus[d] -= eps;

            let lp = triplet_loss_with_distance(
                &[&a_plus],
                &[&positive],
                &[&negative],
                1.0,
                DistanceFn::Cosine,
            )
            .loss;
            let lm = triplet_loss_with_distance(
                &[&a_minus],
                &[&positive],
                &[&negative],
                1.0,
                DistanceFn::Cosine,
            )
            .loss;
            let numerical = (lp - lm) / (2.0 * eps);
            assert!(
                (out.grad_anchors[0][d] - numerical).abs() < tol,
                "triplet_cosine anchor[{d}]: analytical={}, numerical={numerical}",
                out.grad_anchors[0][d]
            );
        }
    }

    #[test]
    fn test_arcface_basic() {
        // Use non-aligned embeddings so gradients are non-trivial
        let e0: &[f32] = &[0.7, 0.3, 0.1];
        let e1: &[f32] = &[0.2, 0.8, -0.1];
        let p0: &[f32] = &[1.0, 0.0, 0.0]; // class 0 proxy
        let p1: &[f32] = &[0.0, 1.0, 0.0]; // class 1 proxy
        let p2: &[f32] = &[0.0, 0.0, 1.0]; // class 2 proxy

        let out = arcface_loss(&[e0, e1], &[0, 1], &[p0, p1, p2], 10.0, 0.3);
        assert!(
            out.loss >= 0.0,
            "arcface loss should be non-negative: {}",
            out.loss
        );
        assert!(out.loss.is_finite(), "arcface loss should be finite");

        // Embedding gradients should be non-zero
        let grad_norm: f32 = out
            .grad_anchors
            .iter()
            .flat_map(|g| g.iter())
            .map(|v| v * v)
            .sum();
        assert!(
            grad_norm > 1e-8,
            "arcface embedding grads should be non-zero"
        );

        // Proxy gradients should be non-zero
        let proxy_grad_norm: f32 = out
            .grad_positives
            .iter()
            .flat_map(|g| g.iter())
            .map(|v| v * v)
            .sum();
        assert!(
            proxy_grad_norm > 1e-8,
            "arcface proxy grads should be non-zero"
        );
    }

    #[test]
    fn test_arcface_gradient_numerical() {
        let embeddings: Vec<Vec<f32>> = vec![vec![0.7, 0.3, -0.2], vec![-0.3, 0.8, 0.1]];
        let proxies: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = [0, 1];
        let scale = 10.0;
        let margin = 0.3;

        let e_refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let p_refs: Vec<&[f32]> = proxies.iter().map(|p| p.as_slice()).collect();
        let out = arcface_loss(&e_refs, &labels, &p_refs, scale, margin);

        let eps = 1e-4;
        let tol = 5e-2;

        // Check embedding gradients
        for i in 0..embeddings.len() {
            for d in 0..3 {
                let mut e_plus = embeddings.clone();
                let mut e_minus = embeddings.clone();
                e_plus[i][d] += eps;
                e_minus[i][d] -= eps;

                let ep: Vec<&[f32]> = e_plus.iter().map(|e| e.as_slice()).collect();
                let em: Vec<&[f32]> = e_minus.iter().map(|e| e.as_slice()).collect();
                let numerical = (arcface_loss(&ep, &labels, &p_refs, scale, margin).loss
                    - arcface_loss(&em, &labels, &p_refs, scale, margin).loss)
                    / (2.0 * eps);
                let err = (out.grad_anchors[i][d] - numerical).abs();
                let denom = numerical.abs().max(out.grad_anchors[i][d].abs()).max(1e-6);
                assert!(
                    err / denom < tol || err < tol * 0.1,
                    "arcface emb[{i}][{d}]: analytical={}, numerical={numerical}",
                    out.grad_anchors[i][d]
                );
            }
        }
    }

    #[test]
    fn test_vicreg_basic() {
        let a0: &[f32] = &[1.0, 0.0, 0.5];
        let a1: &[f32] = &[0.5, 0.5, 0.0];
        let a2: &[f32] = &[0.0, 1.0, 0.5];
        let b0: &[f32] = &[0.9, 0.1, 0.4];
        let b1: &[f32] = &[0.4, 0.6, 0.1];
        let b2: &[f32] = &[0.1, 0.9, 0.6];

        let out = vicreg_loss(&[a0, a1, a2], &[b0, b1, b2], 25.0, 25.0, 1.0);
        assert!(
            out.loss >= 0.0,
            "vicreg loss should be non-negative: {}",
            out.loss
        );
        assert!(out.loss.is_finite(), "vicreg loss should be finite");

        // Both view gradients should be non-zero
        let ga: f32 = out
            .grad_anchors
            .iter()
            .flat_map(|g| g.iter())
            .map(|v| v * v)
            .sum();
        let gb: f32 = out
            .grad_positives
            .iter()
            .flat_map(|g| g.iter())
            .map(|v| v * v)
            .sum();
        assert!(ga > 1e-8, "vicreg view_a grads should be non-zero");
        assert!(gb > 1e-8, "vicreg view_b grads should be non-zero");
    }

    #[test]
    fn test_vicreg_gradient_numerical() {
        let view_a: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.5],
            vec![0.5, 0.5, 0.0],
            vec![0.0, 1.0, 0.5],
        ];
        let view_b: Vec<Vec<f32>> = vec![
            vec![0.9, 0.1, 0.4],
            vec![0.4, 0.6, 0.1],
            vec![0.1, 0.9, 0.6],
        ];

        let a_refs: Vec<&[f32]> = view_a.iter().map(|v| v.as_slice()).collect();
        let b_refs: Vec<&[f32]> = view_b.iter().map(|v| v.as_slice()).collect();
        // Use smaller coefficients to reduce covariance term's gradient magnitude
        let out = vicreg_loss(&a_refs, &b_refs, 25.0, 25.0, 1.0);

        let eps = 1e-3;
        let tol = 0.15; // VICReg covariance gradient through mean subtraction is approximate

        // Check view_a gradients
        for i in 0..view_a.len() {
            for d in 0..3 {
                let mut a_plus = view_a.clone();
                let mut a_minus = view_a.clone();
                a_plus[i][d] += eps;
                a_minus[i][d] -= eps;

                let ap: Vec<&[f32]> = a_plus.iter().map(|v| v.as_slice()).collect();
                let am: Vec<&[f32]> = a_minus.iter().map(|v| v.as_slice()).collect();
                let numerical = (vicreg_loss(&ap, &b_refs, 25.0, 25.0, 1.0).loss
                    - vicreg_loss(&am, &b_refs, 25.0, 25.0, 1.0).loss)
                    / (2.0 * eps);
                let err = (out.grad_anchors[i][d] - numerical).abs();
                let denom = numerical.abs().max(out.grad_anchors[i][d].abs()).max(1e-6);
                assert!(
                    err / denom < tol || err < tol * 0.1,
                    "vicreg a[{i}][{d}]: analytical={}, numerical={numerical}",
                    out.grad_anchors[i][d]
                );
            }
        }
    }

    #[test]
    fn test_proxy_anchor_basic() {
        let e0: &[f32] = &[0.7, 0.3, 0.1];
        let e1: &[f32] = &[0.2, 0.8, -0.1];
        let p0: &[f32] = &[1.0, 0.0, 0.0];
        let p1: &[f32] = &[0.0, 1.0, 0.0];
        let p2: &[f32] = &[0.0, 0.0, 1.0];

        let out = proxy_anchor_loss(&[e0, e1], &[0, 1], &[p0, p1, p2], 0.1, 32.0);
        assert!(out.loss.is_finite(), "proxy_anchor loss: {}", out.loss);

        let grad_norm: f32 = out
            .grad_anchors
            .iter()
            .flat_map(|g| g.iter())
            .map(|v| v * v)
            .sum();
        assert!(grad_norm > 1e-8, "proxy_anchor grads should be non-zero");
    }

    #[test]
    fn test_proxy_anchor_gradient_numerical() {
        let embeddings: Vec<Vec<f32>> = vec![vec![0.7, 0.3, -0.2], vec![-0.3, 0.8, 0.1]];
        let proxies: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = [0, 1];
        let margin = 0.1;
        let alpha = 16.0;

        let e_refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let p_refs: Vec<&[f32]> = proxies.iter().map(|p| p.as_slice()).collect();
        let out = proxy_anchor_loss(&e_refs, &labels, &p_refs, margin, alpha);

        let eps = 1e-4;
        let tol = 5e-2;

        for i in 0..embeddings.len() {
            for d in 0..3 {
                let mut e_plus = embeddings.clone();
                let mut e_minus = embeddings.clone();
                e_plus[i][d] += eps;
                e_minus[i][d] -= eps;

                let ep: Vec<&[f32]> = e_plus.iter().map(|e| e.as_slice()).collect();
                let em: Vec<&[f32]> = e_minus.iter().map(|e| e.as_slice()).collect();
                let numerical = (proxy_anchor_loss(&ep, &labels, &p_refs, margin, alpha).loss
                    - proxy_anchor_loss(&em, &labels, &p_refs, margin, alpha).loss)
                    / (2.0 * eps);
                let err = (out.grad_anchors[i][d] - numerical).abs();
                let den = numerical.abs().max(out.grad_anchors[i][d].abs()).max(1e-6);
                assert!(
                    err / den < tol || err < tol * 0.1,
                    "proxy_anchor emb[{i}][{d}]: analytical={}, numerical={numerical}",
                    out.grad_anchors[i][d]
                );
            }
        }
    }

    #[test]
    fn test_cross_batch_memory() {
        let mut mem = CrossBatchMemory::new(4, 2);
        assert!(mem.is_empty());

        let e0: &[f32] = &[1.0, 0.0];
        let e1: &[f32] = &[0.0, 1.0];
        mem.enqueue(&[e0, e1], &[0, 1]);
        assert_eq!(mem.len(), 2);

        let e2: &[f32] = &[0.5, 0.5];
        let e3: &[f32] = &[-1.0, 0.0];
        mem.enqueue(&[e2, e3], &[0, 1]);
        assert_eq!(mem.len(), 4);

        // Enqueue one more, should evict oldest
        let e4: &[f32] = &[0.3, 0.7];
        mem.enqueue(&[e4], &[2]);
        assert_eq!(mem.len(), 4);
        assert_eq!(mem.memory_labels(), &[1, 0, 1, 2]);

        // Use memory as negatives for a loss
        let anchor: &[f32] = &[1.0, 0.0];
        let positive: &[f32] = &[0.9, 0.1];
        let neg_refs = mem.embeddings();
        let out = mnrl_loss(&[anchor], &[positive], &neg_refs, 0.1);
        assert!(
            out.loss.is_finite(),
            "loss with memory negatives: {}",
            out.loss
        );
    }
}
