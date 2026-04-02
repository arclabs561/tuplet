//! Candle-compatible loss functions for metric learning.
//!
//! These operate on [`candle_core::Tensor`] and participate in autograd.
//! Enable with `features = ["candle"]`.
//!
//! Unlike the `f32`-slice losses in [`crate::losses`], these do not compute
//! analytical gradients -- candle's autograd handles backpropagation.

use candle_core::{Result, Tensor};

/// Numerically stable log-sum-exp along a dimension.
fn log_sum_exp(t: &Tensor, dim: usize) -> Result<Tensor> {
    let max = t.max(dim)?.unsqueeze(dim)?;
    let shifted = t.broadcast_sub(&max)?;
    let lse = shifted.exp()?.sum(dim)?.log()?.unsqueeze(dim)?;
    max.add(&lse)?.squeeze(dim)
}

/// Pairwise cosine similarity matrix. Returns `[n, n]` tensor.
pub fn pairwise_cosine(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let normalized = embeddings.broadcast_div(&norm.unsqueeze(1)?)?;
    normalized.matmul(&normalized.t()?)
}

/// Triplet loss: `mean(max(0, d(a,p) - d(a,n) + margin))`.
///
/// Uses Euclidean distance. Inputs are `[batch, dim]` tensors.
pub fn triplet_loss(
    anchors: &Tensor,
    positives: &Tensor,
    negatives: &Tensor,
    margin: f32,
) -> Result<Tensor> {
    let d_ap = anchors.sub(positives)?.sqr()?.sum(1)?.sqrt()?;
    let d_an = anchors.sub(negatives)?.sqr()?.sum(1)?.sqrt()?;
    d_ap.sub(&d_an)?
        .affine(1.0, margin as f64)?
        .relu()?
        .mean_all()
}

/// InfoNCE (NT-Xent) loss with in-batch negatives.
///
/// `embeddings_a` and `embeddings_b` are `[batch, dim]`. The positive for row i
/// is the corresponding row in the other tensor; all other rows are negatives.
pub fn infonce_loss(
    embeddings_a: &Tensor,
    embeddings_b: &Tensor,
    temperature: f32,
) -> Result<Tensor> {
    let n = embeddings_a.dim(0)?;

    // Cosine similarity matrix [n, n]
    let norm_a = embeddings_a
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let norm_b = embeddings_b
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let a_normed = embeddings_a.broadcast_div(&norm_a.unsqueeze(1)?)?;
    let b_normed = embeddings_b.broadcast_div(&norm_b.unsqueeze(1)?)?;
    let sims = a_normed
        .matmul(&b_normed.t()?)?
        .affine(1.0 / temperature as f64, 0.0)?;

    // Cross-entropy with targets = diagonal (identity)
    let log_softmax = {
        let lse = log_sum_exp(&sims, 1)?;
        sims.broadcast_sub(&lse.unsqueeze(1)?)?
    };

    // Loss = -mean(diag(log_softmax))
    // Extract diagonal: gather indices [0,1,...,n-1] from each row
    let indices = Tensor::arange(0u32, n as u32, sims.device())?
        .to_dtype(candle_core::DType::U32)?
        .unsqueeze(1)?;
    let diag = log_softmax.gather(&indices, 1)?.squeeze(1)?;
    diag.neg()?.mean_all()
}

/// Pairwise contrastive loss (Hadsell et al. 2006).
///
/// `labels`: 1.0 for similar pairs, 0.0 for dissimilar. All tensors are `[batch, dim]`.
pub fn contrastive_loss(
    embeddings_a: &Tensor,
    embeddings_b: &Tensor,
    labels: &Tensor,
    margin: f32,
) -> Result<Tensor> {
    let diff = embeddings_a.sub(embeddings_b)?;
    let d_sq = diff.sqr()?.sum(1)?;
    let d = d_sq.sqrt()?;

    // Similar: label * d^2
    let pos_loss = labels.mul(&d_sq)?;

    // Dissimilar: (1 - label) * max(0, margin - d)^2
    let ones = labels.ones_like()?;
    let neg_labels = ones.sub(labels)?;
    let gap = d.affine(-1.0, margin as f64)?.relu()?;
    let neg_loss = neg_labels.mul(&gap.sqr()?)?;

    pos_loss.add(&neg_loss)?.mean_all()
}

/// Supervised Contrastive Loss (Khosla et al. 2020).
///
/// `embeddings`: `[n, dim]`, `labels`: `[n]` integer labels.
pub fn supcon_loss(embeddings: &Tensor, labels: &Tensor, temperature: f32) -> Result<Tensor> {
    let n = embeddings.dim(0)?;
    let device = embeddings.device();

    // Cosine similarity matrix
    let sim_matrix = pairwise_cosine(embeddings)?.affine(1.0 / temperature as f64, 0.0)?;

    // Mask: labels[i] == labels[j]
    let labels_col = labels.unsqueeze(1)?.expand(&[n, n])?;
    let labels_row = labels.unsqueeze(0)?.expand(&[n, n])?;
    let pos_mask = labels_col
        .eq(&labels_row)?
        .to_dtype(candle_core::DType::F32)?;

    // Self-mask (exclude diagonal)
    let eye = Tensor::eye(n, candle_core::DType::F32, device)?;
    let not_self = eye.affine(-1.0, 1.0)?;
    let pos_mask = pos_mask.mul(&not_self)?;

    // Log-softmax over all j != i
    let large_neg = eye.affine(1e9, 0.0)?;
    let masked_sims = sim_matrix.sub(&large_neg)?;
    let lse = log_sum_exp(&masked_sims, 1)?;
    let log_prob = sim_matrix.broadcast_sub(&lse.unsqueeze(1)?)?;

    // Mean of log_prob over positive pairs
    let pos_log_prob = log_prob.mul(&pos_mask)?;
    let n_pos = pos_mask.sum(1)?.clamp(1.0, f64::MAX)?;
    let per_anchor = pos_log_prob.sum(1)?.div(&n_pos)?;

    per_anchor.neg()?.mean_all()
}

/// Multi-Similarity Loss (Wang et al. 2019).
///
/// `embeddings`: `[n, dim]`, `labels`: `[n]` integer labels.
pub fn multi_similarity_loss(
    embeddings: &Tensor,
    labels: &Tensor,
    alpha: f32,
    beta: f32,
    base: f32,
) -> Result<Tensor> {
    let n = embeddings.dim(0)?;
    let device = embeddings.device();

    let sims = pairwise_cosine(embeddings)?;

    // Positive/negative masks
    let labels_col = labels.unsqueeze(1)?.expand(&[n, n])?;
    let labels_row = labels.unsqueeze(0)?.expand(&[n, n])?;
    let pos_mask = labels_col
        .eq(&labels_row)?
        .to_dtype(candle_core::DType::F32)?;
    let eye = Tensor::eye(n, candle_core::DType::F32, device)?;
    let pos_mask = pos_mask.mul(&eye.affine(-1.0, 1.0)?)?; // exclude self
    let neg_mask = pos_mask.affine(-1.0, 1.0)?.mul(&eye.affine(-1.0, 1.0)?)?;

    // Positive term: (1/alpha) * log(1 + sum_p exp(-alpha * (s_ip - base)))
    let pos_exp_arg = sims
        .affine(-1.0, 0.0)?
        .affine(alpha as f64, alpha as f64 * base as f64)?; // -alpha * (s - base)
    // Mask out non-positives with large negative
    let large_neg_mask = neg_mask.affine(1e9, 0.0)?.add(&eye.affine(1e9, 0.0)?)?;
    let pos_logits = pos_exp_arg.sub(&large_neg_mask)?;
    // log(1 + sum exp(...)) = log(exp(0) + sum exp(...))
    let zero_col = Tensor::zeros(&[n, 1], candle_core::DType::F32, device)?;
    let pos_with_one = Tensor::cat(&[&pos_logits, &zero_col], 1)?;
    let pos_term = log_sum_exp(&pos_with_one, 1)?.affine(1.0 / alpha as f64, 0.0)?;

    // Negative term: (1/beta) * log(1 + sum_n exp(beta * (s_in - base)))
    let neg_exp_arg = sims.affine(beta as f64, -(beta as f64) * base as f64)?;
    let large_neg_mask2 = pos_mask.affine(1e9, 0.0)?.add(&eye.affine(1e9, 0.0)?)?;
    let neg_logits = neg_exp_arg.sub(&large_neg_mask2)?;
    let neg_with_one = Tensor::cat(&[&neg_logits, &zero_col], 1)?;
    let neg_term = log_sum_exp(&neg_with_one, 1)?.affine(1.0 / beta as f64, 0.0)?;

    pos_term.add(&neg_term)?.mean_all()
}

/// Circle Loss (Sun et al. 2020).
///
/// `embeddings`: `[n, dim]`, `labels`: `[n]` integer labels.
pub fn circle_loss(
    embeddings: &Tensor,
    labels: &Tensor,
    margin: f32,
    gamma: f32,
) -> Result<Tensor> {
    let n = embeddings.dim(0)?;
    let device = embeddings.device();

    let sims = pairwise_cosine(embeddings)?;

    let o_p = 1.0 + margin;
    let o_n = -margin;
    let delta_p = 1.0 - margin;
    let delta_n = margin;

    // Masks
    let labels_col = labels.unsqueeze(1)?.expand(&[n, n])?;
    let labels_row = labels.unsqueeze(0)?.expand(&[n, n])?;
    let pos_mask = labels_col
        .eq(&labels_row)?
        .to_dtype(candle_core::DType::F32)?;
    let eye = Tensor::eye(n, candle_core::DType::F32, device)?;
    let pos_mask = pos_mask.mul(&eye.affine(-1.0, 1.0)?)?;
    let neg_mask = pos_mask.affine(-1.0, 1.0)?.mul(&eye.affine(-1.0, 1.0)?)?;

    // Alpha weights (detached, matching reference implementation)
    let sims_detached = sims.detach();
    let alpha_p = sims_detached.affine(-1.0, o_p as f64)?.relu()?;
    let alpha_n = sims_detached.affine(1.0, -(o_n as f64))?.relu()?;

    // Positive logits: -gamma * alpha_p * (s_p - delta_p)
    let pos_logits = sims
        .affine(1.0, -(delta_p as f64))?
        .mul(&alpha_p)?
        .affine(-(gamma as f64), 0.0)?;
    let large_neg_mask = neg_mask.affine(1e9, 0.0)?.add(&eye.affine(1e9, 0.0)?)?;
    let pos_logits = pos_logits.sub(&large_neg_mask)?;

    // Negative logits: gamma * alpha_n * (s_n - delta_n)
    let neg_logits = sims
        .affine(1.0, -(delta_n as f64))?
        .mul(&alpha_n)?
        .affine(gamma as f64, 0.0)?;
    let large_neg_mask2 = pos_mask.affine(1e9, 0.0)?.add(&eye.affine(1e9, 0.0)?)?;
    let neg_logits = neg_logits.sub(&large_neg_mask2)?;

    // softplus(logsumexp(neg) + logsumexp(pos))
    let lse_neg = log_sum_exp(&neg_logits, 1)?;
    let lse_pos = log_sum_exp(&pos_logits, 1)?;
    let combined = lse_neg.add(&lse_pos)?;
    // softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    // This formulation avoids exp overflow for both large positive and large negative x.
    let abs_combined = combined.abs()?;
    let softplus = combined
        .relu()?
        .add(&abs_combined.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;

    softplus.mean_all()
}

/// N-pairs loss (Sohn 2016).
///
/// `anchors` and `positives` are `[batch, dim]`. Negatives are all other positives in the batch.
pub fn n_pairs_loss(anchors: &Tensor, positives: &Tensor, temperature: f32) -> Result<Tensor> {
    let n = anchors.dim(0)?;

    // Dot-product similarity matrix [n, n]
    let sims = anchors
        .matmul(&positives.t()?)?
        .affine(1.0 / temperature as f64, 0.0)?;

    // Cross-entropy with target = identity (each anchor matches its positive)
    let lse = log_sum_exp(&sims, 1)?;
    let n_u32 = n as u32;
    let indices = Tensor::arange(0u32, n_u32, sims.device())?
        .to_dtype(candle_core::DType::U32)?
        .unsqueeze(1)?;
    let diag = sims.gather(&indices, 1)?.squeeze(1)?;
    let losses = lse.sub(&diag)?;

    losses.mean_all()
}

/// Cosine embedding loss for labeled pairs.
///
/// `labels`: 1.0 for similar, -1.0 or 0.0 for dissimilar.
/// Positive: `1 - cos(a, b)`. Negative: `max(0, cos(a, b) - margin)`.
pub fn cosine_embedding_loss(
    embeddings_a: &Tensor,
    embeddings_b: &Tensor,
    labels: &Tensor,
    margin: f32,
) -> Result<Tensor> {
    let norm_a = embeddings_a
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let norm_b = embeddings_b
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let a_normed = embeddings_a.broadcast_div(&norm_a.unsqueeze(1)?)?;
    let b_normed = embeddings_b.broadcast_div(&norm_b.unsqueeze(1)?)?;
    let cos_sim = a_normed.mul(&b_normed)?.sum(1)?;

    // Positive loss: label * (1 - cos)
    let is_pos = labels.clamp(0.0, 1.0)?; // treat >0 as positive
    let pos_loss = cos_sim.affine(-1.0, 1.0)?.mul(&is_pos)?;

    // Negative loss: (1 - label) * max(0, cos - margin)
    let is_neg = is_pos.affine(-1.0, 1.0)?;
    let neg_loss = cos_sim
        .affine(1.0, -(margin as f64))?
        .relu()?
        .mul(&is_neg)?;

    pos_loss.add(&neg_loss)?.mean_all()
}

/// Multiple Negatives Ranking Loss (MNRL).
///
/// Like InfoNCE but with additional explicit negatives beyond in-batch negatives.
/// `anchors` and `positives` are `[batch, dim]`. `negatives` is `[n_neg, dim]`.
pub fn mnrl_loss(
    anchors: &Tensor,
    positives: &Tensor,
    negatives: &Tensor,
    temperature: f32,
) -> Result<Tensor> {
    let batch = anchors.dim(0)?;

    // Normalize all
    let norm = |t: &Tensor| -> Result<Tensor> {
        let n = t.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX as f64)?;
        t.broadcast_div(&n.unsqueeze(1)?)
    };
    let a_n = norm(anchors)?;
    let p_n = norm(positives)?;
    let n_n = norm(negatives)?;

    // Similarities: [batch, batch+n_neg]
    let sim_pos = a_n.matmul(&p_n.t()?)?;
    let sim_neg = a_n.matmul(&n_n.t()?)?;
    let sims = Tensor::cat(&[&sim_pos, &sim_neg], 1)?.affine(1.0 / temperature as f64, 0.0)?;

    // Cross-entropy with target = diagonal of positive block
    let lse = log_sum_exp(&sims, 1)?;
    let indices = Tensor::arange(0u32, batch as u32, sims.device())?
        .to_dtype(candle_core::DType::U32)?
        .unsqueeze(1)?;
    let diag = sims.gather(&indices, 1)?.squeeze(1)?;
    lse.sub(&diag)?.mean_all()
}

/// Matryoshka loss wrapper for candle tensors.
///
/// Computes the inner loss at multiple truncated dimensions and sums weighted results.
pub fn matryoshka_loss<F>(
    loss_fn: F,
    embeddings_a: &Tensor,
    embeddings_b: &Tensor,
    dims: &[usize],
    weights: &[f32],
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
{
    assert_eq!(dims.len(), weights.len());
    let device = embeddings_a.device();
    let mut total = Tensor::zeros(&[], candle_core::DType::F32, device)?;
    for (idx, &trunc_dim) in dims.iter().enumerate() {
        let a_trunc = embeddings_a.narrow(1, 0, trunc_dim)?;
        let b_trunc = embeddings_b.narrow(1, 0, trunc_dim)?;
        let loss = loss_fn(&a_trunc, &b_trunc)?;
        total = total.add(&loss.affine(weights[idx] as f64, 0.0)?)?;
    }
    Ok(total)
}

/// ArcFace loss (Deng et al. 2019).
///
/// Angular margin penalty for classification-to-embedding training.
/// `embeddings`: `[batch, dim]`, `labels`: `[batch]` (u32 class indices),
/// `proxies`: `[num_classes, dim]` (learnable class centroids).
pub fn arcface_loss(
    embeddings: &Tensor,
    labels: &Tensor,
    proxies: &Tensor,
    scale: f32,
    margin: f32,
) -> Result<Tensor> {
    let batch = embeddings.dim(0)?;

    // L2-normalize embeddings and proxies
    let norm_e = embeddings
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let e_normed = embeddings.broadcast_div(&norm_e.unsqueeze(1)?)?;
    let norm_p = proxies
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let p_normed = proxies.broadcast_div(&norm_p.unsqueeze(1)?)?;

    // Cosine logits: [batch, num_classes]
    let logits = e_normed.matmul(&p_normed.t()?)?;

    // Extract target cosines and apply angular margin
    let target_idx = labels.to_dtype(candle_core::DType::U32)?.unsqueeze(1)?;
    let cos_target = logits.gather(&target_idx, 1)?.squeeze(1)?;

    // cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
    let cos_m = Tensor::new(&[margin.cos()], logits.device())?;
    let sin_m = Tensor::new(&[margin.sin()], logits.device())?;
    let sin_target = cos_target
        .sqr()?
        .affine(-1.0, 1.0)?
        .clamp(0.0, f64::MAX)?
        .sqrt()?;
    let cos_target_margin = cos_target
        .broadcast_mul(&cos_m)?
        .sub(&sin_target.broadcast_mul(&sin_m)?)?;

    // Replace target logits with margined version
    // One-hot scatter: create one-hot, multiply by (margined - original), add to logits
    let num_classes = proxies.dim(0)?;
    let one_hot = {
        let zeros = Tensor::zeros(
            &[batch, num_classes],
            candle_core::DType::F32,
            logits.device(),
        )?;
        let ones = Tensor::ones(&[batch, 1], candle_core::DType::F32, logits.device())?;
        zeros.scatter_add(&target_idx, &ones, 1)?
    };
    let diff = cos_target_margin.sub(&cos_target)?.unsqueeze(1)?;
    let logits = logits.add(&one_hot.broadcast_mul(&diff)?)?;

    // Scale + cross-entropy
    let scaled = logits.affine(scale as f64, 0.0)?;
    let lse = log_sum_exp(&scaled, 1)?;
    let target_scaled = scaled.gather(&target_idx, 1)?.squeeze(1)?;
    lse.sub(&target_scaled)?.mean_all()
}

/// VICReg loss (Bardes, Ponce, LeCun 2022).
///
/// Self-supervised loss: invariance (MSE) + variance (hinge on std) + covariance (decorrelation).
/// `view_a` and `view_b` are `[batch, dim]` embeddings of the same inputs under two augmentations.
pub fn vicreg_loss(
    view_a: &Tensor,
    view_b: &Tensor,
    lambda_inv: f32,
    mu_var: f32,
    nu_cov: f32,
) -> Result<Tensor> {
    let n = view_a.dim(0)? as f64;
    let dim = view_a.dim(1)? as f64;

    // Invariance: MSE between paired views
    let inv_loss = view_a.sub(view_b)?.sqr()?.mean_all()?;

    // Center both views
    let mean_a = view_a.mean(0)?;
    let mean_b = view_b.mean(0)?;
    let a_centered = view_a.broadcast_sub(&mean_a)?;
    let b_centered = view_b.broadcast_sub(&mean_b)?;

    // Variance: hinge on std per dimension, target std >= 1
    let var_loss = |centered: &Tensor| -> Result<Tensor> {
        let var = centered.sqr()?.mean(0)?;
        let std = var.affine(1.0, 1e-4)?.sqrt()?;
        std.affine(-1.0, 1.0)?.relu()?.mean_all()
    };
    let var_a = var_loss(&a_centered)?;
    let var_b = var_loss(&b_centered)?;
    let var_total = var_a.add(&var_b)?;

    // Covariance: penalize off-diagonal of covariance matrix
    let cov_loss = |centered: &Tensor| -> Result<Tensor> {
        // cov = (centered^T centered) / n
        let cov = centered.t()?.matmul(centered)?.affine(1.0 / n, 0.0)?;
        // Zero the diagonal, square, sum, normalize by dim
        let eye = Tensor::eye(centered.dim(1)?, candle_core::DType::F32, centered.device())?;
        let off_diag = cov.sub(&cov.mul(&eye)?)?;
        off_diag.sqr()?.sum_all()?.affine(1.0 / dim, 0.0)
    };
    let cov_a = cov_loss(&a_centered)?;
    let cov_b = cov_loss(&b_centered)?;
    let cov_total = cov_a.add(&cov_b)?;

    // Weighted sum
    let total = inv_loss
        .affine(lambda_inv as f64, 0.0)?
        .add(&var_total.affine(mu_var as f64, 0.0)?)?
        .add(&cov_total.affine(nu_cov as f64, 0.0)?)?;
    Ok(total)
}

/// Proxy-Anchor Loss (Kim et al. 2020).
///
/// `embeddings`: `[batch, dim]`, `labels`: `[batch]` u32, `proxies`: `[num_classes, dim]`.
pub fn proxy_anchor_loss(
    embeddings: &Tensor,
    labels: &Tensor,
    proxies: &Tensor,
    margin: f32,
    alpha: f32,
) -> Result<Tensor> {
    let n = embeddings.dim(0)?;
    let num_classes = proxies.dim(0)?;
    let device = embeddings.device();

    // L2-normalize
    let norm_e = embeddings
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let e_normed = embeddings.broadcast_div(&norm_e.unsqueeze(1)?)?;
    let norm_p = proxies
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-8, f32::MAX as f64)?;
    let p_normed = proxies.broadcast_div(&norm_p.unsqueeze(1)?)?;

    // Cosine similarity: [num_classes, batch]
    let sims = p_normed.matmul(&e_normed.t()?)?;

    // Build class masks: [num_classes, batch]
    let labels_expanded = labels
        .to_dtype(candle_core::DType::U32)?
        .unsqueeze(0)?
        .expand(&[num_classes, n])?;
    let class_indices = Tensor::arange(0u32, num_classes as u32, device)?
        .unsqueeze(1)?
        .expand(&[num_classes, n])?;
    let pos_mask = class_indices
        .eq(&labels_expanded)?
        .to_dtype(candle_core::DType::F32)?;
    let neg_mask = pos_mask.affine(-1.0, 1.0)?;

    // Positive term per proxy: softplus(-alpha * lse(alpha * (s - margin)) over pos)
    let pos_logits = sims.affine(alpha as f64, -(alpha as f64 * margin as f64))?;
    let masked_pos = pos_logits.broadcast_add(
        &neg_mask.affine(-1e9, 0.0)?, // mask negatives to -inf
    )?;
    let lse_pos = log_sum_exp(&masked_pos, 1)?;
    let sp_pos = lse_pos
        .neg()?
        .relu()?
        .add(&lse_pos.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;

    // Negative term per proxy: softplus(alpha * lse(alpha * (s + margin)) over neg)
    let neg_logits = sims.affine(alpha as f64, alpha as f64 * margin as f64)?;
    let masked_neg = neg_logits.broadcast_add(
        &pos_mask.affine(-1e9, 0.0)?, // mask positives to -inf
    )?;
    let lse_neg = log_sum_exp(&masked_neg, 1)?;
    let sp_neg = lse_neg
        .relu()?
        .add(&lse_neg.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;

    sp_pos.add(&sp_neg)?.mean_all()
}

/// Lifted Structured Loss (Song et al. 2016).
///
/// `embeddings`: `[n, dim]`, `labels`: `[n]` integer labels.
pub fn lifted_structured_loss(embeddings: &Tensor, labels: &Tensor, margin: f32) -> Result<Tensor> {
    let n = embeddings.dim(0)?;
    let device = embeddings.device();

    // Pairwise Euclidean distances
    let diff = embeddings
        .unsqueeze(1)?
        .broadcast_sub(&embeddings.unsqueeze(0)?)?;
    let dists = diff.sqr()?.sum(2)?.sqrt()?;

    // Masks
    let labels_col = labels.unsqueeze(1)?.expand(&[n, n])?;
    let labels_row = labels.unsqueeze(0)?.expand(&[n, n])?;
    let pos_mask = labels_col
        .eq(&labels_row)?
        .to_dtype(candle_core::DType::F32)?;
    let eye = Tensor::eye(n, candle_core::DType::F32, device)?;
    let pos_mask_no_diag = pos_mask.mul(&eye.affine(-1.0, 1.0)?)?;
    // For each positive pair (i,j): J = lse(margin - d_ik for neg k of i) + lse(margin - d_jl for neg l of j) + d_ij
    // loss = 0.5 * mean(max(0, J)^2) over positive pairs

    // margin - dists, masked for negatives (positives and self -> -inf)
    let neg_terms = dists.affine(-1.0, margin as f64)?;
    let masked_neg = neg_terms
        .broadcast_add(&pos_mask.affine(-1e9, 0.0)?)?
        .broadcast_add(&eye.affine(-1e9, 0.0)?)?;
    let lse_per_anchor = log_sum_exp(&masked_neg, 1)?;

    // For each positive pair (i,j): J_ij = lse_i + lse_j + d_ij
    let j_matrix = lse_per_anchor
        .unsqueeze(1)?
        .broadcast_add(&lse_per_anchor.unsqueeze(0)?)?
        .add(&dists)?;

    // Mask to positive pairs only (upper triangle to avoid double counting)
    let mut upper_data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            upper_data[i * n + j] = 1.0;
        }
    }
    let upper_mask = Tensor::from_slice(&upper_data, &[n, n], device)?;
    let pair_mask = pos_mask_no_diag.mul(&upper_mask)?;

    // 0.5 * mean(relu(J)^2) over positive pairs
    let j_active = j_matrix.relu()?.sqr()?.mul(&pair_mask)?;
    let n_pairs = pair_mask.sum_all()?.clamp(1.0, f64::MAX)?;
    j_active.sum_all()?.div(&n_pairs)?.affine(0.5, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn tensor(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(data, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn test_triplet_loss_basic() {
        let anchors = tensor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let positives = tensor(&[0.9, 0.1, 0.0, 0.1, 0.9, 0.0], &[2, 3]);
        let negatives = tensor(&[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0], &[2, 3]);

        let loss = triplet_loss(&anchors, &positives, &negatives, 0.5).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val >= 0.0, "triplet loss should be non-negative, got {val}");
    }

    #[test]
    fn test_infonce_loss_basic() {
        let a = tensor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let b = tensor(&[0.9, 0.1, 0.0, 0.1, 0.9, 0.0], &[2, 3]);

        let loss = infonce_loss(&a, &b, 0.1).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val > 0.0, "infonce loss should be > 0, got {val}");
        assert!(val.is_finite(), "loss should be finite");
    }

    #[test]
    fn test_contrastive_loss_basic() {
        let a = tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = tensor(&[0.9, 0.1, -1.0, 0.0], &[2, 2]);
        let labels = tensor(&[1.0, 0.0], &[2]);

        let loss = contrastive_loss(&a, &b, &labels, 1.0).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(
            val >= 0.0,
            "contrastive loss should be non-negative, got {val}"
        );
    }

    #[test]
    fn test_supcon_loss_basic() {
        let emb = tensor(&[1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9], &[4, 2]);
        let labels = Tensor::from_slice(&[0u32, 0, 1, 1], &[4], &Device::Cpu).unwrap();

        let loss = supcon_loss(&emb, &labels, 0.1).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val.is_finite(), "supcon loss should be finite, got {val}");
    }

    #[test]
    fn test_multi_similarity_loss_basic() {
        let emb = tensor(&[1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9], &[4, 2]);
        let labels = Tensor::from_slice(&[0u32, 0, 1, 1], &[4], &Device::Cpu).unwrap();

        let loss = multi_similarity_loss(&emb, &labels, 2.0, 50.0, 0.5).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val.is_finite(), "ms loss should be finite, got {val}");
    }

    #[test]
    fn test_circle_loss_basic() {
        let emb = tensor(&[1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9], &[4, 2]);
        let labels = Tensor::from_slice(&[0u32, 0, 1, 1], &[4], &Device::Cpu).unwrap();

        let loss = circle_loss(&emb, &labels, 0.25, 64.0).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val >= 0.0, "circle loss should be non-negative, got {val}");
        assert!(val.is_finite(), "circle loss should be finite");
    }

    #[test]
    fn test_n_pairs_loss_basic() {
        let a = tensor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let p = tensor(&[0.9, 0.1, 0.0, 0.1, 0.9, 0.0], &[2, 3]);

        let loss = n_pairs_loss(&a, &p, 0.1).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val > 0.0, "n_pairs loss should be > 0, got {val}");
    }

    #[test]
    fn test_infonce_backward() {
        let a_var =
            candle_core::Var::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
        let a = a_var.as_tensor();
        let b = tensor(&[0.9, 0.1, 0.1, 0.9], &[2, 2]);

        let loss = infonce_loss(a, &b, 0.1).unwrap();
        let grads = loss.backward().unwrap();
        let grad_a = grads.get(a).unwrap();
        let grad_sum: f32 = grad_a
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|x| x.abs())
            .sum();
        assert!(
            grad_sum > 1e-6,
            "gradients should be non-zero, got sum={grad_sum}"
        );
    }

    /// Helper to check candle backward produces non-zero gradients.
    fn check_backward(name: &str, loss: &Tensor, var: &candle_core::Var) {
        let grads = loss.backward().unwrap();
        let grad = grads.get(var.as_tensor()).unwrap();
        let grad_sum: f32 = grad
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|x| x.abs())
            .sum();
        assert!(
            grad_sum > 1e-6,
            "{name}: gradients should be non-zero, got sum={grad_sum}"
        );
    }

    #[test]
    fn test_supcon_backward() {
        let var = candle_core::Var::from_slice(
            &[1.0f32, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9],
            &[4, 2],
            &Device::Cpu,
        )
        .unwrap();
        let labels = Tensor::from_slice(&[0u32, 0, 1, 1], &[4], &Device::Cpu).unwrap();
        let loss = supcon_loss(var.as_tensor(), &labels, 0.1).unwrap();
        check_backward("supcon", &loss, &var);
    }

    #[test]
    fn test_multi_similarity_backward() {
        let var = candle_core::Var::from_slice(
            &[1.0f32, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9],
            &[4, 2],
            &Device::Cpu,
        )
        .unwrap();
        let labels = Tensor::from_slice(&[0u32, 0, 1, 1], &[4], &Device::Cpu).unwrap();
        let loss = multi_similarity_loss(var.as_tensor(), &labels, 2.0, 50.0, 0.5).unwrap();
        check_backward("multi_similarity", &loss, &var);
    }

    #[test]
    fn test_circle_backward() {
        let var = candle_core::Var::from_slice(
            &[1.0f32, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9],
            &[4, 2],
            &Device::Cpu,
        )
        .unwrap();
        let labels = Tensor::from_slice(&[0u32, 0, 1, 1], &[4], &Device::Cpu).unwrap();
        let loss = circle_loss(var.as_tensor(), &labels, 0.25, 64.0).unwrap();
        check_backward("circle", &loss, &var);
    }

    #[test]
    fn test_cross_backend_infonce() {
        // Verify f32 and candle produce the same loss value for identical inputs
        let a_data = [0.7f32, 0.3, -0.2, -0.3, 0.8, 0.1];
        let p_data = [0.5f32, 0.4, -0.1, -0.2, 0.7, 0.3];
        let temperature = 0.1;

        // f32 path
        let a_refs: Vec<&[f32]> = vec![&a_data[0..3], &a_data[3..6]];
        let p_refs: Vec<&[f32]> = vec![&p_data[0..3], &p_data[3..6]];
        let f32_loss = crate::losses::infonce_loss(&a_refs, &p_refs, temperature).loss;

        // candle path
        let a_tensor = tensor(&a_data, &[2, 3]);
        let p_tensor = tensor(&p_data, &[2, 3]);
        let candle_loss = infonce_loss(&a_tensor, &p_tensor, temperature)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        let rel_err = (f32_loss - candle_loss).abs() / f32_loss.abs().max(1e-6);
        assert!(
            rel_err < 1e-3,
            "cross-backend infonce: f32={f32_loss}, candle={candle_loss}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_cross_backend_triplet() {
        let a_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let p_data = [0.9f32, 0.1, 0.0, 0.1, 0.9, 0.0];
        let n_data = [-1.0f32, 0.0, 0.0, 0.0, -1.0, 0.0];
        let margin = 0.5;

        let a_refs: Vec<&[f32]> = vec![&a_data[0..3], &a_data[3..6]];
        let p_refs: Vec<&[f32]> = vec![&p_data[0..3], &p_data[3..6]];
        let n_refs: Vec<&[f32]> = vec![&n_data[0..3], &n_data[3..6]];
        let f32_loss = crate::losses::triplet_loss(&a_refs, &p_refs, &n_refs, margin).loss;

        let candle_loss = triplet_loss(
            &tensor(&a_data, &[2, 3]),
            &tensor(&p_data, &[2, 3]),
            &tensor(&n_data, &[2, 3]),
            margin,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

        let rel_err = (f32_loss - candle_loss).abs() / f32_loss.abs().max(1e-6);
        assert!(
            rel_err < 1e-3,
            "cross-backend triplet: f32={f32_loss}, candle={candle_loss}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_cross_backend_n_pairs() {
        let a_data = [0.7f32, 0.3, -0.2, -0.3, 0.8, 0.1];
        let p_data = [0.5f32, 0.4, -0.1, -0.2, 0.7, 0.3];
        let temperature = 0.5;

        let a_refs: Vec<&[f32]> = vec![&a_data[0..3], &a_data[3..6]];
        let p_refs: Vec<&[f32]> = vec![&p_data[0..3], &p_data[3..6]];
        let f32_loss = crate::losses::n_pairs_loss(&a_refs, &p_refs, temperature).loss;

        let candle_loss = n_pairs_loss(
            &tensor(&a_data, &[2, 3]),
            &tensor(&p_data, &[2, 3]),
            temperature,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

        let rel_err = (f32_loss - candle_loss).abs() / f32_loss.abs().max(1e-6);
        assert!(
            rel_err < 1e-3,
            "cross-backend n_pairs: f32={f32_loss}, candle={candle_loss}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_mnrl_loss_basic() {
        let a = tensor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let p = tensor(&[0.9, 0.1, 0.0, 0.1, 0.9, 0.0], &[2, 3]);
        let n = tensor(&[-1.0, 0.0, 0.0], &[1, 3]);

        let loss = mnrl_loss(&a, &p, &n, 0.1).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val > 0.0 && val.is_finite(), "mnrl loss: {val}");
    }

    #[test]
    fn test_matryoshka_wrapper() {
        let a = tensor(&[1.0, 0.0, 0.5, 0.2, 0.0, 1.0, 0.3, 0.1], &[2, 4]);
        let b = tensor(&[0.9, 0.1, 0.4, 0.3, 0.1, 0.9, 0.2, 0.2], &[2, 4]);

        let loss =
            matryoshka_loss(|a, b| infonce_loss(a, b, 0.1), &a, &b, &[2, 4], &[1.0, 1.0]).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val.is_finite(), "matryoshka loss: {val}");
    }

    #[test]
    fn test_arcface_basic() {
        // Non-aligned embeddings so gradients are non-trivial
        let emb = tensor(&[0.7, 0.3, 0.2, 0.8], &[2, 2]);
        let proxies = tensor(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], &[3, 2]);
        let labels = Tensor::from_slice(&[0u32, 1], &[2], &Device::Cpu).unwrap();

        let loss = arcface_loss(&emb, &labels, &proxies, 10.0, 0.3).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val >= 0.0 && val.is_finite(), "arcface loss: {val}");
    }

    #[test]
    fn test_arcface_backward() {
        let var =
            candle_core::Var::from_slice(&[0.7f32, 0.3, 0.2, 0.8], &[2, 2], &Device::Cpu).unwrap();
        let proxies = tensor(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], &[3, 2]);
        let labels = Tensor::from_slice(&[0u32, 1], &[2], &Device::Cpu).unwrap();

        let loss = arcface_loss(var.as_tensor(), &labels, &proxies, 10.0, 0.3).unwrap();
        check_backward("arcface", &loss, &var);
    }

    #[test]
    fn test_vicreg_basic() {
        let a = tensor(&[1.0, 0.0, 0.5, 0.5, 0.0, 1.0], &[3, 2]);
        let b = tensor(&[0.9, 0.1, 0.4, 0.6, 0.1, 0.9], &[3, 2]);

        let loss = vicreg_loss(&a, &b, 25.0, 25.0, 1.0).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val >= 0.0 && val.is_finite(), "vicreg loss: {val}");
    }

    #[test]
    fn test_vicreg_backward() {
        let var =
            candle_core::Var::from_slice(&[1.0f32, 0.0, 0.5, 0.5, 0.0, 1.0], &[3, 2], &Device::Cpu)
                .unwrap();
        let b = tensor(&[0.9, 0.1, 0.4, 0.6, 0.1, 0.9], &[3, 2]);

        let loss = vicreg_loss(var.as_tensor(), &b, 25.0, 25.0, 1.0).unwrap();
        check_backward("vicreg", &loss, &var);
    }

    #[test]
    fn test_proxy_anchor_basic() {
        let emb = tensor(&[0.7, 0.3, 0.2, 0.8], &[2, 2]);
        let proxies = tensor(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], &[3, 2]);
        let labels = Tensor::from_slice(&[0u32, 1], &[2], &Device::Cpu).unwrap();

        let loss = proxy_anchor_loss(&emb, &labels, &proxies, 0.1, 32.0).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val.is_finite(), "proxy_anchor loss: {val}");
    }

    #[test]
    fn test_proxy_anchor_backward() {
        let var =
            candle_core::Var::from_slice(&[0.7f32, 0.3, 0.2, 0.8], &[2, 2], &Device::Cpu).unwrap();
        let proxies = tensor(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], &[3, 2]);
        let labels = Tensor::from_slice(&[0u32, 1], &[2], &Device::Cpu).unwrap();

        let loss = proxy_anchor_loss(var.as_tensor(), &labels, &proxies, 0.1, 32.0).unwrap();
        check_backward("proxy_anchor", &loss, &var);
    }

    #[test]
    fn test_lifted_structured_basic() {
        let emb = tensor(&[1.0, 0.5, 0.9, 0.6, 0.0, 1.0, 0.1, 0.9], &[4, 2]);
        let labels = Tensor::from_slice(&[0u32, 0, 1, 1], &[4], &Device::Cpu).unwrap();

        let loss = lifted_structured_loss(&emb, &labels, 1.0).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val >= 0.0 && val.is_finite(), "lifted loss: {val}");
    }
}
