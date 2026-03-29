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
    // softplus: log(1 + exp(x)), use the identity: softplus(x) = x + log(1 + exp(-x))
    // for numerical stability when x is large
    let softplus = combined
        .neg()?
        .exp()?
        .affine(1.0, 1.0)?
        .log()?
        .add(&combined)?;

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
}
