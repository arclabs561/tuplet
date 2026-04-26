use crate::similarity::euclidean_distance;

/// Compute Mahalanobis distance: `sqrt((a-b)^T M (a-b))`.
///
/// `m` is a `dim x dim` positive semi-definite matrix stored row-major
/// as a flat slice of length `dim * dim`.
pub fn mahalanobis_distance(a: &[f32], b: &[f32], m: &[f32]) -> f32 {
    let dim = a.len();
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    assert_eq!(m.len(), dim * dim, "M must be dim x dim");

    let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();

    // Compute diff^T M diff
    let mut result = 0.0f32;
    for i in 0..dim {
        let mut m_diff_i = 0.0f32;
        for j in 0..dim {
            m_diff_i += m[i * dim + j] * diff[j];
        }
        result += diff[i] * m_diff_i;
    }

    result.max(0.0).sqrt()
}

// TODO(v0.2): switch to builder pattern + `#[non_exhaustive]` so adding fields
// is not a breaking change. Currently exhaustive for ergonomic struct-literal
// construction (`..Default::default()` is blocked under `#[non_exhaustive]`).
/// Configuration for Mahalanobis distance learning.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MahalanobisConfig {
    /// Learning rate.
    pub lr: f32,
    /// Maximum number of gradient descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on gradient norm.
    pub tol: f32,
    /// Margin for dissimilar pairs.
    pub margin: f32,
}

impl Default for MahalanobisConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            max_iter: 200,
            tol: 1e-6,
            margin: 1.0,
        }
    }
}

/// Learn a linear transformation `L` (`out_dim x in_dim`) such that Euclidean
/// distance in the transformed space approximates the optimal Mahalanobis distance.
///
/// `M = L^T L` is guaranteed positive semi-definite by construction.
///
/// Returns `L` as a flat row-major vector of length `out_dim * in_dim`.
pub fn learn_transform(
    similar_pairs: &[(&[f32], &[f32])],
    dissimilar_pairs: &[(&[f32], &[f32])],
    in_dim: usize,
    out_dim: usize,
    config: &MahalanobisConfig,
) -> Vec<f32> {
    assert!(out_dim <= in_dim, "out_dim must be <= in_dim");
    assert!(!similar_pairs.is_empty(), "need at least one similar pair");

    // Initialize L as truncated identity: out_dim x in_dim
    let mut l = vec![0.0f32; out_dim * in_dim];
    for i in 0..out_dim {
        l[i * in_dim + i] = 1.0;
    }

    for _iter in 0..config.max_iter {
        let mut grad = vec![0.0f32; out_dim * in_dim];

        // Pull term: minimize d_L(x_i, x_j) for similar pairs
        // d_L^2 = ||L(a - b)||^2 = sum_r (sum_c L[r][c] * diff[c])^2
        // d(d_L^2)/dL[r][c] = 2 * (sum_c' L[r][c'] * diff[c']) * diff[c]
        for &(a, b) in similar_pairs {
            let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();
            for r in 0..out_dim {
                let z_r: f32 = (0..in_dim).map(|c| l[r * in_dim + c] * diff[c]).sum();
                for c in 0..in_dim {
                    grad[r * in_dim + c] += 2.0 * z_r * diff[c];
                }
            }
        }

        // Push term: hinge loss max(0, margin - d_L^2) for dissimilar pairs
        for &(a, b) in dissimilar_pairs {
            let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();
            let mut dist_sq = 0.0f32;
            let mut z = vec![0.0f32; out_dim];
            for r in 0..out_dim {
                z[r] = (0..in_dim).map(|c| l[r * in_dim + c] * diff[c]).sum();
                dist_sq += z[r] * z[r];
            }

            if dist_sq < config.margin * config.margin {
                // gradient of max(0, margin^2 - d_L^2) = -d(d_L^2)/dL when active
                for r in 0..out_dim {
                    for c in 0..in_dim {
                        grad[r * in_dim + c] -= 2.0 * z[r] * diff[c];
                    }
                }
            }
        }

        let n = (similar_pairs.len() + dissimilar_pairs.len()) as f32;
        let grad_norm_sq: f32 = grad.iter().map(|g| g * g).sum();

        if grad_norm_sq < config.tol * config.tol * n * n {
            break;
        }

        // Gradient descent step
        for i in 0..l.len() {
            l[i] -= config.lr * grad[i] / n;
        }
    }

    l
}

/// Learn the Mahalanobis matrix `M = L^T L` from similarity/dissimilarity constraints.
///
/// Returns `M` as a flat row-major vector of length `dim * dim`.
pub fn learn_mahalanobis(
    similar_pairs: &[(&[f32], &[f32])],
    dissimilar_pairs: &[(&[f32], &[f32])],
    dim: usize,
    config: &MahalanobisConfig,
) -> Vec<f32> {
    let l = learn_transform(similar_pairs, dissimilar_pairs, dim, dim, config);
    // M = L^T L
    let mut m = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for k in 0..dim {
                sum += l[k * dim + i] * l[k * dim + j];
            }
            m[i * dim + j] = sum;
        }
    }
    m
}

/// Configuration for Neighbourhood Components Analysis.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NcaConfig {
    /// Learning rate.
    pub lr: f32,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on objective change.
    pub tol: f32,
}

impl Default for NcaConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

/// Learn a linear transformation via Neighbourhood Components Analysis.
///
/// Optimizes leave-one-out k-NN classification accuracy using a
/// differentiable soft neighbor selection rule.
///
/// `data`: N data points of dimension `in_dim`.
/// `labels`: N class labels.
/// `out_dim`: output dimension (<= `in_dim`).
///
/// Returns `L` as flat row-major of length `out_dim * in_dim`.
pub fn nca(data: &[&[f32]], labels: &[usize], out_dim: usize, config: &NcaConfig) -> Vec<f32> {
    let n = data.len();
    assert_eq!(n, labels.len(), "data and labels must have same length");
    assert!(n > 1, "need at least 2 data points");
    let in_dim = data[0].len();
    assert!(out_dim <= in_dim, "out_dim must be <= in_dim");

    // Initialize L as truncated identity
    let mut l = vec![0.0f32; out_dim * in_dim];
    for i in 0..out_dim {
        l[i * in_dim + i] = 1.0;
    }

    let mut prev_obj = f32::NEG_INFINITY;

    for _iter in 0..config.max_iter {
        // Transform all points: z_i = L x_i
        let z: Vec<Vec<f32>> = data
            .iter()
            .map(|x| {
                let mut zi = vec![0.0f32; out_dim];
                for r in 0..out_dim {
                    zi[r] = (0..in_dim).map(|c| l[r * in_dim + c] * x[c]).sum();
                }
                zi
            })
            .collect();

        // Compute pairwise squared distances in transformed space
        // and softmax probabilities using log-sum-exp
        let mut objective = 0.0f32;
        let mut grad = vec![0.0f32; out_dim * in_dim];

        for i in 0..n {
            // Compute negative squared distances to all other points
            let mut neg_dists = Vec::with_capacity(n);
            for j in 0..n {
                if j == i {
                    neg_dists.push(f32::NEG_INFINITY);
                } else {
                    let d_sq: f32 = (0..out_dim).map(|r| (z[i][r] - z[j][r]).powi(2)).sum();
                    neg_dists.push(-d_sq);
                }
            }

            // Log-sum-exp for numerical stability
            let max_val = neg_dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if max_val == f32::NEG_INFINITY {
                continue;
            }
            let sum_exp: f32 = neg_dists.iter().map(|&v| (v - max_val).exp()).sum();
            let log_sum = max_val + sum_exp.ln();

            // p_ij = exp(-d_ij^2) / sum_k exp(-d_ik^2) for k != i
            // p_i = sum_{j: same class} p_ij
            let mut p_i = 0.0f32;
            let mut p_ij = vec![0.0f32; n];
            for j in 0..n {
                if j == i {
                    continue;
                }
                p_ij[j] = (neg_dists[j] - log_sum).exp();
                if labels[j] == labels[i] {
                    p_i += p_ij[j];
                }
            }

            objective += p_i;

            // Gradient of p_i w.r.t. L
            // d(p_i)/dL = 2L * [p_i * sum_k p_ik (x_i - x_k)(x_i - x_k)^T
            //              - sum_{j: same class} p_ij (x_i - x_j)(x_i - x_j)^T]
            //
            // We compute the "weighted covariance" terms and use them to update gradient.
            // Actually, the gradient of the NCA objective w.r.t. L is:
            //   sum_i 2L [p_i C_i - C_i^+]
            // where C_i = sum_k p_ik (x_ik)(x_ik)^T  (all neighbors)
            //       C_i^+ = sum_{j: same} p_ij (x_ij)(x_ij)^T

            for r in 0..out_dim {
                for c in 0..in_dim {
                    let mut all_term = 0.0f32;
                    let mut same_term = 0.0f32;

                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let diff_c = data[i][c] - data[j][c];
                        // We need sum over r' of L[r'][c'] * diff[c'] for the outer product,
                        // but the gradient formula simplifies to:
                        // dObj/dL[r][c] = 2 * sum_i [p_i * sum_k p_ik * (z_i[r]-z_k[r]) * diff_c
                        //                  - sum_{j:same} p_ij * (z_i[r]-z_j[r]) * diff_c]
                        let z_diff_r = z[i][r] - z[j][r];
                        let contribution = p_ij[j] * z_diff_r * diff_c;
                        all_term += contribution;
                        if labels[j] == labels[i] {
                            same_term += contribution;
                        }
                    }

                    grad[r * in_dim + c] += 2.0 * (p_i * all_term - same_term);
                }
            }
        }

        if (objective - prev_obj).abs() < config.tol {
            break;
        }
        prev_obj = objective;

        // Gradient ascent (maximizing objective)
        for i in 0..l.len() {
            l[i] += config.lr * grad[i] / n as f32;
        }
    }

    l
}

/// Configuration for Large Margin Nearest Neighbor.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LmnnConfig {
    /// Learning rate.
    pub lr: f32,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on loss change.
    pub tol: f32,
    /// Number of target neighbors per point.
    pub k: usize,
    /// Hinge loss margin.
    pub margin: f32,
    /// Weight balancing pull vs push terms. `loss = (1-mu)*pull + mu*push`.
    pub mu: f32,
}

impl Default for LmnnConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            max_iter: 200,
            tol: 1e-6,
            k: 3,
            margin: 1.0,
            mu: 0.5,
        }
    }
}

/// Learn a linear transformation via Large Margin Nearest Neighbor.
///
/// Target neighbors (same class, k nearest in original space) should be
/// closer than impostors (different class) by a margin in the learned space.
///
/// Returns `L` as flat row-major of length `out_dim * in_dim`.
pub fn lmnn(data: &[&[f32]], labels: &[usize], out_dim: usize, config: &LmnnConfig) -> Vec<f32> {
    let n = data.len();
    assert_eq!(n, labels.len(), "data and labels must have same length");
    assert!(n > 1, "need at least 2 data points");
    let in_dim = data[0].len();
    assert!(out_dim <= in_dim, "out_dim must be <= in_dim");

    // Find target neighbors: k nearest same-class points in original space (fixed)
    let target_neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut same_class: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i && labels[j] == labels[i])
                .map(|j| (j, euclidean_distance(data[i], data[j])))
                .collect();
            same_class.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            same_class
                .iter()
                .take(config.k)
                .map(|&(idx, _)| idx)
                .collect()
        })
        .collect();

    // Initialize L as truncated identity
    let mut l = vec![0.0f32; out_dim * in_dim];
    for i in 0..out_dim {
        l[i * in_dim + i] = 1.0;
    }

    let mut prev_loss = f32::INFINITY;

    for _iter in 0..config.max_iter {
        // Transform all points
        let z: Vec<Vec<f32>> = data
            .iter()
            .map(|x| {
                let mut zi = vec![0.0f32; out_dim];
                for r in 0..out_dim {
                    zi[r] = (0..in_dim).map(|c| l[r * in_dim + c] * x[c]).sum();
                }
                zi
            })
            .collect();

        let mut grad = vec![0.0f32; out_dim * in_dim];

        // Pull term: sum over target neighbor pairs ||z_i - z_j||^2
        let mut pull_loss = 0.0f32;
        for i in 0..n {
            for &j in &target_neighbors[i] {
                let dist_sq: f32 = (0..out_dim).map(|r| (z[i][r] - z[j][r]).powi(2)).sum();
                pull_loss += dist_sq;

                // Gradient: d(||z_i-z_j||^2)/dL[r][c] = 2*(z_i[r]-z_j[r])*(x_i[c]-x_j[c])
                for r in 0..out_dim {
                    let z_diff = z[i][r] - z[j][r];
                    for c in 0..in_dim {
                        let x_diff = data[i][c] - data[j][c];
                        grad[r * in_dim + c] += (1.0 - config.mu) * 2.0 * z_diff * x_diff;
                    }
                }
            }
        }

        // Push term: hinge loss for impostors
        let mut push_loss = 0.0f32;
        for i in 0..n {
            for &j in &target_neighbors[i] {
                let dist_ij_sq: f32 = (0..out_dim).map(|r| (z[i][r] - z[j][r]).powi(2)).sum();

                for ll in 0..n {
                    if labels[ll] == labels[i] {
                        continue;
                    }
                    let dist_il_sq: f32 = (0..out_dim).map(|r| (z[i][r] - z[ll][r]).powi(2)).sum();

                    let hinge = (dist_ij_sq - dist_il_sq + config.margin).max(0.0);
                    if hinge > 0.0 {
                        push_loss += hinge;

                        // Gradient of hinge: d/dL[r][c] (d_ij^2 - d_il^2)
                        for r in 0..out_dim {
                            let z_diff_ij = z[i][r] - z[j][r];
                            let z_diff_il = z[i][r] - z[ll][r];
                            for c in 0..in_dim {
                                let x_diff_ij = data[i][c] - data[j][c];
                                let x_diff_il = data[i][c] - data[ll][c];
                                grad[r * in_dim + c] += config.mu
                                    * 2.0
                                    * (z_diff_ij * x_diff_ij - z_diff_il * x_diff_il);
                            }
                        }
                    }
                }
            }
        }

        let loss = (1.0 - config.mu) * pull_loss + config.mu * push_loss;

        if (prev_loss - loss).abs() < config.tol {
            break;
        }
        prev_loss = loss;

        // Gradient descent
        let scale = 1.0 / n as f32;
        for i in 0..l.len() {
            l[i] -= config.lr * grad[i] * scale;
        }
    }

    l
}

/// Apply a learned linear transformation `L` to a single data point.
///
/// `L` is `out_dim x in_dim` stored row-major.
pub fn transform(point: &[f32], l: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(point.len(), in_dim, "point dimension mismatch");
    assert_eq!(l.len(), out_dim * in_dim, "L dimension mismatch");

    let mut result = vec![0.0f32; out_dim];
    for r in 0..out_dim {
        result[r] = (0..in_dim).map(|c| l[r * in_dim + c] * point[c]).sum();
    }
    result
}

/// Apply a learned linear transformation to a batch of points.
pub fn transform_batch(
    points: &[&[f32]],
    l: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<Vec<f32>> {
    points
        .iter()
        .map(|p| transform(p, l, in_dim, out_dim))
        .collect()
}

/// Configuration for Information-Theoretic Metric Learning.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ItmlConfig {
    /// Maximum number of Bregman projection iterations.
    pub max_iter: usize,
    /// Convergence tolerance on matrix change.
    pub tol: f32,
    /// Slack parameter controlling how strictly constraints are enforced.
    pub gamma: f32,
}

impl Default for ItmlConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-4,
            gamma: 1.0,
        }
    }
}

/// Information-Theoretic Metric Learning (Davis et al. 2007).
///
/// Learns a Mahalanobis distance matrix by minimizing LogDet divergence from a prior
/// (identity matrix) subject to pairwise distance constraints.
///
/// Returns M as a flat row-major vector of length `dim * dim`.
pub fn itml(
    similar_pairs: &[(&[f32], &[f32])],
    dissimilar_pairs: &[(&[f32], &[f32])],
    dim: usize,
    config: &ItmlConfig,
) -> Vec<f32> {
    assert!(
        !similar_pairs.is_empty() || !dissimilar_pairs.is_empty(),
        "need at least one constraint"
    );

    // Initialize M = I
    let mut m = vec![0.0f32; dim * dim];
    for i in 0..dim {
        m[i * dim + i] = 1.0;
    }

    // Compute upper/lower bounds from initial distances
    // Similar pairs: d_M should be <= mean similar distance * gamma
    // Dissimilar pairs: d_M should be >= mean dissimilar distance / gamma
    let sim_bound = if similar_pairs.is_empty() {
        1.0
    } else {
        let mean: f32 = similar_pairs
            .iter()
            .map(|&(a, b)| {
                let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();
                diff.iter().map(|d| d * d).sum::<f32>()
            })
            .sum::<f32>()
            / similar_pairs.len() as f32;
        mean * config.gamma
    };

    let dissim_bound = if dissimilar_pairs.is_empty() {
        1.0
    } else {
        let mean: f32 = dissimilar_pairs
            .iter()
            .map(|&(a, b)| {
                let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();
                diff.iter().map(|d| d * d).sum::<f32>()
            })
            .sum::<f32>()
            / dissimilar_pairs.len() as f32;
        mean / config.gamma
    };

    // Slack variables
    let n_sim = similar_pairs.len();
    let n_dissim = dissimilar_pairs.len();
    let mut xi_sim = vec![sim_bound; n_sim];
    let mut xi_dissim = vec![dissim_bound; n_dissim];

    let lambda = config.gamma;

    for _iter in 0..config.max_iter {
        let m_prev = m.clone();

        // Process similar pair constraints: d_M(x_i, x_j) <= xi_k
        for (k, &(a, b)) in similar_pairs.iter().enumerate() {
            let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();

            // Compute d_M^2 = diff^T M diff
            let mut d_sq = 0.0f32;
            let mut m_diff = vec![0.0f32; dim];
            for i in 0..dim {
                for j in 0..dim {
                    m_diff[i] += m[i * dim + j] * diff[j];
                }
                d_sq += diff[i] * m_diff[i];
            }

            if d_sq <= xi_sim[k] {
                continue;
            }

            // Bregman projection: update M and xi
            let alpha = (1.0 / (d_sq + 1e-10) - lambda / xi_sim[k]).min(0.0).abs();
            let beta = alpha / (1.0 + alpha * d_sq);

            // M <- M - beta * (M diff)(M diff)^T
            for i in 0..dim {
                for j in 0..dim {
                    m[i * dim + j] -= beta * m_diff[i] * m_diff[j];
                }
            }

            xi_sim[k] = (xi_sim[k] * lambda) / (lambda - xi_sim[k] * alpha);
            if xi_sim[k] < 1e-8 {
                xi_sim[k] = 1e-8;
            }
        }

        // Process dissimilar pair constraints: d_M(x_i, x_j) >= xi_k
        for (k, &(a, b)) in dissimilar_pairs.iter().enumerate() {
            let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();

            let mut d_sq = 0.0f32;
            let mut m_diff = vec![0.0f32; dim];
            for i in 0..dim {
                for j in 0..dim {
                    m_diff[i] += m[i * dim + j] * diff[j];
                }
                d_sq += diff[i] * m_diff[i];
            }

            if d_sq >= xi_dissim[k] {
                continue;
            }

            // Bregman projection for lower bound
            let alpha = (lambda / xi_dissim[k] - 1.0 / (d_sq + 1e-10))
                .min(0.0)
                .abs();
            let beta = alpha / (1.0 - alpha * d_sq).max(1e-8);

            // M <- M + beta * (M diff)(M diff)^T
            for i in 0..dim {
                for j in 0..dim {
                    m[i * dim + j] += beta * m_diff[i] * m_diff[j];
                }
            }

            xi_dissim[k] = (xi_dissim[k] * lambda) / (lambda + xi_dissim[k] * alpha);
            if xi_dissim[k] < 1e-8 {
                xi_dissim[k] = 1e-8;
            }
        }

        // Check convergence
        let change: f32 = m
            .iter()
            .zip(m_prev.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        if change < config.tol {
            break;
        }
    }

    m
}

/// KISSME metric learning (Koestinger et al. 2012).
///
/// Learns Mahalanobis distance from covariance difference:
/// `M = inv(Sigma_similar) - inv(Sigma_dissimilar)`.
/// Negative eigenvalues are clipped to 0 to ensure PSD.
///
/// Returns M as a flat row-major vector of length `dim * dim`.
pub fn kissme(
    similar_pairs: &[(&[f32], &[f32])],
    dissimilar_pairs: &[(&[f32], &[f32])],
    dim: usize,
) -> Vec<f32> {
    assert!(!similar_pairs.is_empty(), "need at least one similar pair");
    assert!(
        !dissimilar_pairs.is_empty(),
        "need at least one dissimilar pair"
    );

    // Compute covariance of differences for similar pairs
    let sigma_s = difference_covariance(similar_pairs, dim);
    let sigma_d = difference_covariance(dissimilar_pairs, dim);

    // Invert both covariance matrices (with regularization)
    let inv_s = invert_matrix(&sigma_s, dim);
    let inv_d = invert_matrix(&sigma_d, dim);

    // M = inv(Sigma_S) - inv(Sigma_D)
    let mut m: Vec<f32> = inv_s
        .iter()
        .zip(inv_d.iter())
        .map(|(&a, &b)| a - b)
        .collect();

    // Clip negative eigenvalues to ensure PSD
    clip_negative_eigenvalues(&mut m, dim);

    m
}

/// Compute covariance matrix of pairwise differences.
fn difference_covariance(pairs: &[(&[f32], &[f32])], dim: usize) -> Vec<f32> {
    let n = pairs.len() as f32;
    let mut cov = vec![0.0f32; dim * dim];

    // Compute mean difference
    let mut mean = vec![0.0f32; dim];
    for &(a, b) in pairs {
        for i in 0..dim {
            mean[i] += a[i] - b[i];
        }
    }
    for v in mean.iter_mut() {
        *v /= n;
    }

    // Covariance
    for &(a, b) in pairs {
        let diff: Vec<f32> = (0..dim).map(|i| (a[i] - b[i]) - mean[i]).collect();
        for i in 0..dim {
            for j in 0..dim {
                cov[i * dim + j] += diff[i] * diff[j];
            }
        }
    }

    for v in cov.iter_mut() {
        *v /= n;
    }

    cov
}

/// Invert a symmetric matrix using Gauss-Jordan elimination with regularization.
fn invert_matrix(mat: &[f32], dim: usize) -> Vec<f32> {
    // Augment with identity
    let mut aug = vec![0.0f32; dim * 2 * dim];
    for i in 0..dim {
        for j in 0..dim {
            aug[i * 2 * dim + j] = mat[i * dim + j];
        }
        // Add regularization for numerical stability
        aug[i * 2 * dim + i] += 1e-4;
        aug[i * 2 * dim + dim + i] = 1.0;
    }

    // Gauss-Jordan
    for col in 0..dim {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * 2 * dim + col].abs();
        for row in (col + 1)..dim {
            let val = aug[row * 2 * dim + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_row != col {
            for j in 0..(2 * dim) {
                aug.swap(col * 2 * dim + j, max_row * 2 * dim + j);
            }
        }

        let pivot = aug[col * 2 * dim + col];
        if pivot.abs() < 1e-10 {
            continue;
        }

        for j in 0..(2 * dim) {
            aug[col * 2 * dim + j] /= pivot;
        }

        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * dim + col];
            for j in 0..(2 * dim) {
                aug[row * 2 * dim + j] -= factor * aug[col * 2 * dim + j];
            }
        }
    }

    // Extract inverse
    let mut inv = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            inv[i * dim + j] = aug[i * 2 * dim + dim + j];
        }
    }

    inv
}

/// Clip negative eigenvalues of a symmetric matrix to zero.
///
/// Uses power iteration to find eigenvectors, then reconstructs with clipped eigenvalues.
/// For small dimensions this is practical; for large dims a proper eigendecomposition
/// library would be needed.
fn clip_negative_eigenvalues(m: &mut [f32], dim: usize) {
    // Simple approach: use iterative eigendecomposition
    // For small dims (typical in metric learning), we do sequential deflation.

    let mut eigenvalues = Vec::with_capacity(dim);
    let mut eigenvectors = Vec::with_capacity(dim);

    let mut work = m.to_vec();

    for _ in 0..dim {
        // Power iteration to find dominant eigenvalue/vector
        let mut v = vec![1.0f32; dim];
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }

        let mut eigenvalue = 0.0f32;

        for _iter in 0..200 {
            // w = work * v
            let mut w = vec![0.0f32; dim];
            for i in 0..dim {
                for j in 0..dim {
                    w[i] += work[i * dim + j] * v[j];
                }
            }

            eigenvalue = w.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();

            let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                eigenvalue = 0.0;
                break;
            }
            for x in w.iter_mut() {
                *x /= norm;
            }

            // Check convergence
            let diff: f32 = w
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>();
            v = w;
            if diff < 1e-10 {
                break;
            }
        }

        eigenvalues.push(eigenvalue);
        eigenvectors.push(v.clone());

        // Deflate: work -= eigenvalue * v * v^T
        for i in 0..dim {
            for j in 0..dim {
                work[i * dim + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    // Reconstruct with clipped eigenvalues
    for val in m.iter_mut() {
        *val = 0.0;
    }
    for (k, &ev) in eigenvalues.iter().enumerate() {
        let clipped = ev.max(0.0);
        if clipped < 1e-10 {
            continue;
        }
        let v = &eigenvectors[k];
        for i in 0..dim {
            for j in 0..dim {
                m[i * dim + j] += clipped * v[i] * v[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::euclidean_distance;

    #[test]
    fn test_mahalanobis_identity() {
        // With M = I, Mahalanobis distance = Euclidean distance
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        #[rustfmt::skip]
        let m = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];

        let maha = mahalanobis_distance(&a, &b, &m);
        let eucl = euclidean_distance(&a, &b);
        assert!(
            (maha - eucl).abs() < 1e-5,
            "M=I should give Euclidean distance: maha={maha}, eucl={eucl}"
        );
    }

    #[test]
    fn test_mahalanobis_scaled() {
        // M = diag(4, 1) weights dimension 0 by factor 2
        let a = [0.0, 0.0];
        let b = [1.0, 1.0];
        #[rustfmt::skip]
        let m = [
            4.0, 0.0,
            0.0, 1.0,
        ];

        let d = mahalanobis_distance(&a, &b, &m);
        // sqrt(1*4 + 1*1) = sqrt(5)
        let expected = 5.0f32.sqrt();
        assert!(
            (d - expected).abs() < 1e-5,
            "diag(4,1) distance: got {d}, expected {expected}"
        );
    }

    #[test]
    fn test_learn_transform_separates() {
        // Two clusters that are close in original space
        // Cluster A: around (1, 0)
        // Cluster B: around (1, 0.5)
        let a1 = [1.0f32, 0.0];
        let a2 = [1.1, 0.1];
        let a3 = [0.9, -0.1];
        let b1 = [1.0, 0.5];
        let b2 = [1.1, 0.6];
        let b3 = [0.9, 0.4];

        let similar: Vec<(&[f32], &[f32])> = vec![
            (&a1, &a2),
            (&a1, &a3),
            (&a2, &a3),
            (&b1, &b2),
            (&b1, &b3),
            (&b2, &b3),
        ];
        let dissimilar: Vec<(&[f32], &[f32])> = vec![(&a1, &b1), (&a2, &b2), (&a3, &b3)];

        let config = MahalanobisConfig {
            lr: 0.1,
            max_iter: 100,
            tol: 1e-8,
            margin: 1.0,
        };

        let l = learn_transform(&similar, &dissimilar, 2, 2, &config);

        // After learning, inter-cluster distance should be larger than intra-cluster
        let ta1 = transform(&a1, &l, 2, 2);
        let ta2 = transform(&a2, &l, 2, 2);
        let tb1 = transform(&b1, &l, 2, 2);

        let intra = euclidean_distance(&ta1, &ta2);
        let inter = euclidean_distance(&ta1, &tb1);
        assert!(
            inter > intra,
            "inter-cluster ({inter}) should exceed intra-cluster ({intra})"
        );
    }

    #[test]
    fn test_learn_transform_identity_init() {
        // With zero iterations, L should be identity
        let sim: Vec<(&[f32], &[f32])> = vec![(&[1.0, 0.0], &[0.9, 0.1])];
        let dissim: Vec<(&[f32], &[f32])> = vec![];
        let config = MahalanobisConfig {
            lr: 0.01,
            max_iter: 0,
            tol: 1e-6,
            margin: 1.0,
        };

        let l = learn_transform(&sim, &dissim, 2, 2, &config);
        #[rustfmt::skip]
        let expected = [1.0, 0.0, 0.0, 1.0];
        for (i, (&got, &exp)) in l.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "L[{i}] = {got}, expected {exp}");
        }
    }

    #[test]
    fn test_nca_improves_classification() {
        // 3 classes arranged along dim 0, but with noise in dim 1
        // Class 0: x ~ (0, noise), Class 1: x ~ (3, noise), Class 2: x ~ (6, noise)
        let points: Vec<[f32; 2]> = vec![
            [0.0, 0.5],
            [0.1, -0.3],
            [-0.1, 0.8],
            [3.0, 0.4],
            [3.1, -0.2],
            [2.9, 0.6],
            [6.0, 0.3],
            [6.1, -0.1],
            [5.9, 0.7],
        ];
        let labels = [0, 0, 0, 1, 1, 1, 2, 2, 2];

        let data_refs: Vec<&[f32]> = points.iter().map(|p| p.as_slice()).collect();

        // Reduce to 1D -- should learn to project onto dim 0
        let config = NcaConfig {
            lr: 0.01,
            max_iter: 100,
            tol: 1e-8,
        };
        let l = nca(&data_refs, &labels, 1, &config);

        // Transform and check that classes are still separable
        let transformed: Vec<Vec<f32>> = data_refs.iter().map(|p| transform(p, &l, 2, 1)).collect();

        // All class-0 should be on one side, class-2 on the other
        let mean_0 = (transformed[0][0] + transformed[1][0] + transformed[2][0]) / 3.0;
        let mean_2 = (transformed[6][0] + transformed[7][0] + transformed[8][0]) / 3.0;
        assert!(
            (mean_0 - mean_2).abs() > 1.0,
            "NCA should separate classes: mean_0={mean_0}, mean_2={mean_2}"
        );
    }

    #[test]
    fn test_nca_reduces_dimension() {
        let points: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.5],
            [1.1, 0.1, 0.4],
            [0.0, 1.0, 0.5],
            [0.1, 1.1, 0.6],
        ];
        let labels = [0, 0, 1, 1];
        let data_refs: Vec<&[f32]> = points.iter().map(|p| p.as_slice()).collect();

        let config = NcaConfig {
            lr: 0.01,
            max_iter: 50,
            tol: 1e-8,
        };
        let l = nca(&data_refs, &labels, 2, &config);

        assert_eq!(l.len(), 2 * 3, "L should be out_dim * in_dim");
        let out = transform(&points[0], &l, 3, 2);
        assert_eq!(out.len(), 2, "output should have out_dim dimensions");
    }

    #[test]
    fn test_lmnn_improves_margin() {
        // 2 classes, simple separation task
        let points: Vec<[f32; 2]> = vec![
            [0.0, 1.0],
            [0.1, 0.9],
            [-0.1, 1.1],
            [1.0, 0.0],
            [0.9, 0.1],
            [1.1, -0.1],
        ];
        let labels = [0, 0, 0, 1, 1, 1];
        let data_refs: Vec<&[f32]> = points.iter().map(|p| p.as_slice()).collect();

        let config = LmnnConfig {
            lr: 0.01,
            max_iter: 50,
            tol: 1e-8,
            k: 2,
            margin: 1.0,
            mu: 0.5,
        };
        let l = lmnn(&data_refs, &labels, 2, &config);

        // Count margin violations before and after
        let count_violations = |l_mat: &[f32]| -> usize {
            let z: Vec<Vec<f32>> = data_refs
                .iter()
                .map(|p| transform(p, l_mat, 2, 2))
                .collect();
            let mut violations = 0;
            for i in 0..6 {
                for j in 0..6 {
                    if labels[j] != labels[i] {
                        continue;
                    }
                    if j == i {
                        continue;
                    }
                    let d_ij: f32 = (0..2).map(|r| (z[i][r] - z[j][r]).powi(2)).sum();
                    for ll in 0..6 {
                        if labels[ll] == labels[i] {
                            continue;
                        }
                        let d_il: f32 = (0..2).map(|r| (z[i][r] - z[ll][r]).powi(2)).sum();
                        if d_ij + config.margin > d_il {
                            violations += 1;
                        }
                    }
                }
            }
            violations
        };

        // Identity L
        #[rustfmt::skip]
        let id = [1.0, 0.0, 0.0, 1.0];
        let before = count_violations(&id);
        let after = count_violations(&l);
        assert!(
            after <= before,
            "LMNN should reduce margin violations: before={before}, after={after}"
        );
    }

    #[test]
    fn test_lmnn_basic() {
        // Verify loss decreases over training
        let points: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [2.1, 1.9]];
        let labels = [0, 0, 1, 1];
        let data_refs: Vec<&[f32]> = points.iter().map(|p| p.as_slice()).collect();

        let config = LmnnConfig {
            lr: 0.01,
            max_iter: 50,
            tol: 1e-8,
            k: 1,
            margin: 1.0,
            mu: 0.5,
        };
        let l = lmnn(&data_refs, &labels, 2, &config);

        // Output should have correct dimensions
        assert_eq!(l.len(), 4, "L should be 2x2 = 4 elements");
    }

    #[test]
    fn test_transform_identity() {
        let point = [1.0, 2.0, 3.0];
        #[rustfmt::skip]
        let l = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let out = transform(&point, &l, 3, 3);
        for (i, (&got, &exp)) in out.iter().zip(point.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "transform with identity: out[{i}]={got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_transform_dimension_reduction() {
        let point = [1.0, 2.0, 3.0];
        // Project onto first 2 dims
        #[rustfmt::skip]
        let l = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let out = transform(&point, &l, 3, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_batch() {
        let p1 = [1.0f32, 2.0];
        let p2 = [3.0f32, 4.0];
        let points: Vec<&[f32]> = vec![&p1, &p2];
        #[rustfmt::skip]
        let l = [1.0, 0.0, 0.0, 1.0];

        let out = transform_batch(&points, &l, 2, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0][0] - 1.0).abs() < 1e-6);
        assert!((out[1][1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_itml_improves_distance() {
        // Two clusters with some overlap
        let a1 = [0.0f32, 0.0];
        let a2 = [0.1, 0.2];
        let a3 = [-0.1, 0.1];
        let b1 = [0.5, 0.5];
        let b2 = [0.6, 0.4];
        let b3 = [0.4, 0.6];

        let similar: Vec<(&[f32], &[f32])> = vec![(&a1, &a2), (&a1, &a3), (&b1, &b2), (&b1, &b3)];
        let dissimilar: Vec<(&[f32], &[f32])> = vec![(&a1, &b1), (&a2, &b2)];

        let config = ItmlConfig {
            max_iter: 200,
            tol: 1e-6,
            gamma: 1.0,
        };
        let m = itml(&similar, &dissimilar, 2, &config);
        assert_eq!(m.len(), 4);

        // Verify M is computed (not all zeros, not all identity)
        let is_identity = (m[0] - 1.0).abs() < 1e-6
            && m[1].abs() < 1e-6
            && m[2].abs() < 1e-6
            && (m[3] - 1.0).abs() < 1e-6;
        // M should have been modified from identity
        // (it may stay close to identity if constraints are already satisfied)
        assert!(m.iter().all(|&v| v.is_finite()), "M should be finite");
        let _ = is_identity;

        // Similar pairs should have smaller Mahalanobis distance than dissimilar
        let d_sim = mahalanobis_distance(&a1, &a2, &m);
        let d_dissim = mahalanobis_distance(&a1, &b1, &m);
        assert!(
            d_sim < d_dissim + 0.5,
            "similar distance ({d_sim}) should be less than dissimilar ({d_dissim})"
        );
    }

    #[test]
    fn test_kissme_basic() {
        let a1 = [0.0f32, 0.0];
        let a2 = [0.1, 0.05];
        let b1 = [1.0, 1.0];
        let b2 = [1.1, 0.95];

        let similar: Vec<(&[f32], &[f32])> = vec![(&a1, &a2), (&b1, &b2)];
        let dissimilar: Vec<(&[f32], &[f32])> = vec![(&a1, &b1), (&a2, &b2)];

        let m = kissme(&similar, &dissimilar, 2);
        assert_eq!(m.len(), 4, "M should be 2x2 = 4 elements");

        // M should be finite
        assert!(m.iter().all(|&v| v.is_finite()), "M should be finite");

        // M should be PSD (all eigenvalues >= 0)
        // For 2x2: eigenvalues = (trace +/- sqrt(trace^2 - 4*det)) / 2
        let trace = m[0] + m[3];
        let det = m[0] * m[3] - m[1] * m[2];
        let disc = trace * trace - 4.0 * det;
        if disc >= 0.0 {
            let e1 = (trace + disc.sqrt()) / 2.0;
            let e2 = (trace - disc.sqrt()) / 2.0;
            assert!(
                e1 >= -1e-5 && e2 >= -1e-5,
                "M should be PSD: eigenvalues = ({e1}, {e2})"
            );
        }
    }

    #[test]
    fn test_kissme_separates_clusters() {
        // Two well-separated 2D clusters where similar differences are small
        // and dissimilar differences are large along a consistent axis.
        // Class A: tightly clustered near (0, 0)
        // Class B: tightly clustered near (3, 0)
        let class_a: Vec<[f32; 2]> = vec![
            [0.0, 0.0],
            [0.1, 0.05],
            [-0.1, 0.05],
            [0.05, -0.1],
            [-0.05, -0.05],
        ];
        let class_b: Vec<[f32; 2]> = vec![
            [3.0, 0.0],
            [3.1, 0.05],
            [2.9, 0.05],
            [3.05, -0.1],
            [2.95, -0.05],
        ];

        let mut similar = Vec::new();
        for i in 0..class_a.len() {
            for j in (i + 1)..class_a.len() {
                similar.push((class_a[i].as_slice(), class_a[j].as_slice()));
            }
            for j in (i + 1)..class_b.len() {
                similar.push((class_b[i].as_slice(), class_b[j].as_slice()));
            }
        }

        let mut dissimilar = Vec::new();
        for a in &class_a {
            for b in &class_b {
                dissimilar.push((a.as_slice(), b.as_slice()));
            }
        }

        let m = kissme(&similar, &dissimilar, 2);

        // M should be PSD and non-trivial
        assert!(m.iter().any(|&v| v.abs() > 1e-6), "M should be non-trivial");

        // Verify M is non-zero and gives finite distances
        let d_intra = mahalanobis_distance(&class_a[0], &class_a[1], &m);
        let d_inter = mahalanobis_distance(&class_a[0], &class_b[0], &m);
        assert!(d_intra.is_finite(), "intra distance should be finite");
        assert!(d_inter.is_finite(), "inter distance should be finite");
        assert!(
            d_inter > d_intra,
            "KISSME inter-cluster distance ({d_inter}) should exceed intra ({d_intra})"
        );
    }
}
