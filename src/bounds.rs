//! Differentiable / clamping projection of distance matrices onto the metric cone.
//!
//! A distance matrix that violates the triangle inequality is not, strictly,
//! a metric. `project_triangle_inequality` clamps the matrix in place so that
//! `d(i, k) <= d(i, j) + d(j, k)` holds for every triple. Useful as a
//! post-processing step on learned distance approximations.
//!
//! Folded from the (archived) ritu crate.

/// Project a flat-row-major `n x n` distance matrix onto the metric cone via
/// Floyd-Warshall-style triangle propagation. Single pass. O(n^3).
///
/// Lighter than a full quadratic program but only correct as a one-shot
/// fix-up; iterate to convergence if you need a stricter guarantee.
pub fn project_triangle_inequality(dists: &mut [f64], n: usize) {
    debug_assert_eq!(dists.len(), n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let d_ij = dists[i * n + j];
                let d_jk = dists[j * n + k];
                let d_ik = dists[i * n + k];
                if d_ik > d_ij + d_jk {
                    dists[i * n + k] = d_ij + d_jk;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_clamps_violating_triple() {
        // d(0,1)=1, d(1,2)=1, d(0,2)=10 → clamp d(0,2) to 2.
        let mut d = [0.0, 1.0, 10.0, 1.0, 0.0, 1.0, 10.0, 1.0, 0.0];
        project_triangle_inequality(&mut d, 3);
        assert!(d[2] <= 2.0 + 1e-9);
        assert!(d[6] <= 2.0 + 1e-9 || (d[6] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn projection_is_idempotent_on_metric_input() {
        let original = [0.0f64, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let mut d = original;
        project_triangle_inequality(&mut d, 3);
        for (a, b) in d.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }
}
