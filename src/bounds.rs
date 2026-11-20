//! Differentiable / clamping projection of distance matrices onto the metric cone.
//!
//! A distance matrix that violates the triangle inequality is not, strictly,
//! a metric. `project_triangle_inequality` clamps the matrix in place so that
//! `d(i, k) <= d(i, j) + d(j, k)` holds for every triple. Useful as a
//! post-processing step on learned distance approximations.
//!
//! Folded from the (archived) ritu crate.

/// Project a flat-row-major `n x n` distance matrix onto the metric cone by
/// replacing every entry with the shortest-path distance (Floyd-Warshall).
/// O(n^3).
///
/// The result satisfies `d(i, k) <= d(i, j) + d(j, k)` for every triple, and
/// a symmetric input stays symmetric. Entries only decrease. This is the
/// largest metric below the input, not the nearest one in a least-squares
/// sense.
pub fn project_triangle_inequality(dists: &mut [f64], n: usize) {
    debug_assert_eq!(dists.len(), n * n);
    // The intermediate vertex must be the outermost loop: after iteration `j`,
    // every entry is the shortest path using intermediates `0..=j`. With `j`
    // inside, a pair is relaxed through vertices whose own entries are not
    // final yet, which left violations and broke symmetry.
    for j in 0..n {
        for i in 0..n {
            let d_ij = dists[i * n + j];
            for k in 0..n {
                let via = d_ij + dists[j * n + k];
                if dists[i * n + k] > via {
                    dists[i * n + k] = via;
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

    #[test]
    fn random_symmetric_inputs_become_symmetric_metrics() {
        // xorshift; no dev-dependency needed.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1000) as f64 / 100.0 + 0.01
        };
        for n in [3usize, 4, 5, 7] {
            for _ in 0..2000 {
                let mut d = vec![0.0; n * n];
                for i in 0..n {
                    for j in (i + 1)..n {
                        let v = next();
                        d[i * n + j] = v;
                        d[j * n + i] = v;
                    }
                }
                let before = d.clone();
                project_triangle_inequality(&mut d, n);
                for i in 0..n {
                    for j in 0..n {
                        assert!((d[i * n + j] - d[j * n + i]).abs() < 1e-12, "asymmetric");
                        assert!(d[i * n + j] <= before[i * n + j] + 1e-12, "entry grew");
                        for k in 0..n {
                            assert!(
                                d[i * n + k] <= d[i * n + j] + d[j * n + k] + 1e-9,
                                "triangle violated"
                            );
                        }
                    }
                }
            }
        }
    }
}
