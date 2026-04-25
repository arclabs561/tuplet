//! Metric axiom utilities for sanity-checking learned distances.
//!
//! These functions take a distance closure and a sample of points; they don't
//! depend on any trait. That sidesteps the abstraction the alga / ritu crates
//! tried and were under-used for: callers can plug any distance function in
//! without committing to a `Metric` trait surface.
//!
//! Folded from the (archived) ritu crate.

/// Worst triangle-inequality gap over an ordered triple of points:
/// `max(0, d(a, c) - (d(a, b) + d(b, c)))`.
///
/// Returns 0 when the triangle inequality holds. Positive values indicate
/// violation magnitude. Useful as a regression check on learned Mahalanobis
/// matrices and quasimetrics.
pub fn triangle_violation<F>(distance: F, a: &[f64], b: &[f64], c: &[f64]) -> f64
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let d_ac = distance(a, c);
    let d_ab = distance(a, b);
    let d_bc = distance(b, c);
    (d_ac - (d_ab + d_bc)).max(0.0)
}

/// Maximum triangle-inequality violation over all ordered triples drawn from
/// `points`. O(n^3). Use on small samples (`n <= ~50`) as a property check.
pub fn max_triangle_violation<F>(distance: F, points: &[&[f64]]) -> f64
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let n = points.len();
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                if i == j || j == k || i == k {
                    continue;
                }
                let v = triangle_violation(&distance, points[i], points[j], points[k]);
                if v > worst {
                    worst = v;
                }
            }
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn euclid(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn euclidean_satisfies_triangle_inequality() {
        let pts: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let refs: Vec<&[f64]> = pts.iter().map(|p| p.as_slice()).collect();
        assert_eq!(max_triangle_violation(euclid, &refs), 0.0);
    }

    #[test]
    fn squared_euclidean_violates_triangle_inequality() {
        // d(a,c)=2 but d(a,b)+d(b,c)=1+1=2, so squared d(a,c)=4 > 1+1=2.
        fn sq(a: &[f64], b: &[f64]) -> f64 {
            a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
        }
        let a: &[f64] = &[0.0, 0.0];
        let b: &[f64] = &[1.0, 0.0];
        let c: &[f64] = &[2.0, 0.0];
        assert!(triangle_violation(sq, a, b, c) > 0.0);
    }
}
