/// Dot product of two vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 { 0.0 } else { d / denom }
}

/// Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Pairwise cosine similarity matrix for a batch of embeddings.
///
/// Returns n x n matrix where `result[i][j] = cosine_similarity(batch[i], batch[j])`.
pub fn pairwise_cosine(batch: &[&[f32]]) -> Vec<Vec<f32>> {
    let n = batch.len();
    let mut result = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        result[i][i] = 1.0;
        for j in (i + 1)..n {
            let sim = cosine_similarity(batch[i], batch[j]);
            result[i][j] = sim;
            result[j][i] = sim;
        }
    }
    result
}

/// L2 normalize a vector in-place. Returns the original norm.
pub fn l2_normalize(v: &mut [f32]) -> f32 {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let v = [1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "cos(v, v) should be 1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "cos([1,0], [0,1]) should be 0.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "cos(v, -v) should be -1.0, got {sim}"
        );
    }

    #[test]
    fn test_euclidean_zero() {
        let v = [1.0, 2.0, 3.0];
        let d = euclidean_distance(&v, &v);
        assert!(d.abs() < 1e-6, "d(v, v) should be 0.0, got {d}");
    }

    #[test]
    fn test_pairwise_cosine() {
        let a: &[f32] = &[1.0, 0.0];
        let b: &[f32] = &[0.0, 1.0];
        let c: &[f32] = &[1.0, 1.0];
        let batch: Vec<&[f32]> = vec![a, b, c];
        let mat = pairwise_cosine(&batch);

        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);

        // Diagonal should be 1.0
        for (i, row) in mat.iter().enumerate() {
            assert!(
                (row[i] - 1.0).abs() < 1e-6,
                "diagonal [{i}][{i}] should be 1.0, got {}",
                row[i]
            );
        }

        // Symmetry
        for (i, row_i) in mat.iter().enumerate() {
            for (j, row_j) in mat.iter().enumerate() {
                assert!(
                    (row_i[j] - row_j[i]).abs() < 1e-6,
                    "matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = [3.0, 4.0];
        let norm = l2_normalize(&mut v);
        assert!((norm - 5.0).abs() < 1e-6, "original norm should be 5.0");

        let new_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (new_norm - 1.0).abs() < 1e-6,
            "norm after normalization should be 1.0, got {new_norm}"
        );
    }
}
