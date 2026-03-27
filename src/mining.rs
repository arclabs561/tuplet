use crate::similarity::cosine_similarity;
use rand::seq::SliceRandom;

/// Selects negative indices from a candidate pool.
pub trait NegativeMiner {
    /// Returns indices into `pool` of selected negatives, excluding indices in `exclude`.
    fn mine(&self, anchor: &[f32], pool: &[&[f32]], exclude: &[usize]) -> Vec<usize>;
}

/// Uses all non-excluded items in the pool as negatives.
#[derive(Debug, Clone)]
pub struct InBatchMiner;

impl NegativeMiner for InBatchMiner {
    fn mine(&self, _anchor: &[f32], pool: &[&[f32]], exclude: &[usize]) -> Vec<usize> {
        (0..pool.len()).filter(|i| !exclude.contains(i)).collect()
    }
}

/// Selects the single hardest (most similar) non-excluded negative.
#[derive(Debug, Clone)]
pub struct HardestMiner;

impl NegativeMiner for HardestMiner {
    fn mine(&self, anchor: &[f32], pool: &[&[f32]], exclude: &[usize]) -> Vec<usize> {
        let mut best_idx = None;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, candidate) in pool.iter().enumerate() {
            if exclude.contains(&i) {
                continue;
            }
            let sim = cosine_similarity(anchor, candidate);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        best_idx.into_iter().collect()
    }
}

/// Selects negatives within a semi-hard margin band.
///
/// A semi-hard negative satisfies: `d(a, p) < d(a, n) < d(a, p) + margin`,
/// where distance is `1 - cosine_similarity`.
#[derive(Debug, Clone)]
pub struct SemiHardMiner {
    /// Width of the margin band.
    pub margin: f32,
    /// Precomputed `d(anchor, positive)` as cosine distance (`1 - cos`).
    pub positive_distance: f32,
}

impl NegativeMiner for SemiHardMiner {
    fn mine(&self, anchor: &[f32], pool: &[&[f32]], exclude: &[usize]) -> Vec<usize> {
        let d_ap = self.positive_distance;
        let upper = d_ap + self.margin;

        (0..pool.len())
            .filter(|i| {
                if exclude.contains(i) {
                    return false;
                }
                let d_an = 1.0 - cosine_similarity(anchor, pool[*i]);
                d_an > d_ap && d_an < upper
            })
            .collect()
    }
}

/// Selects `k` random non-excluded negatives.
#[derive(Debug, Clone)]
pub struct RandomMiner {
    /// Number of negatives to sample.
    pub k: usize,
}

impl NegativeMiner for RandomMiner {
    fn mine(&self, _anchor: &[f32], pool: &[&[f32]], exclude: &[usize]) -> Vec<usize> {
        let mut candidates: Vec<usize> = (0..pool.len()).filter(|i| !exclude.contains(i)).collect();
        let mut rng = rand::rng();
        candidates.shuffle(&mut rng);
        candidates.truncate(self.k);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_batch_miner() {
        let anchor: &[f32] = &[1.0, 0.0];
        let pool: Vec<&[f32]> = vec![&[0.9, 0.1], &[0.0, 1.0], &[0.5, 0.5]];
        let exclude = vec![0];

        let miner = InBatchMiner;
        let result = miner.mine(anchor, &pool, &exclude);

        assert_eq!(result.len(), 2);
        assert!(!result.contains(&0));
        assert!(result.contains(&1));
        assert!(result.contains(&2));
    }

    #[test]
    fn test_hardest_miner() {
        let anchor: &[f32] = &[1.0, 0.0];
        let pool: Vec<&[f32]> = vec![
            &[0.0, 1.0],  // orthogonal, cos=0
            &[0.9, 0.1],  // very similar, cos~0.99
            &[-1.0, 0.0], // opposite, cos=-1
        ];
        let exclude = vec![];

        let miner = HardestMiner;
        let result = miner.mine(anchor, &pool, &exclude);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1, "should select the most similar candidate");
    }

    #[test]
    fn test_semi_hard_miner() {
        let anchor: &[f32] = &[1.0, 0.0];

        // positive_distance = 0.1 (cosine distance)
        // margin = 0.3
        // Want negatives with 0.1 < d < 0.4
        let pool: Vec<&[f32]> = vec![
            &[0.99, 0.14], // cos~0.99, d~0.01 -- too easy (d < d_ap)
            &[0.87, 0.50], // cos~0.87, d~0.13 -- semi-hard (in band)
            &[0.0, 1.0],   // cos~0.0, d~1.0 -- too hard (d > upper)
        ];
        let exclude = vec![];

        let miner = SemiHardMiner {
            margin: 0.3,
            positive_distance: 0.1,
        };
        let result = miner.mine(anchor, &pool, &exclude);

        // Only index 1 should be in the band
        assert!(
            result.contains(&1),
            "semi-hard miner should find index 1 in the margin band, got {:?}",
            result
        );
        assert!(
            !result.contains(&0),
            "index 0 is too easy (d < d_ap), should be excluded"
        );
        assert!(
            !result.contains(&2),
            "index 2 is too hard (d > upper), should be excluded"
        );
    }

    #[test]
    fn test_random_miner() {
        let anchor: &[f32] = &[1.0, 0.0];
        let pool: Vec<&[f32]> = vec![
            &[0.9, 0.1],
            &[0.0, 1.0],
            &[0.5, 0.5],
            &[-1.0, 0.0],
            &[0.3, 0.7],
        ];
        let exclude = vec![2];

        let miner = RandomMiner { k: 2 };
        let result = miner.mine(anchor, &pool, &exclude);

        assert_eq!(result.len(), 2, "should return exactly k items");
        assert!(
            !result.contains(&2),
            "excluded index should not appear in result"
        );
        for &idx in &result {
            assert!(idx < pool.len(), "index should be in range");
        }
    }
}
