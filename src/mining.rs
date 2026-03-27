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

/// Selects informative pairs based on the multi-similarity criterion (Wang et al. 2019).
///
/// For each anchor, selects positive pairs where `sim(a, p) < max(sim(a, neg)) + epsilon`
/// and negative pairs where `sim(a, n) > min(sim(a, pos)) - epsilon`.
#[derive(Debug, Clone)]
pub struct MultiSimilarityMiner {
    /// Margin for pair selection.
    pub epsilon: f32,
}

impl MultiSimilarityMiner {
    /// Mine informative positive and negative indices for the given anchor.
    ///
    /// `positive_indices` and `negative_indices` partition `pool` into positives and negatives.
    /// Returns `(informative_pos, informative_neg)` as index vectors into `pool`.
    pub fn mine(
        &self,
        anchor: &[f32],
        pool: &[&[f32]],
        positive_indices: &[usize],
        negative_indices: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        if positive_indices.is_empty() || negative_indices.is_empty() {
            return (vec![], vec![]);
        }

        // Compute similarities
        let pos_sims: Vec<(usize, f32)> = positive_indices
            .iter()
            .map(|&i| (i, cosine_similarity(anchor, pool[i])))
            .collect();
        let neg_sims: Vec<(usize, f32)> = negative_indices
            .iter()
            .map(|&i| (i, cosine_similarity(anchor, pool[i])))
            .collect();

        let max_neg_sim = neg_sims
            .iter()
            .map(|&(_, s)| s)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_pos_sim = pos_sims
            .iter()
            .map(|&(_, s)| s)
            .fold(f32::INFINITY, f32::min);

        // Informative positives: harder than the hardest negative (with margin)
        let informative_pos: Vec<usize> = pos_sims
            .iter()
            .filter(|&&(_, s)| s < max_neg_sim + self.epsilon)
            .map(|&(i, _)| i)
            .collect();

        // Informative negatives: harder than the easiest positive (with margin)
        let informative_neg: Vec<usize> = neg_sims
            .iter()
            .filter(|&&(_, s)| s > min_pos_sim - self.epsilon)
            .map(|&(i, _)| i)
            .collect();

        (informative_pos, informative_neg)
    }
}

/// Samples pairs with probability proportional to distance (Wu et al. 2017).
///
/// Filters pairs by distance thresholds to avoid degenerate sampling.
#[derive(Debug, Clone)]
pub struct DistanceWeightedMiner {
    /// Minimum distance threshold -- pairs closer than this are excluded.
    pub cutoff: f32,
    /// Maximum distance -- pairs farther than this are excluded.
    pub nonzero_loss_cutoff: f32,
}

impl NegativeMiner for DistanceWeightedMiner {
    fn mine(&self, anchor: &[f32], pool: &[&[f32]], exclude: &[usize]) -> Vec<usize> {
        (0..pool.len())
            .filter(|i| {
                if exclude.contains(i) {
                    return false;
                }
                let d = 1.0 - cosine_similarity(anchor, pool[*i]);
                d >= self.cutoff && d <= self.nonzero_loss_cutoff
            })
            .collect()
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

    #[test]
    fn test_multi_similarity_miner_selects_informative() {
        let anchor: &[f32] = &[1.0, 0.0];
        let pool: Vec<&[f32]> = vec![
            &[0.99, 0.14], // idx 0: positive, very similar (easy positive)
            &[0.5, 0.87],  // idx 1: positive, less similar (informative positive)
            &[0.0, 1.0],   // idx 2: negative, orthogonal (easy negative)
            &[0.87, 0.5],  // idx 3: negative, fairly similar (informative negative)
        ];
        let pos_idx = vec![0, 1];
        let neg_idx = vec![2, 3];

        let miner = MultiSimilarityMiner { epsilon: 0.1 };
        let (inf_pos, inf_neg) = miner.mine(anchor, &pool, &pos_idx, &neg_idx);

        // Index 1 (less similar positive) should be selected as informative
        assert!(
            inf_pos.contains(&1),
            "less similar positive should be informative, got {:?}",
            inf_pos
        );
        // Index 3 (similar negative) should be selected as informative
        assert!(
            inf_neg.contains(&3),
            "similar negative should be informative, got {:?}",
            inf_neg
        );
    }

    #[test]
    fn test_distance_weighted_miner() {
        let anchor: &[f32] = &[1.0, 0.0];
        let pool: Vec<&[f32]> = vec![
            &[0.99, 0.14], // very close, cos~0.99, d~0.01
            &[0.5, 0.87],  // medium, cos~0.5, d~0.5
            &[-1.0, 0.0],  // far, cos=-1, d=2.0
        ];

        let miner = DistanceWeightedMiner {
            cutoff: 0.1,
            nonzero_loss_cutoff: 1.5,
        };
        let result = miner.mine(anchor, &pool, &[]);

        // Index 0 is too close (d < cutoff)
        assert!(
            !result.contains(&0),
            "very close pair should be excluded by cutoff"
        );
        // Index 1 should be in range
        assert!(
            result.contains(&1),
            "medium distance pair should be included"
        );
        // Index 2 is too far (d > nonzero_loss_cutoff)
        assert!(
            !result.contains(&2),
            "very far pair should be excluded by nonzero_loss_cutoff"
        );
    }
}
