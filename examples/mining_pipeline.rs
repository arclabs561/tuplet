//! End-to-end mining + loss pipeline.
//!
//! Demonstrates negative mining strategies feeding into loss functions:
//! hardest negatives -> triplet loss, semi-hard -> triplet loss,
//! multi-similarity mining -> multi-similarity loss.
//!
//! Run: `cargo run --example mining_pipeline`

use tuplet::*;

fn main() {
    // Pool of embeddings with class labels
    // Class 0: upper-right quadrant, Class 1: lower-left
    let pool_data: Vec<[f32; 3]> = vec![
        [0.9, 0.1, 0.0],     // 0: class 0
        [0.8, 0.2, 0.1],     // 1: class 0
        [0.85, 0.15, 0.05],  // 2: class 0
        [-0.1, 0.9, 0.0],    // 3: class 1
        [-0.2, 0.8, 0.1],    // 4: class 1
        [-0.15, 0.85, 0.05], // 5: class 1
        [0.5, 0.5, 0.1],     // 6: class 0 (ambiguous, near boundary)
        [-0.5, 0.5, 0.1],    // 7: class 1 (ambiguous, near boundary)
    ];
    let labels = [0, 0, 0, 1, 1, 1, 0, 1];
    let pool_refs: Vec<&[f32]> = pool_data.iter().map(|p| p.as_slice()).collect();

    println!("=== Hardest Negative Mining + Triplet Loss ===\n");
    {
        let miner = HardestMiner;
        let anchor_idx = 0; // class 0 anchor
        let positive_idx = 1; // class 0 positive
        // Exclude same-class items -- only mine from other classes
        let exclude: Vec<usize> = (0..pool_data.len())
            .filter(|&i| labels[i] == labels[anchor_idx])
            .collect();
        let neg_indices = miner.mine(pool_refs[anchor_idx], &pool_refs, &exclude);

        println!("Anchor: idx {anchor_idx} (class {})", labels[anchor_idx]);
        println!(
            "Positive: idx {positive_idx} (class {})",
            labels[positive_idx]
        );
        println!(
            "Hardest negative: idx {:?} (class {})",
            neg_indices, labels[neg_indices[0]]
        );

        let out = triplet_loss(
            &[pool_refs[anchor_idx]],
            &[pool_refs[positive_idx]],
            &[pool_refs[neg_indices[0]]],
            0.5,
        );
        println!("Triplet loss: {:.4}", out.loss);
        println!(
            "Anchor grad norm: {:.4}",
            out.grad_anchors[0]
                .iter()
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt()
        );
    }

    println!("\n=== Semi-Hard Mining + Triplet Loss ===\n");
    {
        let anchor_idx = 0;
        let positive_idx = 1;
        let d_ap = 1.0 - cosine_similarity(pool_refs[anchor_idx], pool_refs[positive_idx]);
        let miner = SemiHardMiner { margin: 0.5 };

        // Only mine from other-class items
        let other_class: Vec<usize> = (0..pool_data.len())
            .filter(|&i| labels[i] != labels[anchor_idx])
            .collect();
        let other_refs: Vec<&[f32]> = other_class.iter().map(|&i| pool_refs[i]).collect();
        let neg_indices = miner.mine(pool_refs[anchor_idx], &other_refs, &[], d_ap);

        println!("Positive distance (cosine): {d_ap:.4}");
        println!("Semi-hard band: [{d_ap:.4}, {:.4}]", d_ap + 0.5);
        println!(
            "Semi-hard negatives found: {} (indices: {:?})",
            neg_indices.len(),
            neg_indices
                .iter()
                .map(|&i| other_class[i])
                .collect::<Vec<_>>()
        );

        if !neg_indices.is_empty() {
            let neg_idx = other_class[neg_indices[0]];
            let out = triplet_loss(
                &[pool_refs[anchor_idx]],
                &[pool_refs[positive_idx]],
                &[pool_refs[neg_idx]],
                0.5,
            );
            println!("Triplet loss (semi-hard): {:.4}", out.loss);
        }
    }

    println!("\n=== Multi-Similarity Mining + Multi-Similarity Loss ===\n");
    {
        let anchor_idx = 6; // boundary point
        let pos_indices: Vec<usize> = (0..pool_data.len())
            .filter(|&i| i != anchor_idx && labels[i] == labels[anchor_idx])
            .collect();
        let neg_indices: Vec<usize> = (0..pool_data.len())
            .filter(|&i| labels[i] != labels[anchor_idx])
            .collect();

        let miner = MultiSimilarityMiner { epsilon: 0.1 };
        let (inf_pos, inf_neg) = miner.mine(
            pool_refs[anchor_idx],
            &pool_refs,
            &pos_indices,
            &neg_indices,
        );

        println!("Anchor: idx {anchor_idx} (boundary class-0 point)");
        println!("Informative positives: {:?}", inf_pos);
        println!("Informative negatives: {:?}", inf_neg);

        // Run full multi-similarity loss on the batch
        let out = multi_similarity_loss(&pool_refs, &labels, 2.0, 50.0, 0.5);
        println!("Multi-similarity loss (full batch): {:.4}", out.loss);
        println!(
            "Mean grad norm: {:.4}",
            out.grad_anchors
                .iter()
                .map(|g| g.iter().map(|v| v * v).sum::<f32>().sqrt())
                .sum::<f32>()
                / pool_data.len() as f32
        );
    }

    println!("\n=== Distance-Weighted Mining ===\n");
    {
        let anchor_idx = 0;
        let miner = DistanceWeightedMiner {
            cutoff: 0.1,
            nonzero_loss_cutoff: 1.5,
        };
        let neg_indices = miner.mine(pool_refs[anchor_idx], &pool_refs, &[anchor_idx]);
        println!("Distance-weighted negatives (cutoff=0.1, max=1.5):");
        for &i in &neg_indices {
            let d = 1.0 - cosine_similarity(pool_refs[anchor_idx], pool_refs[i]);
            println!("  idx {i} (class {}, cosine_dist={d:.4})", labels[i]);
        }
    }
}
