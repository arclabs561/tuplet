//! Pure-Rust embedding training loop without any tensor framework.
//!
//! Learns a linear projection via NCA, then fine-tunes embeddings
//! using InfoNCE loss with manual gradient descent.
//!
//! Run: `cargo run --example embedding_training`

use tuplet::*;

fn main() {
    // 3 classes of 2D points, 5 samples each
    let cluster_centers = [[0.0f32, 0.0], [2.0, 0.0], [1.0, 2.0]];
    let offsets = [
        [0.1, 0.05],
        [-0.1, 0.1],
        [0.05, -0.1],
        [-0.05, 0.05],
        [0.15, -0.05],
    ];

    let mut data: Vec<[f32; 2]> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for (class, center) in cluster_centers.iter().enumerate() {
        for offset in &offsets {
            data.push([center[0] + offset[0], center[1] + offset[1]]);
            labels.push(class);
        }
    }

    let data_refs: Vec<&[f32]> = data.iter().map(|p| p.as_slice()).collect();

    println!("=== Phase 1: NCA projection ===\n");
    let mut nca_config = NcaConfig::new();
    nca_config.tol = 1e-8;
    let l = nca(&data_refs, &labels, 2, &nca_config);
    let projected: Vec<Vec<f32>> = data_refs.iter().map(|p| transform(p, &l, 2, 2)).collect();
    let proj_refs: Vec<&[f32]> = projected.iter().map(|v| v.as_slice()).collect();
    println!(
        "NCA 1-NN accuracy: {:.1}%\n",
        nn_accuracy(&proj_refs, &labels) * 100.0
    );

    println!("=== Phase 2: InfoNCE fine-tuning ===\n");

    // Create trainable embeddings (copy of projected data)
    let n = data.len();
    let mut embeddings: Vec<Vec<f32>> = projected.clone();
    let lr = 0.01;

    for epoch in 0..50 {
        // Build same-class pairs as anchor/positive
        let mut anchors = Vec::new();
        let mut positives = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if labels[i] == labels[j] {
                    anchors.push(i);
                    positives.push(j);
                    break; // one positive per anchor
                }
            }
        }

        let a_refs: Vec<&[f32]> = anchors.iter().map(|&i| embeddings[i].as_slice()).collect();
        let p_refs: Vec<&[f32]> = positives
            .iter()
            .map(|&i| embeddings[i].as_slice())
            .collect();

        let out = infonce_loss(&a_refs, &p_refs, 0.5);

        if epoch % 10 == 0 {
            let emb_refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
            println!(
                "epoch {epoch:3}: loss={:.4}, 1-NN acc={:.1}%",
                out.loss,
                nn_accuracy(&emb_refs, &labels) * 100.0
            );
        }

        // Apply gradients to the anchor and positive embeddings
        for (batch_idx, &orig_idx) in anchors.iter().enumerate() {
            for (emb_d, grad_d) in embeddings[orig_idx]
                .iter_mut()
                .zip(&out.grad_anchors[batch_idx])
            {
                *emb_d -= lr * grad_d;
            }
        }
        for (batch_idx, &orig_idx) in positives.iter().enumerate() {
            for (emb_d, grad_d) in embeddings[orig_idx]
                .iter_mut()
                .zip(&out.grad_positives[batch_idx])
            {
                *emb_d -= lr * grad_d;
            }
        }
    }

    let final_refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
    println!(
        "\nFinal 1-NN accuracy: {:.1}%",
        nn_accuracy(&final_refs, &labels) * 100.0
    );
}

fn nn_accuracy(data: &[&[f32]], labels: &[usize]) -> f32 {
    let n = data.len();
    let mut correct = 0;
    for i in 0..n {
        let mut best_j = 0;
        let mut best_dist = f32::INFINITY;
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = euclidean_distance(data[i], data[j]);
            if d < best_dist {
                best_dist = d;
                best_j = j;
            }
        }
        if labels[best_j] == labels[i] {
            correct += 1;
        }
    }
    correct as f32 / n as f32
}
