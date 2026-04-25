//! Demonstrates the metric learning lineage from Mahalanobis to InfoNCE.
//!
//! Generates 2D data with 3 overlapping clusters and applies progressively
//! more sophisticated methods, printing nearest-neighbor accuracy for each.
//!
//! Run: `cargo run --example metric_learning`

use tuplet::*;

fn main() {
    // Generate 3 overlapping 2D clusters
    let cluster_centers = [[0.0f32, 0.0], [1.0, 0.5], [0.5, 1.0]];
    let offsets = [
        [0.1, 0.05],
        [-0.1, 0.1],
        [0.05, -0.1],
        [-0.05, 0.05],
        [0.15, -0.05],
    ];

    let mut data = Vec::new();
    let mut labels = Vec::new();
    for (class, center) in cluster_centers.iter().enumerate() {
        for offset in &offsets {
            data.push([center[0] + offset[0], center[1] + offset[1]]);
            labels.push(class);
        }
    }

    let data_refs: Vec<&[f32]> = data.iter().map(|p| p.as_slice()).collect();
    let n = data.len();

    // Baseline: Euclidean 1-NN accuracy
    let baseline_acc = nn_accuracy(&data_refs, &labels);
    println!("=== Metric Learning Lineage ===\n");
    println!("Baseline (Euclidean):       {:.1}%", baseline_acc * 100.0);

    // 1. Mahalanobis distance learning
    let (similar, dissimilar) = build_pairs(&data_refs, &labels);
    let m_config = MahalanobisConfig {
        lr: 0.05,
        max_iter: 200,
        tol: 1e-8,
        margin: 1.0,
    };
    let l_maha = learn_transform(&similar, &dissimilar, 2, 2, &m_config);
    let transformed: Vec<Vec<f32>> = data_refs
        .iter()
        .map(|p| transform(p, &l_maha, 2, 2))
        .collect();
    let t_refs: Vec<&[f32]> = transformed.iter().map(|v| v.as_slice()).collect();
    println!(
        "Mahalanobis learning:      {:.1}%",
        nn_accuracy(&t_refs, &labels) * 100.0
    );

    // 2. NCA
    let nca_config = NcaConfig {
        lr: 0.01,
        max_iter: 200,
        tol: 1e-8,
    };
    let l_nca = nca(&data_refs, &labels, 2, &nca_config);
    let transformed: Vec<Vec<f32>> = data_refs
        .iter()
        .map(|p| transform(p, &l_nca, 2, 2))
        .collect();
    let t_refs: Vec<&[f32]> = transformed.iter().map(|v| v.as_slice()).collect();
    println!(
        "NCA:                       {:.1}%",
        nn_accuracy(&t_refs, &labels) * 100.0
    );

    // 3. LMNN
    let lmnn_config = LmnnConfig {
        lr: 0.01,
        max_iter: 200,
        tol: 1e-8,
        k: 2,
        margin: 1.0,
        mu: 0.5,
    };
    let l_lmnn = lmnn(&data_refs, &labels, 2, &lmnn_config);
    let transformed: Vec<Vec<f32>> = data_refs
        .iter()
        .map(|p| transform(p, &l_lmnn, 2, 2))
        .collect();
    let t_refs: Vec<&[f32]> = transformed.iter().map(|v| v.as_slice()).collect();
    println!(
        "LMNN:                      {:.1}%",
        nn_accuracy(&t_refs, &labels) * 100.0
    );

    // 4. KISSME
    let m_kissme = kissme(&similar, &dissimilar, 2);
    println!(
        "KISSME:                    (M computed, {} non-zero entries)",
        m_kissme.iter().filter(|&&v| v.abs() > 1e-6).count()
    );

    // 5. ITML
    let itml_config = ItmlConfig {
        max_iter: 200,
        tol: 1e-6,
        gamma: 1.0,
    };
    let m_itml = itml(&similar, &dissimilar, 2, &itml_config);
    println!(
        "ITML:                      (M computed, trace={:.4})",
        m_itml[0] + m_itml[3]
    );

    // 6. Contrastive losses on the same data (report loss values)
    println!("\n=== Loss Functions ===\n");

    // Build anchor/positive/negative triplets
    let mut anchors = Vec::new();
    let mut positives = Vec::new();
    let mut negatives = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if j != i && labels[j] == labels[i] {
                for k in 0..n {
                    if labels[k] != labels[i] {
                        anchors.push(data_refs[i]);
                        positives.push(data_refs[j]);
                        negatives.push(data_refs[k]);
                        break;
                    }
                }
                break;
            }
        }
    }

    let triplet_out = triplet_loss(&anchors, &positives, &negatives, 0.5);
    println!("Triplet loss:              {:.4}", triplet_out.loss);

    let contrastive_pairs: Vec<(&[f32], &[f32])> = similar
        .iter()
        .map(|&(a, b)| (a, b))
        .chain(dissimilar.iter().map(|&(a, b)| (a, b)))
        .collect();
    let contrastive_labels: Vec<bool> = (0..similar.len())
        .map(|_| true)
        .chain((0..dissimilar.len()).map(|_| false))
        .collect();
    let contrastive_out = contrastive_loss(&contrastive_pairs, &contrastive_labels, 1.0);
    println!("Contrastive loss:          {:.4}", contrastive_out.loss);

    let infonce_out = infonce_loss(&anchors, &positives, 0.1);
    println!("InfoNCE loss:              {:.4}", infonce_out.loss);

    let n_pairs_out = n_pairs_loss(&anchors, &positives, 0.1);
    println!("N-pairs loss:              {:.4}", n_pairs_out.loss);

    let supcon_out = supcon_loss(&data_refs, &labels, 0.1);
    println!("SupCon loss:               {:.4}", supcon_out.loss);

    let ms_out = multi_similarity_loss(&data_refs, &labels, 2.0, 50.0, 0.5);
    println!("Multi-Similarity loss:     {:.4}", ms_out.loss);

    let circle_out = circle_loss(&data_refs, &labels, 0.25, 256.0);
    println!("Circle loss:               {:.4}", circle_out.loss);

    let lifted_out = lifted_structured_loss(&data_refs, &labels, 1.0);
    println!("Lifted Structured loss:    {:.4}", lifted_out.loss);
}

/// 1-nearest-neighbor accuracy.
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

type PairList<'a> = Vec<(&'a [f32], &'a [f32])>;

/// Build similar/dissimilar pair lists from labeled data.
fn build_pairs<'a>(data: &[&'a [f32]], labels: &[usize]) -> (PairList<'a>, PairList<'a>) {
    let n = data.len();
    let mut similar = Vec::new();
    let mut dissimilar = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if labels[i] == labels[j] {
                similar.push((data[i], data[j]));
            } else {
                dissimilar.push((data[i], data[j]));
            }
        }
    }
    (similar, dissimilar)
}
