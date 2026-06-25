//! Supervised metric learning on Fashion-MNIST with SupCon loss.
//!
//! Learns a 2-layer MLP embedding (784 -> 128 -> ReLU -> 64) by minimizing
//! tuplet's `supcon_loss` on labeled mini-batches. SupCon returns the gradient
//! w.r.t. each output embedding; the example backprops that through the MLP by
//! hand and updates with SGD. Evaluates by kNN classification accuracy on the
//! test split (learned embedding vs raw pixels).
//!
//! ```sh
//! ./scripts/fetch_fashion_mnist.sh
//! cargo run --release --example fashion_metric_learning
//! ```

// Index loops are the clearest form for the flat-matrix MLP forward/backward.
#![allow(clippy::needless_range_loop)]

use std::path::Path;
use std::process::ExitCode;

use tuplet::losses::supcon_loss;

const IN_DIM: usize = 784;
const HID: usize = 128;
const EMB_DIM: usize = 64;
const KNN: usize = 5;
const TEMP: f32 = 0.1;

fn be_u32(b: &[u8], o: usize) -> usize {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

fn load_images(path: &Path, limit: usize) -> std::io::Result<Vec<Vec<f32>>> {
    let b = std::fs::read(path)?;
    let n = be_u32(&b, 4).min(limit);
    let d = be_u32(&b, 8) * be_u32(&b, 12);
    Ok((0..n)
        .map(|i| {
            let s = 16 + i * d;
            b[s..s + d].iter().map(|&p| p as f32 / 255.0).collect()
        })
        .collect())
}

fn load_labels(path: &Path, limit: usize) -> std::io::Result<Vec<usize>> {
    let b = std::fs::read(path)?;
    let n = be_u32(&b, 4).min(limit);
    Ok(b[8..8 + n].iter().map(|&l| l as usize).collect())
}

/// Two-layer MLP: h_pre = W1 x, h = relu(h_pre), emb = W2 h.
/// W1 is [HID][IN_DIM], W2 is [EMB_DIM][HID], both row-major.
struct Mlp {
    w1: Vec<f32>,
    w2: Vec<f32>,
}

impl Mlp {
    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32; HID];
        for r in 0..HID {
            let row = &self.w1[r * IN_DIM..(r + 1) * IN_DIM];
            let pre: f32 = row.iter().zip(x).map(|(a, b)| a * b).sum();
            h[r] = pre.max(0.0);
        }
        let mut emb = vec![0.0f32; EMB_DIM];
        for r in 0..EMB_DIM {
            let row = &self.w2[r * HID..(r + 1) * HID];
            emb[r] = row.iter().zip(&h).map(|(a, b)| a * b).sum();
        }
        (emb, h)
    }

    fn embed(&self, x: &[f32]) -> Vec<f32> {
        self.forward(x).0
    }

    /// Accumulate grads for one sample given dL/demb, then return them folded
    /// into the running grad buffers (descent applied by the caller).
    fn backward(&self, x: &[f32], h: &[f32], g_emb: &[f32], g1: &mut [f32], g2: &mut [f32]) {
        // dL/dW2 = g_emb (x) h ; dL/dh = W2^T g_emb
        let mut g_h = vec![0.0f32; HID];
        for r in 0..EMB_DIM {
            let ge = g_emb[r];
            let w2row = &self.w2[r * HID..(r + 1) * HID];
            let g2row = &mut g2[r * HID..(r + 1) * HID];
            for c in 0..HID {
                g2row[c] += ge * h[c];
                g_h[c] += ge * w2row[c];
            }
        }
        // through ReLU (h>0) then dL/dW1 = (g_h ⊙ relu') (x) x
        for r in 0..HID {
            if h[r] <= 0.0 {
                continue;
            }
            let gh = g_h[r];
            if gh == 0.0 {
                continue;
            }
            let g1row = &mut g1[r * IN_DIM..(r + 1) * IN_DIM];
            for (gv, &xv) in g1row.iter_mut().zip(x) {
                *gv += gh * xv;
            }
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn knn_accuracy(
    queries: &[Vec<f32>],
    q_labels: &[usize],
    gallery: &[Vec<f32>],
    g_labels: &[usize],
) -> f64 {
    let mut correct = 0;
    for (q, &ql) in queries.iter().zip(q_labels) {
        let mut sims: Vec<(f32, usize)> = gallery
            .iter()
            .zip(g_labels)
            .map(|(g, &gl)| (cosine(q, g), gl))
            .collect();
        sims.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut votes = [0usize; 10];
        for &(_, gl) in sims.iter().take(KNN) {
            votes[gl] += 1;
        }
        let pred = votes.iter().enumerate().max_by_key(|&(_, &v)| v).unwrap().0;
        if pred == ql {
            correct += 1;
        }
    }
    correct as f64 / queries.len() as f64
}

fn shuffle(v: &mut [usize], state: &mut u64) {
    for i in (1..v.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        v.swap(i, (*state % (i as u64 + 1)) as usize);
    }
}

/// Small deterministic uniform init scaled by `scale`.
fn init(buf: &mut [f32], seed: &mut u64, scale: f32) {
    for w in buf.iter_mut() {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *w = ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale;
    }
}

fn main() -> ExitCode {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/fashion");
    if !dir.join("train-images-idx3-ubyte").exists() {
        eprintln!(
            "dataset not found at {}\nrun: ./scripts/fetch_fashion_mnist.sh",
            dir.display()
        );
        return ExitCode::SUCCESS;
    }

    let train_x = load_images(&dir.join("train-images-idx3-ubyte"), 6000).unwrap();
    let train_y = load_labels(&dir.join("train-labels-idx1-ubyte"), 6000).unwrap();
    let test_x = load_images(&dir.join("t10k-images-idx3-ubyte"), 2000).unwrap();
    let test_y = load_labels(&dir.join("t10k-labels-idx1-ubyte"), 2000).unwrap();
    println!(
        "train: {}  test: {}  emb_dim: {EMB_DIM}",
        train_x.len(),
        test_x.len()
    );

    let raw_acc = knn_accuracy(&test_x, &test_y, &train_x, &train_y);
    println!("raw-pixel kNN@{KNN} accuracy:      {raw_acc:.4}");

    let mut seed = 0xABCDEF1234567u64;
    let mut model = Mlp {
        w1: vec![0.0; HID * IN_DIM],
        w2: vec![0.0; EMB_DIM * HID],
    };
    init(&mut model.w1, &mut seed, 0.08);
    init(&mut model.w2, &mut seed, 0.12);

    let epochs = 60;
    let batch = 512;
    let lr = 10.0f32;
    let mut order: Vec<usize> = (0..train_x.len()).collect();
    let mut rng = 0x2468ACE0u64;

    for epoch in 1..=epochs {
        shuffle(&mut order, &mut rng);
        let mut epoch_loss = 0.0f32;
        let mut nb = 0;
        for chunk in order.chunks(batch) {
            let mut embs = Vec::with_capacity(chunk.len());
            let mut hs = Vec::with_capacity(chunk.len());
            for &i in chunk {
                let (e, h) = model.forward(&train_x[i]);
                embs.push(e);
                hs.push(h);
            }
            let labels: Vec<usize> = chunk.iter().map(|&i| train_y[i]).collect();
            let emb_refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();

            let out = supcon_loss(&emb_refs, &labels, TEMP);
            epoch_loss += out.loss;
            nb += 1;

            let mut g1 = vec![0.0f32; HID * IN_DIM];
            let mut g2 = vec![0.0f32; EMB_DIM * HID];
            for (bi, &i) in chunk.iter().enumerate() {
                model.backward(
                    &train_x[i],
                    &hs[bi],
                    &out.grad_anchors[bi],
                    &mut g1,
                    &mut g2,
                );
            }
            let scale = lr / chunk.len() as f32;
            for (w, g) in model.w1.iter_mut().zip(&g1) {
                *w -= scale * g;
            }
            for (w, g) in model.w2.iter_mut().zip(&g2) {
                *w -= scale * g;
            }
        }
        if epoch % 10 == 0 || epoch == 1 {
            println!(
                "epoch {epoch:>2}  supcon loss {:.4}",
                epoch_loss / nb as f32
            );
        }
    }

    let train_emb: Vec<Vec<f32>> = train_x.iter().map(|x| model.embed(x)).collect();
    let test_emb: Vec<Vec<f32>> = test_x.iter().map(|x| model.embed(x)).collect();
    let learned_acc = knn_accuracy(&test_emb, &test_y, &train_emb, &train_y);

    println!("learned-embedding kNN@{KNN} accuracy: {learned_acc:.4}");
    println!(
        "improvement:                       {:+.4}",
        learned_acc - raw_acc
    );

    ExitCode::SUCCESS
}
