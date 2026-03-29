//! Candle-based training loop using InfoNCE and SupCon losses.
//!
//! Demonstrates the candle feature: tensor-based losses with autograd.
//! A small linear model learns to separate 3 classes.
//!
//! Run: `cargo run --features candle --example candle_training`

use candle_core::{Device, Tensor, Var};

fn main() -> candle_core::Result<()> {
    let device = &Device::Cpu;

    // Generate 3-class 2D data
    let class_data: Vec<Vec<[f32; 4]>> = vec![
        // Class 0: near origin
        vec![
            [0.1, 0.05, 0.0, 0.1],
            [-0.1, 0.1, 0.05, -0.05],
            [0.05, -0.1, 0.1, 0.0],
        ],
        // Class 1: shifted right
        vec![
            [2.1, 0.05, 2.0, 0.1],
            [1.9, 0.1, 2.05, -0.05],
            [2.05, -0.1, 1.9, 0.0],
        ],
        // Class 2: shifted up
        vec![
            [0.1, 2.05, 0.0, 2.1],
            [-0.1, 1.9, 0.05, 1.95],
            [0.05, 2.1, 0.1, 2.0],
        ],
    ];

    let mut all_data = Vec::new();
    let mut all_labels = Vec::new();
    for (class, samples) in class_data.iter().enumerate() {
        for sample in samples {
            all_data.extend_from_slice(sample);
            all_labels.push(class as u32);
        }
    }
    let n = 9;
    let in_dim = 4;
    let out_dim = 2;

    let data = Tensor::from_slice(&all_data, &[n, in_dim], device)?;
    let labels = Tensor::from_slice(&all_labels, &[n], device)?;

    // Learnable linear projection: [in_dim, out_dim]
    let w = Var::from_slice(
        &[0.5f32, 0.0, 0.0, 0.5, 0.1, -0.1, -0.1, 0.1],
        &[in_dim, out_dim],
        device,
    )?;

    let lr = 0.01;

    println!("=== InfoNCE training ===\n");

    // Build anchor/positive pairs (same class)
    let mut anchor_indices = Vec::new();
    let mut positive_indices = Vec::new();
    for class in 0..3 {
        let start = class * 3;
        anchor_indices.push(start as u32);
        positive_indices.push((start + 1) as u32);
        anchor_indices.push((start + 1) as u32);
        positive_indices.push((start + 2) as u32);
    }
    let a_idx = Tensor::from_slice(&anchor_indices, &[6], device)?;
    let p_idx = Tensor::from_slice(&positive_indices, &[6], device)?;

    for epoch in 0..100 {
        let projected = data.matmul(w.as_tensor())?;
        let anchors = projected.index_select(&a_idx, 0)?;
        let positives = projected.index_select(&p_idx, 0)?;

        let loss = mahala::candle_losses::infonce_loss(&anchors, &positives, 0.5)?;

        if epoch % 20 == 0 {
            let val = loss.to_scalar::<f32>()?;
            println!("epoch {epoch:3}: infonce_loss = {val:.4}");
        }

        let grads = loss.backward()?;
        let grad_w = grads.get(w.as_tensor()).unwrap();
        w.set(&w.as_tensor().sub(&grad_w.affine(lr, 0.0)?)?)?;
    }

    println!("\n=== SupCon training ===\n");

    // Reset weights
    w.set(&Tensor::from_slice(
        &[0.5f32, 0.0, 0.0, 0.5, 0.1, -0.1, -0.1, 0.1],
        &[in_dim, out_dim],
        device,
    )?)?;

    for epoch in 0..100 {
        let projected = data.matmul(w.as_tensor())?;
        let loss = mahala::candle_losses::supcon_loss(&projected, &labels, 0.5)?;

        if epoch % 20 == 0 {
            let val = loss.to_scalar::<f32>()?;
            println!("epoch {epoch:3}: supcon_loss = {val:.4}");
        }

        let grads = loss.backward()?;
        let grad_w = grads.get(w.as_tensor()).unwrap();
        w.set(&w.as_tensor().sub(&grad_w.affine(lr, 0.0)?)?)?;
    }

    // Print final embeddings
    let final_proj = data.matmul(w.as_tensor())?;
    println!("\nFinal embeddings:");
    for (i, label) in all_labels.iter().enumerate() {
        let emb = final_proj.get(i)?.to_vec1::<f32>()?;
        println!("  class {label}: [{:.3}, {:.3}]", emb[0], emb[1]);
    }

    Ok(())
}
