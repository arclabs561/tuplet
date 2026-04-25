//! Burn-based metric learning: forward + backward smoke test.
//!
//! Computes triplet and InfoNCE losses on the `NdArray` autodiff backend and
//! verifies gradients flow. The same code runs on `burn-wgpu`, `burn-tch`, or
//! `burn-cuda` by changing only the backend type alias.
//!
//! A full training-loop example with `Module` + AdamW will land once all 16
//! losses are ported and the abstraction surface settles.
//!
//! Run: `cargo run --features burn-ndarray --example burn_training`

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_ndarray::NdArray;
use tuplet::burn_losses;

type B = Autodiff<NdArray>;

fn batch(rows: &[&[f32]], device: &<B as Backend>::Device) -> Tensor<B, 2> {
    let n = rows.len();
    let d = rows[0].len();
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::<B, 1>::from_data(TensorData::new(flat, [n * d]), device).reshape([n, d])
}

fn main() {
    let device = Default::default();

    let anchors = batch(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]], &device).require_grad();
    let positives = batch(&[&[0.9, 0.1], &[0.1, 0.9], &[0.95, 0.95]], &device);
    let negatives = batch(&[&[0.0, 1.0], &[1.0, 0.0], &[-1.0, 0.0]], &device);

    let triplet = burn_losses::triplet_loss(anchors.clone(), positives.clone(), negatives, 0.5);
    let triplet_val: f32 = triplet.clone().into_scalar();
    println!("triplet_loss = {triplet_val:.4}");
    let _ = triplet.backward();

    let infonce = burn_losses::infonce_loss(anchors.clone(), positives, 0.07);
    let infonce_val: f32 = infonce.clone().into_scalar();
    println!("infonce_loss = {infonce_val:.4}");
    let _ = infonce.backward();

    let a = batch(&[&[1.0, 0.0], &[0.9, 0.1]], &device).require_grad();
    let b = batch(&[&[0.95, 0.05], &[0.0, 1.0]], &device);
    let same_class = Tensor::<B, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![1i64, 0i64], [2]),
        &device,
    );
    let contrastive = burn_losses::contrastive_loss(a, b, same_class, 1.0);
    let contrastive_val: f32 = contrastive.clone().into_scalar();
    println!("contrastive_loss = {contrastive_val:.4}");
    let _ = contrastive.backward();

    println!("\nAll three losses computed forward + backward on Autodiff<NdArray>.");
}
