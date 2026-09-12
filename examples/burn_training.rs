//! Burn-based metric learning: forward + backward smoke test.
//!
//! The loss API uses Burn 0.22's backend-independent tensors. This example
//! selects the pure-Rust Flex CPU runtime explicitly and enables autodiff on
//! that device; replace `Device::flex()` with another configured Burn device
//! to run the same work on a different backend. Batch tensors request `f32`
//! explicitly, independent of device defaults.
//!
//! Run: `cargo run --features burn-flex --example burn_training`

use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};
use tuplet::burn_losses;

fn batch(rows: &[&[f32]], device: &Device) -> Tensor<2> {
    let n = rows.len();
    let d = rows[0].len();
    let flat: Vec<f32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    Tensor::<1>::from_data(
        TensorData::new(flat, [n * d]),
        (device, FloatDType::F32.into()),
    )
    .reshape([n, d])
}

fn main() {
    let device = Device::flex().autodiff();

    let anchors = batch(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]], &device).require_grad();
    let positives = batch(&[&[0.9, 0.1], &[0.1, 0.9], &[0.95, 0.95]], &device);
    let negatives = batch(&[&[0.0, 1.0], &[1.0, 0.0], &[-1.0, 0.0]], &device);

    let triplet = burn_losses::triplet_loss(anchors.clone(), positives.clone(), negatives, 0.5);
    let triplet_value: f32 = triplet.clone().into_scalar();
    println!("triplet_loss = {triplet_value:.4}");
    let _ = triplet.backward();

    let infonce = burn_losses::infonce_loss(anchors.clone(), positives, 0.07);
    let infonce_value: f32 = infonce.clone().into_scalar();
    println!("infonce_loss = {infonce_value:.4}");
    let _ = infonce.backward();

    let a = batch(&[&[1.0, 0.0], &[0.9, 0.1]], &device).require_grad();
    let b = batch(&[&[0.95, 0.05], &[0.0, 1.0]], &device);
    let same_class = Tensor::<1, Int>::from_ints([1, 0], &device);
    let contrastive = burn_losses::contrastive_loss(a, b, same_class, 1.0);
    let contrastive_value: f32 = contrastive.clone().into_scalar();
    println!("contrastive_loss = {contrastive_value:.4}");
    let _ = contrastive.backward();

    println!("\nAll three losses computed forward + backward on Flex with f32 autodiff.");
}
