use tuplet::multi_similarity_loss;

#[test]
fn finite_logits_keep_loss_and_gradients_finite() {
    let unit: &[f32] = &[1.0, 0.0];
    let out = multi_similarity_loss(&[unit, unit, unit, unit], &[0, 0, 1, 1], 2.0, 200.0, 0.5);
    // Each anchor has one positive logit -1 and two negative logits 100.
    // Evaluate the defining expression in f64, whose exponent range covers these logits.
    let expected = (1.0 + (-1.0_f64).exp()).ln() / 2.0 + (1.0 + 2.0 * 100.0_f64.exp()).ln() / 200.0;
    assert!(out.loss.is_finite(), "finite logits produced {}", out.loss);
    assert!((f64::from(out.loss) - expected).abs() < 1e-6);
    // Cosine has zero derivative between identical nonzero vectors.
    for grad in &out.grad_anchors {
        for &value in grad {
            assert!(value.is_finite());
            assert!(value.abs() < 1e-6);
        }
    }
}

// Direct defining formula in f64: independent of the production shifted sum.
// Test logits stay below the f64 exponential limit while exceeding f32's.
fn reference(embeddings: &[Vec<f64>], labels: &[usize], alpha: f64, beta: f64, base: f64) -> f64 {
    let cosine = |a: &[f64], b: &[f64]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let aa: f64 = a.iter().map(|x| x * x).sum();
        let bb: f64 = b.iter().map(|x| x * x).sum();
        dot / (aa * bb).sqrt()
    };
    let mut loss = 0.0;
    for (i, a) in embeddings.iter().enumerate() {
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for (j, b) in embeddings.iter().enumerate() {
            if i == j {
                continue;
            }
            let similarity = cosine(a, b);
            if labels[i] == labels[j] {
                pos.push((-alpha * (similarity - base)).exp());
            } else {
                neg.push((beta * (similarity - base)).exp());
            }
        }
        if !pos.is_empty() && !neg.is_empty() {
            loss +=
                pos.iter().sum::<f64>().ln_1p() / alpha + neg.iter().sum::<f64>().ln_1p() / beta;
        }
    }
    loss / embeddings.len() as f64
}

#[test]
fn high_positive_and_negative_logits_have_nonzero_correct_gradients() {
    let embeddings = [
        vec![1.0_f32, 0.0],
        vec![-0.8, 0.6],
        vec![0.8, 0.6],
        vec![0.0, -1.0],
    ];
    let labels = [0, 0, 1, 1];
    let refs: Vec<_> = embeddings.iter().map(Vec::as_slice).collect();
    let out = multi_similarity_loss(&refs, &labels, 100.0, 200.0, 0.0);
    let wide: Vec<Vec<f64>> = embeddings
        .iter()
        .map(|e| e.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let expected = reference(&wide, &labels, 100.0, 200.0, 0.0);
    assert!((f64::from(out.loss) - expected).abs() < 2e-6);
    let mut nonzero = false;
    for i in 0..embeddings.len() {
        for k in 0..2 {
            let mut plus = wide.clone();
            let mut minus = wide.clone();
            let h = 1e-5;
            plus[i][k] += h;
            minus[i][k] -= h;
            let numerical = (reference(&plus, &labels, 100.0, 200.0, 0.0)
                - reference(&minus, &labels, 100.0, 200.0, 0.0))
                / (2.0 * h);
            nonzero |= numerical.abs() > 0.01;
            let actual = f64::from(out.grad_anchors[i][k]);
            assert!(actual.is_finite());
            assert!(
                (actual - numerical).abs() < 2e-4,
                "gradient ({i},{k}): {actual} vs {numerical}"
            );
        }
    }
    assert!(
        nonzero,
        "fixture must detect a zero-gradient implementation"
    );
}

#[test]
fn positive_exponential_overflow_is_stable() {
    let a: &[f32] = &[1.0, 0.0];
    let b: &[f32] = &[-1.0, 0.0];
    let out = multi_similarity_loss(&[a, b, a, b], &[0, 0, 1, 1], 100.0, 50.0, 0.5);
    let wide = vec![
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
    ];
    assert!((f64::from(out.loss) - reference(&wide, &[0, 0, 1, 1], 100.0, 50.0, 0.5)).abs() < 1e-6);
    assert!(out.grad_anchors.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn tiny_losses_and_skipped_anchors_preserve_the_defining_reduction() {
    let a: &[f32] = &[1.0, 0.0];
    let b: &[f32] = &[-1.0, 0.0];
    let tiny = multi_similarity_loss(&[a, a, b, b], &[0, 0, 1, 1], 40.0, 40.0, 0.0);
    let expected = ((-40.0_f64).exp().ln_1p() + (2.0 * (-40.0_f64).exp()).ln_1p()) / 40.0;
    assert!(tiny.loss > 0.0);
    assert!((f64::from(tiny.loss) / expected - 1.0).abs() < 1e-6);
    let skipped = multi_similarity_loss(&[a, a, b], &[0, 0, 1], 2.0, 3.0, 0.5);
    let wide = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![-1.0, 0.0]];
    assert!((f64::from(skipped.loss) - reference(&wide, &[0, 0, 1], 2.0, 3.0, 0.5)).abs() < 1e-7);
    for labels in [[0, 0, 0], [0, 1, 2]] {
        let none = multi_similarity_loss(&[a, a, b], &labels, 2.0, 3.0, 0.5);
        assert_eq!(none.loss, 0.0);
        assert!(none.grad_anchors.iter().flatten().all(|&v| v == 0.0));
    }
}
