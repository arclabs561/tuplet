//! Loss values against their papers, plus a finite-difference gradient check.
// `!(err <= tol)` is deliberate: it also fails on NaN, which `err > tol` would pass.
#![allow(clippy::type_complexity, clippy::neg_cmp_op_on_partial_ord)]
use tuplet::*;

fn data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn refs(v: &[Vec<f32>]) -> Vec<&[f32]> {
    v.iter().map(|x| x.as_slice()).collect()
}

/// `f(groups)` returns a LossOutput; `fields[g]` picks which gradient field
/// belongs to group g (0 = grad_anchors, 1 = grad_positives, 2 = grad_negatives).
fn fd_check(
    name: &str,
    groups: Vec<Vec<Vec<f32>>>,
    fields: &[usize],
    f: &dyn Fn(&[Vec<Vec<f32>>]) -> LossOutput,
) -> Vec<String> {
    let out = f(&groups);
    let mut errs = vec![];
    let h = 1e-3f32;
    for (g, &field) in fields.iter().enumerate() {
        let grads = match field {
            0 => &out.grad_anchors,
            1 => &out.grad_positives,
            _ => &out.grad_negatives,
        };
        for i in 0..groups[g].len() {
            for d in 0..groups[g][i].len() {
                let mut gp = groups.clone();
                gp[g][i][d] += h;
                let mut gm = groups.clone();
                gm[g][i][d] -= h;
                let num = (f(&gp).loss - f(&gm).loss) / (2.0 * h);
                let ana = grads
                    .get(i)
                    .and_then(|r| r.get(d))
                    .copied()
                    .unwrap_or(f32::NAN);
                if !((num - ana).abs() <= 2e-3 + 3e-2 * num.abs().max(ana.abs())) {
                    errs.push(format!(
                        "{name}: group {g} [{i}][{d}] analytic {ana} numeric {num}"
                    ));
                }
            }
        }
    }
    errs
}

const LABELS: [usize; 6] = [0, 0, 1, 1, 2, 2];

#[test]
fn all_slice_losses_match_finite_difference_gradients() {
    let e = data(6, 4, 1);
    let p = data(6, 4, 2);
    let n = data(6, 4, 3);
    let proxies = data(3, 4, 4);
    let mut errs = vec![];
    errs.extend(fd_check(
        "triplet_euclid",
        vec![e.clone(), p.clone(), n.clone()],
        &[0, 1, 2],
        &|g| triplet_loss(&refs(&g[0]), &refs(&g[1]), &refs(&g[2]), 1.0),
    ));
    errs.extend(fd_check(
        "triplet_cos",
        vec![e.clone(), p.clone(), n.clone()],
        &[0, 1, 2],
        &|g| {
            triplet_loss_with_distance(
                &refs(&g[0]),
                &refs(&g[1]),
                &refs(&g[2]),
                0.5,
                DistanceFn::Cosine,
            )
        },
    ));
    errs.extend(fd_check(
        "infonce",
        vec![e.clone(), p.clone()],
        &[0, 1],
        &|g| infonce_loss(&refs(&g[0]), &refs(&g[1]), 0.5),
    ));
    errs.extend(fd_check(
        "mnrl",
        vec![e.clone(), p.clone(), n.clone()],
        &[0, 1, 2],
        &|g| mnrl_loss(&refs(&g[0]), &refs(&g[1]), &refs(&g[2]), 0.5),
    ));
    let labels_b = [true, false, true, false, false, true];
    errs.extend(fd_check(
        "cosine_embedding",
        vec![e.clone(), p.clone()],
        &[0, 1],
        &|g| {
            let pairs: Vec<(&[f32], &[f32])> = g[0]
                .iter()
                .zip(&g[1])
                .map(|(a, b)| (a.as_slice(), b.as_slice()))
                .collect();
            cosine_embedding_loss(&pairs, &labels_b, -0.5)
        },
    ));
    errs.extend(fd_check(
        "contrastive",
        vec![e.clone(), p.clone()],
        &[0, 1],
        &|g| {
            let pairs: Vec<(&[f32], &[f32])> = g[0]
                .iter()
                .zip(&g[1])
                .map(|(a, b)| (a.as_slice(), b.as_slice()))
                .collect();
            contrastive_loss(&pairs, &labels_b, 3.0)
        },
    ));
    errs.extend(fd_check("multi_similarity", vec![e.clone()], &[0], &|g| {
        multi_similarity_loss(&refs(&g[0]), &LABELS, 2.0, 50.0, 0.5)
    }));
    errs.extend(fd_check("supcon", vec![e.clone()], &[0], &|g| {
        supcon_loss(&refs(&g[0]), &LABELS, 0.5)
    }));
    errs.extend(fd_check("circle", vec![e.clone()], &[0], &|g| {
        circle_loss(&refs(&g[0]), &LABELS, 0.25, 4.0)
    }));
    errs.extend(fd_check("lifted", vec![e.clone()], &[0], &|g| {
        lifted_structured_loss(&refs(&g[0]), &LABELS, 1.0)
    }));
    errs.extend(fd_check(
        "n_pairs",
        vec![e.clone(), p.clone()],
        &[0, 1],
        &|g| n_pairs_loss(&refs(&g[0]), &refs(&g[1]), 0.5),
    ));
    let lab3 = [0usize, 1, 2, 0, 1, 2];
    errs.extend(fd_check(
        "arcface",
        vec![e.clone(), proxies.clone()],
        &[0, 1],
        &|g| arcface_loss(&refs(&g[0]), &lab3, &refs(&g[1]), 4.0, 0.3),
    ));
    errs.extend(fd_check(
        "vicreg",
        vec![e.clone(), p.clone()],
        &[0, 1],
        &|g| vicreg_loss(&refs(&g[0]), &refs(&g[1]), 25.0, 25.0, 1.0),
    ));
    errs.extend(fd_check(
        "proxy_anchor",
        vec![e.clone(), proxies.clone()],
        &[0, 1],
        &|g| proxy_anchor_loss(&refs(&g[0]), &lab3, &refs(&g[1]), 0.1, 8.0),
    ));
    assert!(
        errs.is_empty(),
        "{} mismatches:\n{}",
        errs.len(),
        errs.join("\n")
    );
}

fn cos64(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    d / (na * nb)
}

/// Kim et al. 2020 Eq. 4: positive term log(1 + sum_{x in X_p+} exp(-alpha (s - delta))),
/// averaged over |P+|; negative term averaged over all proxies |P|.
#[test]
fn proxy_anchor_matches_kim_2020_eq4() {
    let e = data(6, 4, 11);
    let proxies = data(4, 4, 12); // class 3 has no samples in the batch
    let labels = [0usize, 0, 0, 1, 1, 2];
    let (delta, alpha) = (0.1f64, 4.0f64);
    let mut pos_sum = 0.0;
    let mut n_pos_proxies = 0;
    let mut neg_sum = 0.0;
    for (c, p) in proxies.iter().enumerate() {
        let pos: Vec<&Vec<f32>> = e
            .iter()
            .zip(&labels)
            .filter(|(_, l)| **l == c)
            .map(|(x, _)| x)
            .collect();
        let neg: Vec<&Vec<f32>> = e
            .iter()
            .zip(&labels)
            .filter(|(_, l)| **l != c)
            .map(|(x, _)| x)
            .collect();
        if !pos.is_empty() {
            n_pos_proxies += 1;
            let s: f64 = pos
                .iter()
                .map(|x| (-alpha * (cos64(x, p) - delta)).exp())
                .sum();
            pos_sum += (1.0 + s).ln();
        }
        let s: f64 = neg
            .iter()
            .map(|x| (alpha * (cos64(x, p) + delta)).exp())
            .sum();
        neg_sum += (1.0 + s).ln();
    }
    let expected = pos_sum / n_pos_proxies as f64 + neg_sum / proxies.len() as f64;
    let got = proxy_anchor_loss(
        &refs(&e),
        &labels,
        &refs(&proxies),
        delta as f32,
        alpha as f32,
    )
    .loss;
    assert!(
        (got as f64 - expected).abs() < 1e-3 * expected.abs().max(1.0),
        "proxy_anchor_loss = {got}, Kim et al. Eq. 4 = {expected}"
    );
}

/// Song et al. 2016 Eq. 4: J_ij = log(sum_k exp(a - D_ik) + sum_l exp(a - D_jl)) + D_ij,
/// loss = 1/(2|P|) sum max(0, J_ij)^2.
#[test]
fn lifted_structured_matches_song_2016_eq4() {
    let e = data(6, 4, 21);
    let labels = LABELS;
    let margin = 1.0f64;
    let d = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut total = 0.0;
    let mut count = 0;
    for i in 0..6 {
        for j in (i + 1)..6 {
            if labels[i] != labels[j] {
                continue;
            }
            let mut s = 0.0;
            for k in 0..6 {
                if labels[k] != labels[i] {
                    s += (margin - d(&e[i], &e[k])).exp();
                }
                if labels[k] != labels[j] {
                    s += (margin - d(&e[j], &e[k])).exp();
                }
            }
            let jij = s.ln() + d(&e[i], &e[j]);
            total += jij.max(0.0).powi(2);
            count += 1;
        }
    }
    let expected = total / (2.0 * count as f64);
    let got = lifted_structured_loss(&refs(&e), &labels, margin as f32).loss;
    assert!(
        (got as f64 - expected).abs() < 1e-3 * expected.abs().max(1.0),
        "lifted_structured_loss = {got}, Song et al. Eq. 4 = {expected}"
    );
}
