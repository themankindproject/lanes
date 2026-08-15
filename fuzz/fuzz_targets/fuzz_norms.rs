//! Fuzz target for the norm/ML composition family (`rms_norm`,
//! `cosine_similarity`).
//!
//! Verifies no panics on arbitrary input, the rms_norm scale-invariance
//! invariant, and the cosine_similarity contract (error on length
//! mismatch, EmptyInput on empty input, 0.0 for zero-norm vectors,
//! [-1, 1] output for finite input).
//!
//! Run with: `cargo +nightly fuzz run fuzz_norms`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct NormsInput {
    a: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

fuzz_target!(|input: NormsInput| {
    let a = &input.a;
    let b = &input.b;
    // Keep eps in a sane range; NaN eps would poison the invariant.
    let eps = if input.eps.is_finite() {
        input.eps.abs().max(1e-12)
    } else {
        1e-5
    };

    // No panic, ever.
    let r = lanes::ml::f32::rms_norm(a, eps);
    assert_eq!(r.len(), a.len());

    // rms_norm scale invariance: rms_norm(c·x, c²·eps) == rms_norm(x, eps),
    // since x/sqrt(m+eps) is homogeneous only when eps scales with c².
    // Guard the scaled sum-of-squares too: if it overflows to inf, the
    // ratio legitimately becomes 0 and invariance doesn't hold.
    let max_abs = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let scale_safe = max_abs.is_finite()
        && max_abs * 3.5 < (f32::MAX / (a.len() as f32).max(1.0)).sqrt()
        && a.iter().all(|x| x.is_finite());
    // Skip when eps dominates the mean: then the outputs are x/sqrt(eps),
    // which underflow/overflow independently of the scaling property.
    let mean_sq = if a.is_empty() {
        0.0
    } else {
        lanes::stats::f32::sum_sq(a) / a.len() as f32
    };
    if scale_safe && !a.is_empty() && eps <= 0.1 * mean_sq {
        let scaled: Vec<f32> = a.iter().map(|x| x * 3.5).collect();
        let r1 = lanes::ml::f32::rms_norm(a, eps);
        let r2 = lanes::ml::f32::rms_norm(&scaled, eps * 3.5 * 3.5);
        // Skip when either computed norm is non-finite: the SIMD
        // reduction tree can overflow to inf even when a per-element
        // bound holds, and invariance is a formula property, not an
        // IEEE-overflow property.
        let finite = lanes::stats::f32::sum_sq(a).is_finite()
            && lanes::stats::f32::sum_sq(&scaled).is_finite();
        if finite {
            for i in 0..a.len() {
                // Skip the denormal zone: outputs below ~1e-20 flush to
                // zero on some paths (SIMD vs scalar rsqrt rounding).
                if r1[i].abs() < 1e-20 && r2[i].abs() < 1e-20 {
                    continue;
                }
                // Relative tolerance only: the 3.5 factors and three
                // sqrt/rsqrt roundings cost ~1e-5 relative in f32.
                let tol = 1e-4 * r1[i].abs();
                assert!(
                    (r1[i] - r2[i]).abs() <= tol,
                    "scale invariance broken at {i}: {} vs {}, a={a:?} eps={eps}",
                    r1[i],
                    r2[i]
                );
            }
        }
    }

    match lanes::ml::f32::cosine_similarity(a, b) {
        Ok(s) => {
            assert_eq!(a.len(), b.len());
            assert!(!a.is_empty());
            let na = lanes::stats::f32::sum_sq(a);
            let nb = lanes::stats::f32::sum_sq(b);
            if na == 0.0 || nb == 0.0 {
                // Zero norm (including tiny values whose squares underflow
                // to 0 in f32) → 0.0 by convention.
                assert_eq!(s, 0.0, "cosine zero-norm must be 0.0, a={a:?} b={b:?}");
            } else if a.iter().all(|x| x.is_finite())
                && b.iter().all(|x| x.is_finite())
                && na.is_finite()
                && nb.is_finite()
                && na > 1e-30
                && nb > 1e-30
            {
                // [-1, 1] only holds when no intermediate overflows AND both
                // norms stay out of the denormal zone (sqrt/dot rounding at
                // 1e-45-magnitude norms legitimately breaks the bound; the
                // f64 reference gives exactly 1.0 there).
                assert!(
                    (-1.0 - 1e-4..=1.0 + 1e-4).contains(&s),
                    "cosine {s} out of [-1, 1], a={a:?} b={b:?} na={na} nb={nb}"
                );
                assert!(s.is_finite());
            }
        }
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, a.len());
            assert_eq!(actual, b.len());
            assert_ne!(a.len(), b.len());
        }
        Err(lanes::Error::EmptyInput) => {
            assert!(a.is_empty() && b.is_empty());
        }
    }

    // layer_norm: no panic, same length, unit variance on finite, non-empty
    // input whose sum doesn't overflow (the mean is sum/n, so a finite
    // mean needs a finite sum — IEEE overflow legitimately NaNs the rest).
    // Variance is the exact invariant: the output mean is only ~0 when
    // centering is exact (f32 rounding of a huge common offset survives).
    let lnorm = lanes::ml::f32::layer_norm(a, eps);
    assert_eq!(lnorm.len(), a.len());
    if a.len() > 1
        && a.iter().all(|x| x.is_finite())
        && lanes::stats::f32::sum(a).is_finite()
        && lanes::stats::f32::sum_sq(a) > 1e-30
    {
        let out_ss = lanes::stats::f32::sum_sq(&lnorm);
        if out_ss.is_finite() && out_ss > 1e-30 {
            // Exact invariant: output variance = v/(v+eps), where v is the
            // input variance. ~1 when eps ≪ v; tiny when eps dominates.
            let n = a.len() as f32;
            let mean = lanes::stats::f32::sum(a) / n;
            let centered: Vec<f32> = a.iter().map(|x| x - mean).collect();
            let v = lanes::stats::f32::sum_sq(&centered) / n;
            let want = v / (v + eps);
            let var = out_ss / n;
            assert!(
                (var - want).abs() < 1e-3 * want.max(1e-3),
                "layer_norm var {var} want {want} (a={a:?} eps={eps})"
            );
        }
    }

    // layer_norm_into must agree with layer_norm elementwise.
    let mut into = vec![0.0_f32; a.len()];
    lanes::ml::f32::layer_norm_into(a, eps, &mut into);
    for (x, y) in into.iter().zip(lnorm.iter()) {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        assert!(
            (x - y).abs() < 1e-4 * y.abs().max(1.0),
            "layer_norm_into {x} != layer_norm {y} (a={a:?} eps={eps})"
        );
    }

    // logsumexp: no panic; NaN input → NaN output; finite input → finite
    // output. (Inf input may also NaN: max=inf makes x-max = NaN.)
    let lse = lanes::ml::f32::logsumexp(a);
    if !a.is_empty() {
        if a.iter().any(|x| x.is_nan()) {
            assert!(lse.is_nan(), "logsumexp({a:?}) = {lse:e} should be NaN");
        }
        if a.iter().all(|x| x.is_finite()) {
            assert!(
                lse.is_finite(),
                "logsumexp({a:?}) = {lse:e} should be finite"
            );
        }
    }

    // log_softmax_into: no panic, same length; agrees with the Vec wrapper.
    let mut lsm = vec![0.0_f32; a.len()];
    lanes::ml::f32::log_softmax_into(a, &mut lsm);
    let ls_alloc = lanes::ml::f32::log_softmax(a);
    for (x, y) in lsm.iter().zip(ls_alloc.iter()) {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        assert!(
            (x - y).abs() < 1e-4 * y.abs().max(1.0),
            "log_softmax_into {x} != log_softmax {y} (a={a:?})"
        );
    }

    // geometric_mean: None exactly when a value is ≤ 0 or NaN; otherwise
    // positive (finite inputs may overflow to inf: exp(mean(ln x)) with
    // mean(ln x) > 88.7 in f32).
    let gm = lanes::stats::f32::geometric_mean(a);
    if a.iter().any(|x| *x <= 0.0 || x.is_nan()) || a.is_empty() {
        assert!(gm.is_none(), "geometric_mean({a:?}) = {gm:?} should be None");
    } else if a.iter().all(|x| x.is_finite()) {
        let g = gm.unwrap();
        assert!(g > 0.0, "geometric_mean({a:?}) = {g}");
    }
});
