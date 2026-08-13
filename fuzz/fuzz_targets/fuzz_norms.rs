//! Fuzz target for the norm/ML composition family (`rms_norm`,
//! `cosine_similarity`).
//!
//! Verifies no panics on arbitrary input, the rms_norm scale-invariance
//! invariant, and the cosine_similarity contract (error on length
//! mismatch, None for zero vectors, [-1, 1] output for finite input).
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
        Ok(None) => {
            // None requires equal lengths and a zero computed norm. Tiny
            // values whose squares underflow to 0 in f32 also legitimately
            // yield None.
            assert_eq!(a.len(), b.len());
            let na = lanes::stats::f32::sum_sq(a);
            let nb = lanes::stats::f32::sum_sq(b);
            assert!(
                a.is_empty() || na == 0.0 || nb == 0.0,
                "cosine None but non-zero norms, a={a:?} b={b:?}"
            );
        }
        Ok(Some(s)) => {
            assert_eq!(a.len(), b.len());
            assert!(!a.is_empty());
            // [-1, 1] only holds when no intermediate overflows: finite
            // inputs can still produce inf dot or inf norms, making the
            // ratio NaN. Guard on the computed norms, not the inputs.
            let na = lanes::stats::f32::sum_sq(a);
            let nb = lanes::stats::f32::sum_sq(b);
            if a.iter().all(|x| x.is_finite())
                && b.iter().all(|x| x.is_finite())
                && na.is_finite()
                && nb.is_finite()
            {
                assert!(
                    (-1.0 - 1e-4..=1.0 + 1e-4).contains(&s),
                    "cosine {s} out of [-1, 1]"
                );
                assert!(s.is_finite());
            }
        }
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, a.len());
            assert_eq!(actual, b.len());
            assert_ne!(a.len(), b.len());
        }
        Err(_) => panic!("unexpected error variant"),
    }
});
