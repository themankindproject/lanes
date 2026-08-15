//! Fuzz target for the f64 family.
//!
//! Verifies that double-precision kernels never panic on arbitrary input
//! (NaN, inf, denormals, any length) and, where a cheap exact property
//! exists, that the result matches a naive reference.
//!
//! Run with: `cargo +nightly fuzz run fuzz_f64`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct F64Input {
    a: Vec<f64>,
    b: Vec<f64>,
}

fuzz_target!(|input: F64Input| {
    let a = &input.a;
    let b = &input.b;

    // No panic, ever.
    let _ = lanes::stats::f64::sum(a);
    let _ = lanes::stats::f64::prod(a);
    let _ = lanes::stats::f64::min(a);
    let _ = lanes::stats::f64::max(a);
    let _ = lanes::stats::f64::argmax(a);
    let _ = lanes::stats::f64::argmin(a);
    let _ = lanes::stats::f64::mean(a);
    let _ = lanes::distance::f64::l1_norm(a);
    let _ = lanes::distance::f64::max_norm(a);

    // Exact agreements where IEEE order is deterministic.
    if a.iter().all(|x| x.is_finite()) {
        // Empty identity.
        if a.is_empty() {
            assert_eq!(lanes::stats::f64::sum(a), 0.0);
            assert_eq!(lanes::stats::f64::prod(a), 1.0);
        }
        // Naive reference vs SIMD: reordering changes the last ulp, so
        // assert the Higham summation bound (γ·Σ|x|), not bit equality.
        // inf==inf (both overflow) is accepted directly; inf - inf = NaN.
        let naive_sum: f64 = a.iter().sum();
        let got = lanes::stats::f64::sum(a);
        if got == naive_sum {
            return; // bit-equal, done
        }
        let n = a.len().max(1) as f64;
        let u = f64::EPSILON / 2.0;
        let gamma = n * u / (1.0 - n * u);
        let bound = 16.0 * gamma * a.iter().map(|x| x.abs()).sum::<f64>();
        assert!(
            (got - naive_sum).abs() <= bound.max(f64::MIN_POSITIVE),
            "sum: {got} vs naive {naive_sum}, bound {bound}"
        );
    }

    // argmax/argmin consistency: the returned index must point at the
    // extremum (first occurrence, strict >). NaN inputs are ignored unless
    // every element is NaN, in which case index 0 wins.
    if let Some(i) = lanes::stats::f64::argmax(a) {
        assert!(i < a.len());
        let all_nan = !a.is_empty() && a.iter().all(|x| x.is_nan());
        if all_nan {
            assert_eq!(i, 0, "argmax: all-NaN must return first index");
        } else {
            assert!(
                a.iter().enumerate().all(|(j, &x)| !(j < i && x >= a[i])),
                "argmax: index {i} not first occurrence, a={a:?}"
            );
            assert!(
                a.iter().enumerate().all(|(j, &x)| j == i || x <= a[i] || x.is_nan()),
                "argmax: index {i} value {} not maximal, a={a:?}",
                a[i]
            );
        }
    }
    if let Some(i) = lanes::stats::f64::argmin(a) {
        assert!(i < a.len());
        let all_nan = !a.is_empty() && a.iter().all(|x| x.is_nan());
        if all_nan {
            assert_eq!(i, 0, "argmin: all-NaN must return first index");
        } else {
            assert!(
                a.iter().enumerate().all(|(j, &x)| !(j < i && x <= a[i])),
                "argmin: index {i} not first occurrence"
            );
        }
    }

    // dot: length agreement.
    if let Ok(v) = lanes::stats::f64::dot(a, b) {
        assert_eq!(a.len(), b.len());
        let _ = v;
    }

    // std_dev: empty → None, single → 0.0, else finite or NaN for inf input.
    if a.iter().all(|x| x.is_finite()) {
        if a.is_empty() {
            assert_eq!(lanes::stats::f64::std_dev(a), None);
        } else if a.len() == 1 {
            assert_eq!(lanes::stats::f64::std_dev(a), Some(0.0));
        }
    }

    // ML kernels: no panic; `_into` forms agree with the allocating
    // wrappers; log_softmax exp-sum is ~1 when no exp overflows.
    let lse = lanes::ml::f64::logsumexp(a);
    if !a.is_empty() && a.iter().all(|x| x.is_nan()) {
        assert!(lse.is_nan(), "logsumexp all-NaN must be NaN");
    }
    let mut ls = vec![0.0_f64; a.len()];
    lanes::ml::f64::log_softmax_into(a, &mut ls);
    let ls_alloc = lanes::ml::f64::log_softmax(a);
    for (x, y) in ls.iter().zip(ls_alloc.iter()) {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        assert!(
            (x - y).abs() < 1e-9 * y.abs().max(1.0),
            "log_softmax_into {x} != log_softmax {y} (a={a:?})"
        );
    }
    if !a.is_empty()
        && a.iter().all(|x| x.is_finite())
        && lse.is_finite()
        && lse < 700.0
    {
        let sum: f64 = ls.iter().map(|&x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "log_softmax exp-sum {sum}, a={a:?}");
    }
    let eps = 1e-9;
    let mut ln = vec![0.0_f64; a.len()];
    lanes::ml::f64::layer_norm_into(a, eps, &mut ln);
    let ln_alloc = lanes::ml::f64::layer_norm(a, eps);
    for (x, y) in ln.iter().zip(ln_alloc.iter()) {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        assert!(
            (x - y).abs() < 1e-9 * y.abs().max(1.0),
            "layer_norm_into {x} != layer_norm {y} (a={a:?})"
        );
    }
});
