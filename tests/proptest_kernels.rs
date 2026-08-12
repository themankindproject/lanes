//! Property-based tests using `proptest`.
//!
//! These tests generate random inputs and verify that `lanes` functions
//! match naive Rust iterator implementations.
//!
//! Tolerance philosophy: summation error is bounded by Higham's error
//! analysis, `|err| <= gamma_n * sum(|x_i|)` (unit roundoff u = 2^-24). We
//! compare against a tolerance derived from the *input* magnitudes rather
//! than the result magnitude, which stays valid under catastrophic
//! cancellation. Input magnitude bounds only prevent sum overflow (`inf`
//! vs `NaN` order effects, documented as backend-dependent).

use proptest::prelude::*;

/// Strategy for a Vec<f32> of finite values with length 0..1000.
///
/// Magnitudes are bounded so sums of up to 1000 values cannot overflow to
/// infinity. Sum overflow behavior is documented as backend-dependent and
/// is excluded from the equality property.
fn finite_f32_vec() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(
        any::<f32>().prop_filter("bounded finite", |x| x.is_finite() && x.abs() < 1e30),
        0..1000,
    )
}

/// Strategy for softmax inputs: `exp(x)` overflows to `inf` for |x| > ~88.7,
/// which would make `inf/inf = NaN` in softmax. Sample directly from
/// `[-40, 40]` (uniform, no filtering/rejection) with margin so that adding
/// the shift-invariance shift stays < 88 too.
#[cfg(feature = "alloc")]
fn softmax_f32_vec() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-40.0_f32..40.0, 0..1000)
}

/// Strategy for dot-product inputs: products of two values must stay finite
/// and far enough from overflow that 1000 products cannot tip the sum.
fn bounded_f32_vec() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(
        any::<f32>().prop_filter("bounded finite", |x| x.is_finite() && x.abs() < 1e15),
        0..1000,
    )
}

/// Values in the range where f32 arithmetic is well-conditioned for
/// sum-based comparisons: ~24 bits of mantissa, no catastrophic term loss.
/// Used for f64-reference tests, where the reference must be meaningful.
fn mid_f32_vec() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1e6_f32..1e6, 0..512)
}

/// Error bound for reduction over `n` terms of magnitudes `|x_i|`, per
/// Higham: with u = 2^-24 and 16x slack, `|err| <= scale * n * 2^-20`.
/// The bound is the same for any summation order, so SIMD chunked
/// reductions and scalar left-to-right reductions both satisfy it.
fn reduction_tolerance(n: usize, scale: f64) -> f64 {
    scale * (n as f64) * 2_f64.powi(-20) + 1.0 // +1 nanosecond-scale slack for near-zero inputs
}

/// Check approximate equality of two reduction results with a tolerance
/// derived from the input magnitudes (see [`reduction_tolerance`]).
fn approx_reduction_eq(a: f32, b: f32, inputs: &[f32]) -> bool {
    if a == b {
        return true;
    }
    let scale: f64 = inputs.iter().map(|&x| f64::from(x).abs()).sum();
    let tol = reduction_tolerance(inputs.len(), scale);
    (f64::from(a) - f64::from(b)).abs() <= tol
}

proptest! {
    #[test]
    fn prop_sum_matches_naive(values in finite_f32_vec()) {
        let lanes_result = lanes::stats::f32::sum(&values);
        let naive_result: f32 = values.iter().sum();

        prop_assert!(
            approx_reduction_eq(lanes_result, naive_result, &values),
            "sum mismatch: lanes={}, naive={}, len={}",
            lanes_result, naive_result, values.len()
        );
    }
}

proptest! {
    #[test]
    fn prop_prod_matches_naive(values in proptest::collection::vec(1_i32..=4, 0..32)) {
        // Integer values in [1,4]: products are exactly representable in
        // f32 for lengths up to 32 (4^32 ~ 2^64 < 2^24*2^63), so the
        // reduction order cannot change the result. Larger lengths or
        // fractional values would legitimately round differently per
        // backend (a documented non-equality).
        let values: Vec<f32> = values.iter().map(|&x| x as f32).collect();
        let lanes_result = lanes::stats::f32::prod(&values);
        let naive_result: f32 = values.iter().product();
        prop_assert_eq!(lanes_result, naive_result, "prod mismatch for len {}", values.len());
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_softmax_sums_to_one(values in softmax_f32_vec()) {
        let out = lanes::ml::f32::softmax(&values);
        if values.is_empty() {
            prop_assert!(out.is_empty());
        } else {
            // Softmax output sums to ~1 within exp-accumulation tolerance.
            let s: f64 = out.iter().map(|&x| f64::from(x)).sum();
            prop_assert!((s - 1.0).abs() < 1e-4, "sum={s}");
            prop_assert!(out.iter().all(|x| x.is_finite()));
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_softmax_matches_f64_reference(values in softmax_f32_vec()) {
        let out = lanes::ml::f32::softmax(&values);
        if values.is_empty() {
            prop_assert!(out.is_empty());
            return Ok(());
        }
        // Independent f64 ground truth: exp in f64, normalize in f64.
        let max = values.iter().map(|&x| x as f64).fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = values.iter().map(|&x| (x as f64 - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for i in 0..values.len() {
            let want = (exps[i] / sum) as f32;
            // f64 ground truth is itself rounded to f32; allow ~4 ulp.
            let tol = (want.abs() * 1e-6).max(1e-7);
            prop_assert!(
                (out[i] - want).abs() <= tol,
                "lane {i}: softmax got {}, want {want}",
                out[i]
            );
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_sigmoid_matches_f64_reference(values in softmax_f32_vec()) {
        let out = lanes::ml::f32::sigmoid(&values);
        if values.is_empty() {
            prop_assert!(out.is_empty());
            return Ok(());
        }
        for i in 0..values.len() {
            let x = values[i] as f64;
            let want = (1.0 / (1.0 + (-x).exp())) as f32;
            if want.is_nan() {
                continue; // extreme saturation — both sides saturate
            }
            let tol = (want.abs() * 1e-6).max(1e-7);
            prop_assert!(
                (out[i] - want).abs() <= tol,
                "lane {i}: sigmoid({}) got {}, want {want}",
                values[i],
                out[i]
            );
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_sigmoid_in_unit_interval_and_symmetry(values in softmax_f32_vec()) {
        let out = lanes::ml::f32::sigmoid(&values);
        if values.is_empty() {
            prop_assert!(out.is_empty());
            return Ok(());
        }
        for (i, &x) in values.iter().enumerate() {
            let s = out[i];
            // [0, 1] inclusive: sigmoid saturates to exactly 1.0/0.0 for
            // |x| ≳ 16 in f32 (exp under/overflow), well within [-40, 40].
            prop_assert!((0.0..=1.0).contains(&s), "lane {i}: {s} out of [0,1]");
            // sigmoid(x) + sigmoid(-x) == 1 (within exp tolerance).
            let neg = lanes::ml::f32::sigmoid(&[-x]);
            prop_assert!(
                (f64::from(s) + f64::from(neg[0]) - 1.0).abs() < 1e-4,
                "lane {i}: symmetry broken at x={x}"
            );
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_silu_equals_x_times_sigmoid(values in softmax_f32_vec()) {
        let silu_out = lanes::ml::f32::silu(&values);
        let sig = lanes::ml::f32::sigmoid(&values);
        if values.is_empty() {
            prop_assert!(silu_out.is_empty());
            return Ok(());
        }
        for i in 0..values.len() {
            let expected = values[i] * sig[i];
            prop_assert!(
                (f64::from(silu_out[i]) - f64::from(expected)).abs() < 1e-4,
                "lane {i}: silu={} vs x*sig={}",
                silu_out[i],
                expected
            );
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_gelu_linear_in_tail(values in softmax_f32_vec()) {
        let out = lanes::ml::f32::gelu(&values);
        if values.is_empty() {
            prop_assert!(out.is_empty());
            return Ok(());
        }
        for (i, &x) in values.iter().enumerate() {
            // gelu(x) ≈ x for large positive x (tanh → 1), ≈ 0 for large
            // negative; always finite and between 0 and x for x > 0.
            prop_assert!(out[i].is_finite(), "lane {i}: non-finite");
            if x > 0.0 {
                prop_assert!(
                    out[i] > 0.0 && out[i] <= x,
                    "lane {i}: x={x} gelu={}",
                    out[i]
                );
            } else {
                prop_assert!(
                    out[i] <= 0.0 && out[i] >= x,
                    "lane {i}: x={x} gelu={}",
                    out[i]
                );
            }
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_relu_is_max_zero(values in softmax_f32_vec()) {
        let out = lanes::ml::f32::relu(&values);
        if values.is_empty() {
            prop_assert!(out.is_empty());
            return Ok(());
        }
        for i in 0..values.len() {
            prop_assert_eq!(out[i], values[i].max(0.0), "lane {} mismatch", i);
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_softmax_shift_invariance(values in softmax_f32_vec(), shift in -40.0_f32..40.0) {
        if values.is_empty() { return Ok(()); }
        let shifted: Vec<f32> = values.iter().map(|&x| x + shift).collect();
        let a = lanes::ml::f32::softmax(&values);
        let b = lanes::ml::f32::softmax(&shifted);
        for i in 0..a.len() {
            prop_assert!((a[i] - b[i]).abs() < 1e-5, "lane {i}");
        }
    }
}

proptest! {
    #[test]
    fn prop_dot_matches_naive(
        a in bounded_f32_vec()
    ) {
        // Make b the same length as a for a valid dot product.
        let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();

        let lanes_result = lanes::stats::f32::dot(&a, &b).unwrap();
        let naive_result: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

        // Products may themselves round; include a small absolute slack on
        // top of the reduction bound.
        let products: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
        let tol = reduction_tolerance(a.len(), {
            let mut scale = 0.0_f64;
            for &p in &products {
                scale += f64::from(p).abs();
            }
            scale
        });

        prop_assert!(
            (f64::from(lanes_result) - f64::from(naive_result)).abs() <= tol,
            "dot mismatch: lanes={}, naive={}, len={}",
            lanes_result, naive_result, a.len()
        );
    }
}

proptest! {
    #[test]
    fn prop_min_matches_naive(values in finite_f32_vec()) {
        let lanes_result = lanes::stats::f32::min(&values);
        let naive_result = values.iter().copied().reduce(f32::min);

        match (lanes_result, naive_result) {
            (None, None) => {} // both empty
            (Some(l), Some(n)) => {
                // min/max are order-insensitive for finite values; only
                // signed zero may differ (-0.0 == 0.0 in `==`), so exact
                // equality is expected.
                prop_assert_eq!(l, n, "min mismatch for len {}", values.len());
            }
            _ => prop_assert!(false, "min None/Some mismatch"),
        }
    }
}

proptest! {
    #[test]
    fn prop_max_matches_naive(values in finite_f32_vec()) {
        let lanes_result = lanes::stats::f32::max(&values);
        let naive_result = values.iter().copied().reduce(f32::max);

        match (lanes_result, naive_result) {
            (None, None) => {} // both empty
            (Some(l), Some(n)) => {
                prop_assert_eq!(l, n, "max mismatch for len {}", values.len());
            }
            _ => prop_assert!(false, "max None/Some mismatch"),
        }
    }
}

proptest! {
    #[test]
    fn prop_dot_length_mismatch_is_error(
        a in finite_f32_vec(),
        extra in 1_usize..100,
    ) {
        // b is always longer than a, so we get LengthMismatch.
        let b: Vec<f32> = (0..a.len() + extra).map(|i| i as f32).collect();
        let result = lanes::stats::f32::dot(&a, &b);
        prop_assert!(result.is_err());
    }
}

proptest! {
    #[test]
    fn prop_sum_sq_matches_naive(values in finite_f32_vec()) {
        let got = lanes::stats::f32::sum_sq(&values);
        // Exact for values whose squares are exactly representable; use a
        // tolerance for larger magnitudes (reduction order may differ).
        let want: f32 = values.iter().map(|x| x * x).sum();
        // Both overflow to inf identically (IEEE semantics) — skip the
        // comparison then, since inf - inf is NaN.
        if !got.is_finite() || !want.is_finite() {
            return Ok(());
        }
        let tol = values.len() as f32 * f32::EPSILON * want.abs().max(1.0);
        prop_assert!(
            (got - want).abs() <= tol * 8.0,
            "sum_sq mismatch: got {got}, want {want}"
        );
    }
}

proptest! {
    #[test]
    fn prop_l1_norm_matches_naive(values in finite_f32_vec()) {
        let got = lanes::distance::f32::l1_norm(&values);
        let want: f32 = values.iter().map(|x| x.abs()).sum();
        let tol = values.len() as f32 * f32::EPSILON * want.abs().max(1.0);
        prop_assert!(
            (got - want).abs() <= tol * 8.0,
            "l1_norm mismatch: got {got}, want {want}"
        );
    }
}

proptest! {
    #[test]
    fn prop_max_norm_matches_naive(values in finite_f32_vec()) {
        let got = lanes::distance::f32::max_norm(&values);
        let want = values.iter().copied().map(f32::abs).reduce(f32::max);
        prop_assert_eq!(got, want, "max_norm mismatch for len {}", values.len());
    }
}

proptest! {
    #[test]
    fn prop_mean_matches_f64_reference(values in mid_f32_vec()) {
        let got = lanes::stats::f32::mean(&values);
        if values.is_empty() {
            prop_assert!(got.is_none());
            return Ok(());
        }
        // Independent ground truth: accumulate in f64 (not the crate's f32
        // sum path). Catches a subtle bug in `sum` itself, since this
        // reference shares no code with it.
        let want = (values.iter().map(|x| *x as f64).sum::<f64>() / values.len() as f64) as f32;
        if !want.is_finite() {
            return Ok(()); // overflow to inf (IEEE) — skip, inf - inf is NaN
        }
        // The proven reduction tolerance (scale = Σ|x|, per Higham, with
        // slack for rounding the f64 reference to f32).
        let scale: f64 = values.iter().map(|&x| f64::from(x).abs()).sum();
        let tol = reduction_tolerance(values.len(), scale) as f32;
        prop_assert!(
            (got.unwrap() - want).abs() <= tol.max(1e-5),
            "mean mismatch: got {:?}, want {want}, tol {tol}",
            got
        );
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_variance_matches_f64_reference(values in mid_f32_vec()) {
        let got = lanes::stats::f32::variance(&values);
        if values.is_empty() {
            prop_assert!(got.is_none());
            return Ok(());
        }
        // Independent ground truth in f64 (two-pass, same algorithm shape
        // but no shared code with the crate).
        let m = values.iter().map(|x| *x as f64).sum::<f64>() / values.len() as f64;
        let want =
            (values.iter().map(|x| (*x as f64 - m) * (*x as f64 - m)).sum::<f64>() / values.len() as f64)
                as f32;
        let got_v = got.unwrap();
        if !got_v.is_finite() || !want.is_finite() {
            return Ok(()); // overflow to inf (IEEE) — skip
        }
        // Proven reduction tolerance on the centered-squared scale.
        let scale: f64 = values
            .iter()
            .map(|&x| {
                let c = f64::from(x) - m;
                c * c
            })
            .sum();
        let tol = reduction_tolerance(values.len(), scale) as f32;
        prop_assert!(
            (got_v - want).abs() <= tol.max(1e-5),
            "variance mismatch: got {:?}, want {want}, tol {tol}",
            got
        );
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_l2_norm_matches_f64_reference(values in mid_f32_vec()) {
        let got = lanes::distance::f32::l2_norm(&values);
        // Independent f64 ground truth.
        let want = (values.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()) as f32;
        if !got.is_finite() || !want.is_finite() {
            return Ok(()); // overflow to inf (IEEE) — skip
        }
        // Proven reduction tolerance on the squared scale.
        let scale: f64 = values.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let tol = reduction_tolerance(values.len(), scale) as f32;
        prop_assert!(
            (got - want).abs() <= tol.max(1e-5),
            "l2_norm mismatch: got {got}, want {want}, tol {tol}"
        );
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_sqrt_matches_naive(values in finite_f32_vec()) {
        let got = lanes::math::f32::sqrt(&values);
        let want: Vec<f32> = values.iter().map(|x| x.sqrt()).collect();
        for i in 0..values.len() {
            let (g, w) = (got[i], want[i]);
            // NaN inputs → NaN both sides; inf → inf; otherwise ≤ 1 ulp.
            if w.is_nan() {
                prop_assert!(g.is_nan(), "lane {i}: sqrt({}) should be NaN", values[i]);
            } else if w.is_infinite() {
                prop_assert_eq!(g, w, "lane {} inf mismatch", i);
            } else {
                let tol = w.abs() * 2e-7 + 1e-6;
                prop_assert!((g - w).abs() <= tol, "lane {i}: sqrt({}) = {g}, want {w}", values[i]);
            }
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_clip_matches_naive(
        values in finite_f32_vec(),
        lo in -10.0_f32..10.0,
        delta in 0.0_f32..20.0,
    ) {
        let hi = lo + delta; // valid bounds: lo <= hi
        let got = lanes::math::f32::clip(&values, lo, hi);
        let want: Vec<f32> = values.iter().map(|&x| x.clamp(lo, hi)).collect();
        // Exact: clamp is a pure min/max (except NaN, which propagates both
        // sides identically).
        for i in 0..values.len() {
            let (g, w) = (got[i], want[i]);
            if w.is_nan() {
                prop_assert!(g.is_nan(), "lane {i}: clip should be NaN");
            } else {
                prop_assert_eq!(g, w, "lane {} clip({}) in [{}, {}]", i, values[i], lo, hi);
            }
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_rsqrt_matches_naive(values in finite_f32_vec()) {
        let got = lanes::math::f32::rsqrt(&values);
        let want: Vec<f32> = values.iter().map(|&x| 1.0 / x.sqrt()).collect();
        for i in 0..values.len() {
            let (g, w) = (got[i], want[i]);
            if w.is_nan() {
                prop_assert!(g.is_nan(), "lane {i}: rsqrt should be NaN");
            } else if w.is_infinite() {
                // rsqrt(±0) = ±inf, rsqrt(inf) = 0 — check the inf direction.
                prop_assert!(g.is_infinite() || g == 0.0, "lane {i}: rsqrt({}) = {g}", values[i]);
            } else {
                let tol = w.abs() * 2e-7 + 1e-6;
                prop_assert!(
                    (g - w).abs() <= tol,
                    "lane {i}: rsqrt({}) = {g}, want {w}",
                    values[i]
                );
            }
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_exp_matches_naive(values in softmax_f32_vec()) {
        // softmax_f32_vec bounds to [-40, 40] where exp stays finite.
        let got = lanes::math::f32::exp(&values);
        let want: Vec<f32> = values.iter().map(|&x| x.exp()).collect();
        for i in 0..values.len() {
            let (g, w) = (got[i], want[i]);
            // exp is ≤ 2 ulp of f32::exp; allow ~4 ulp relative.
            let tol = w.abs() * 4e-7 + 1e-6;
            prop_assert!(
                (g - w).abs() <= tol,
                "lane {i}: exp({}) = {g}, want {w}",
                values[i]
            );
        }
    }

    #[test]
    fn prop_tanh_matches_naive(values in softmax_f32_vec()) {
        // tanh(x) = 1 - 2/(e^(2x)+1); |x| ≤ 40 keeps e^(2x) finite.
        let got = lanes::math::f32::tanh(&values);
        let want: Vec<f32> = values.iter().map(|&x| x.tanh()).collect();
        for i in 0..values.len() {
            let (g, w) = (got[i], want[i]);
            // tanh is in [-1,1]; absolute tolerance suffices.
            prop_assert!(
                (g - w).abs() <= 2e-6,
                "lane {i}: tanh({}) = {g}, want {w}",
                values[i]
            );
        }
    }

    #[test]
    fn prop_rms_norm_matches_naive(values in bounded_f32_vec(), eps in 1e-6_f32..1e-2) {
        let got = lanes::ml::f32::rms_norm(&values, eps);
        let mean_sq: f32 = values.iter().map(|x| x * x).sum::<f32>() / values.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let want: Vec<f32> = values.iter().map(|&x| x * inv).collect();
        for i in 0..values.len() {
            let tol = want[i].abs() * 2e-6 + 1e-6;
            prop_assert!(
                (got[i] - want[i]).abs() <= tol,
                "lane {i}: rms_norm({}) = {}, want {}",
                values[i], got[i], want[i]
            );
        }
    }

    #[test]
    fn prop_cosine_similarity_matches_naive(a in bounded_f32_vec()) {
        // Derive b from a (same pattern as prop_dot_matches_naive): a
        // second filtered strategy rejects too often.
        let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
        let got = lanes::ml::f32::cosine_similarity(&a, &b).unwrap();
        let naive = {
            let dot: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 { None } else { Some(dot / (na * nb)) }
        };
        match (got, naive) {
            (None, None) => {}
            (Some(g), Some(w)) => {
                let tol = w.abs() * 2e-6 + 1e-6;
                prop_assert!((g - w).abs() <= tol, "cos({g}) vs naive {w}");
            }
            (g, w) => prop_assert!(false, "cos: got {g:?}, naive {w:?}"),
        }
    }
}
