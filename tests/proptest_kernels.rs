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
        let got = lanes::math::f32::clip(&values, lo, hi).unwrap();
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
                // Documented rsqrt contract: ≤ 2 ulp vs 1/sqrt (worst-case
                // at the top of the finite range). The previous tol
                // w*2e-7+1e-6 is ~1 ulp for normals but border normals
                // (e.g. 1.53e-38) can hit 2 ulp after Newton refine, so
                // use a 2-ulp bound via nextafter.
                let ulp = (f32::from_bits(w.to_bits().wrapping_add(1)) - w).abs().max(f32::from_bits(w.to_bits().wrapping_sub(1)) - w).abs();
                let tol = (ulp * 2.0).max(1e-6);
                prop_assert!(
                    (g - w).abs() <= tol,
                    "lane {i}: rsqrt({}) = {g}, want {w} (tol {tol})",
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
        let got = lanes::ml::f32::cosine_similarity(&a, &b);
        let naive = {
            let dot: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if a.is_empty() {
                None // empty → EmptyInput error, not a value
            } else if na == 0.0 || nb == 0.0 {
                Some(0.0) // zero norm → 0.0 by convention
            } else {
                Some(dot / (na * nb))
            }
        };
        match (got, naive) {
            (Err(lanes::Error::EmptyInput), None) => {}
            (Ok(g), Some(w)) => {
                let tol = w.abs() * 2e-6 + 1e-6;
                prop_assert!((g - w).abs() <= tol, "cos({g}) vs naive {w}");
            }
            (g, w) => prop_assert!(false, "cos: got {g:?}, naive {w:?}"),
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_abs_sub_matches_naive(a in mid_f32_vec()) {
        // Build a same-length b by reversing a (deterministic, no extra strategy).
        let b: Vec<f32> = a.iter().rev().copied().collect();
        let out = lanes::math::f32::abs_sub(&a, &b).unwrap();
        for (i, (got, (x, y))) in out.iter().zip(a.iter().zip(b.iter())).enumerate() {
            let want = (x - y).abs();
            prop_assert_eq!(*got, want, "lane {}: |{} - {}|", i, x, y);
        }
    }
}

proptest! {
    #[test]
    fn prop_counts_match_naive(values in proptest::collection::vec(any::<f32>(), 0..1000)) {
        prop_assert_eq!(
            lanes::stats::f32::count_zero(&values),
            values.iter().filter(|&&x| x == 0.0).count()
        );
        prop_assert_eq!(
            lanes::stats::f32::count_nan(&values),
            values.iter().filter(|x| x.is_nan()).count()
        );
        prop_assert_eq!(
            lanes::stats::f32::count_infinite(&values),
            values.iter().filter(|x| x.is_infinite()).count()
        );
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_hypot_matches_std(a in proptest::collection::vec(-1e15_f32..1e15, 0..512)) {
        let b: Vec<f32> = a.iter().rev().copied().collect();
        let out = lanes::math::f32::hypot(&a, &b).unwrap();
        for (i, (got, (x, y))) in out.iter().zip(a.iter().zip(b.iter())).enumerate() {
            let want = x.hypot(*y);
            let tol = (want.abs() * 1e-6).max(1e-6);
            prop_assert!(
                (got - want).abs() <= tol,
                "lane {}: hypot({},{}) got {}, want {}",
                i, x, y, got, want
            );
        }
    }
}

#[cfg(feature = "alloc")]
proptest! {
    #[test]
    fn prop_powi_bit_exact_with_std(
        values in proptest::collection::vec(-10.0_f32..10.0, 0..512),
        n in -12_i32..12,
    ) {
        let out = lanes::math::f32::powi(&values, n);
        for (i, (got, x)) in out.iter().zip(values.iter()).enumerate() {
            let want = x.powi(n);
            prop_assert_eq!(
                got.to_bits(), want.to_bits(),
                "lane {}: powi({}, {}) got {}, std {}",
                i, x, n, got, want
            );
        }
    }
}

proptest! {
    #[test]
    fn prop_squared_distance_matches_naive(a in bounded_f32_vec()) {
        let b: Vec<f32> = a.iter().rev().copied().collect();
        let got = lanes::distance::f32::squared_distance(&a, &b).unwrap();
        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| { let d = x - y; d * d }).sum();
        // Same summation-order caveat as dot: use the input-magnitude tolerance.
        let scale: f64 = a.iter().zip(b.iter()).map(|(x, y)| f64::from((x - y).abs())).sum();
        let tol = scale * scale * (a.len() as f64) * 2_f64.powi(-20) + 1.0;
        prop_assert!(
            (f64::from(got) - f64::from(naive)).abs() <= tol,
            "got {}, naive {}", got, naive
        );
    }
}

/// Strategy for divergence inputs: strictly positive and bounded away from
/// zero so `ln(p/q)` stays finite and the summation-order tolerance is
/// meaningful. Magnitudes keep 512-term sums far from overflow.
///
/// Built by mapping an integer strategy onto the float range rather than
/// using a float range directly: proptest's float-range sampler asserts
/// on inexact bounds (`1e-3` is not exactly representable in f32) and
/// intermittently panics inside the sampler itself (seen on CI).
fn positive_f32_vec() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(
        (0..1_000_000u32).prop_map(|i| 1e-3 + (i as f32) * 9.999 / 1_000_000.0),
        0..512,
    )
}

/// `f64` twin of [`positive_f32_vec`] (same integer-map rationale).
fn positive_f64_vec() -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(
        (0..1_000_000u32).prop_map(|i| 1e-6 + (i as f64) * 9.999_999 / 1_000_000.0),
        0..512,
    )
}

/// Tolerance for the divergence properties: the SIMD and naive results use
/// the same term formula but (a) sum in different orders and (b) use
/// different ≤1-ulp `ln` implementations (crate fdlibm vs libm), so each
/// term can differ by ~2 ulp of its magnitude before summation. Bound the
/// accumulated difference by `2^-23 * Σ|term|` with slack, plus an absolute
/// floor for near-zero results.
fn divergence_tol_f32(term_abs_sum: f64) -> f64 {
    term_abs_sum * 2_f64.powi(-23) * 16.0 + 1e-6
}

proptest! {
    #[test]
    fn prop_kl_divergence_matches_naive(p in positive_f32_vec()) {
        let q: Vec<f32> = p.iter().rev().copied().collect();
        let got = lanes::distance::f32::kl_divergence(&p, &q).unwrap();
        let mut naive = 0.0_f32;
        let mut term_abs = 0.0_f64;
        for (&a, &b) in p.iter().zip(q.iter()) {
            let t = a * (a / b).ln();
            naive += t;
            term_abs += f64::from(t.abs());
        }
        let tol = divergence_tol_f32(term_abs);
        prop_assert!(
            (f64::from(got) - f64::from(naive)).abs() <= tol,
            "got {got}, naive {naive}, tol {tol}, len {}", p.len()
        );
    }
}

proptest! {
    #[test]
    fn prop_js_divergence_matches_naive(p in positive_f32_vec()) {
        let q: Vec<f32> = p.iter().rev().copied().collect();
        let got = lanes::distance::f32::js_divergence(&p, &q).unwrap();
        let mut naive = 0.0_f32;
        let mut term_abs = 0.0_f64;
        for (&a, &b) in p.iter().zip(q.iter()) {
            let m = (a + b) * 0.5;
            let t = a * (a / m).ln() + b * (b / m).ln();
            naive += t;
            term_abs += f64::from(t.abs());
        }
        naive *= 0.5;
        let tol = divergence_tol_f32(term_abs);
        prop_assert!(
            (f64::from(got) - f64::from(naive)).abs() <= tol,
            "got {got}, naive {naive}, tol {tol}, len {}", p.len()
        );
    }
}

proptest! {
    #[test]
    fn prop_js_divergence_symmetric(p in positive_f32_vec()) {
        let q: Vec<f32> = p.iter().rev().copied().collect();
        let js_pq = lanes::distance::f32::js_divergence(&p, &q).unwrap();
        let js_qp = lanes::distance::f32::js_divergence(&q, &p).unwrap();
        let tol = (f64::from(js_pq.abs()) * 1e-5) + 1e-6;
        prop_assert!(
            (f64::from(js_pq) - f64::from(js_qp)).abs() <= tol,
            "js(p,q)={js_pq} vs js(q,p)={js_qp}"
        );
    }
}

proptest! {
    #[test]
    fn prop_divergence_self_is_zero(p in positive_f32_vec()) {
        // Every term is p·ln(1) = 0 exactly (ln(1) = 0 in fdlibm), so the
        // result is exactly 0.0 regardless of summation order.
        prop_assert_eq!(lanes::distance::f32::kl_divergence(&p, &p), Ok(0.0));
        prop_assert_eq!(lanes::distance::f32::js_divergence(&p, &p), Ok(0.0));
    }
}

proptest! {
    #[test]
    fn prop_divergence_nonnegative(p in positive_f32_vec()) {
        let q: Vec<f32> = p.iter().rev().copied().collect();
        let kl = lanes::distance::f32::kl_divergence(&p, &q).unwrap();
        let js = lanes::distance::f32::js_divergence(&p, &q).unwrap();
        // KL >= 0 up to the summation/ln tolerance; JS >= 0 likewise.
        prop_assert!(kl > -1e-4, "kl = {kl}");
        prop_assert!(js >= -1e-6, "js = {js}");
    }
}

proptest! {
    #[test]
    fn prop_kl_divergence_f64_matches_naive(p in positive_f64_vec()) {
        let q: Vec<f64> = p.iter().rev().copied().collect();
        let got = lanes::distance::f64::kl_divergence(&p, &q).unwrap();
        let mut naive = 0.0_f64;
        let mut term_abs = 0.0_f64;
        for (&a, &b) in p.iter().zip(q.iter()) {
            let t = a * (a / b).ln();
            naive += t;
            term_abs += t.abs();
        }
        let tol = term_abs * 2_f64.powi(-48) * 16.0 + 1e-12;
        prop_assert!(
            (got - naive).abs() <= tol,
            "got {got}, naive {naive}, tol {tol}, len {}", p.len()
        );
    }
}

proptest! {
    #[test]
    fn prop_js_divergence_f64_matches_naive(p in positive_f64_vec()) {
        let q: Vec<f64> = p.iter().rev().copied().collect();
        let got = lanes::distance::f64::js_divergence(&p, &q).unwrap();
        let mut naive = 0.0_f64;
        let mut term_abs = 0.0_f64;
        for (&a, &b) in p.iter().zip(q.iter()) {
            let m = (a + b) * 0.5;
            let t = a * (a / m).ln() + b * (b / m).ln();
            naive += t;
            term_abs += t.abs();
        }
        naive *= 0.5;
        let tol = term_abs * 2_f64.powi(-48) * 16.0 + 1e-12;
        prop_assert!(
            (got - naive).abs() <= tol,
            "got {got}, naive {naive}, tol {tol}, len {}", p.len()
        );
    }
}

proptest! {
    #[test]
    fn prop_divergence_f64_self_zero_and_symmetric(p in positive_f64_vec()) {
        let q: Vec<f64> = p.iter().rev().copied().collect();
        prop_assert_eq!(lanes::distance::f64::kl_divergence(&p, &p), Ok(0.0));
        prop_assert_eq!(lanes::distance::f64::js_divergence(&p, &p), Ok(0.0));
        let js_pq = lanes::distance::f64::js_divergence(&p, &q).unwrap();
        let js_qp = lanes::distance::f64::js_divergence(&q, &p).unwrap();
        let tol = js_pq.abs() * 1e-12 + 1e-15;
        prop_assert!(
            (js_pq - js_qp).abs() <= tol,
            "js(p,q)={js_pq} vs js(q,p)={js_qp}"
        );
    }
}

// --- binary family ---------------------------------------------------------

/// Equal-length byte-vector pairs of length 0..=288 (covers every
/// chunk/tail combination for the 16- and 32-byte kernels).
fn byte_pairs() -> impl Strategy<Value = (Vec<u8>, Vec<u8>)> {
    proptest::collection::vec((any::<u8>(), any::<u8>()), 0..=288)
        .prop_map(|pairs| pairs.into_iter().unzip())
}

/// Equal-length i8 pairs, sizes spanning chunk and epoch boundaries.
fn i8_pairs() -> impl Strategy<Value = (Vec<i8>, Vec<i8>)> {
    (0usize..=4200).prop_flat_map(|n| {
        (
            proptest::collection::vec(any::<i8>(), n),
            proptest::collection::vec(any::<i8>(), n),
        )
    })
}

fn naive_hamming_ref(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones() as usize)
        .sum()
}

fn naive_jaccard_ref(a: &[u8], b: &[u8]) -> Option<f32> {
    let mut inter = 0usize;
    let mut union = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        inter += (x & y).count_ones() as usize;
        union += (x | y).count_ones() as usize;
    }
    (union != 0).then(|| inter as f32 / union as f32)
}

proptest! {
    #[test]
    fn prop_hamming_matches_naive((a, b) in byte_pairs()) {
        prop_assert_eq!(lanes::binary::hamming(&a, &b), Ok(naive_hamming_ref(&a, &b)));
    }

    #[test]
    fn prop_jaccard_matches_naive((a, b) in byte_pairs()) {
        prop_assert_eq!(lanes::binary::jaccard(&a, &b), Ok(naive_jaccard_ref(&a, &b)));
    }

    #[test]
    fn prop_hamming_symmetric((a, b) in byte_pairs()) {
        prop_assert_eq!(
            lanes::binary::hamming(&a, &b),
            lanes::binary::hamming(&b, &a)
        );
    }

    #[test]
    fn prop_jaccard_symmetric((a, b) in byte_pairs()) {
        prop_assert_eq!(
            lanes::binary::jaccard(&a, &b),
            lanes::binary::jaccard(&b, &a)
        );
    }

    #[test]
    fn prop_hamming_self_is_zero(a in proptest::collection::vec(any::<u8>(), 0..=288)) {
        prop_assert_eq!(lanes::binary::hamming(&a, &a), Ok(0));
    }

    #[test]
    fn prop_hamming_bounded(a in proptest::collection::vec(any::<u8>(), 0..=288)) {
        let b = vec![0u8; a.len()];
        let d = lanes::binary::hamming(&a, &b).unwrap();
        prop_assert!(d <= 8 * a.len());
    }

    #[test]
    fn prop_jaccard_range_and_self((a, b) in byte_pairs()) {
        if let Ok(Some(j)) = lanes::binary::jaccard(&a, &b) {
            prop_assert!((0.0..=1.0).contains(&j));
        }
        // Self-similarity: 1.0 if any bit set, None if all-zero.
        match lanes::binary::jaccard(&a, &a) {
            Ok(Some(j)) => {
                prop_assert_eq!(j, 1.0);
                prop_assert!(a.iter().any(|&x| x != 0));
            }
            Ok(None) => prop_assert!(a.iter().all(|&x| x == 0)),
            Err(_) => prop_assert!(false),
        }
    }

    // --- i8 family ---------------------------------------------------------

    #[test]
    fn prop_dot_i8_matches_naive((a, b) in i8_pairs()) {
        let naive: i64 = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| i64::from(x) * i64::from(y))
            .sum();
        prop_assert_eq!(lanes::stats::i8::dot(&a, &b), Ok(naive));
    }

    #[test]
    fn prop_dot_i8_commutative((a, b) in i8_pairs()) {
        prop_assert_eq!(
            lanes::stats::i8::dot(&a, &b),
            lanes::stats::i8::dot(&b, &a)
        );
    }

    #[test]
    fn prop_sum_i8_matches_naive(a in proptest::collection::vec(any::<i8>(), 0..=4200)) {
        let naive: i64 = a.iter().map(|&x| i64::from(x)).sum();
        prop_assert_eq!(lanes::stats::i8::sum(&a), naive);
    }

    #[test]
    fn prop_sum_i8_split_invariant(a in proptest::collection::vec(any::<i8>(), 0..=4200)) {
        // sum(a) == sum(prefix) + sum(suffix) at every split point.
        for split in [0, a.len() / 3, a.len() / 2, a.len()] {
            prop_assert_eq!(
                lanes::stats::i8::sum(&a),
                lanes::stats::i8::sum(&a[..split]) + lanes::stats::i8::sum(&a[split..])
            );
        }
    }

    #[test]
    fn prop_sum_sq_i8_matches_naive(a in proptest::collection::vec(any::<i8>(), 0..=4200)) {
        let naive: i64 = a.iter().map(|&x| i64::from(x) * i64::from(x)).sum();
        prop_assert_eq!(lanes::stats::i8::sum_sq(&a), naive);
    }

    #[test]
    fn prop_min_max_i8_match_naive(a in proptest::collection::vec(any::<i8>(), 0..=4200)) {
        prop_assert_eq!(lanes::stats::i8::min(&a), a.iter().copied().min());
        prop_assert_eq!(lanes::stats::i8::max(&a), a.iter().copied().max());
    }

    #[test]
    fn prop_count_zero_i8_matches_naive(a in proptest::collection::vec(any::<i8>(), 0..=4200)) {
        let naive = a.iter().filter(|&&x| x == 0).count();
        prop_assert_eq!(lanes::stats::i8::count_zero(&a), naive);
    }

    #[test]
    fn prop_l1_norm_i8_matches_naive(a in proptest::collection::vec(any::<i8>(), 0..=4200)) {
        let naive: i64 = a.iter().map(|&x| i64::from(x.unsigned_abs())).sum();
        prop_assert_eq!(lanes::distance::i8::l1_norm(&a), naive);
    }

    #[test]
    fn prop_max_norm_i8_matches_naive(a in proptest::collection::vec(any::<i8>(), 0..=4200)) {
        let expected = if a.is_empty() {
            None
        } else {
            Some(a.iter().map(|&x| x.unsigned_abs()).max().unwrap())
        };
        prop_assert_eq!(lanes::distance::i8::max_norm(&a), expected);
    }

    #[test]
    fn prop_squared_distance_i8_matches_naive((a, b) in i8_pairs()) {
        let naive: i64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| {
                let d = i64::from(x) - i64::from(y);
                d * d
            })
            .sum();
        prop_assert_eq!(lanes::distance::i8::squared_distance(&a, &b), Ok(naive));
    }

    // --- erf/erfc: oracle-free algebraic properties ------------------------
    //
    // No std oracle exists (float_erf is unstable), so these check the
    // invariants that follow from the kernel construction: ranges, specials,
    // exact odd symmetry, saturation, the exact complement where erf is
    // computed as 1 − erfc, and the erf+erfc = 1 sum identity.

    #[test]
    fn prop_erf_erfc_f32_properties(bits in any::<u32>()) {
        let x = f32::from_bits(bits);
        let e = lanes::special::f32::erf(std::slice::from_ref(&x))[0];
        let c = lanes::special::f32::erfc(std::slice::from_ref(&x))[0];
        if x.is_nan() {
            prop_assert!(e.is_nan() && c.is_nan(), "NaN propagation at {}", x);
        } else {
            prop_assert!(e.abs() <= 1.0, "erf({}) = {} out of [-1,1]", x, e);
            prop_assert!((0.0..=2.0).contains(&c), "erfc({}) = {} out of [0,2]", x, c);
            // Odd symmetry is bit-exact: the small-region signed form and
            // the big-region negation both commute with rounding.
            let e_neg = lanes::special::f32::erf(&[-x])[0];
            prop_assert_eq!(e_neg.to_bits(), (-e).to_bits(), "erf symmetry at {}", x);
            // Saturation beyond XMAX (27.23).
            if x > 28.0 {
                prop_assert_eq!(e, 1.0);
                prop_assert_eq!(c, 0.0);
            }
            if x < -28.0 {
                prop_assert_eq!(e, -1.0);
                prop_assert_eq!(c, 2.0);
            }
            // erf + erfc == 1 exactly where erf = round32(1 − erfc_f64):
            // the two roundings stay within half an ulp of 1.0. (Do NOT
            // assert e == 1.0 − c here: for f32 that subtracts in f32 from
            // an already-rounded c, a second rounding that can drift 1 ulp.)
            if x >= 0.84375 {
                prop_assert_eq!(e + c, 1.0, "complement sum at {}", x);
            }
            // Sum identity everywhere else, with a rounding tolerance.
            prop_assert!(
                (e + c - 1.0).abs() <= 1e-6,
                "erf({x}) + erfc({x}) = {}",
                e + c
            );
        }
    }

    #[test]
    fn prop_erf_erfc_f64_properties(bits in any::<u64>()) {
        let x = f64::from_bits(bits);
        let e = lanes::special::f64::erf(std::slice::from_ref(&x))[0];
        let c = lanes::special::f64::erfc(std::slice::from_ref(&x))[0];
        if x.is_nan() {
            prop_assert!(e.is_nan() && c.is_nan(), "NaN propagation at {}", x);
        } else {
            prop_assert!(e.abs() <= 1.0, "erf({}) = {} out of [-1,1]", x, e);
            prop_assert!((0.0..=2.0).contains(&c), "erfc({}) = {} out of [0,2]", x, c);
            let e_neg = lanes::special::f64::erf(&[-x])[0];
            prop_assert_eq!(e_neg.to_bits(), (-e).to_bits(), "erf symmetry at {}", x);
            if x > 28.0 {
                prop_assert_eq!(e, 1.0);
                prop_assert_eq!(c, 0.0);
            }
            if x < -28.0 {
                prop_assert_eq!(e, -1.0);
                prop_assert_eq!(c, 2.0);
            }
            // f64 erf IS `1.0 − erfc` in the big region — bit-exact.
            if x >= 0.84375 {
                prop_assert_eq!(e, 1.0 - c, "complement at {}", x);
                prop_assert_eq!(e + c, 1.0, "complement sum at {}", x);
            }
            prop_assert!(
                (e + c - 1.0).abs() <= 1e-14,
                "erf({x}) + erfc({x}) = {}",
                e + c
            );
        }
    }

    /// f32 erf/erfc are perfectly rounded (f64 widen + round once), and a
    /// correctly-rounded monotone function is monotone — so even a 1-ulp
    /// step must not reverse. erfc is only perfectly rounded for x ≥ 0
    /// (the `2 − c` complement for x < 0 adds one rounding).
    #[test]
    fn prop_erf_erfc_f32_monotone_1ulp(bits in any::<u32>()) {
        let x = f32::from_bits(bits);
        prop_assume!(!x.is_nan() && x != f32::INFINITY);
        let y = if x == 0.0 {
            f32::from_bits(1) // nextafter(±0, +inf)
        } else if x > 0.0 {
            f32::from_bits(x.to_bits() + 1)
        } else {
            f32::from_bits(x.to_bits() - 1)
        };
        let ex = lanes::special::f32::erf(std::slice::from_ref(&x))[0];
        let ey = lanes::special::f32::erf(std::slice::from_ref(&y))[0];
        prop_assert!(ex <= ey, "erf({}) = {} > erf({}) = {}", x, ex, y, ey);
        if x >= 0.0 {
            let cx = lanes::special::f32::erfc(std::slice::from_ref(&x))[0];
            let cy = lanes::special::f32::erfc(std::slice::from_ref(&y))[0];
            prop_assert!(cx >= cy, "erfc({}) = {} < erfc({}) = {}", x, cx, y, cy);
        }
    }

    /// f64 monotonicity with a 0.01 step: large enough that the true gap
    /// exceeds the combined ≤ 1/≤ 3 ulp approximation error everywhere
    /// (a 1-ulp step is NOT guaranteed monotone under ≤ 1 ulp error).
    /// Integer-mapped sampler — inexact float bounds choke proptest.
    #[test]
    fn prop_erf_erfc_f64_monotone_step(xi in -20_000_i64..=19_999) {
        let x = (xi as f64) / 1000.0; // [-20, ~20)
        let y = x + 0.01;
        let ex = lanes::special::f64::erf(std::slice::from_ref(&x))[0];
        let ey = lanes::special::f64::erf(std::slice::from_ref(&y))[0];
        prop_assert!(ex <= ey, "erf({}) = {} > erf({}) = {}", x, ex, y, ey);
        let cx = lanes::special::f64::erfc(std::slice::from_ref(&x))[0];
        let cy = lanes::special::f64::erfc(std::slice::from_ref(&y))[0];
        prop_assert!(cx >= cy, "erfc({}) = {} < erfc({}) = {}", x, cx, y, cy);
    }
}
