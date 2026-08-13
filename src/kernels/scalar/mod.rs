//! Scalar (portable) kernel implementations.
//!
//! These serve as the universal fallback when no SIMD backend is available
//! and as reference implementations for correctness testing.
//!
//! The activation kernels below (softmax/sigmoid/silu/gelu/relu) are gated
//! on `alloc`: their only public callers (`lanes::ml::*`) return a `Vec`.
//! The exp they use (`kernels::exp`) is fully `no_std`.

use crate::kernels::exp;

/// Constants for the GELU tanh approximation.
const GELU_A: f32 = 0.797_884_6; // sqrt(2/pi) in f32
const GELU_B: f32 = 0.044_715;

/// Scalar softmax reference. Writes into `out` (same length as `values`).
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn softmax(values: &[f32], out: &mut [f32]) {
    let Some(max) = values.iter().copied().reduce(f32::max) else {
        return;
    };
    let mut sum = 0.0;
    for (i, &v) in values.iter().enumerate() {
        let e = exp::exp(v - max);
        out[i] = e;
        sum += e;
    }
    if sum != 0.0 {
        let inv = 1.0 / sum;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
}

/// Scalar sigmoid reference. Writes into `out` (same length as `values`).
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn sigmoid(values: &[f32], out: &mut [f32]) {
    for (i, &v) in values.iter().enumerate() {
        out[i] = 1.0 / (1.0 + exp::exp(-v));
    }
}

/// Scalar `SiLU` reference. Writes into `out` (same length as `values`).
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn silu(values: &[f32], out: &mut [f32]) {
    for (i, &v) in values.iter().enumerate() {
        out[i] = v / (1.0 + exp::exp(-v));
    }
}

/// Scalar GELU reference (tanh approximation). Writes into `out`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn gelu(values: &[f32], out: &mut [f32]) {
    for (i, &x) in values.iter().enumerate() {
        let z = GELU_A * (x + GELU_B * x * x * x);
        // tanh(z) = 1 - 2/(exp(2z)+1); saturates to ±1 via exp under/overflow.
        let tanh_z = 1.0 - 2.0 / (exp::exp(2.0 * z) + 1.0);
        out[i] = 0.5 * x * (1.0 + tanh_z);
    }
}

/// Scalar `ReLU` reference. Writes into `out` (same length as `values`).
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn relu(values: &[f32], out: &mut [f32]) {
    for (i, &v) in values.iter().enumerate() {
        out[i] = v.max(0.0);
    }
}

/// Elementwise `tanh` into `out`.
///
/// Piecewise: for `|x| < 0.1` a Taylor series `x - x³/3 + 2x⁵/15` is used
/// (the `1 - 2/(exp(2x)+1)` form catastrophically cancels to 0 there);
/// beyond that the exp form saturates correctly to ±1. NaN propagates.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn tanh(values: &[f32], out: &mut [f32]) {
    map(values, out, |x| {
        let a = x.abs();
        if a > 9.011 {
            x.signum() // tanh(x) = ±1 to f32 precision (e^-2x < 2^-24)
        } else if a < 2e-4 {
            x // tanh(x) ≈ x, error x³/3 < 1 ulp
        } else if a < 0.1 {
            let x2 = x * x;
            x * (1.0 - x2 / 3.0 + 2.0 * x2 * x2 / 15.0)
        } else {
            let e = exp::exp(2.0 * x);
            (e - 1.0) / (e + 1.0) // Sterbenz-exact for e in [1, 2]
        }
    });
}

/// RMS norm into `out`: `x_i * rsqrt(mean(x²) + eps)`. Empty input leaves
/// `out` untouched. The epsilon guards the all-zero case.
#[cfg(feature = "alloc")]
#[inline]
#[allow(clippy::cast_precision_loss)] // `len as f32` is inherent to the mean
pub(crate) fn rms_norm(values: &[f32], eps: f32, out: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mean_sq = values.iter().map(|x| x * x).sum::<f32>() / values.len() as f32;
    let inv = 1.0 / crate::kernels::sqrt::sqrt(mean_sq + eps);
    for (i, &v) in values.iter().enumerate() {
        out[i] = v * inv;
    }
}

/// Compute the sum of all elements in a slice.
///
/// Returns `0.0` for an empty slice.
#[inline]
pub(crate) fn sum(values: &[f32]) -> f32 {
    values.iter().sum()
}

/// Shared elementwise map: applies `op` to each element of `values`, writing
/// into `out` (same length). The single skeleton for all scalar map kernels.
#[cfg(feature = "alloc")]
#[inline]
fn map(values: &[f32], out: &mut [f32], op: impl Fn(f32) -> f32) {
    debug_assert_eq!(values.len(), out.len());
    for (v, o) in values.iter().zip(out) {
        *o = op(*v);
    }
}

/// Elementwise square root into `out` (same length as `values`).
///
/// Uses the std-free `kernels::sqrt::sqrt` (IEEE-correct within 1 ulp).
/// Gated on `alloc`: its only caller (`dispatch_sqrt`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn sqrt(values: &[f32], out: &mut [f32]) {
    map(values, out, crate::kernels::sqrt::sqrt);
}

/// Elementwise clip into `out`: `clamp(x, lo, hi)`.
///
/// NaN inputs yield NaN (`f32::clamp` propagates NaN). Gated on `alloc`: its
/// only caller (`dispatch_clip`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn clip(values: &[f32], lo: f32, hi: f32, out: &mut [f32]) {
    map(values, out, |x| x.clamp(lo, hi));
}

/// Elementwise reciprocal square root into `out`: `1/sqrt(x)`.
///
/// Gated on `alloc`: its only caller (`dispatch_rsqrt`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn sub_scalar(values: &[f32], p: f32, _p2: f32, out: &mut [f32]) {
    for (o, &x) in out.iter_mut().zip(values) {
        *o = x - p;
    }
}

#[cfg(feature = "alloc")]
pub(crate) fn rsqrt(values: &[f32], out: &mut [f32]) {
    map(values, out, |x| 1.0 / crate::kernels::sqrt::sqrt(x));
}

/// Elementwise exponential into `out`.
///
/// Uses the std-free `kernels::exp::exp` (≤ 2 ulp vs `f32::exp`). Gated on
/// `alloc`: its only caller (`dispatch_exp`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn exp(values: &[f32], out: &mut [f32]) {
    map(values, out, crate::kernels::exp::exp);
}

/// Elementwise natural logarithm into `out`.
///
/// Uses the std-free `kernels::ln::ln` (≤ 1 ulp vs `f32::ln`, fdlibm
/// algorithm). Gated on `alloc`: its only caller (`dispatch_ln`) is
/// alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn ln(values: &[f32], out: &mut [f32]) {
    map(values, out, crate::kernels::ln::ln);
}

/// Elementwise natural logarithm into `out` (`f64`).
///
/// Uses the std-free `kernels::ln::ln_f64` (≤ 1 ulp vs `f64::ln`).
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn ln_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        *o = crate::kernels::ln::ln_f64(*v);
    }
}

/// Compute the sum of squares of all elements in a slice.
///
/// Returns `0.0` for an empty slice.
#[inline]
pub(crate) fn sum_sq(values: &[f32]) -> f32 {
    values.iter().map(|x| x * x).sum()
}

/// Compute the L1 norm (sum of absolute values) of a slice.
///
/// Returns `0.0` for an empty slice.
#[inline]
pub(crate) fn l1_norm(values: &[f32]) -> f32 {
    values.iter().copied().map(f32::abs).sum()
}

/// Compute the maximum absolute value (max norm) of a slice.
///
/// Returns `None` for an empty slice.
#[inline]
pub(crate) fn max_norm(values: &[f32]) -> Option<f32> {
    values.iter().copied().map(f32::abs).max_by(f32::total_cmp)
}

/// Compute the product of all elements in a slice.
///
/// Returns `1.0` for an empty slice (the multiplicative identity).
#[inline]
pub(crate) fn prod(values: &[f32]) -> f32 {
    values.iter().product()
}

/// Find the minimum element in a slice using IEEE 754 `minNum` semantics.
///
/// Returns [`None`] if the slice is empty. Uses [`f32::min`]: a NaN input is
/// ignored unless every input is NaN, and `min(-0.0, +0.0)` is `-0.0`.
#[inline]
pub(crate) fn min(values: &[f32]) -> Option<f32> {
    values.iter().copied().reduce(f32::min)
}

/// Find the maximum element in a slice using IEEE 754 `maxNum` semantics.
///
/// Returns [`None`] if the slice is empty. Uses [`f32::max`]: a NaN input is
/// ignored unless every input is NaN, and `max(-0.0, +0.0)` is `+0.0`.
#[inline]
pub(crate) fn max(values: &[f32]) -> Option<f32> {
    values.iter().copied().reduce(f32::max)
}

/// Compute the dot product of two equal-length slices.
///
/// # Panics
/// Does not panic — if `b` is shorter than `a`, the zip stops at the
/// shorter length. The caller is expected to guarantee equal lengths.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Find the index of the maximum element in a slice, returning `(value, index)`.
///
/// Caller must guarantee the slice is non-empty. Ties resolve to the first
/// occurrence (strict `>` keeps the earliest index). NaN handling: a NaN
/// never compares greater, so it is ignored unless every element is NaN.
#[inline]
pub(crate) fn argmax(values: &[f32]) -> (f32, usize) {
    let mut idx = 0;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() && (values[idx].is_nan() || v > values[idx]) {
            idx = i;
        }
    }
    (values[idx], idx)
}

/// Find the index of the minimum element in a slice, returning `(value, index)`.
///
/// Caller must guarantee the slice is non-empty. Ties resolve to the first
/// occurrence (strict `<` keeps the earliest index). NaN handling: a NaN
/// never compares less, so it is ignored unless every element is NaN.
#[inline]
pub(crate) fn argmin(values: &[f32]) -> (f32, usize) {
    let mut idx = 0;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() && (values[idx].is_nan() || v < values[idx]) {
            idx = i;
        }
    }
    (values[idx], idx)
}

// ===========================================================================
// f64 kernels (double-precision). Same contracts as the f32 versions above.
// ===========================================================================

/// Compute the sum of all elements in a slice. Returns `0.0` for empty.
#[inline]
pub(crate) fn sum_f64(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Compute the product of all elements in a slice. Returns `1.0` for empty.
#[inline]
pub(crate) fn prod_f64(values: &[f64]) -> f64 {
    values.iter().product()
}

/// Find the minimum element in a slice (IEEE 754 `minNum` semantics).
#[inline]
pub(crate) fn min_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::min)
}

/// Find the maximum element in a slice (IEEE 754 `maxNum` semantics).
#[inline]
pub(crate) fn max_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::max)
}

/// Compute the sum of squares of all elements in a slice. Returns `0.0` for empty.
#[inline]
pub(crate) fn sum_sq_f64(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum()
}

/// Compute the L1 norm (sum of absolute values) of a slice.
#[inline]
pub(crate) fn l1_norm_f64(values: &[f64]) -> f64 {
    values.iter().copied().map(f64::abs).sum()
}

/// Compute the maximum absolute value (max norm) of a slice.
#[inline]
pub(crate) fn max_norm_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().map(f64::abs).max_by(f64::total_cmp)
}

/// Compute the dot product of two equal-length slices.
#[inline]
pub(crate) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Find the index of the maximum element in a slice, returning `(value, index)`.
///
/// Caller must guarantee the slice is non-empty. Ties resolve to the first
/// occurrence (strict `>` keeps the earliest index).
#[inline]
pub(crate) fn argmax_f64(values: &[f64]) -> (f64, usize) {
    let mut idx = 0;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() && (values[idx].is_nan() || v > values[idx]) {
            idx = i;
        }
    }
    (values[idx], idx)
}

/// Find the index of the minimum element in a slice, returning `(value, index)`.
///
/// Caller must guarantee the slice is non-empty. Ties resolve to the first
/// occurrence (strict `<` keeps the earliest index).
#[inline]
pub(crate) fn argmin_f64(values: &[f64]) -> (f64, usize) {
    let mut idx = 0;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() && (values[idx].is_nan() || v < values[idx]) {
            idx = i;
        }
    }
    (values[idx], idx)
}

/// Elementwise square root into `out` (same length as `values`).
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn sqrt_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        *o = crate::kernels::sqrt::sqrt_f64(*v);
    }
}

/// Elementwise clip into `out`: `clamp(x, lo, hi)`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn clip_f64(values: &[f64], lo: f64, hi: f64, out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        *o = v.clamp(lo, hi);
    }
}

/// Elementwise reciprocal square root into `out`: `1/sqrt(x)`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn rsqrt_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        *o = 1.0 / crate::kernels::sqrt::sqrt_f64(*v);
    }
}

/// Elementwise subtract a scalar into `out`: `x - p`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn sub_scalar_f64(values: &[f64], p: f64, _p2: f64, out: &mut [f64]) {
    for (o, &x) in out.iter_mut().zip(values) {
        *o = x - p;
    }
}

pub(crate) fn exp_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        *o = crate::kernels::exp::exp_f64(*v);
    }
}

/// Numerically-stable softmax into `out`.
#[cfg(feature = "alloc")]
pub(crate) fn softmax_f64(values: &[f64], out: &mut [f64]) {
    let Some(max) = values.iter().copied().reduce(f64::max) else {
        return;
    };
    let mut sum = 0.0;
    for (i, &v) in values.iter().enumerate() {
        let e = crate::kernels::exp::exp_f64(v - max);
        out[i] = e;
        sum += e;
    }
    if sum != 0.0 {
        let inv = 1.0 / sum;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
}

/// Elementwise sigmoid into `out`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn sigmoid_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        *o = 1.0 / (1.0 + crate::kernels::exp::exp_f64(-*v));
    }
}

/// Elementwise `SiLU` (`x * sigmoid(x)`) into `out`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn silu_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        *o = *v / (1.0 + crate::kernels::exp::exp_f64(-*v));
    }
}

/// Elementwise GELU (tanh approximation) into `out`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn gelu_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        let z = 0.797_884_560_802_865_4 * (*v + 0.044_715 * *v * *v * *v);
        let tanh_z = 1.0 - 2.0 / (crate::kernels::exp::exp_f64(2.0 * z) + 1.0);
        *o = 0.5 * *v * (1.0 + tanh_z);
    }
}

/// Elementwise `ReLU` (`max(x, 0)`) into `out`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn relu_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        *o = (*v).max(0.0);
    }
}

/// Elementwise `tanh` into `out`. Piecewise: Horner Taylor series (through
/// x¹³) for `|x| < 0.1` — truncation there is < 0.1 ulp, while the exp form
/// `1 - 2/(e^2x+1)` cancels catastrophically; beyond, the exp form is
/// cancellation-safe and saturates correctly to ±1. NaN propagates.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn tanh_f64(values: &[f64], out: &mut [f64]) {
    for (v, o) in values.iter().zip(out) {
        let a = v.abs();
        *o = if a > 19.062 {
            v.signum() // tanh(x) = ±1 to f64 precision (e^-2x < 2^-54)
        } else if a < 2e-8 {
            *v // tanh(x) ≈ x, error x³/3 < 1 ulp
        } else if a < 0.1 {
            // tanh(x) = x·P(x²), P = Σ c_k·x^{2k} (odd-powers Taylor).
            let y = v * v;
            let p = 0.003_592_128_572_437_055_f64; // 21844/6081075
            let p = p * y - 0.008_863_235_529_902_197; // -1382/155925
            let p = p * y + 0.021_869_488_536_155_2; // 62/2835
            let p = p * y - 0.053_968_253_968_253_97; // -17/315
            let p = p * y + 0.133_333_333_333_333_33; // 2/15
            let p = p * y - 0.333_333_333_333_333_3; // -1/3
            v * (p * y + 1.0)
        } else {
            let e = crate::kernels::exp::exp_f64(2.0 * v);
            (e - 1.0) / (e + 1.0) // Sterbenz-exact for e in [1, 2]
        };
    }
}

/// RMS norm into `out`: `x_i * rsqrt(mean(x²) + eps)`. Empty input leaves
/// `out` untouched.
#[cfg(feature = "alloc")]
#[inline]
#[allow(clippy::cast_precision_loss)] // `len as f64` is inherent to the mean
pub(crate) fn rms_norm_f64(values: &[f64], eps: f64, out: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let mean_sq = values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64;
    let inv = 1.0 / crate::kernels::sqrt::sqrt_f64(mean_sq + eps);
    for (i, &v) in values.iter().enumerate() {
        out[i] = v * inv;
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn prod_empty() {
        assert_eq!(prod(&[]), 1.0);
    }

    #[test]
    fn prod_single() {
        assert_eq!(prod(&[5.0]), 5.0);
    }

    #[test]
    fn prod_multiple() {
        assert_eq!(prod(&[2.0, 3.0, 4.0]), 24.0);
    }

    #[test]
    fn prod_with_zero() {
        assert_eq!(prod(&[2.0, 0.0, 4.0]), 0.0);
    }

    #[test]
    fn softmax_empty() {
        let mut out = [1.0_f32; 2];
        softmax(&[], &mut out[..0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let v = [1.0_f32, 2.0, 3.0];
        let mut out = [0.0_f32; 3];
        softmax(&v, &mut out);
        let s: f32 = out.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum={s}");
    }

    #[test]
    fn softmax_single() {
        let mut out = [0.0_f32];
        softmax(&[7.0], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_shift_invariance() {
        // Adding a constant to all inputs must not change the output.
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [11.0_f32, 12.0, 13.0, 14.0];
        let mut oa = [0.0_f32; 4];
        let mut ob = [0.0_f32; 4];
        softmax(&a, &mut oa);
        softmax(&b, &mut ob);
        for i in 0..4 {
            assert!((oa[i] - ob[i]).abs() < 1e-6, "lane {i}");
        }
    }

    #[test]
    fn sigmoid_known_values() {
        let v = [0.0_f32, 1.0, -1.0];
        let mut out = [0.0_f32; 3];
        sigmoid(&v, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6, "sigmoid(0)={}", out[0]);
        assert!((out[1] - 0.731_058_6).abs() < 1e-6, "sigmoid(1)={}", out[1]);
        assert!(
            (out[2] - 0.268_941_4).abs() < 1e-6,
            "sigmoid(-1)={}",
            out[2]
        );
    }

    #[test]
    fn sigmoid_saturates() {
        // Large positive → 1.0, large negative → 0.0, monotone.
        let v = [100.0_f32, -100.0, 0.0, 10.0, -10.0];
        let mut out = [0.0_f32; 5];
        sigmoid(&v, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6, "sigmoid(100)={}", out[0]);
        assert!(out[1].abs() < 1e-6, "sigmoid(-100)={}", out[1]);
        assert!(out[3] > out[4], "sigmoid must be monotone");
        assert!((out[2] - 0.5).abs() < 1e-6, "sigmoid(0)={}", out[2]);
    }

    #[test]
    fn sigmoid_empty() {
        let mut out = [1.0_f32; 2];
        sigmoid(&[], &mut out[..0]);
    }

    #[test]
    fn sigmoid_identity() {
        // sigmoid(x) + sigmoid(-x) == 1 (symmetry).
        for x in [-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
            let mut o = [0.0_f32; 2];
            sigmoid(&[x, -x], &mut o);
            assert!(
                (o[0] + o[1] - 1.0).abs() < 1e-5,
                "x={x}: {} + {} != 1",
                o[0],
                o[1]
            );
        }
    }

    #[test]
    fn silu_known_values() {
        let v = [0.0_f32, 1.0, -1.0];
        let mut out = [0.0_f32; 3];
        silu(&v, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6, "silu(0)={}", out[0]);
        assert!((out[1] - 0.731_058_6).abs() < 1e-6, "silu(1)={}", out[1]);
        assert!((out[2] + 0.268_941_4).abs() < 1e-6, "silu(-1)={}", out[2]);
    }

    #[test]
    fn silu_saturates() {
        // Large positive → x, large negative → 0.
        let v = [100.0_f32, -100.0, 5.0, -5.0];
        let mut out = [0.0_f32; 4];
        silu(&v, &mut out);
        assert!((out[0] - 100.0).abs() < 1e-4, "silu(100)={}", out[0]);
        assert!(out[1].abs() < 1e-4, "silu(-100)={}", out[1]);
        assert!(out[2] > 0.0, "silu(5)>0");
        assert!(out[3] < 0.0, "silu(-5)<0");
    }

    #[test]
    fn silu_empty() {
        let mut out = [1.0_f32; 2];
        silu(&[], &mut out[..0]);
    }

    #[test]
    fn silu_matches_x_times_sigmoid() {
        // silu(x) == x * sigmoid(x) for a few values.
        for x in [-3.0_f32, -1.5, 0.0, 0.5, 2.0] {
            let mut o = [0.0_f32; 1];
            silu(&[x], &mut o);
            let expected = x / (1.0 + exp::exp(-x));
            assert!(
                (o[0] - expected).abs() < 1e-6,
                "x={x}: {} vs {expected}",
                o[0]
            );
        }
    }

    #[test]
    fn gelu_known_values() {
        // gelu(0) = 0, gelu(1) ≈ 0.8413, gelu(-1) ≈ -0.1587 (exact GELU);
        // tanh approx: gelu(1) ≈ 0.84119, gelu(-1) ≈ -0.15881.
        let v = [0.0_f32, 1.0, -1.0];
        let mut out = [0.0_f32; 3];
        gelu(&v, &mut out);
        assert!(out[0].abs() < 1e-6, "gelu(0)={}", out[0]);
        assert!((out[1] - 0.841_19).abs() < 2e-4, "gelu(1)={}", out[1]);
        assert!((out[2] + 0.158_81).abs() < 2e-4, "gelu(-1)={}", out[2]);
    }

    #[test]
    fn gelu_saturates() {
        // Large positive → x, large negative → ~0.
        let v = [100.0_f32, -100.0, 5.0, -5.0];
        let mut out = [0.0_f32; 4];
        gelu(&v, &mut out);
        assert!((out[0] - 100.0).abs() < 1e-3, "gelu(100)={}", out[0]);
        assert!(out[1].abs() < 1e-3, "gelu(-100)={}", out[1]);
        assert!(out[2] > 4.99, "gelu(5)={}", out[2]);
        assert!(out[3].abs() < 1e-3, "gelu(-5)={}", out[3]);
    }

    #[test]
    fn gelu_empty() {
        let mut out = [1.0_f32; 2];
        gelu(&[], &mut out[..0]);
    }

    #[test]
    fn gelu_matches_exact_at_known_points() {
        // Exact GELU = x·Φ(x) (erf-based), computed in f64. The tanh approx
        // is within ~1e-3 absolute in the bulk; at the far tails (|x| > 3)
        // its relative deviation grows but the values are tiny (< 0.005).
        // Tolerances reflect that.
        let points = [
            (-3.0_f32, -0.004_05),
            (-1.0, -0.158_66),
            (0.0, 0.0),
            (1.0, 0.841_34),
            (3.0, 2.995_95),
        ];
        for (x, exact) in points {
            let mut o = [0.0_f32; 1];
            gelu(&[x], &mut o);
            assert!(
                (o[0] - exact).abs() < 5e-3,
                "x={x}: gelu={} vs exact {exact}",
                o[0]
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact: relu is a pure clamp
    fn relu_known_values() {
        let v = [-3.0_f32, -0.5, 0.0, 1.0, 5.0];
        let mut out = [0.0_f32; 5];
        relu(&v, &mut out);
        // Exact: relu is a pure clamp, so equality is exact.
        for (got, want) in out.iter().zip([0.0_f32, 0.0, 0.0, 1.0, 5.0]) {
            assert_eq!(got, &want);
        }
    }

    #[test]
    fn relu_empty() {
        let mut out = [1.0_f32; 2];
        relu(&[], &mut out[..0]);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact: relu maps -0.0 to +0.0
    fn relu_negative_zero() {
        let mut out = [0.0_f32; 1];
        relu(&[-0.0], &mut out);
        assert_eq!(out[0], 0.0);
    }
}
