//! Scalar (portable) kernel implementations.
//!
//! These serve as the universal fallback when no SIMD backend is available
//! and as reference implementations for correctness testing.

// Softmax's only public caller (`lanes::ml::softmax`) returns a `Vec`, so it
// is gated on `alloc` (available with `std`, or `no_std` + `alloc` feature).
// The exp it uses (`kernels/exp`) is fully no_std.
#[cfg(feature = "alloc")]
mod softmax;

#[cfg(feature = "alloc")]
pub(crate) use softmax::softmax;

// Sigmoid: same gating as softmax (public caller is `lanes::ml::sigmoid`,
// which returns a `Vec`; the exp it uses is fully no_std).
#[cfg(feature = "alloc")]
mod sigmoid;

#[cfg(feature = "alloc")]
pub(crate) use sigmoid::sigmoid;

// `SiLU`: same gating as sigmoid.
#[cfg(feature = "alloc")]
mod silu;

#[cfg(feature = "alloc")]
pub(crate) use silu::silu;

// `GELU`: same gating as sigmoid (public caller is `lanes::ml::gelu`).
#[cfg(feature = "alloc")]
mod gelu;

#[cfg(feature = "alloc")]
pub(crate) use gelu::gelu;

// `ReLU`: same gating (public caller is `lanes::ml::relu`).
#[cfg(feature = "alloc")]
mod relu;

#[cfg(feature = "alloc")]
pub(crate) use relu::relu;

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
        if v > values[idx] {
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
        if v < values[idx] {
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
        if v > values[idx] {
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
        if v < values[idx] {
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

/// Elementwise exponential into `out`.
#[cfg(feature = "alloc")]
#[inline]
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
}
