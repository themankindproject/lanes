//! Scalar (portable) kernel implementations.
//!
//! These serve as the universal fallback when no SIMD backend is available
//! and as reference implementations for correctness testing.
//!
//! The activation kernels below (softmax/sigmoid/silu/gelu/relu) are gated
//! on `alloc`: their only public callers (`lanes::ml::*`) return a `Vec`.
//! The exp they use (`kernels::exp`) is fully `no_std`.

#[cfg(feature = "alloc")]
use crate::kernels::exp;

/// Constants for the GELU tanh approximation.
#[cfg(feature = "alloc")]
const GELU_A: f32 = 0.797_884_6; // sqrt(2/pi) in f32
#[cfg(feature = "alloc")]
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

/// Scalar log-sum-exp: `max + ln(Σ exp(x − max))`. Empty input yields
/// `-infinity`. The max shift prevents overflow for large inputs. Gated on
/// `alloc`: its only caller (`dispatch_logsumexp`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn logsumexp(values: &[f32]) -> f32 {
    let Some(m) = values.iter().copied().reduce(f32::max) else {
        return f32::NEG_INFINITY;
    };
    let sum = values
        .iter()
        .map(|&x| crate::kernels::exp::exp(x - m))
        .sum::<f32>();
    m + crate::kernels::ln::ln(sum)
}

/// Scalar log-softmax into `out`: `x_i − logsumexp(x)`. Empty input leaves
/// `out` untouched. The `ln(sum)` term is subtracted from `(x_i − m)`
/// separately — never folded into `m` — so it does not vanish in the ulp
/// of a large `m`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn log_softmax(values: &[f32], out: &mut [f32]) {
    let Some(m) = values.iter().copied().reduce(f32::max) else {
        return;
    };
    let sum = values
        .iter()
        .map(|&x| crate::kernels::exp::exp(x - m))
        .sum::<f32>();
    let log_sum = crate::kernels::ln::ln(sum);
    for (o, &x) in out.iter_mut().zip(values) {
        *o = (x - m) - log_sum;
    }
}

/// Scalar layer norm into `out`: `(x_i − mean) / sqrt(var + eps)` with
/// population variance. Empty input leaves `out` untouched. NaNs propagate.
#[cfg(feature = "alloc")]
#[inline]
#[allow(clippy::cast_precision_loss)] // `len as f32` is inherent to the mean
pub(crate) fn layer_norm(values: &[f32], eps: f32, out: &mut [f32]) {
    let len = values.len();
    if len == 0 {
        return;
    }
    let mean = values.iter().sum::<f32>() / len as f32;
    let mut sum_sq = 0.0;
    for (i, &x) in values.iter().enumerate() {
        let c = x - mean;
        out[i] = c;
        sum_sq += c * c;
    }
    let inv = 1.0 / crate::kernels::sqrt::sqrt(sum_sq / len as f32 + eps);
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// Scalar softplus reference: `ln(1 + e^x)` via the overflow-free form
/// `max(x, 0) + ln1p(e^-|x|)` — no exp overflow, no precision loss for
/// large |x|. Writes into `out` (same length as `values`).
///
/// The `ln1p` here uses the identity `ln(1+z) = ln(u)·z/(u-1)` with
/// `u = 1+z` (musl/fdlibm `s_log1pf` form) so the result stays accurate
/// when `e^-|x|` underflows toward 0.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn softplus(values: &[f32], out: &mut [f32]) {
    map(values, out, |x| {
        let a = x.abs();
        let z = crate::kernels::exp::exp(-a);
        x.max(0.0) + log1p(z)
    });
}

/// `ln(1+z)` for `z >= 0` (`musl s_log1pf.c` identity). Shared by the softplus
/// scalar tails on every backend.
#[cfg(feature = "alloc")]
#[inline]
#[allow(clippy::float_cmp)] // u == 1.0 is the musl underflow branch
pub(crate) fn log1p(z: f32) -> f32 {
    let u = 1.0 + z;
    if u == 1.0 {
        z
    } else {
        crate::kernels::ln::ln(u) * z / (u - 1.0)
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

/// `map` for `f64`, alloc-gated like its `f32` twin (its users are all
/// alloc-gated dispatch paths).
#[cfg(feature = "alloc")]
#[inline]
fn map_f64(values: &[f64], out: &mut [f64], op: impl Fn(f64) -> f64) {
    debug_assert_eq!(values.len(), out.len());
    for (v, o) in values.iter().zip(out) {
        *o = op(*v);
    }
}

/// Shared two-input elementwise map: applies `op(a[i], b[i])`, writing into
/// `out` (same length as both inputs). Skeleton for two-input scalar maps.
#[cfg(feature = "alloc")]
#[inline]
fn map2(a: &[f32], b: &[f32], out: &mut [f32], op: impl Fn(f32, f32) -> f32) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = op(x, y);
    }
}

/// `map2` for `f64`, alloc-gated like its `f32` twin.
#[cfg(feature = "alloc")]
#[inline]
fn map2_f64(a: &[f64], b: &[f64], out: &mut [f64], op: impl Fn(f64, f64) -> f64) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = op(x, y);
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

/// Elementwise absolute difference into `out`: `|a[i] - b[i]|`.
///
/// NaN inputs yield NaN (`abs` propagates NaN). Gated on `alloc`: its only
/// caller (`dispatch_abs_sub`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn abs_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    map2(a, b, out, |x, y| (x - y).abs());
}

/// Elementwise overflow-safe hypotenuse into `out`: `hypot(a[i], b[i])`.
///
/// Matches `f32::hypot` semantics (overflow-safe, `hypot(inf, nan) == inf`).
/// Gated on `alloc`: its only caller (`dispatch_hypot`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn hypot(a: &[f32], b: &[f32], out: &mut [f32]) {
    map2(a, b, out, crate::kernels::hypot::hypot);
}

/// Elementwise integer power into `out`: `values[i].powi(n)`.
///
/// Bit-exact with `f32::powi` (same squaring loop). Gated on `alloc`: its
/// only caller (`dispatch_powi`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn powi(values: &[f32], n: i32, out: &mut [f32]) {
    map(values, out, |x| crate::kernels::powi::powi(x, n));
}

/// Elementwise reciprocal square root into `out`: `1/sqrt(x)`.
///
/// Gated on `alloc`: its only caller (`dispatch_rsqrt`) is alloc-gated.
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
    map_f64(values, out, crate::kernels::ln::ln_f64);
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

/// Squared Euclidean distance of two equal-length slices:
/// `sum((a[i] - b[i])²)`. Returns `0.0` for empty inputs.
///
/// Caller guarantees equal lengths (zip stops at the shorter otherwise).
#[inline]
pub(crate) fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Hamming distance kernel for packed bitmaps: `sum(popcount(a[i] ^ b[i]))`,
/// i.e. the number of differing **bits**. Returns `0` for empty inputs.
///
/// Caller guarantees equal lengths (zip stops at the shorter otherwise).
#[inline]
pub(crate) fn hamming_popcount(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones() as usize)
        .sum()
}

/// Jaccard counts kernel for packed bitmaps:
/// `(popcount(a & b), popcount(a | b))` — the intersection and union bit
/// counts. Returns `(0, 0)` for empty inputs. This is the shared
/// intermediate form every backend reduces to; the final similarity
/// `intersection / union` (or `None` on empty union) is applied by the
/// dispatch wrapper.
///
/// Caller guarantees equal lengths (zip stops at the shorter otherwise).
#[inline]
pub(crate) fn jaccard_counts(a: &[u8], b: &[u8]) -> (usize, usize) {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        intersection += (x & y).count_ones() as usize;
        union += (x | y).count_ones() as usize;
    }
    (intersection, union)
}

/// Jaccard kernel for packed bitmaps: reduces
/// `(popcount(a & b), popcount(a | b))` to the similarity
/// `intersection / union`, or `None` when the union is empty (both
/// bitmaps all-zero, including the empty case).
///
/// Caller guarantees equal lengths (zip stops at the shorter otherwise).
#[inline]
pub(crate) fn jaccard(a: &[u8], b: &[u8]) -> Option<f32> {
    super::jaccard_similarity(jaccard_counts(a, b))
}

/// i8 dot product with i64 accumulation: `sum(a[i] * b[i] as i64)`.
/// Returns `0` for empty inputs. Exact for any slice length: the i64
/// accumulator cannot overflow in addressable memory (each product is
/// ≤ 16384, so overflow would need > 5.6e14 elements).
///
/// Caller guarantees equal lengths (zip stops at the shorter otherwise).
#[inline]
pub(crate) fn dot_i8(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

/// i8 sum with i64 accumulation: `sum(values[i] as i64)`.
/// Returns `0` for an empty slice. Exact for any slice length.
#[inline]
pub(crate) fn sum_i8(values: &[i8]) -> i64 {
    values.iter().map(|&x| i64::from(x)).sum()
}

/// Find the minimum i8 element. Returns [`None`] for an empty slice.
#[inline]
pub(crate) fn min_i8(values: &[i8]) -> Option<i8> {
    values.iter().copied().min()
}

/// Find the maximum i8 element. Returns [`None`] for an empty slice.
#[inline]
pub(crate) fn max_i8(values: &[i8]) -> Option<i8> {
    values.iter().copied().max()
}

/// Count i8 elements equal to zero.
#[inline]
pub(crate) fn count_zero_i8(values: &[i8]) -> usize {
    values.iter().filter(|&&x| x == 0).count()
}

/// Kullback–Leibler divergence kernel (f32): `sum(p[i] * ln(p[i] / q[i]))`.
/// Returns `0.0` for empty inputs.
///
/// Scalar reference: strictly left-to-right summation. The term formula
/// (`div → ln → mul`) is identical in every SIMD backend, and all use the
/// same fdlibm `ln` (scalar `kernels::ln::ln`, register-only `vln_*`), so
/// backends agree term-for-term and differ only in summation order.
/// Non-positive entries follow raw IEEE arithmetic through `ln` (see the
/// public wrapper's docs).
#[inline]
pub(crate) fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| pi * crate::kernels::ln::ln(pi / qi))
        .sum()
}

/// Jensen–Shannon divergence kernel (f32): the raw two-sided sum
/// `sum(p[i] * ln(p[i] / m[i]) + q[i] * ln(q[i] / m[i]))` with
/// `m = (p + q) / 2`. The algorithm wrapper applies the final `* 0.5`.
/// Returns `0.0` for empty inputs. Same cross-backend contract as
/// [`kl_divergence`].
#[inline]
pub(crate) fn js_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let m = (pi + qi) * 0.5;
            pi * crate::kernels::ln::ln(pi / m) + qi * crate::kernels::ln::ln(qi / m)
        })
        .sum()
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

/// Count elements equal to `+0.0` or `-0.0` (they compare equal).
#[inline]
pub(crate) fn count_zero(values: &[f32]) -> usize {
    values.iter().filter(|&&x| x == 0.0).count()
}

/// Count NaN elements.
#[inline]
pub(crate) fn count_nan(values: &[f32]) -> usize {
    values.iter().filter(|x| x.is_nan()).count()
}

/// Count infinite (`+inf`/`-inf`) elements.
#[inline]
pub(crate) fn count_infinite(values: &[f32]) -> usize {
    values.iter().filter(|x| x.is_infinite()).count()
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

/// `f64` twin of [`squared_distance`].
#[inline]
pub(crate) fn squared_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// `f64` twin of [`kl_divergence`].
#[inline]
pub(crate) fn kl_divergence_f64(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| pi * crate::kernels::ln::ln_f64(pi / qi))
        .sum()
}

/// `f64` twin of [`js_divergence`] (raw two-sided sum; the wrapper halves).
#[inline]
pub(crate) fn js_divergence_f64(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let m = (pi + qi) * 0.5;
            pi * crate::kernels::ln::ln_f64(pi / m) + qi * crate::kernels::ln::ln_f64(qi / m)
        })
        .sum()
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

/// `f64` twin of [`count_zero`].
#[inline]
pub(crate) fn count_zero_f64(values: &[f64]) -> usize {
    values.iter().filter(|&&x| x == 0.0).count()
}

/// `f64` twin of [`count_nan`].
#[inline]
pub(crate) fn count_nan_f64(values: &[f64]) -> usize {
    values.iter().filter(|x| x.is_nan()).count()
}

/// `f64` twin of [`count_infinite`].
#[inline]
pub(crate) fn count_infinite_f64(values: &[f64]) -> usize {
    values.iter().filter(|x| x.is_infinite()).count()
}

/// Elementwise square root into `out` (same length as `values`).
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn sqrt_f64(values: &[f64], out: &mut [f64]) {
    map_f64(values, out, crate::kernels::sqrt::sqrt_f64);
}

/// Elementwise clip into `out`: `clamp(x, lo, hi)`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn clip_f64(values: &[f64], lo: f64, hi: f64, out: &mut [f64]) {
    map_f64(values, out, |x| x.clamp(lo, hi));
}

/// Elementwise absolute difference into `out` (`f64`): `|a[i] - b[i]|`.
///
/// NaN inputs yield NaN (`abs` propagates NaN). Gated on `alloc`: its only
/// caller (`dispatch_abs_sub_f64`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn abs_sub_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    map2_f64(a, b, out, |x, y| (x - y).abs());
}

/// Elementwise overflow-safe hypotenuse into `out` (`f64`).
///
/// Matches `f64::hypot` semantics. Gated on `alloc`: its only caller
/// (`dispatch_hypot_f64`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn hypot_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    map2_f64(a, b, out, crate::kernels::hypot::hypot_f64);
}

/// Elementwise integer power into `out` (`f64`): `values[i].powi(n)`.
///
/// Bit-exact with `f64::powi` (same squaring loop). Gated on `alloc`: its
/// only caller (`dispatch_powi_f64`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn powi_f64(values: &[f64], n: i32, out: &mut [f64]) {
    map_f64(values, out, |x| crate::kernels::powi::powi_f64(x, n));
}

/// Elementwise reciprocal square root into `out`: `1/sqrt(x)`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn rsqrt_f64(values: &[f64], out: &mut [f64]) {
    map_f64(values, out, |x| 1.0 / crate::kernels::sqrt::sqrt_f64(x));
}

#[cfg(feature = "alloc")]
pub(crate) fn exp_f64(values: &[f64], out: &mut [f64]) {
    map_f64(values, out, crate::kernels::exp::exp_f64);
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

/// Scalar softplus (f64): `ln(1 + e^x)` via the overflow-free form
/// `max(x, 0) + ln1p(e^-|x|)`. See [`softplus`] for the formula rationale.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn softplus_f64(values: &[f64], out: &mut [f64]) {
    for (i, &v) in values.iter().enumerate() {
        let a = v.abs();
        let z = crate::kernels::exp::exp_f64(-a);
        out[i] = v.max(0.0) + log1p_f64(z);
    }
}

/// `ln(1+z)` for `z >= 0` (`musl s_log1p.c` identity). Shared by the softplus
/// scalar tails on every backend.
#[cfg(feature = "alloc")]
#[inline]
#[allow(clippy::float_cmp)] // u == 1.0 is the musl underflow branch
pub(crate) fn log1p_f64(z: f64) -> f64 {
    let u = 1.0 + z;
    if u == 1.0 {
        z
    } else {
        crate::kernels::ln::ln_f64(u) * z / (u - 1.0)
    }
}

/// Scalar log-sum-exp (f64): `max + ln(Σ exp(x − max))`. Empty input yields
/// `-infinity`. The max shift prevents overflow for large inputs. Gated on
/// `alloc`: its only caller (`dispatch_logsumexp_f64`) is alloc-gated.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn logsumexp_f64(values: &[f64]) -> f64 {
    let Some(m) = values.iter().copied().reduce(f64::max) else {
        return f64::NEG_INFINITY;
    };
    let sum = values
        .iter()
        .map(|&x| crate::kernels::exp::exp_f64(x - m))
        .sum::<f64>();
    m + crate::kernels::ln::ln_f64(sum)
}

/// Scalar log-softmax into `out` (f64): `x_i − logsumexp(x)`. Empty input
/// leaves `out` untouched. `ln(sum)` is subtracted from `(x_i − m)`
/// separately — never folded into `m` — so it does not vanish in the ulp of
/// a large `m`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn log_softmax_f64(values: &[f64], out: &mut [f64]) {
    let Some(m) = values.iter().copied().reduce(f64::max) else {
        return;
    };
    let sum = values
        .iter()
        .map(|&x| crate::kernels::exp::exp_f64(x - m))
        .sum::<f64>();
    let log_sum = crate::kernels::ln::ln_f64(sum);
    for (o, &x) in out.iter_mut().zip(values) {
        *o = (x - m) - log_sum;
    }
}

/// Scalar layer norm into `out` (f64): `(x_i − mean) / sqrt(var + eps)` with
/// population variance. Empty input leaves `out` untouched. NaNs propagate.
#[cfg(feature = "alloc")]
#[inline]
#[allow(clippy::cast_precision_loss)] // `len as f64` is inherent to the mean
pub(crate) fn layer_norm_f64(values: &[f64], eps: f64, out: &mut [f64]) {
    let len = values.len();
    if len == 0 {
        return;
    }
    let mean = values.iter().sum::<f64>() / len as f64;
    let mut sum_sq = 0.0;
    for (i, &x) in values.iter().enumerate() {
        let c = x - mean;
        out[i] = c;
        sum_sq += c * c;
    }
    let inv = 1.0 / crate::kernels::sqrt::sqrt_f64(sum_sq / len as f64 + eps);
    for o in out.iter_mut() {
        *o *= inv;
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn alloc_uninit_empty() {
        let buf: alloc::vec::Vec<f32> = crate::kernels::alloc_uninit(0);
        assert!(buf.is_empty());
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[allow(clippy::cast_precision_loss)] // test indices are tiny; exactness irrelevant
    fn alloc_uninit_write_then_read() {
        // Miri coverage for the uninit-allocation helper: the buffer comes
        // back uninitialized and must be fully written before any read.
        // Writing every element then reading them back is the exact contract
        // the map kernels uphold.
        let mut buf: alloc::vec::Vec<f32> = crate::kernels::alloc_uninit(17);
        assert_eq!(buf.len(), 17);
        for (i, x) in buf.iter_mut().enumerate() {
            *x = i as f32;
        }
        assert_eq!(buf[16], 16.0);

        let mut buf64: alloc::vec::Vec<f64> = crate::kernels::alloc_uninit(9);
        assert_eq!(buf64.len(), 9);
        for (i, x) in buf64.iter_mut().enumerate() {
            *x = i as f64 * 2.0;
        }
        assert_eq!(buf64[8], 16.0);
    }

    #[test]
    fn prod_empty() {
        assert_eq!(prod(&[]), 1.0);
    }

    #[test]
    fn scalar_hamming_popcount() {
        assert_eq!(hamming_popcount(&[], &[]), 0);
        assert_eq!(hamming_popcount(&[0b01], &[0b11]), 1);
        assert_eq!(hamming_popcount(&[0xFF; 4], &[0x00; 4]), 32);
        assert_eq!(hamming_popcount(&[0xAA, 0x55], &[0x55, 0xAA]), 16);
    }

    #[test]
    fn scalar_jaccard_counts() {
        assert_eq!(jaccard_counts(&[], &[]), (0, 0));
        assert_eq!(jaccard_counts(&[0x00], &[0x00]), (0, 0));
        assert_eq!(jaccard_counts(&[0xFF], &[0xFF]), (8, 8));
        assert_eq!(jaccard_counts(&[0xF0], &[0x0F]), (0, 8));
        // AND = 0b0010_0010 (2), OR = 0b1110_1110 (6).
        assert_eq!(jaccard_counts(&[0b1010_1010], &[0b0110_0110]), (2, 6));
    }

    #[test]
    fn scalar_jaccard() {
        assert_eq!(jaccard(&[], &[]), None);
        assert_eq!(jaccard(&[0x00], &[0x00]), None);
        assert_eq!(jaccard(&[0xFF], &[0xFF]), Some(1.0));
        assert_eq!(jaccard(&[0xF0], &[0x0F]), Some(0.0));
        // AND = 0b0010_0010 (2), OR = 0b1110_1110 (6) -> 1/3.
        let j = jaccard(&[0b1010_1010], &[0b0110_0110]).unwrap();
        assert!((j - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn scalar_dot_i8() {
        assert_eq!(dot_i8(&[], &[]), 0);
        assert_eq!(dot_i8(&[1, -2, 3, -4], &[5, 3, -1, -2]), 4);
        // Extremes need the i64 accumulator.
        assert_eq!(dot_i8(&[-128, 127], &[-128, 127]), 16384 + 16129);
        assert_eq!(dot_i8(&[7; 8], &[3; 8]), 168);
    }

    #[test]
    fn scalar_sum_i8() {
        assert_eq!(sum_i8(&[]), 0);
        assert_eq!(sum_i8(&[1, -2, 3, -4]), -2);
        assert_eq!(sum_i8(&[127; 100]), 12700);
        assert_eq!(sum_i8(&[-128; 3]), -384);
    }

    #[test]
    fn scalar_min_max_i8() {
        assert_eq!(min_i8(&[]), None);
        assert_eq!(max_i8(&[]), None);
        assert_eq!(min_i8(&[3, 1, 4]), Some(1));
        assert_eq!(max_i8(&[3, 1, 4]), Some(4));
        assert_eq!(min_i8(&[-128, 127]), Some(-128));
        assert_eq!(max_i8(&[-128, 127]), Some(127));
    }

    #[test]
    fn scalar_count_zero_i8() {
        assert_eq!(count_zero_i8(&[]), 0);
        assert_eq!(count_zero_i8(&[0, 1, 0, -1, 0]), 3);
        assert_eq!(count_zero_i8(&[1, 2, 3]), 0);
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

    #[test]
    fn logsumexp_known_values() {
        // Empty → -inf; single → x; two equal → x + ln 2; shift-invariant.
        assert_eq!(logsumexp(&[]), f32::NEG_INFINITY);
        assert_eq!(logsumexp(&[3.5]), 3.5);
        assert!((logsumexp(&[1.0, 1.0]) - (1.0 + 2.0_f32.ln())).abs() < 1e-6);
        // [0, -1]: max 0, so result is ln(1 + e^-1).
        let want = (1.0 + (-1.0_f32).exp()).ln();
        assert!((logsumexp(&[0.0, -1.0]) - want).abs() < 1e-6);
        // Shift invariance: adding C to every input adds C to the result.
        let a = [1.0_f32, 2.0, 3.0];
        let b = [101.0_f32, 102.0, 103.0];
        assert!((logsumexp(&b) - logsumexp(&a) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn log_softmax_sums_to_one() {
        let v = [1.0_f32, 2.0, 3.0];
        let mut out = [0.0_f32; 3];
        log_softmax(&v, &mut out);
        let s: f32 = out.iter().map(|x| x.exp()).sum();
        assert!((s - 1.0).abs() < 1e-6, "exp-sum {s}");
        // Equal inputs → each output is -ln(n).
        let u = [5.0_f32; 4];
        let mut o = [0.0_f32; 4];
        log_softmax(&u, &mut o);
        for x in o {
            assert!((x + 4.0_f32.ln()).abs() < 1e-6, "x={x}");
        }
    }

    #[test]
    fn layer_norm_unit_variance() {
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        layer_norm(&v, 1e-5, &mut out);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        let var: f32 = out.iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
        // Constant input: variance 0, so eps dominates → output ~0.
        let c = [7.0_f32; 4];
        layer_norm(&c, 1e-5, &mut out);
        for x in out {
            assert!(x.abs() < 1e-3, "const input gave {x}");
        }
    }

    #[test]
    fn logsumexp_f64_known_values() {
        assert_eq!(logsumexp_f64(&[]), f64::NEG_INFINITY);
        assert_eq!(logsumexp_f64(&[3.5]), 3.5);
        assert!((logsumexp_f64(&[1.0, 1.0]) - (1.0 + 2.0_f64.ln())).abs() < 1e-12);
        let a = [1.0_f64, 2.0, 3.0];
        let b = [1001.0_f64, 1002.0, 1003.0];
        assert!((logsumexp_f64(&b) - logsumexp_f64(&a) - 1000.0).abs() < 1e-9);
    }
}
