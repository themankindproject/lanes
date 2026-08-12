//! Elementwise math functions over slices.
//!
//! Per-element maps (`f(x)` applied to every lane): `sqrt`. Each function
//! dispatches to the best available SIMD backend; the scalar reference is
//! std-free (`kernels::sqrt::sqrt`, IEEE-correct within 1 ulp).

use crate::dispatch::Backend;
use crate::kernels;
use alloc::vec::Vec;

/// Elementwise square root over a slice.
///
/// Returns a new `Vec` of the same length; an empty slice yields an empty
/// `Vec`. NaN/negative inputs yield NaN, `sqrt(±0) = ±0`, `sqrt(inf) = inf`
/// (IEEE 754).
///
/// Gated on `alloc`: returns a heap-allocated `Vec`.
///
/// # Example
/// ```
/// let v = lanes::math::sqrt(&[1.0_f32, 4.0, 9.0]);
/// for (got, want) in v.iter().zip([1.0, 2.0, 3.0]) {
///     assert!((got - want).abs() < 1e-6);
/// }
/// ```
#[cfg(feature = "alloc")]
#[must_use]
pub fn sqrt(values: &[f32]) -> Vec<f32> {
    let mut out = alloc::vec![0.0_f32; values.len()];
    let backend = Backend::detect();
    kernels::dispatch_sqrt(backend, values, &mut out);
    out
}

/// Elementwise clip over a slice: `clamp(x, lo, hi)` per element.
///
/// Returns a new `Vec` of the same length; an empty slice yields an empty
/// `Vec`. NaN inputs yield NaN; `lo > hi` is not checked (clamp's behavior
/// with inverted bounds is unspecified per [`f32::clamp`]).
///
/// Gated on `alloc`: returns a heap-allocated `Vec`.
///
/// # Example
/// ```
/// let v = lanes::math::clip(&[-5.0_f32, 0.5, 3.0, 10.0], -1.0, 2.0);
/// assert_eq!(v, [-1.0, 0.5, 2.0, 2.0]);
/// ```
#[cfg(feature = "alloc")]
#[must_use]
pub fn clip(values: &[f32], lo: f32, hi: f32) -> Vec<f32> {
    let mut out = alloc::vec![0.0_f32; values.len()];
    let backend = Backend::detect();
    kernels::dispatch_clip(backend, values, lo, hi, &mut out);
    out
}

/// Elementwise reciprocal square root over a slice: `1/sqrt(x)` per element.
///
/// Returns a new `Vec` of the same length; an empty slice yields an empty
/// `Vec`. NaN/negative inputs yield NaN, `rsqrt(±0) = ±inf`,
/// `rsqrt(inf) = 0` (IEEE semantics of the underlying sqrt).
///
/// Gated on `alloc`: returns a heap-allocated `Vec`.
///
/// # Example
/// ```
/// let v = lanes::math::rsqrt(&[1.0_f32, 4.0, 16.0]);
/// for (got, want) in v.iter().zip([1.0, 0.5, 0.25]) {
///     assert!((got - want).abs() < 1e-6);
/// }
/// ```
#[cfg(feature = "alloc")]
#[must_use]
pub fn rsqrt(values: &[f32]) -> Vec<f32> {
    let mut out = alloc::vec![0.0_f32; values.len()];
    let backend = Backend::detect();
    kernels::dispatch_rsqrt(backend, values, &mut out);
    out
}

/// Elementwise exponential over a slice: `e^x` per element.
///
/// Returns a new `Vec` of the same length; an empty slice yields an empty
/// `Vec`. `exp(x)` saturates to `0.0` below `x ≈ -104` and `inf` above
/// `x ≈ 88.7` (IEEE); NaN propagates. Accuracy: ≤ 2 ulp vs `f32::exp`.
///
/// Gated on `alloc`: returns a heap-allocated `Vec`.
///
/// # Example
/// ```
/// let v = lanes::math::exp(&[0.0_f32, 1.0]);
/// assert!((v[0] - 1.0).abs() < 1e-6);
/// assert!((v[1] - std::f32::consts::E).abs() < 1e-5);
/// ```
#[cfg(feature = "alloc")]
#[must_use]
pub fn exp(values: &[f32]) -> Vec<f32> {
    let mut out = alloc::vec![0.0_f32; values.len()];
    let backend = Backend::detect();
    kernels::dispatch_exp(backend, values, &mut out);
    out
}
