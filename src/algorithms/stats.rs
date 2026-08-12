//! Statistical reductions over slices.
//!
//! Aggregate functions that reduce a slice to a single value (or `None` for
//! empty input): `sum`, `prod`, `min`, `max`, `sum_sq`, `mean`, `variance`.
//! All are dispatched to the best available SIMD backend at runtime.

use crate::dispatch::Backend;
use crate::error::Error;
use crate::kernels;

/// Compute the sum of all elements in a slice.
///
/// Returns `0.0` for an empty slice.
///
/// # Example
/// ```
/// assert_eq!(lanes::stats::sum(&[1.0_f32, 2.0, 3.0]), 6.0);
/// ```
#[must_use]
pub fn sum(values: &[f32]) -> f32 {
    let backend = Backend::detect();
    kernels::dispatch_sum(backend, values)
}

/// Compute the product of all elements in a slice.
///
/// Returns `1.0` for an empty slice.
///
/// # Example
/// ```
/// assert_eq!(lanes::stats::prod(&[2.0_f32, 3.0, 4.0]), 24.0);
/// ```
#[must_use]
pub fn prod(values: &[f32]) -> f32 {
    let backend = Backend::detect();
    kernels::dispatch_prod(backend, values)
}

/// Find the minimum element in a slice.
///
/// Returns [`None`] if the slice is empty.
///
/// # Example
/// ```
/// assert_eq!(lanes::stats::min(&[3.0_f32, 1.0, 4.0]), Some(1.0));
/// ```
#[must_use]
pub fn min(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let backend = Backend::detect();
    kernels::dispatch_min(backend, values)
}

/// Find the maximum element in a slice.
///
/// Returns [`None`] if the slice is empty.
///
/// # Example
/// ```
/// assert_eq!(lanes::stats::max(&[3.0_f32, 1.0, 4.0]), Some(4.0));
/// ```
#[must_use]
pub fn max(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let backend = Backend::detect();
    kernels::dispatch_max(backend, values)
}

/// Compute the sum of squares of all elements in a slice.
///
/// Returns `0.0` for an empty slice.
///
/// # Example
/// ```
/// assert_eq!(lanes::stats::sum_sq(&[1.0_f32, 2.0, 3.0]), 14.0);
/// ```
#[must_use]
pub fn sum_sq(values: &[f32]) -> f32 {
    let backend = Backend::detect();
    kernels::dispatch_sum_sq(backend, values)
}

/// Compute the arithmetic mean of a slice.
///
/// Returns [`None`] if the slice is empty.
///
/// # Example
/// ```
/// assert_eq!(lanes::stats::mean(&[1.0_f32, 2.0, 3.0]), Some(2.0));
/// ```
#[allow(clippy::cast_precision_loss)] // `len as f32` is inherent to mean
#[must_use]
pub fn mean(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let backend = Backend::detect();
    Some(kernels::dispatch_sum(backend, values) / values.len() as f32)
}

/// Compute the (population) variance of a slice.
///
/// Returns [`None`] if the slice is empty. Uses the numerically stable
/// two-pass form `sum((x-μ)²)/n`.
///
/// Gated on `alloc`: the second pass needs a heap buffer for the centered
/// values.
///
/// # Example
/// ```
/// let v = lanes::stats::variance(&[1.0_f32, 2.0, 3.0]).unwrap();
/// assert!((v - 2.0 / 3.0).abs() < 1e-6);
/// ```
#[cfg(feature = "alloc")]
#[allow(clippy::cast_precision_loss)] // `len as f32` is inherent to variance
#[must_use]
pub fn variance(values: &[f32]) -> Option<f32> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let backend = Backend::detect();
    let mean = kernels::dispatch_sum(backend, values) / n as f32;
    let centered: alloc::vec::Vec<f32> = values.iter().map(|x| (x - mean) * (x - mean)).collect();
    Some(kernels::dispatch_sum(backend, &centered) / n as f32)
}

/// Compute the dot product of two slices (linear algebra, also exposed at
/// the crate root as `lanes::dot`).
///
/// Returns an error if the slices have different lengths.
///
/// # Errors
/// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
///
/// # Example
/// ```
/// assert_eq!(lanes::stats::dot(&[1.0_f32, 2.0], &[3.0_f32, 4.0]).unwrap(), 11.0);
/// ```
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32, Error> {
    if a.len() != b.len() {
        return Err(Error::LengthMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let backend = Backend::detect();
    Ok(kernels::dispatch_dot(backend, a, b))
}
