//! Public algorithm functions with runtime SIMD dispatch.
//!
//! Each function automatically selects the best available backend
//! for the current CPU. The dispatch decision is cached after the
//! first call (with the `std` feature enabled).

// The `ml` family returns heap-allocated `Vec`s, so it needs `alloc`.
#[cfg(feature = "alloc")]
pub mod ml;

// Statistical reductions (aggregates over slices).
pub mod stats;

// Distance and norm functions.
pub mod distance;

// Elementwise math functions (per-element maps).
#[cfg(feature = "alloc")]
pub mod math;

use crate::error::Error;

/// Compute the sum of all elements in a slice.
///
/// Returns `0.0` for an empty slice.
///
/// # Example
/// ```
/// let values = [1.0_f32, 2.0, 3.0, 4.0];
/// assert_eq!(lanes::sum(&values), 10.0);
/// ```
#[must_use]
pub fn sum(values: &[f32]) -> f32 {
    stats::sum(values)
}

/// Compute the product of all elements in a slice.
///
/// Returns `1.0` for an empty slice.
///
/// # Example
/// ```
/// let values = [2.0_f32, 3.0, 4.0];
/// assert_eq!(lanes::prod(&values), 24.0);
/// ```
#[must_use]
pub fn prod(values: &[f32]) -> f32 {
    stats::prod(values)
}

/// Find the minimum element in a slice.
///
/// Returns [`None`] if the slice is empty.
///
/// # Example
/// ```
/// let values = [3.0_f32, 1.0, 4.0, 1.5];
/// assert_eq!(lanes::min(&values), Some(1.0));
/// assert_eq!(lanes::min(&[] as &[f32]), None);
/// ```
#[must_use]
pub fn min(values: &[f32]) -> Option<f32> {
    stats::min(values)
}

/// Find the maximum element in a slice.
///
/// Returns [`None`] if the slice is empty.
///
/// # Example
/// ```
/// let values = [3.0_f32, 1.0, 4.0, 1.5];
/// assert_eq!(lanes::max(&values), Some(4.0));
/// assert_eq!(lanes::max(&[] as &[f32]), None);
/// ```
#[must_use]
pub fn max(values: &[f32]) -> Option<f32> {
    stats::max(values)
}

/// Compute the dot product of two slices.
///
/// Returns an error if the slices have different lengths.
///
/// # Errors
/// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
///
/// # Example
/// ```
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [4.0_f32, 5.0, 6.0];
/// assert_eq!(lanes::dot(&a, &b).unwrap(), 32.0);
/// ```
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32, Error> {
    stats::dot(a, b)
}
