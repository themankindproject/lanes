//! Small power-of-two sorts with SIMD-accelerated compare-exchange networks.
//!
//! Provides deterministic ascending sorts for slices of length 8, 16, 32 — the
//! regime where a sorting network beats `std sort` by avoiding branches and
//! keeping the whole input in registers. Other lengths fall back to
//! `sort_unstable_by(total_cmp)`.
//!
//! ## References
//!
//! - [1] K. Batcher, "Sorting networks and their applications," AFIPS 1968.
//! - [2] B. Dobbelaere, "Sorting networks" — minimal-size tables up to n=32
//!   (<https://bertdobbelaere.github.io/sorting_networks.html>).
//! - [3] Intel `x86-simd-sort` — optimal network enlistings used verbatim
//!   (`xss-optimal-networks.hpp`, BSD-3, generated from [2]) via `curl` on
//!   2026-08-22.
//! - [4] R. Sedgewick, *Algorithms* — total order for IEEE 754 (`total_cmp`).
//!
//! Every kernel preserves [`f32::total_cmp`]/[`f64::total_cmp`] ordering:
//! `-inf < … < -0.0 < 0.0 < … < inf < NaN` (all NaN bit patterns sort last,
//! by payload). This matches the scalar reference and the proptest oracle.
//!
//! The family is `no_std`-clean (in-place, no `Vec`).

#![forbid(unsafe_code)]

use crate::dispatch::Backend;
use crate::kernels;

/// Single-precision (f32) sorts.
pub mod f32 {
    use super::{Backend, kernels};

    /// Sort `values` ascending by [`f32::total_cmp`] in place.
    ///
    /// For `values.len()` in `{8, 16, 32}` an optimal sorting network is
    /// dispatched (19, 60, 80 compare-exchanges respectively from [2] via [3],
    /// depth 6/10/10). Every other length falls back to
    /// `values.sort_unstable_by(f32::total_cmp)`.
    ///
    /// Deterministic: result is bit-for-bit identical to the fallback for any
    /// input, including NaNs and signed zeros.
    ///
    /// # References
    ///
    /// - Batcher [1] for the bitonic construction; optimal size bounds from
    ///   Dobbelaere [2]; enlistings from Intel [3].
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = [3.0_f32, 1.0, f32::NAN, -0.0, 0.0, 2.0, 1.5, -1.0];
    /// lanes::sort::f32::bitonic_sort(&mut v);
    /// let mut want = [3.0_f32, 1.0, f32::NAN, -0.0, 0.0, 2.0, 1.5, -1.0];
    /// want.sort_unstable_by(f32::total_cmp);
    /// assert!(v.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
    /// ```
    pub fn bitonic_sort(values: &mut [f32]) {
        let backend = Backend::detect();
        kernels::dispatch_bitonic_sort_f32(backend, values);
    }
}

/// Double-precision (f64) sorts.
pub mod f64 {
    use super::{Backend, kernels};

    /// Sort `values` ascending by [`f64::total_cmp`] in place.
    ///
    /// Same contract as [`f32::bitonic_sort`] with the same lengths and
    /// fallback; networks are instantiated for `f64` lanes.
    ///
    /// # References
    ///
    /// See [`f32::bitonic_sort`] references [1]–[3].
    pub fn bitonic_sort(values: &mut [f64]) {
        let backend = Backend::detect();
        kernels::dispatch_bitonic_sort_f64(backend, values);
    }
}
