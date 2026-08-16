//! Special functions over slices: `erf` (error function) and `erfc`
//! (complementary error function).
//!
//! Per-element maps dispatched to the best available SIMD backend; the
//! scalar reference is std-free. Accuracy contracts (validated against an
//! arbitrary-precision mpmath oracle — see `scripts/gen_erf_reference.py`
//! and `tests/erf_reference.rs`):
//!
//! * f64 `erf`: ≤ 1 ulp over every finite input.
//! * f64 `erfc`: ≤ 3 ulp — the structural floor of the exp-product tail
//!   form (the small/middle regions are ≤ 2 ulp).
//! * f32 `erf`/`erfc`: perfectly rounded — computed in f64 and rounded
//!   once (0 ulp measured over every non-negative f32 input).
//!
//! ## Performance
//!
//! `erf`/`erfc` are piecewise, so per-element cost depends on where the
//! input lands:
//!
//! * `|x| < 0.84375` (small): one degree-13 Horner, no `exp` — the
//!   cheapest region, runs at full SIMD speed.
//! * `0.84375 ≤ |x| < 1.25` (middle): one rational `P/Q`, no `exp`.
//! * `1.25 ≤ |x| ≤ 27.23` (tail): two correctly-rounded vector `exp`s
//!   plus a rational — several times the per-element cost of the small
//!   region. That is the price of the accuracy contract: the
//!   single-`exp` tail alternative was measured at 249 ulp and rejected.
//! * `|x| > 27.23`: saturated (±1 / 0 / 2), nearly free.
//!
//! SIMD chunks whose lanes all fall in one region take a fast path that
//! skips the other regions' work entirely, so uniform inputs (the common
//! case — e.g. GELU-style workloads concentrated near zero) are the fast
//! case, while inputs that alternate regions every element are the worst
//! case. A tail-heavy distribution can run *slower* than a low-accuracy
//! polynomial approximation; see the benchmark table in the README for
//! measured numbers.
//!
//! Every map comes in two forms: the allocating form returns a new `Vec`,
//! and the allocation-free `_into` form writes into a caller-provided
//! buffer. Length mismatches are reported as [`Error::LengthMismatch`] —
//! nothing in this module panics on bad input.
//!
//! Precision is selected by the submodule: [`f32`] for single-precision,
//! [`f64`] for double-precision.
//!
//! [`Error::LengthMismatch`]: crate::Error::LengthMismatch
pub mod f32 {
    //! Single-precision (`f32`) special functions.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;
    use alloc::vec::Vec;

    /// Elementwise error function, written into `out` (allocation-free
    /// variant of [`erf`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`erf`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f32, 1.0, 2.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::special::f32::erf_into(&v, &mut out).unwrap();
    /// for (got, want) in out.iter().zip([0.0, 0.8427008, 0.9953223]) {
    ///     assert!((got - want).abs() < 1e-6);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn erf_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_erf(backend, values, out);
        Ok(())
    }

    /// Elementwise error function `erf(x) = (2/√π) ∫₀ˣ e^(−t²) dt` over a
    /// slice.
    ///
    /// Perfectly rounded: computed in f64 and rounded once. `erf(-x) =
    /// -erf(x)`, `erf(±0) = ±0`, `erf(±∞) = ±1`, NaN propagates.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`erf_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::special::f32::erf(&[0.0_f32, 1.0, 2.0]);
    /// for (got, want) in v.iter().zip([0.0, 0.8427008, 0.9953223]) {
    ///     assert!((got - want).abs() < 1e-6);
    /// }
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn erf(values: &[f32]) -> Vec<f32> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = erf_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Elementwise complementary error function, written into `out`
    /// (allocation-free variant of [`erfc`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`erfc`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f32, 1.0, 3.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::special::f32::erfc_into(&v, &mut out).unwrap();
    /// for (got, want) in out.iter().zip([1.0, 0.1572992, 0.0000220905]) {
    ///     assert!((got - want).abs() < 1e-6);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn erfc_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_erfc(backend, values, out);
        Ok(())
    }

    /// Elementwise complementary error function `erfc(x) = 1 − erf(x)`
    /// over a slice, computed directly (not as `1 − erf(x)`) so no
    /// precision is lost where `erf(x)` is near 1.
    ///
    /// Perfectly rounded: computed in f64 and rounded once. `erfc(+∞) = 0`,
    /// `erfc(−∞) = 2`, `erfc(±0) = 1`, NaN propagates. Large arguments
    /// underflow to `+0.0` (e.g. `erfc(28) = 0` in f64, hence also f32).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`erfc_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::special::f32::erfc(&[0.0_f32, 1.0, 3.0]);
    /// for (got, want) in v.iter().zip([1.0, 0.1572992, 0.0000220905]) {
    ///     assert!((got - want).abs() < 1e-6);
    /// }
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn erfc(values: &[f32]) -> Vec<f32> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = erfc_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }
}

pub mod f64 {
    //! Double-precision (`f64`) special functions.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;
    use alloc::vec::Vec;

    /// Elementwise error function, written into `out` (allocation-free
    /// variant of [`erf`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`erf`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f64, 1.0, 2.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::special::f64::erf_into(&v, &mut out).unwrap();
    /// for (got, want) in out.iter().zip([0.0, 0.8427007929497149, 0.9953222650189527]) {
    ///     assert!((got - want).abs() < 1e-12);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn erf_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_erf_f64(backend, values, out);
        Ok(())
    }

    /// Elementwise error function `erf(x) = (2/√π) ∫₀ˣ e^(−t²) dt` over a
    /// slice.
    ///
    /// Accuracy: ≤ 1 ulp over every finite input (measured against an
    /// arbitrary-precision oracle). `erf(-x) = -erf(x)`, `erf(±0) = ±0`,
    /// `erf(±∞) = ±1`, NaN propagates.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`erf_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::special::f64::erf(&[0.0_f64, 1.0, 2.0]);
    /// for (got, want) in v.iter().zip([0.0, 0.8427007929497149, 0.9953222650189527]) {
    ///     assert!((got - want).abs() < 1e-12);
    /// }
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn erf(values: &[f64]) -> Vec<f64> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = erf_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Elementwise complementary error function, written into `out`
    /// (allocation-free variant of [`erfc`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`erfc`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f64, 1.0, 3.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::special::f64::erfc_into(&v, &mut out).unwrap();
    /// for (got, want) in out.iter().zip([1.0, 0.15729920705028513, 2.2090496998585441e-5]) {
    ///     assert!((got - want).abs() < 1e-12);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn erfc_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_erfc_f64(backend, values, out);
        Ok(())
    }

    /// Elementwise complementary error function `erfc(x) = 1 − erf(x)`
    /// over a slice, computed directly (not as `1 − erf(x)`) so no
    /// precision is lost where `erf(x)` is near 1.
    ///
    /// Accuracy: ≤ 3 ulp — the structural floor of the exp-product tail
    /// form used for large arguments (the small/middle regions are ≤ 2
    /// ulp); measured against an arbitrary-precision oracle.
    /// `erfc(+∞) = 0`, `erfc(−∞) = 2`, `erfc(±0) = 1`, NaN propagates.
    /// Large arguments underflow to `+0.0` (e.g. `erfc(28) = 0`).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`erfc_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::special::f64::erfc(&[0.0_f64, 1.0, 3.0]);
    /// for (got, want) in v.iter().zip([1.0, 0.15729920705028513, 2.2090496998585441e-5]) {
    ///     assert!((got - want).abs() < 1e-12);
    /// }
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn erfc(values: &[f64]) -> Vec<f64> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = erfc_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }
}
