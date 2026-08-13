//! Elementwise math functions over slices.
//!
//! Per-element maps (`f(x)` applied to every lane): `sqrt`, `clip`, `rsqrt`,
//! `exp`. Each function dispatches to the best available SIMD backend; the
//! scalar reference is std-free and IEEE-correct within 1 ulp.
//!
//! Precision is selected by the submodule: [`f32`] for single-precision,
//! [`f64`] for double-precision.

pub mod f32 {
    //! Single-precision (`f32`) elementwise math functions.

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
    /// let v = lanes::math::f32::sqrt(&[1.0_f32, 4.0, 9.0]);
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
    /// let v = lanes::math::f32::clip(&[-5.0_f32, 0.5, 3.0, 10.0], -1.0, 2.0);
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

    /// Elementwise reciprocal square root over a slice: `1/sqrt(x)` per
    /// element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. NaN/negative inputs yield NaN, `rsqrt(±0) = ±inf`,
    /// `rsqrt(inf) = 0` (IEEE semantics of the underlying sqrt).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f32::rsqrt(&[1.0_f32, 4.0, 16.0]);
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

    /// Elementwise hyperbolic tangent: `tanh(x) = 1 - 2/(e^(2x) + 1)`.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. Saturates to ±1 (via exp overflow/underflow); NaN propagates.
    /// Accuracy follows the crate's `exp` kernel (≤ 2 ulp).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f32::tanh(&[0.0_f32, 10.0, -10.0]);
    /// assert!(v[0].abs() < 1e-6);
    /// assert!((v[1] - 1.0).abs() < 1e-6);
    /// assert!((v[2] + 1.0).abs() < 1e-6);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn tanh(values: &[f32]) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_tanh(backend, values, &mut out);
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
    /// let v = lanes::math::f32::exp(&[0.0_f32, 1.0]);
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

    /// Elementwise natural logarithm over a slice: `ln(x)` per element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. Follows IEEE 754: `ln(±0) = -inf`, `ln(x < 0) = NaN`,
    /// `ln(+inf) = +inf`, `ln(NaN) = NaN`. Accuracy: ≤ 1 ulp vs `f32::ln`
    /// (fdlibm algorithm).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f32::ln(&[1.0_f32, std::f32::consts::E]);
    /// assert!(v[0].abs() < 1e-6);
    /// assert!((v[1] - 1.0).abs() < 1e-6);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn ln(values: &[f32]) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_ln(backend, values, &mut out);
        out
    }
}

pub mod f64 {
    //! Double-precision (`f64`) elementwise math functions.

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
    /// let v = lanes::math::f64::sqrt(&[1.0_f64, 4.0, 9.0]);
    /// for (got, want) in v.iter().zip([1.0, 2.0, 3.0]) {
    ///     assert!((got - want).abs() < 1e-12);
    /// }
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn sqrt(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_sqrt_f64(backend, values, &mut out);
        out
    }

    /// Elementwise clip over a slice: `clamp(x, lo, hi)` per element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. NaN inputs yield NaN; `lo > hi` is not checked (clamp's behavior
    /// with inverted bounds is unspecified per [`f64::clamp`]).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f64::clip(&[-5.0_f64, 0.5, 3.0, 10.0], -1.0, 2.0);
    /// assert_eq!(v, [-1.0, 0.5, 2.0, 2.0]);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn clip(values: &[f64], lo: f64, hi: f64) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_clip_f64(backend, values, lo, hi, &mut out);
        out
    }

    /// Elementwise reciprocal square root over a slice: `1/sqrt(x)` per
    /// element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. NaN/negative inputs yield NaN, `rsqrt(±0) = ±inf`,
    /// `rsqrt(inf) = 0` (IEEE semantics of the underlying sqrt).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f64::rsqrt(&[1.0_f64, 4.0, 16.0]);
    /// for (got, want) in v.iter().zip([1.0, 0.5, 0.25]) {
    ///     assert!((got - want).abs() < 1e-12);
    /// }
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn rsqrt(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_rsqrt_f64(backend, values, &mut out);
        out
    }

    /// Elementwise exponential over a slice: `e^x` per element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. `exp(x)` saturates to `0.0` below `x ≈ -745.1` and `inf` above
    /// `x ≈ 709.8` (IEEE); NaN propagates. Accuracy: ≤ 1 ulp vs `f64::exp`.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f64::exp(&[0.0_f64, 1.0]);
    /// assert!((v[0] - 1.0).abs() < 1e-12);
    /// assert!((v[1] - std::f64::consts::E).abs() < 1e-12);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn exp(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_exp_f64(backend, values, &mut out);
        out
    }

    /// Elementwise natural logarithm over a slice: `ln(x)` per element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. Follows IEEE 754: `ln(±0) = -inf`, `ln(x < 0) = NaN`,
    /// `ln(+inf) = +inf`, `ln(NaN) = NaN`. Accuracy: ≤ 1 ulp vs `f64::ln`
    /// (fdlibm algorithm).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f64::ln(&[1.0_f64, std::f64::consts::E]);
    /// assert!(v[0].abs() < 1e-12);
    /// assert!((v[1] - 1.0).abs() < 1e-12);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn ln(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_ln_f64(backend, values, &mut out);
        out
    }

    /// Elementwise hyperbolic tangent: `tanh(x) = 1 - 2/(e^(2x) + 1)`.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. Saturates to ±1 (via exp overflow/underflow); NaN propagates.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f64::tanh(&[0.0_f64, 20.0, -20.0]);
    /// assert!(v[0].abs() < 1e-12);
    /// assert!((v[1] - 1.0).abs() < 1e-12);
    /// assert!((v[2] + 1.0).abs() < 1e-12);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn tanh(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_tanh_f64(backend, values, &mut out);
        out
    }
}
