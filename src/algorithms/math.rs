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

    /// Elementwise square root, written into `out` (allocation-free variant of [`sqrt`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`sqrt`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f32, 4.0, 9.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::math::f32::sqrt_into(&v, &mut out);
    /// for (got, want) in out.iter().zip([1.0, 2.0, 3.0]) {
    ///     assert!((got - want).abs() < 1e-6);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn sqrt_into(values: &[f32], out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_sqrt(backend, values, out);
    }

    /// Elementwise square root over a slice.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. NaN/negative inputs yield NaN, `sqrt(±0) = ±0`, `sqrt(inf) = inf`
    /// (IEEE 754).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`sqrt_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        sqrt_into(values, &mut out);
        out
    }

    /// Elementwise clip (`clamp(x, lo, hi)`), written into `out` (allocation-free variant of [`clip`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`clip`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [-5.0_f32, 0.5, 3.0, 10.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::math::f32::clip_into(&v, -1.0, 2.0, &mut out);
    /// assert_eq!(out, [-1.0, 0.5, 2.0, 2.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn clip_into(values: &[f32], lo: f32, hi: f32, out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_clip(backend, values, lo, hi, out);
    }

    /// Elementwise clip over a slice: `clamp(x, lo, hi)` per element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. NaN inputs yield NaN; `lo > hi` is not checked (clamp's behavior
    /// with inverted bounds is unspecified per [`f32::clamp`]).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`clip_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        clip_into(values, lo, hi, &mut out);
        out
    }

    /// Elementwise absolute difference `|a[i] - b[i]|`, written into `out`
    /// (allocation-free variant of [`abs_sub`]).
    ///
    /// Allocation-free: `a`, `b`, and `out` must all have the same length;
    /// reuse `out` across calls to avoid per-call allocation in hot loops.
    ///
    /// # Example
    /// ```
    /// let a = [1.0_f32, 5.0, -3.0];
    /// let b = [4.0_f32, 2.0, -8.0];
    /// let mut out = vec![0.0_f32; a.len()];
    /// lanes::math::f32::abs_sub_into(&a, &b, &mut out);
    /// assert_eq!(out, [3.0, 3.0, 5.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()` or `out.len() != a.len()`.
    pub fn abs_sub_into(a: &[f32], b: &[f32], out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` (or mismatched inputs) would be UB.
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_abs_sub(backend, a, b, out);
    }

    /// Elementwise absolute difference over two slices: `|a[i] - b[i]|`.
    ///
    /// Returns a new `Vec` of length `a.len()`; an empty pair yields an
    /// empty `Vec`. NaN inputs yield NaN (`abs` propagates NaN).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f32::abs_sub(&[1.0_f32, 5.0], &[4.0_f32, 2.0]);
    /// assert_eq!(v, [3.0, 3.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()`.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn abs_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        let mut out = alloc::vec![0.0_f32; a.len()];
        abs_sub_into(a, b, &mut out);
        out
    }

    /// Elementwise overflow-safe hypotenuse `sqrt(a[i]² + b[i]²)`, written
    /// into `out` (allocation-free variant of [`hypot`]).
    ///
    /// Scales by `max(|a[i]|, |b[i]|)` instead of squaring directly, so it
    /// does not overflow for large magnitudes (matches [`f32::hypot`]).
    ///
    /// # Example
    /// ```
    /// let a = [3.0_f32];
    /// let b = [4.0_f32];
    /// let mut out = vec![0.0_f32; 1];
    /// lanes::math::f32::hypot_into(&a, &b, &mut out);
    /// assert!((out[0] - 5.0).abs() < 1e-6);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()` or `out.len() != a.len()`.
    pub fn hypot_into(a: &[f32], b: &[f32], out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` (or mismatched inputs) would be UB.
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_hypot(backend, a, b, out);
    }

    /// Elementwise overflow-safe hypotenuse over two slices.
    ///
    /// Returns a new `Vec`; matches `f32::hypot` within 1–2 ulp with
    /// identical NaN/inf propagation (`hypot(inf, nan) == inf`).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f32::hypot(&[3.0_f32], &[4.0_f32]);
    /// assert!((v[0] - 5.0).abs() < 1e-6);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()`.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn hypot(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        let mut out = alloc::vec![0.0_f32; a.len()];
        hypot_into(a, b, &mut out);
        out
    }

    /// Elementwise integer power `values[i].powi(n)`, written into `out`
    /// (allocation-free variant of [`powi`]).
    ///
    /// Bit-exact with [`f32::powi`]: `powi(x, 0) == 1` for every `x`
    /// (including NaN/inf), negative `n` takes the reciprocal, and
    /// `powi(x, i32::MIN)` is `1 / x^(2^31)`.
    ///
    /// # Example
    /// ```
    /// let v = [2.0_f32, 3.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::math::f32::powi_into(&v, 3, &mut out);
    /// assert_eq!(out, [8.0, 27.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn powi_into(values: &[f32], n: i32, out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_powi(backend, values, n, out);
    }

    /// Elementwise integer power over a slice: `values[i].powi(n)`.
    ///
    /// Returns a new `Vec`; bit-exact with [`f32::powi`] on every backend.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f32::powi(&[2.0_f32, 3.0], 3);
    /// assert_eq!(v, [8.0, 27.0]);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn powi(values: &[f32], n: i32) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        powi_into(values, n, &mut out);
        out
    }

    /// Elementwise reciprocal square root, written into `out` (allocation-free variant of [`rsqrt`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`rsqrt`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f32, 4.0, 16.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::math::f32::rsqrt_into(&v, &mut out);
    /// for (got, want) in out.iter().zip([1.0, 0.5, 0.25]) {
    ///     assert!((got - want).abs() < 1e-6);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn rsqrt_into(values: &[f32], out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_rsqrt(backend, values, out);
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
    /// Convenience wrapper around [`rsqrt_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        rsqrt_into(values, &mut out);
        out
    }

    /// Elementwise hyperbolic tangent, written into `out` (allocation-free variant of [`tanh`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`tanh`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f32, 10.0, -10.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::math::f32::tanh_into(&v, &mut out);
    /// assert!(out[0].abs() < 1e-6);
    /// assert!((out[1] - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn tanh_into(values: &[f32], out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_tanh(backend, values, out);
    }

    /// Elementwise hyperbolic tangent: `tanh(x) = 1 - 2/(e^(2x) + 1)`.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. Saturates to ±1 (via exp overflow/underflow); NaN propagates.
    /// Accuracy follows the crate's `exp` kernel (≤ 2 ulp).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`tanh_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        tanh_into(values, &mut out);
        out
    }

    /// Elementwise exponential, written into `out` (allocation-free variant of [`exp`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`exp`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f32, 1.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::math::f32::exp_into(&v, &mut out);
    /// assert!((out[0] - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn exp_into(values: &[f32], out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_exp(backend, values, out);
    }

    /// Elementwise exponential over a slice: `e^x` per element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. `exp(x)` saturates to `0.0` below `x ≈ -104` and `inf` above
    /// `x ≈ 88.7` (IEEE); NaN propagates. Accuracy: ≤ 2 ulp vs `f32::exp`.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`exp_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        exp_into(values, &mut out);
        out
    }

    /// Elementwise natural logarithm, written into `out` (allocation-free variant of [`ln`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`ln`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f32, std::f32::consts::E];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::math::f32::ln_into(&v, &mut out);
    /// assert!(out[0].abs() < 1e-6);
    /// assert!((out[1] - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn ln_into(values: &[f32], out: &mut [f32]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_ln(backend, values, out);
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
    /// Convenience wrapper around [`ln_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        ln_into(values, &mut out);
        out
    }
}

pub mod f64 {
    //! Double-precision (`f64`) elementwise math functions.

    use crate::dispatch::Backend;
    use crate::kernels;
    use alloc::vec::Vec;

    /// Elementwise square root, written into `out` (allocation-free variant of [`sqrt`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`sqrt`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f64, 4.0, 9.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::math::f64::sqrt_into(&v, &mut out);
    /// for (got, want) in out.iter().zip([1.0, 2.0, 3.0]) {
    ///     assert!((got - want).abs() < 1e-12);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn sqrt_into(values: &[f64], out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_sqrt_f64(backend, values, out);
    }

    /// Elementwise square root over a slice.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. NaN/negative inputs yield NaN, `sqrt(±0) = ±0`, `sqrt(inf) = inf`
    /// (IEEE 754).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`sqrt_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        sqrt_into(values, &mut out);
        out
    }

    /// Elementwise clip (`clamp(x, lo, hi)`), written into `out` (allocation-free variant of [`clip`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`clip`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [-5.0_f64, 0.5, 3.0, 10.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::math::f64::clip_into(&v, -1.0, 2.0, &mut out);
    /// assert_eq!(out, [-1.0, 0.5, 2.0, 2.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn clip_into(values: &[f64], lo: f64, hi: f64, out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_clip_f64(backend, values, lo, hi, out);
    }

    /// Elementwise clip over a slice: `clamp(x, lo, hi)` per element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. NaN inputs yield NaN; `lo > hi` is not checked (clamp's behavior
    /// with inverted bounds is unspecified per [`f64::clamp`]).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`clip_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        clip_into(values, lo, hi, &mut out);
        out
    }

    /// Elementwise absolute difference `|a[i] - b[i]|`, written into `out`
    /// (allocation-free variant of [`abs_sub`]).
    ///
    /// Allocation-free: `a`, `b`, and `out` must all have the same length;
    /// reuse `out` across calls to avoid per-call allocation in hot loops.
    ///
    /// # Example
    /// ```
    /// let a = [1.0_f64, 5.0, -3.0];
    /// let b = [4.0_f64, 2.0, -8.0];
    /// let mut out = vec![0.0_f64; a.len()];
    /// lanes::math::f64::abs_sub_into(&a, &b, &mut out);
    /// assert_eq!(out, [3.0, 3.0, 5.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()` or `out.len() != a.len()`.
    pub fn abs_sub_into(a: &[f64], b: &[f64], out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` (or mismatched inputs) would be UB.
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_abs_sub_f64(backend, a, b, out);
    }

    /// Elementwise absolute difference over two slices: `|a[i] - b[i]|`.
    ///
    /// Returns a new `Vec` of length `a.len()`; an empty pair yields an
    /// empty `Vec`. NaN inputs yield NaN (`abs` propagates NaN).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f64::abs_sub(&[1.0_f64, 5.0], &[4.0_f64, 2.0]);
    /// assert_eq!(v, [3.0, 3.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()`.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn abs_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len());
        let mut out = alloc::vec![0.0_f64; a.len()];
        abs_sub_into(a, b, &mut out);
        out
    }

    /// Elementwise overflow-safe hypotenuse `sqrt(a[i]² + b[i]²)`, written
    /// into `out` (allocation-free variant of [`hypot`]).
    ///
    /// Scales by `max(|a[i]|, |b[i]|)` instead of squaring directly, so it
    /// does not overflow for large magnitudes (matches [`f64::hypot`]).
    ///
    /// # Example
    /// ```
    /// let a = [3.0_f64];
    /// let b = [4.0_f64];
    /// let mut out = vec![0.0_f64; 1];
    /// lanes::math::f64::hypot_into(&a, &b, &mut out);
    /// assert!((out[0] - 5.0).abs() < 1e-12);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()` or `out.len() != a.len()`.
    pub fn hypot_into(a: &[f64], b: &[f64], out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` (or mismatched inputs) would be UB.
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_hypot_f64(backend, a, b, out);
    }

    /// Elementwise overflow-safe hypotenuse over two slices.
    ///
    /// Returns a new `Vec`; matches `f64::hypot` within 1–2 ulp with
    /// identical NaN/inf propagation (`hypot(inf, nan) == inf`).
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f64::hypot(&[3.0_f64], &[4.0_f64]);
    /// assert!((v[0] - 5.0).abs() < 1e-12);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()`.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn hypot(a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len());
        let mut out = alloc::vec![0.0_f64; a.len()];
        hypot_into(a, b, &mut out);
        out
    }

    /// Elementwise integer power `values[i].powi(n)`, written into `out`
    /// (allocation-free variant of [`powi`]).
    ///
    /// Bit-exact with [`f64::powi`]: `powi(x, 0) == 1` for every `x`
    /// (including NaN/inf), negative `n` takes the reciprocal, and
    /// `powi(x, i32::MIN)` is `1 / x^(2^31)`.
    ///
    /// # Example
    /// ```
    /// let v = [2.0_f64, 3.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::math::f64::powi_into(&v, 3, &mut out);
    /// assert_eq!(out, [8.0, 27.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn powi_into(values: &[f64], n: i32, out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_powi_f64(backend, values, n, out);
    }

    /// Elementwise integer power over a slice: `values[i].powi(n)`.
    ///
    /// Returns a new `Vec`; bit-exact with [`f64::powi`] on every backend.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::math::f64::powi(&[2.0_f64, 3.0], 3);
    /// assert_eq!(v, [8.0, 27.0]);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn powi(values: &[f64], n: i32) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        powi_into(values, n, &mut out);
        out
    }

    /// Elementwise reciprocal square root, written into `out` (allocation-free variant of [`rsqrt`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`rsqrt`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f64, 4.0, 16.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::math::f64::rsqrt_into(&v, &mut out);
    /// for (got, want) in out.iter().zip([1.0, 0.5, 0.25]) {
    ///     assert!((got - want).abs() < 1e-12);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn rsqrt_into(values: &[f64], out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_rsqrt_f64(backend, values, out);
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
    /// Convenience wrapper around [`rsqrt_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        rsqrt_into(values, &mut out);
        out
    }

    /// Elementwise exponential, written into `out` (allocation-free variant of [`exp`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`exp`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f64, 1.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::math::f64::exp_into(&v, &mut out);
    /// assert!((out[0] - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn exp_into(values: &[f64], out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_exp_f64(backend, values, out);
    }

    /// Elementwise exponential over a slice: `e^x` per element.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. `exp(x)` saturates to `0.0` below `x ≈ -745.1` and `inf` above
    /// `x ≈ 709.8` (IEEE); NaN propagates. Accuracy: ≤ 1 ulp vs `f64::exp`.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`exp_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        exp_into(values, &mut out);
        out
    }

    /// Elementwise natural logarithm, written into `out` (allocation-free variant of [`ln`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`ln`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f64, std::f64::consts::E];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::math::f64::ln_into(&v, &mut out);
    /// assert!(out[0].abs() < 1e-12);
    /// assert!((out[1] - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn ln_into(values: &[f64], out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_ln_f64(backend, values, out);
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
    /// Convenience wrapper around [`ln_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        ln_into(values, &mut out);
        out
    }

    /// Elementwise hyperbolic tangent, written into `out` (allocation-free variant of [`tanh`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`tanh`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f64, 20.0, -20.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::math::f64::tanh_into(&v, &mut out);
    /// assert!(out[0].abs() < 1e-12);
    /// assert!((out[1] - 1.0).abs() < 1e-12);
    /// assert!((out[2] + 1.0).abs() < 1e-12);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != values.len()`.
    pub fn tanh_into(values: &[f64], out: &mut [f64]) {
        // Always-on check: the backend kernels use unchecked writes, so a
        // short `out` would be UB; panicking here keeps the safe API sound.
        assert_eq!(values.len(), out.len());
        let backend = Backend::detect();
        kernels::dispatch_tanh_f64(backend, values, out);
    }

    /// Elementwise hyperbolic tangent: `tanh(x) = 1 - 2/(e^(2x) + 1)`.
    ///
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`. Saturates to ±1 (via exp overflow/underflow); NaN propagates.
    ///
    /// Gated on `alloc`: returns a heap-allocated `Vec`.
    ///
    /// Convenience wrapper around [`tanh_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        tanh_into(values, &mut out);
        out
    }
}
