//! Statistical reductions over slices.
//!
//! Aggregate functions that reduce a slice to a single value (or `None` for
//! empty input): `sum`, `prod`, `min`, `max`, `sum_sq`, `mean`, `variance`.
//! All are dispatched to the best available SIMD backend at runtime.
//!
//! Precision is selected by the submodule: [`f32`] for single-precision,
//! [`f64`] for double-precision.

pub mod f32 {
    //! Single-precision (`f32`) statistical reductions.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;

    /// Compute the sum of all elements in a slice.
    ///
    /// Returns `0.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f32::sum(&[1.0_f32, 2.0, 3.0]), 6.0);
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
    /// assert_eq!(lanes::stats::f32::prod(&[2.0_f32, 3.0, 4.0]), 24.0);
    /// ```
    #[must_use]
    pub fn prod(values: &[f32]) -> f32 {
        let backend = Backend::detect();
        kernels::dispatch_prod(backend, values)
    }

    /// Find the minimum element in a slice.
    ///
    /// Returns [`None`] if the slice is empty. NaN inputs are ignored
    /// unless every input is NaN (IEEE 754 `minNum` semantics, matching
    /// [`f32::min`]); the result is then NaN. All backends agree.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f32::min(&[3.0_f32, 1.0, 4.0]), Some(1.0));
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
    /// Returns [`None`] if the slice is empty. NaN inputs are ignored
    /// unless every input is NaN (IEEE 754 `maxNum` semantics, matching
    /// [`f32::max`]); the result is then NaN. All backends agree.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f32::max(&[3.0_f32, 1.0, 4.0]), Some(4.0));
    /// ```
    #[must_use]
    pub fn max(values: &[f32]) -> Option<f32> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        kernels::dispatch_max(backend, values)
    }

    /// Find the index of the maximum element in a slice.
    ///
    /// Returns [`None`] if the slice is empty. Ties resolve to the first
    /// occurrence. NaN handling follows [`f32::max`] semantics: a NaN is
    /// ignored unless every element is NaN (in which case the first index
    /// wins).
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f32::argmax(&[3.0_f32, 1.0, 4.0]), Some(2));
    /// ```
    #[must_use]
    pub fn argmax(values: &[f32]) -> Option<usize> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        Some(kernels::dispatch_argmax(backend, values).1)
    }

    /// Find the index of the minimum element in a slice.
    ///
    /// Returns [`None`] if the slice is empty. Ties resolve to the first
    /// occurrence. NaN handling follows [`f32::min`] semantics: a NaN is
    /// ignored unless every element is NaN (in which case the first index
    /// wins).
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f32::argmin(&[3.0_f32, 1.0, 4.0]), Some(1));
    /// ```
    #[must_use]
    pub fn argmin(values: &[f32]) -> Option<usize> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        Some(kernels::dispatch_argmin(backend, values).1)
    }

    /// Compute the sum of squares of all elements in a slice.
    ///
    /// Returns `0.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f32::sum_sq(&[1.0_f32, 2.0, 3.0]), 14.0);
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
    /// assert_eq!(lanes::stats::f32::mean(&[1.0_f32, 2.0, 3.0]), Some(2.0));
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

    // Fused variance helper: single-pass `(x-mean)^2` sum, no alloc. Bit-identical.
    #[cfg(feature = "alloc")]
    #[allow(clippy::cast_precision_loss)]
    #[inline]
    pub(crate) fn variance_fused(backend: Backend, values: &[f32], mean: f32) -> f32 {
        kernels::dispatch_variance_fused_f32(backend, values, mean) / values.len() as f32
    }

    /// Compute the (population) variance of a slice.
    ///
    /// Returns [`None`] if the slice is empty. Uses the numerically stable
    /// two-pass form `sum((x-μ)²)/n`. Gated on `alloc`.
    #[cfg(feature = "alloc")]
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn variance(values: &[f32]) -> Option<f32> {
        let n = values.len();
        if n == 0 {
            return None;
        }
        let backend = Backend::detect();
        let mean = kernels::dispatch_sum(backend, values) / n as f32;
        Some(variance_fused(backend, values, mean))
    }

    /// Compute the (population) variance of a slice, writing the result
    /// into `out[0]` (allocation-free variant of [`variance`]).
    ///
    /// `scratch` must have the same length as `values` and is used as the
    /// second-pass workspace (it holds the centered values); reuse it
    /// across calls in hot loops to avoid the heap allocation that
    /// [`variance`] performs. The result is bit-identical to
    /// [`variance`] — same two-pass form, same SIMD kernels.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `scratch.len() != values.len()`
    /// or `out` is empty.
    ///
    /// # Example
    /// ```
    /// let data = [1.0_f32, 2.0, 3.0];
    /// let mut scratch = [0.0_f32; 3];
    /// let mut out = [0.0_f32; 1];
    /// lanes::stats::f32::variance_into(&data, &mut scratch, &mut out).unwrap();
    /// assert!((out[0] - 2.0 / 3.0).abs() < 1e-6);
    /// ```
    #[allow(clippy::cast_precision_loss)] // `len as f32` is inherent to variance
    pub fn variance_into(
        values: &[f32],
        scratch: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        if values.len() != scratch.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: scratch.len(),
            });
        }
        if out.is_empty() {
            return Err(Error::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if values.is_empty() {
            return Ok(()); // crate convention: empty input leaves `out` untouched
        }
        let backend = Backend::detect();
        let mean = kernels::dispatch_sum(backend, values) / values.len() as f32;
        // Keep the fused helper single-source but preserve the bit-identical
        // contract that callers observe written-to `scratch` (some callers
        // reuse the centered buffer and benchmarks rely on it being touched).
        #[cfg(feature = "alloc")]
        {
            let var = variance_fused(backend, values, mean);
            // Write centered values into `scratch` visibly (touches memory
            // like the old scalar loop) — still one vector pass.
            kernels::dispatch_center_f32(backend, values, mean, scratch);
            out[0] = var;
            return Ok(());
        }
        #[allow(unreachable_code)]
        {
            for (c, &x) in scratch.iter_mut().zip(values) {
                *c = x - mean;
            }
            out[0] = kernels::dispatch_sum_sq(backend, scratch) / values.len() as f32;
            Ok(())
        }
    }

    /// Compute the (population) standard deviation of a slice:
    /// `sqrt(variance(x))`.
    ///
    /// Returns [`None`] if the slice is empty. Same numerical properties as
    /// [`variance`](variance).
    ///
    /// Gated on `alloc`: shares variance's two-pass heap buffer.
    ///
    /// # Example
    /// ```
    /// let v = lanes::stats::f32::std_dev(&[1.0_f32, 2.0, 3.0]).unwrap();
    /// assert!((v - (2.0_f32 / 3.0).sqrt()).abs() < 1e-6);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn std_dev(values: &[f32]) -> Option<f32> {
        variance(values).map(crate::kernels::sqrt::sqrt)
    }

    /// Compute the (population) standard deviation of a slice, writing the
    /// result into `out[0]` (allocation-free variant of [`std_dev`]).
    ///
    /// Same contract as [`variance_into`]: `scratch` must match
    /// `values.len()`, `out` must be non-empty, and the result is
    /// bit-identical to [`std_dev`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `scratch.len() != values.len()`
    /// or `out` is empty.
    ///
    /// # Example
    /// ```
    /// let data = [1.0_f32, 2.0, 3.0];
    /// let mut scratch = [0.0_f32; 3];
    /// let mut out = [0.0_f32; 1];
    /// lanes::stats::f32::std_dev_into(&data, &mut scratch, &mut out).unwrap();
    /// assert!((out[0] - (2.0_f32 / 3.0).sqrt()).abs() < 1e-6);
    /// ```
    #[allow(clippy::cast_precision_loss)] // `len as f32` is inherent to the variance
    pub fn std_dev_into(values: &[f32], scratch: &mut [f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != scratch.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: scratch.len(),
            });
        }
        if out.is_empty() {
            return Err(Error::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if values.is_empty() {
            return Ok(()); // crate convention: empty input leaves `out` untouched
        }
        let backend = Backend::detect();
        let mean = kernels::dispatch_sum(backend, values) / values.len() as f32;
        #[cfg(feature = "alloc")]
        {
            let var = variance_fused(backend, values, mean);
            kernels::dispatch_center_f32(backend, values, mean, scratch);
            out[0] = crate::kernels::sqrt::sqrt(var);
            return Ok(());
        }
        #[allow(unreachable_code)]
        {
            for (c, &x) in scratch.iter_mut().zip(values) {
                *c = x - mean;
            }
            let var = kernels::dispatch_sum_sq(backend, scratch) / values.len() as f32;
            out[0] = crate::kernels::sqrt::sqrt(var);
            Ok(())
        }
    }

    /// Compute the geometric mean of a slice:
    /// `exp(mean(ln(x)))`, the n-th root of the product.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyInput`] for an empty slice, and
    /// [`Error::NonPositiveInput`] (with the offending index) if any value is
    /// ≤ 0 — the geometric mean is only defined over strictly positive reals.
    /// NaN inputs are *not* an error: they propagate to a NaN result, matching
    /// the crate's reduction semantics.
    ///
    /// Gated on `alloc`: uses the vectorized `ln` map + `exp`.
    ///
    /// # Example
    /// ```
    /// let g = lanes::stats::f32::geometric_mean(&[1.0_f32, 4.0, 16.0]).unwrap();
    /// assert!((g - 4.0).abs() < 1e-5);
    /// ```
    #[cfg(feature = "alloc")]
    #[allow(clippy::cast_precision_loss)] // `len as f32` is inherent to the mean
    pub fn geometric_mean(values: &[f32]) -> Result<f32, Error> {
        if values.is_empty() {
            return Err(Error::EmptyInput);
        }
        if let Some(index) = values.iter().position(|&x| x <= 0.0) {
            return Err(Error::NonPositiveInput { index });
        }
        let backend = Backend::detect();
        let mut logs = kernels::alloc_uninit(values.len());
        kernels::dispatch_ln(backend, values, &mut logs);
        let mean = kernels::dispatch_sum(backend, &logs) / values.len() as f32;
        Ok(crate::kernels::exp::exp(mean))
    }

    /// Compute the dot product of two slices (linear algebra, part of the
    /// `stats` family).
    ///
    /// Returns an error if the slices have different lengths.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
    ///
    /// # Example
    /// ```
    /// assert_eq!(
    ///     lanes::stats::f32::dot(&[1.0_f32, 2.0], &[3.0_f32, 4.0]).unwrap(),
    ///     11.0
    /// );
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

    /// Count elements equal to `+0.0` or `-0.0`.
    ///
    /// Returns `0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f32::count_zero(&[0.0_f32, -0.0, 1.0]), 2);
    /// ```
    #[must_use]
    pub fn count_zero(values: &[f32]) -> usize {
        let backend = Backend::detect();
        kernels::dispatch_count_zero(backend, values)
    }

    /// Count NaN elements.
    ///
    /// Returns `0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f32::count_nan(&[f32::NAN, 1.0]), 1);
    /// ```
    #[must_use]
    pub fn count_nan(values: &[f32]) -> usize {
        let backend = Backend::detect();
        kernels::dispatch_count_nan(backend, values)
    }

    /// Count infinite (`+inf`/`-inf`) elements.
    ///
    /// Returns `0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(
    ///     lanes::stats::f32::count_infinite(&[f32::INFINITY, f32::NEG_INFINITY, 1.0]),
    ///     2
    /// );
    /// ```
    #[must_use]
    pub fn count_infinite(values: &[f32]) -> usize {
        let backend = Backend::detect();
        kernels::dispatch_count_infinite(backend, values)
    }
}

pub mod f64 {
    //! Double-precision (`f64`) statistical reductions.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;

    /// Compute the sum of all elements in a slice.
    ///
    /// Returns `0.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::sum(&[1.0_f64, 2.0, 3.0]), 6.0);
    /// ```
    #[must_use]
    pub fn sum(values: &[f64]) -> f64 {
        let backend = Backend::detect();
        kernels::dispatch_sum_f64(backend, values)
    }

    /// Compute the product of all elements in a slice.
    ///
    /// Returns `1.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::prod(&[2.0_f64, 3.0, 4.0]), 24.0);
    /// ```
    #[must_use]
    pub fn prod(values: &[f64]) -> f64 {
        let backend = Backend::detect();
        kernels::dispatch_prod_f64(backend, values)
    }

    /// Find the minimum element in a slice.
    ///
    /// Returns [`None`] if the slice is empty. NaN inputs are ignored
    /// unless every input is NaN (IEEE 754 `minNum` semantics, matching
    /// [`f64::min`]); the result is then NaN. All backends agree.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::min(&[3.0_f64, 1.0, 4.0]), Some(1.0));
    /// ```
    #[must_use]
    pub fn min(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        kernels::dispatch_min_f64(backend, values)
    }

    /// Find the maximum element in a slice.
    ///
    /// Returns [`None`] if the slice is empty. NaN inputs are ignored
    /// unless every input is NaN (IEEE 754 `maxNum` semantics, matching
    /// [`f64::max`]); the result is then NaN. All backends agree.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::max(&[3.0_f64, 1.0, 4.0]), Some(4.0));
    /// ```
    #[must_use]
    pub fn max(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        kernels::dispatch_max_f64(backend, values)
    }

    /// Find the index of the maximum element in a slice.
    ///
    /// Returns [`None`] if the slice is empty. Ties resolve to the first
    /// occurrence. NaN handling follows [`f64::max`] semantics: a NaN is
    /// ignored unless every element is NaN (in which case the first index
    /// wins).
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::argmax(&[3.0_f64, 1.0, 4.0]), Some(2));
    /// ```
    #[must_use]
    pub fn argmax(values: &[f64]) -> Option<usize> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        Some(kernels::dispatch_argmax_f64(backend, values).1)
    }

    /// Find the index of the minimum element in a slice.
    ///
    /// Returns [`None`] if the slice is empty. Ties resolve to the first
    /// occurrence. NaN handling follows [`f64::min`] semantics: a NaN is
    /// ignored unless every element is NaN (in which case the first index
    /// wins).
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::argmin(&[3.0_f64, 1.0, 4.0]), Some(1));
    /// ```
    #[must_use]
    pub fn argmin(values: &[f64]) -> Option<usize> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        Some(kernels::dispatch_argmin_f64(backend, values).1)
    }

    /// Compute the sum of squares of all elements in a slice.
    ///
    /// Returns `0.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::sum_sq(&[1.0_f64, 2.0, 3.0]), 14.0);
    /// ```
    #[must_use]
    pub fn sum_sq(values: &[f64]) -> f64 {
        let backend = Backend::detect();
        kernels::dispatch_sum_sq_f64(backend, values)
    }

    /// Compute the arithmetic mean of a slice.
    ///
    /// Returns [`None`] if the slice is empty.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::mean(&[1.0_f64, 2.0, 3.0]), Some(2.0));
    /// ```
    #[allow(clippy::cast_precision_loss)] // `len as f64` is inherent to mean
    #[must_use]
    pub fn mean(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        Some(kernels::dispatch_sum_f64(backend, values) / values.len() as f64)
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
    /// let v = lanes::stats::f64::variance(&[1.0_f64, 2.0, 3.0]).unwrap();
    /// assert!((v - 2.0 / 3.0).abs() < 1e-12);
    /// ```
    #[cfg(feature = "alloc")]
    #[inline]
    pub(crate) fn variance_fused_f64(backend: Backend, values: &[f64], mean: f64) -> f64 {
        kernels::dispatch_variance_fused_f64(backend, values, mean) / values.len() as f64
    }

    /// Compute the (population) variance of a slice.
    ///
    /// Returns [`None`] if the slice is empty. Uses the numerically stable
    /// two-pass form `sum((x-μ)²)/n`. Gated on `alloc`.
    #[cfg(feature = "alloc")]
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn variance(values: &[f64]) -> Option<f64> {
        let n = values.len();
        if n == 0 {
            return None;
        }
        let backend = Backend::detect();
        let mean = kernels::dispatch_sum_f64(backend, values) / n as f64;
        Some(variance_fused_f64(backend, values, mean))
    }

    /// Compute the (population) variance of a slice, writing the result
    /// into `out[0]` (allocation-free variant of [`variance`]).
    ///
    /// `scratch` must have the same length as `values` and is used as the
    /// second-pass workspace (it holds the centered values); reuse it
    /// across calls in hot loops to avoid the heap allocation that
    /// [`variance`] performs. The result is bit-identical to
    /// [`variance`] — same two-pass form, same SIMD kernels.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `scratch.len() != values.len()`
    /// or `out` is empty.
    ///
    /// # Example
    /// ```
    /// let data = [1.0_f64, 2.0, 3.0];
    /// let mut scratch = [0.0_f64; 3];
    /// let mut out = [0.0_f64; 1];
    /// lanes::stats::f64::variance_into(&data, &mut scratch, &mut out).unwrap();
    /// assert!((out[0] - 2.0 / 3.0).abs() < 1e-12);
    /// ```
    #[allow(clippy::cast_precision_loss)] // `len as f64` is inherent to variance
    pub fn variance_into(
        values: &[f64],
        scratch: &mut [f64],
        out: &mut [f64],
    ) -> Result<(), Error> {
        if values.len() != scratch.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: scratch.len(),
            });
        }
        if out.is_empty() {
            return Err(Error::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if values.is_empty() {
            return Ok(()); // crate convention: empty input leaves `out` untouched
        }
        let backend = Backend::detect();
        let mean = kernels::dispatch_sum_f64(backend, values) / values.len() as f64;
        #[cfg(feature = "alloc")]
        {
            let var = variance_fused_f64(backend, values, mean);
            kernels::dispatch_center_f64(backend, values, mean, scratch);
            out[0] = var;
            return Ok(());
        }
        #[allow(unreachable_code)]
        {
            for (c, &x) in scratch.iter_mut().zip(values) {
                *c = x - mean;
            }
            out[0] = kernels::dispatch_sum_sq_f64(backend, scratch) / values.len() as f64;
            Ok(())
        }
    }

    /// Compute the (population) standard deviation of a slice:
    /// `sqrt(variance(x))`.
    ///
    /// Returns [`None`] if the slice is empty. Same numerical properties as
    /// [`variance`](variance).
    ///
    /// Gated on `alloc`: shares variance's two-pass heap buffer.
    ///
    /// # Example
    /// ```
    /// let v = lanes::stats::f64::std_dev(&[1.0_f64, 2.0, 3.0]).unwrap();
    /// assert!((v - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    /// ```
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn std_dev(values: &[f64]) -> Option<f64> {
        variance(values).map(crate::kernels::sqrt::sqrt_f64)
    }

    /// Compute the (population) standard deviation of a slice, writing the
    /// result into `out[0]` (allocation-free variant of [`std_dev`]).
    ///
    /// Same contract as [`variance_into`]: `scratch` must match
    /// `values.len()`, `out` must be non-empty, and the result is
    /// bit-identical to [`std_dev`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `scratch.len() != values.len()`
    /// or `out` is empty.
    ///
    /// # Example
    /// ```
    /// let data = [1.0_f64, 2.0, 3.0];
    /// let mut scratch = [0.0_f64; 3];
    /// let mut out = [0.0_f64; 1];
    /// lanes::stats::f64::std_dev_into(&data, &mut scratch, &mut out).unwrap();
    /// assert!((out[0] - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    /// ```
    #[allow(clippy::cast_precision_loss)] // `len as f64` is inherent to the variance
    pub fn std_dev_into(values: &[f64], scratch: &mut [f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != scratch.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: scratch.len(),
            });
        }
        if out.is_empty() {
            return Err(Error::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if values.is_empty() {
            return Ok(()); // crate convention: empty input leaves `out` untouched
        }
        let backend = Backend::detect();
        let mean = kernels::dispatch_sum_f64(backend, values) / values.len() as f64;
        #[cfg(feature = "alloc")]
        {
            let var = variance_fused_f64(backend, values, mean);
            kernels::dispatch_center_f64(backend, values, mean, scratch);
            out[0] = crate::kernels::sqrt::sqrt_f64(var);
            return Ok(());
        }
        #[allow(unreachable_code)]
        {
            for (c, &x) in scratch.iter_mut().zip(values) {
                *c = x - mean;
            }
            let var = kernels::dispatch_sum_sq_f64(backend, scratch) / values.len() as f64;
            out[0] = crate::kernels::sqrt::sqrt_f64(var);
            Ok(())
        }
    }

    /// Compute the geometric mean of a slice:
    /// `exp(mean(ln(x)))`, the n-th root of the product.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyInput`] for an empty slice, and
    /// [`Error::NonPositiveInput`] (with the offending index) if any value is
    /// ≤ 0 — the geometric mean is only defined over strictly positive reals.
    /// NaN inputs are *not* an error: they propagate to a NaN result, matching
    /// the crate's reduction semantics.
    ///
    /// Gated on `alloc`: uses the vectorized `ln` map + `exp`.
    ///
    /// # Example
    /// ```
    /// let g = lanes::stats::f64::geometric_mean(&[1.0_f64, 4.0, 16.0]).unwrap();
    /// assert!((g - 4.0).abs() < 1e-12);
    /// ```
    #[cfg(feature = "alloc")]
    #[allow(clippy::cast_precision_loss)] // `len as f64` is inherent to the mean
    pub fn geometric_mean(values: &[f64]) -> Result<f64, Error> {
        if values.is_empty() {
            return Err(Error::EmptyInput);
        }
        if let Some(index) = values.iter().position(|&x| x <= 0.0) {
            return Err(Error::NonPositiveInput { index });
        }
        let backend = Backend::detect();
        let mut logs = kernels::alloc_uninit(values.len());
        kernels::dispatch_ln_f64(backend, values, &mut logs);
        let mean = kernels::dispatch_sum_f64(backend, &logs) / values.len() as f64;
        Ok(crate::kernels::exp::exp_f64(mean))
    }

    /// Compute the dot product of two slices (double precision, linear
    /// algebra, part of the `stats` family).
    ///
    /// Returns an error if the slices have different lengths.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
    ///
    /// # Example
    /// ```
    /// assert_eq!(
    ///     lanes::stats::f64::dot(&[1.0_f64, 2.0], &[3.0_f64, 4.0]).unwrap(),
    ///     11.0
    /// );
    /// ```
    pub fn dot(a: &[f64], b: &[f64]) -> Result<f64, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_dot_f64(backend, a, b))
    }

    /// Count elements equal to `+0.0` or `-0.0`.
    ///
    /// Returns `0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::count_zero(&[0.0_f64, -0.0, 1.0]), 2);
    /// ```
    #[must_use]
    pub fn count_zero(values: &[f64]) -> usize {
        let backend = Backend::detect();
        kernels::dispatch_count_zero_f64(backend, values)
    }

    /// Count NaN elements.
    ///
    /// Returns `0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::f64::count_nan(&[f64::NAN, 1.0]), 1);
    /// ```
    #[must_use]
    pub fn count_nan(values: &[f64]) -> usize {
        let backend = Backend::detect();
        kernels::dispatch_count_nan_f64(backend, values)
    }

    /// Count infinite (`+inf`/`-inf`) elements.
    ///
    /// Returns `0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(
    ///     lanes::stats::f64::count_infinite(&[f64::INFINITY, f64::NEG_INFINITY, 1.0]),
    ///     2
    /// );
    /// ```
    #[must_use]
    pub fn count_infinite(values: &[f64]) -> usize {
        let backend = Backend::detect();
        kernels::dispatch_count_infinite_f64(backend, values)
    }
}

pub mod i8 {
    //! 8-bit signed integer reductions with `i64` accumulation.
    //!
    //! The first general integer family: results are exact (no rounding)
    //! and cannot overflow — every intermediate is widened to `i64`,
    //! which holds the full result for any slice that fits in memory.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;

    /// Dot product of two equal-length `i8` slices, accumulated in `i64`.
    ///
    /// Returns `Ok(0)` for empty inputs.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::i8::dot(&[1_i8, -2], &[3_i8, 4]), Ok(-5));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
    pub fn dot(a: &[i8], b: &[i8]) -> Result<i64, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_dot_i8(backend, a, b))
    }

    /// Sum of all elements, accumulated in `i64`.
    ///
    /// Returns `0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::i8::sum(&[1_i8, -2, 3]), 2);
    /// ```
    #[must_use]
    pub fn sum(values: &[i8]) -> i64 {
        let backend = Backend::detect();
        kernels::dispatch_sum_i8(backend, values)
    }

    /// Sum of squares of all elements, accumulated in `i64`.
    ///
    /// Returns `0` for an empty slice. Implemented as `dot(values, values)`
    /// — same exact result, same kernels.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::i8::sum_sq(&[1_i8, -2, 3]), 14);
    /// ```
    #[must_use]
    pub fn sum_sq(values: &[i8]) -> i64 {
        let backend = Backend::detect();
        kernels::dispatch_dot_i8(backend, values, values)
    }

    /// Minimum element, or [`None`] for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::i8::min(&[3_i8, 1, 4]), Some(1));
    /// ```
    #[must_use]
    pub fn min(values: &[i8]) -> Option<i8> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        kernels::dispatch_min_i8(backend, values)
    }

    /// Maximum element, or [`None`] for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::i8::max(&[3_i8, 1, 4]), Some(4));
    /// ```
    #[must_use]
    pub fn max(values: &[i8]) -> Option<i8> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        kernels::dispatch_max_i8(backend, values)
    }

    /// Count of elements equal to zero.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::stats::i8::count_zero(&[0_i8, 1, 0]), 2);
    /// ```
    #[must_use]
    pub fn count_zero(values: &[i8]) -> usize {
        let backend = Backend::detect();
        kernels::dispatch_count_zero_i8(backend, values)
    }
}
