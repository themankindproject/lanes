//! Distance and norm functions over slices.
//!
//! Norms reduce a slice to a single magnitude: `l1_norm`, `l2_norm`,
//! `max_norm`. All are dispatched to the best available SIMD backend.
//!
//! Precision is selected by the submodule: [`f32`] for single-precision,
//! [`f64`] for double-precision.

pub mod f32 {
    //! Single-precision (`f32`) distance and norm functions.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;

    /// Compute the L1 norm (sum of absolute values) of a slice.
    ///
    /// Returns `0.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::distance::f32::l1_norm(&[-3.0_f32, 4.0]), 7.0);
    /// ```
    #[must_use]
    pub fn l1_norm(values: &[f32]) -> f32 {
        let backend = Backend::detect();
        kernels::dispatch_l1_norm(backend, values)
    }

    /// Compute the L2 (Euclidean) norm of a slice.
    ///
    /// Returns `0.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// let n = lanes::distance::f32::l2_norm(&[3.0_f32, 4.0]);
    /// assert!((n - 5.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn l2_norm(values: &[f32]) -> f32 {
        let backend = Backend::detect();
        kernels::sqrt::sqrt(kernels::dispatch_sum_sq(backend, values))
    }

    /// Compute the maximum absolute value (max norm) of a slice.
    ///
    /// Returns [`None`] if the slice is empty. If any input is NaN the
    /// result is NaN (matching the scalar `total_cmp` reference, where
    /// NaN sorts above all values). All backends agree.
    ///
    /// # Example
    /// ```
    /// assert_eq!(
    ///     lanes::distance::f32::max_norm(&[-3.0_f32, 4.0, -9.0]),
    ///     Some(9.0)
    /// );
    /// ```
    #[must_use]
    pub fn max_norm(values: &[f32]) -> Option<f32> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        kernels::dispatch_max_norm(backend, values)
    }

    /// Compute the squared Euclidean distance between two slices:
    /// `sum((a[i] - b[i])²)`.
    ///
    /// Returns `Ok(0.0)` for two empty slices (same policy as
    /// `dot`/`cosine_similarity`).
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
    ///
    /// # Example
    /// ```
    /// let d = lanes::distance::f32::squared_distance(&[1.0_f32, 2.0], &[4.0_f32, 6.0]);
    /// assert_eq!(d, Ok(25.0));
    /// ```
    pub fn squared_distance(a: &[f32], b: &[f32]) -> Result<f32, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_squared_distance(backend, a, b))
    }
}

pub mod f64 {
    //! Double-precision (`f64`) distance and norm functions.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;

    /// Compute the L1 norm (sum of absolute values) of a slice.
    ///
    /// Returns `0.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::distance::f64::l1_norm(&[-3.0_f64, 4.0]), 7.0);
    /// ```
    #[must_use]
    pub fn l1_norm(values: &[f64]) -> f64 {
        let backend = Backend::detect();
        kernels::dispatch_l1_norm_f64(backend, values)
    }

    /// Compute the L2 (Euclidean) norm of a slice.
    ///
    /// Returns `0.0` for an empty slice.
    ///
    /// # Example
    /// ```
    /// let n = lanes::distance::f64::l2_norm(&[3.0_f64, 4.0]);
    /// assert!((n - 5.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn l2_norm(values: &[f64]) -> f64 {
        let backend = Backend::detect();
        kernels::sqrt::sqrt_f64(kernels::dispatch_sum_sq_f64(backend, values))
    }

    /// Compute the maximum absolute value (max norm) of a slice.
    ///
    /// Returns [`None`] if the slice is empty. If any input is NaN the
    /// result is NaN (matching the scalar `total_cmp` reference, where
    /// NaN sorts above all values). All backends agree.
    ///
    /// # Example
    /// ```
    /// assert_eq!(
    ///     lanes::distance::f64::max_norm(&[-3.0_f64, 4.0, -9.0]),
    ///     Some(9.0)
    /// );
    /// ```
    #[must_use]
    pub fn max_norm(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        kernels::dispatch_max_norm_f64(backend, values)
    }

    /// Compute the squared Euclidean distance between two slices:
    /// `sum((a[i] - b[i])²)`.
    ///
    /// Returns `Ok(0.0)` for two empty slices (same policy as
    /// `dot`/`cosine_similarity`).
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
    ///
    /// # Example
    /// ```
    /// let d = lanes::distance::f64::squared_distance(&[1.0_f64, 2.0], &[4.0_f64, 6.0]);
    /// assert_eq!(d, Ok(25.0));
    /// ```
    pub fn squared_distance(a: &[f64], b: &[f64]) -> Result<f64, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_squared_distance_f64(backend, a, b))
    }
}
