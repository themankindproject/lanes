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
    /// Returns [`None`] if the slice is empty.
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
}

pub mod f64 {
    //! Double-precision (`f64`) distance and norm functions.

    use crate::dispatch::Backend;
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
    /// Returns [`None`] if the slice is empty.
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
}
