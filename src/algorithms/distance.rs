//! Distance and norm functions over slices.
//!
//! Norms reduce a slice to a single magnitude: `l1_norm`, `l2_norm`,
//! `max_norm`. All are dispatched to the best available SIMD backend.
//!
//! Precision is selected by the submodule: [`f32`] for single-precision,
//! [`f64`] for double-precision.
//!
//! # NaN handling
//!
//! `l1_norm`, `l2_norm`, `squared_distance`, `kl_divergence`, and
//! `js_divergence` propagate NaN — any NaN input yields a NaN result.
//! `max_norm` returns NaN if any input is NaN. See each function's docs.
//!
//! # Precision
//!
//! Reduction order is backend-dependent: results are deterministic *within*
//! a backend but may differ in the last ulp *across* backends.

pub mod f32 {
    //! Single-precision (`f32`) distance and norm functions.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;

    /// Compute the L1 norm (sum of absolute values) of a slice.
    ///
    /// Returns `0.0` for an empty slice. NaN inputs propagate (any NaN
    /// input yields a NaN result).
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
    /// Returns `0.0` for an empty slice. NaN inputs propagate (any NaN
    /// input yields a NaN result).
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

    /// Compute the Kullback–Leibler divergence
    /// `KL(p ‖ q) = Σ pᵢ · ln(pᵢ / qᵢ)`.
    ///
    /// Returns `Ok(0.0)` for two empty slices (same policy as
    /// `dot`/`squared_distance`).
    ///
    /// # Semantics
    ///
    /// Inputs must be valid probability distributions (non-negative, and
    /// the caller owns normalization — this function does **not**
    /// normalize, unlike `scipy.spatial.distance.jensenshannon`).
    /// Non-positive entries follow raw IEEE arithmetic through the natural
    /// log (`ln(0) = -inf`, `ln(x < 0) = NaN`, NaN propagates):
    ///
    /// * `pᵢ = 0`, `qᵢ > 0` contributes `0 · ln(0) = 0 · -inf = NaN`
    ///   (note: scipy's `rel_entr` convention defines this term as 0);
    /// * `pᵢ > 0`, `qᵢ = 0` yields `+inf` (the divergence is unbounded);
    /// * any NaN input propagates to a NaN result.
    ///
    /// Non-negative for valid distributions with matching support, and
    /// **asymmetric**: in general `kl_divergence(p, q) ≠ kl_divergence(q, p)`.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `p.len() != q.len()`.
    ///
    /// # Example
    /// ```
    /// let p = [0.1_f32, 0.9];
    /// let q = [0.2_f32, 0.8];
    /// let d = lanes::distance::f32::kl_divergence(&p, &q).unwrap();
    /// assert!((d - 0.0367).abs() < 1e-3);
    /// ```
    pub fn kl_divergence(p: &[f32], q: &[f32]) -> Result<f32, Error> {
        if p.len() != q.len() {
            return Err(Error::LengthMismatch {
                expected: p.len(),
                actual: q.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_kl_divergence(backend, p, q))
    }

    /// Compute the Jensen–Shannon divergence
    /// `JS(p, q) = (KL(p ‖ m) + KL(q ‖ m)) / 2` with `m = (p + q) / 2`.
    ///
    /// Returns `Ok(0.0)` for two empty slices (same policy as
    /// `dot`/`squared_distance`).
    ///
    /// # Semantics
    ///
    /// This is the Jensen–Shannon **divergence** (range `[0, ln 2]` for
    /// valid distributions), not its square root. The Jensen–Shannon
    /// **distance** — the metric returned by
    /// `scipy.spatial.distance.jensenshannon` — is
    /// `sqrt(js_divergence(p, q))`; note also that scipy normalizes its
    /// inputs first, while this function does **not** (callers own
    /// normalization).
    ///
    /// Symmetric: `js_divergence(p, q) == js_divergence(q, p)` up to
    /// rounding, and `js_divergence(p, p) == 0`. Non-positive entries
    /// follow the same raw IEEE arithmetic as [`kl_divergence`] (a zero in
    /// one input with a positive value in the other contributes
    /// `0 · ln(0) = NaN`; NaN inputs propagate).
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `p.len() != q.len()`.
    ///
    /// # Example
    /// ```
    /// let p = [0.1_f32, 0.9];
    /// let q = [0.2_f32, 0.8];
    /// let d = lanes::distance::f32::js_divergence(&p, &q).unwrap();
    /// assert!((d - 0.00997).abs() < 1e-3);
    /// ```
    pub fn js_divergence(p: &[f32], q: &[f32]) -> Result<f32, Error> {
        if p.len() != q.len() {
            return Err(Error::LengthMismatch {
                expected: p.len(),
                actual: q.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_js_divergence(backend, p, q) * 0.5)
    }
}

pub mod f64 {
    //! Double-precision (`f64`) distance and norm functions.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;

    /// Compute the L1 norm (sum of absolute values) of a slice.
    ///
    /// Returns `0.0` for an empty slice. NaN inputs propagate (any NaN
    /// input yields a NaN result).
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
    /// Returns `0.0` for an empty slice. NaN inputs propagate (any NaN
    /// input yields a NaN result).
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

    /// Compute the Kullback–Leibler divergence
    /// `KL(p ‖ q) = Σ pᵢ · ln(pᵢ / qᵢ)`.
    ///
    /// `f64` twin of [`super::f32::kl_divergence`] — same semantics: no input
    /// normalization, raw IEEE zero/NaN behavior, `Ok(0.0)` for two empty
    /// slices, asymmetric in `(p, q)`.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `p.len() != q.len()`.
    ///
    /// # Example
    /// ```
    /// let p = [0.1_f64, 0.9];
    /// let q = [0.2_f64, 0.8];
    /// let d = lanes::distance::f64::kl_divergence(&p, &q).unwrap();
    /// assert!((d - 0.0367).abs() < 1e-3);
    /// ```
    pub fn kl_divergence(p: &[f64], q: &[f64]) -> Result<f64, Error> {
        if p.len() != q.len() {
            return Err(Error::LengthMismatch {
                expected: p.len(),
                actual: q.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_kl_divergence_f64(backend, p, q))
    }

    /// Compute the Jensen–Shannon divergence
    /// `JS(p, q) = (KL(p ‖ m) + KL(q ‖ m)) / 2` with `m = (p + q) / 2`.
    ///
    /// `f64` twin of [`super::f32::js_divergence`] — same semantics: returns the
    /// divergence (not the sqrt-distance), no input normalization,
    /// symmetric, `Ok(0.0)` for two empty slices.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `p.len() != q.len()`.
    ///
    /// # Example
    /// ```
    /// let p = [0.1_f64, 0.9];
    /// let q = [0.2_f64, 0.8];
    /// let d = lanes::distance::f64::js_divergence(&p, &q).unwrap();
    /// assert!((d - 0.00997).abs() < 1e-3);
    /// ```
    pub fn js_divergence(p: &[f64], q: &[f64]) -> Result<f64, Error> {
        if p.len() != q.len() {
            return Err(Error::LengthMismatch {
                expected: p.len(),
                actual: q.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_js_divergence_f64(backend, p, q) * 0.5)
    }
}

pub mod i8 {
    //! 8-bit signed integer distance and norm functions with `i64`
    //! accumulation.
    //!
    //! Results are exact (no rounding) and cannot overflow — every
    //! intermediate is widened before the operation that could overflow
    //! (notably `|i8::MIN| = 128`, which does not fit in `i8`).

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;

    /// L1 norm (sum of absolute values), accumulated in `i64`.
    ///
    /// Returns `0` for an empty slice. `|i8::MIN| = 128` is handled
    /// exactly.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::distance::i8::l1_norm(&[-3_i8, 4]), 7);
    /// assert_eq!(lanes::distance::i8::l1_norm(&[i8::MIN]), 128);
    /// ```
    #[must_use]
    pub fn l1_norm(values: &[i8]) -> i64 {
        let backend = Backend::detect();
        kernels::dispatch_l1_norm_i8(backend, values)
    }

    /// Max norm (maximum absolute value), or [`None`] for an empty slice.
    ///
    /// Returns `u8` — the minimal exact type, since `|i8::MIN| = 128`
    /// does not fit in `i8`.
    ///
    /// Single-pass `max(|v|)` via a dedicated `max_abs_i8` kernel (one SIMD
    /// scan instead of two). Result is `u8` to hold `|i8::MIN| = 128`.
    ///
    /// # Example
    /// ```
    /// assert_eq!(lanes::distance::i8::max_norm(&[-3_i8, 4]), Some(4));
    /// assert_eq!(lanes::distance::i8::max_norm(&[i8::MIN]), Some(128));
    /// ```
    #[must_use]
    pub fn max_norm(values: &[i8]) -> Option<u8> {
        if values.is_empty() {
            return None;
        }
        let backend = Backend::detect();
        kernels::dispatch_max_abs_i8(backend, values)
    }

    /// Squared Euclidean distance `sum((a[i] - b[i])²)`, accumulated in
    /// `i64`.
    ///
    /// Returns `Ok(0)` for two empty slices (same policy as the float
    /// variants). Each difference fits in `i16` and each square in `i32`,
    /// so the result is exact for any slice length.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
    ///
    /// # Example
    /// ```
    /// let d = lanes::distance::i8::squared_distance(&[1_i8, 2], &[4_i8, 6]);
    /// assert_eq!(d, Ok(25));
    /// ```
    pub fn squared_distance(a: &[i8], b: &[i8]) -> Result<i64, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        let backend = Backend::detect();
        Ok(kernels::dispatch_squared_distance_i8(backend, a, b))
    }
}
