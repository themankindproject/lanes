//! Machine-learning kernels built on the `lanes` core.
//!
//! These functions compose the core reductions (`sum`, `max`, `dot`) into
//! higher-level ML ops (softmax, activations, `rms_norm`,
//! `cosine_similarity`).
//! Each is dispatched to the best SIMD backend at runtime, exactly like the
//! core functions.
//!
//! Precision is selected by the submodule: [`f32`] for single-precision,
//! [`f64`] for double-precision.

pub mod f32 {
    //! Single-precision (`f32`) ML kernels.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;
    use alloc::vec::Vec;

    /// Numerically-stable softmax over a slice.
    ///
    /// Computes `softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
    /// The max subtraction prevents overflow for large inputs. Returns a new
    /// `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::softmax(&[1.0_f32, 2.0, 3.0]);
    /// let s: f32 = v.iter().sum();
    /// assert!((s - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn softmax(values: &[f32]) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_softmax(backend, values, &mut out);
        out
    }

    /// Sigmoid activation over a slice.
    ///
    /// Computes `sigmoid(x)_i = 1 / (1 + exp(-x_i))` elementwise. Outputs are
    /// in `(0, 1)`, monotone, with `sigmoid(0) = 0.5`. Large positive inputs
    /// saturate to 1.0, large negative to 0.0 (via the exp saturation — no
    /// overflow). Returns a new `Vec` of the same length; an empty slice
    /// yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::sigmoid(&[0.0_f32, 1.0, -1.0]);
    /// assert!((v[0] - 0.5).abs() < 1e-6);
    /// assert!((v[1] - 0.731_058_6).abs() < 1e-6);
    /// assert!((v[2] - 0.268_941_4).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn sigmoid(values: &[f32]) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_sigmoid(backend, values, &mut out);
        out
    }

    /// `SiLU` (`Swish`) activation over a slice.
    ///
    /// Computes `silu(x)_i = x_i / (1 + exp(-x_i))` elementwise — the smooth
    /// LLM activation (Llama, Qwen, etc.). Saturates to `x` for large positive
    /// x, to 0 for large negative x; minimum ≈ −0.278 at x ≈ −1.28. Returns a
    /// new `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::silu(&[0.0_f32, 1.0, -1.0]);
    /// assert!(v[0].abs() < 1e-6);
    /// assert!((v[1] - 0.731_058_6).abs() < 1e-6);
    /// assert!((v[2] + 0.268_941_4).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn silu(values: &[f32]) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_silu(backend, values, &mut out);
        out
    }

    /// GELU activation over a slice.
    ///
    /// Computes the tanh approximation
    /// `gelu(x)_i = 0.5·x_i·(1 + tanh(√(2/π)·(x_i + 0.044715·x_i³)))` — the
    /// production LLM activation (GPT-2 etc.), accurate to ~1e-3 of exact
    /// GELU. `tanh` is derived from `exp`, so no extra transcendental. Returns
    /// a new `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::gelu(&[0.0_f32, 1.0, -1.0]);
    /// assert!(v[0].abs() < 1e-6);
    /// assert!((v[1] - 0.84119).abs() < 2e-4);
    /// assert!((v[2] + 0.15881).abs() < 2e-4);
    /// ```
    #[must_use]
    pub fn gelu(values: &[f32]) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_gelu(backend, values, &mut out);
        out
    }

    /// `ReLU` activation over a slice.
    ///
    /// Computes `relu(x)_i = max(x_i, 0)` elementwise. Returns a new `Vec` of
    /// the same length; an empty slice yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::relu(&[-3.0_f32, 0.0, 5.0]);
    /// assert_eq!(v, [0.0, 0.0, 5.0]);
    /// ```
    #[must_use]
    pub fn relu(values: &[f32]) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_relu(backend, values, &mut out);
        out
    }

    /// RMS norm over a slice: `x_i / sqrt(mean(x²) + eps)`.
    ///
    /// The standard LLM normalization (Llama, Qwen). `eps` guards against
    /// division by zero for all-zero input; a typical value is `1e-5`.
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::rms_norm(&[3.0_f32, 4.0], 0.0);
    /// let r = 12.5_f32.sqrt(); // sqrt(mean(9, 16))
    /// assert!((v[0] - 3.0 / r).abs() < 1e-6);
    /// assert!((v[1] - 4.0 / r).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn rms_norm(values: &[f32], eps: f32) -> Vec<f32> {
        let mut out = alloc::vec![0.0_f32; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_rms_norm(backend, values, eps, &mut out);
        out
    }

    /// Cosine similarity between two equal-length slices:
    /// `dot(a, b) / (|a|·|b|)`.
    ///
    /// Returns an error if the slices have different lengths. Returns `None`
    /// if either vector has zero length (so the angle is undefined). The
    /// result is in `[-1, 1]` up to rounding.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
    ///
    /// # Example
    /// ```
    /// let s = lanes::ml::f32::cosine_similarity(&[1.0_f32, 0.0], &[1.0_f32, 0.0]);
    /// assert_eq!(s, Ok(Some(1.0)));
    /// ```
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<Option<f32>, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        if a.is_empty() {
            return Ok(None);
        }
        let backend = Backend::detect();
        let dot = kernels::dispatch_dot(backend, a, b);
        let na = kernels::sqrt::sqrt(kernels::dispatch_sum_sq(backend, a));
        let nb = kernels::sqrt::sqrt(kernels::dispatch_sum_sq(backend, b));
        if na == 0.0 || nb == 0.0 {
            return Ok(None);
        }
        Ok(Some(dot / (na * nb)))
    }
}

pub mod f64 {
    //! Double-precision (`f64`) ML kernels.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;
    use alloc::vec::Vec;

    /// Numerically-stable softmax over a slice.
    ///
    /// Computes `softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
    /// The max subtraction prevents overflow for large inputs. Returns a new
    /// `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::softmax(&[1.0_f64, 2.0, 3.0]);
    /// let s: f64 = v.iter().sum();
    /// assert!((s - 1.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn softmax(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_softmax_f64(backend, values, &mut out);
        out
    }

    /// Sigmoid activation over a slice.
    ///
    /// Computes `sigmoid(x)_i = 1 / (1 + exp(-x_i))` elementwise. Outputs are
    /// in `(0, 1)`, monotone, with `sigmoid(0) = 0.5`. Large positive inputs
    /// saturate to 1.0, large negative to 0.0 (via the exp saturation — no
    /// overflow). Returns a new `Vec` of the same length; an empty slice
    /// yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::sigmoid(&[0.0_f64, 1.0, -1.0]);
    /// assert!((v[0] - 0.5).abs() < 1e-12);
    /// assert!((v[1] - 0.731_058_578_630_092_5).abs() < 1e-12);
    /// assert!((v[2] - 0.268_941_421_369_907_5).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn sigmoid(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_sigmoid_f64(backend, values, &mut out);
        out
    }

    /// `SiLU` (`Swish`) activation over a slice.
    ///
    /// Computes `silu(x)_i = x_i / (1 + exp(-x_i))` elementwise — the smooth
    /// LLM activation (Llama, Qwen, etc.). Saturates to `x` for large positive
    /// x, to 0 for large negative x; minimum ≈ −0.278 at x ≈ −1.28. Returns a
    /// new `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::silu(&[0.0_f64, 1.0, -1.0]);
    /// assert!(v[0].abs() < 1e-12);
    /// assert!((v[1] - 0.731_058_578_630_092_5).abs() < 1e-12);
    /// assert!((v[2] + 0.268_941_421_369_907_5).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn silu(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_silu_f64(backend, values, &mut out);
        out
    }

    /// GELU activation over a slice.
    ///
    /// Computes the tanh approximation
    /// `gelu(x)_i = 0.5·x_i·(1 + tanh(√(2/π)·(x_i + 0.044715·x_i³)))` — the
    /// production LLM activation (GPT-2 etc.). `tanh` is derived from `exp`,
    /// so no extra transcendental. Returns a new `Vec` of the same length; an
    /// empty slice yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::gelu(&[0.0_f64, 1.0, -1.0]);
    /// assert!(v[0].abs() < 1e-12);
    /// assert!((v[1] - 0.841_192_029_433_373).abs() < 2e-4);
    /// assert!((v[2] + 0.158_807_970_566_627).abs() < 2e-4);
    /// ```
    #[must_use]
    pub fn gelu(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_gelu_f64(backend, values, &mut out);
        out
    }

    /// `ReLU` activation over a slice.
    ///
    /// Computes `relu(x)_i = max(x_i, 0)` elementwise. Returns a new `Vec` of
    /// the same length; an empty slice yields an empty `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::relu(&[-3.0_f64, 0.0, 5.0]);
    /// assert_eq!(v, [0.0, 0.0, 5.0]);
    /// ```
    #[must_use]
    pub fn relu(values: &[f64]) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_relu_f64(backend, values, &mut out);
        out
    }

    /// RMS norm over a slice: `x_i / sqrt(mean(x²) + eps)`.
    ///
    /// The standard LLM normalization (Llama, Qwen). `eps` guards against
    /// division by zero for all-zero input; a typical value is `1e-5`.
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::rms_norm(&[3.0_f64, 4.0], 0.0);
    /// let r = 12.5_f64.sqrt(); // sqrt(mean(9, 16))
    /// assert!((v[0] - 3.0 / r).abs() < 1e-12);
    /// assert!((v[1] - 4.0 / r).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn rms_norm(values: &[f64], eps: f64) -> Vec<f64> {
        let mut out = alloc::vec![0.0_f64; values.len()];
        let backend = Backend::detect();
        kernels::dispatch_rms_norm_f64(backend, values, eps, &mut out);
        out
    }

    /// Cosine similarity between two equal-length slices:
    /// `dot(a, b) / (|a|·|b|)`.
    ///
    /// Returns an error if the slices have different lengths. Returns `None`
    /// if either vector has zero length (so the angle is undefined). The
    /// result is in `[-1, 1]` up to rounding.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
    ///
    /// # Example
    /// ```
    /// let s = lanes::ml::f64::cosine_similarity(&[1.0_f64, 0.0], &[1.0_f64, 0.0]);
    /// assert_eq!(s, Ok(Some(1.0)));
    /// ```
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<Option<f64>, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        if a.is_empty() {
            return Ok(None);
        }
        let backend = Backend::detect();
        let dot = kernels::dispatch_dot_f64(backend, a, b);
        let na = kernels::sqrt::sqrt_f64(kernels::dispatch_sum_sq_f64(backend, a));
        let nb = kernels::sqrt::sqrt_f64(kernels::dispatch_sum_sq_f64(backend, b));
        if na == 0.0 || nb == 0.0 {
            return Ok(None);
        }
        Ok(Some(dot / (na * nb)))
    }
}
