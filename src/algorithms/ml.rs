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

    /// Numerically-stable softmax, written into `out` (allocation-free variant of [`softmax`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`softmax`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f32, 2.0, 3.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::softmax_into(&v, &mut out).unwrap();
    /// let s: f32 = out.iter().sum();
    /// assert!((s - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn softmax_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_softmax(backend, values, out);
        Ok(())
    }

    /// Numerically-stable softmax over a slice.
    ///
    /// Computes `softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
    /// NaN propagates. `softmax([+inf, x...]) = [1, 0, ...]`.
    /// The max subtraction prevents overflow for large inputs. Returns a new
    /// `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`softmax_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::softmax(&[1.0_f32, 2.0, 3.0]);
    /// let s: f32 = v.iter().sum();
    /// assert!((s - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn softmax(values: &[f32]) -> Vec<f32> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = softmax_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Sigmoid activation, written into `out` (allocation-free variant of [`sigmoid`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`sigmoid`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f32, 1.0, -1.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::sigmoid_into(&v, &mut out).unwrap();
    /// assert!((out[0] - 0.5).abs() < 1e-6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn sigmoid_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_sigmoid(backend, values, out);
        Ok(())
    }

    /// Sigmoid activation over a slice.
    ///
    /// Computes `sigmoid(x)_i = 1 / (1 + exp(-x_i))` elementwise. Outputs are
    /// in `(0, 1)`, monotone, with `sigmoid(0) = 0.5`. Large positive inputs
    /// saturate to 1.0, large negative to 0.0 (via the exp saturation — no
    /// overflow). Returns a new `Vec` of the same length; an empty slice
    /// yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`sigmoid_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        let mut out = kernels::alloc_uninit(values.len());
        let _ = sigmoid_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Softplus activation (`ln(1 + e^x)`), written into `out` (allocation-free variant of [`softplus`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`softplus`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f32, 1.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::softplus_into(&v, &mut out).unwrap();
    /// assert!((out[0] - std::f32::consts::LN_2).abs() < 1e-6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn softplus_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_softplus(backend, values, out);
        Ok(())
    }

    /// Softplus activation over a slice: `ln(1 + e^x)` elementwise.
    ///
    /// Computed with the overflow-free form `max(x, 0) + ln1p(e^-|x|)` so
    /// large `x` cannot overflow `exp` and the result is exact to ~1 ulp
    /// across the full range (as `x → ∞` it approaches `x`, as `x → −∞` it
    /// approaches 0). Returns a new `Vec` of the same length; an empty
    /// slice yields an empty `Vec`.
    ///
    /// Reference: the `log1pexp` formulation and the `ln1p` identity
    /// `ln(1+z) = z·ln(1+z)/((1+z)−1)` from musl libc's `s_log1pf.c`
    /// (<https://musl.libc.org>) and fdlibm's `s_log1p.c`
    /// (<https://www.netlib.org/fdlibm>); see also the CUDA softplus
    /// optimization guide (<https://www.rightnowai.co/guides/cuda-operations/softplus>).
    ///
    /// Convenience wrapper around [`softplus_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::softplus(&[0.0_f32, 1.0, -1.0, 100.0]);
    /// assert!((v[0] - std::f32::consts::LN_2).abs() < 1e-6);
    /// assert!((v[1] - 1.313_261_7).abs() < 1e-6);
    /// assert!((v[2] - 0.313_261_7).abs() < 1e-6);
    /// assert!((v[3] - 100.0).abs() < 1e-4);
    /// ```
    #[must_use]
    pub fn softplus(values: &[f32]) -> Vec<f32> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = softplus_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Log-softmax over a slice: `x_i − logsumexp(x)` elementwise, written
    /// into `out`.
    ///
    /// Numerically stable via the max-shift: `x_i − max(x) − ln(Σ_j exp(x_j −
    /// max(x)))`. This is the primitive `PyTorch`'s [`nn.LogSoftmax`] computes
    /// (paired with [`nn.NLLLoss`] it forms [`nn.CrossEntropyLoss`]).
    /// Allocation-free: `out` must have the same length as `values` (a length
    /// mismatch returns [`Error::LengthMismatch`]); reuse it across calls to
    /// avoid per-call allocation in hot loops. An empty slice leaves `out`
    /// untouched.
    ///
    /// Reference: the max-subtraction trick and the fused
    /// log-softmax/NLL/cross-entropy decomposition as documented in
    /// `PyTorch`'s [`nn.LogSoftmax`]
    /// (<https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html>).
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f32, 2.0, 3.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::log_softmax_into(&v, &mut out).unwrap();
    /// // exp(log_softmax) sums to 1 — it IS softmax, logged.
    /// let s: f32 = out.iter().map(|x| x.exp()).sum();
    /// assert!((s - 1.0).abs() < 1e-6);
    /// assert!(out[2] > out[1] && out[1] > out[0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn log_softmax_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_log_softmax(backend, values, out);
        Ok(())
    }

    /// Log-softmax over a slice, returning a new `Vec`. Convenience wrapper
    /// around [`log_softmax_into`]; prefer the `_into` form in hot loops.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::log_softmax(&[1.0_f32, 2.0, 3.0]);
    /// let s: f32 = v.iter().map(|x| x.exp()).sum();
    /// assert!((s - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn log_softmax(values: &[f32]) -> Vec<f32> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = log_softmax_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// `SiLU` (`Swish`) activation, written into `out` (allocation-free variant of [`silu`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`silu`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f32, 1.0, -1.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::silu_into(&v, &mut out).unwrap();
    /// assert!(out[0].abs() < 1e-6);
    /// assert!((out[1] - 0.731_058_6).abs() < 1e-6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn silu_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_silu(backend, values, out);
        Ok(())
    }

    /// `SiLU` (`Swish`) activation over a slice.
    ///
    /// Computes `silu(x)_i = x_i / (1 + exp(-x_i))` elementwise — the smooth
    /// LLM activation (Llama, Qwen, etc.). Saturates to `x` for large positive
    /// x, to 0 for large negative x; minimum ≈ −0.278 at x ≈ −1.28. Returns a
    /// new `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`silu_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        let mut out = kernels::alloc_uninit(values.len());
        let _ = silu_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// GELU activation (tanh approximation), written into `out` (allocation-free variant of [`gelu`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`gelu`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f32, 1.0, -1.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::gelu_into(&v, &mut out).unwrap();
    /// assert!(out[0].abs() < 1e-6);
    /// assert!((out[1] - 0.84119).abs() < 2e-4);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn gelu_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_gelu(backend, values, out);
        Ok(())
    }

    /// GELU activation over a slice.
    ///
    /// Computes the tanh approximation
    /// `gelu(x)_i = 0.5·x_i·(1 + tanh(√(2/π)·(x_i + 0.044715·x_i³)))` — the
    /// production LLM activation (GPT-2 etc.), accurate to ~1e-3 of exact
    /// GELU. `tanh` is derived from `exp`, so no extra transcendental. Returns
    /// a new `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`gelu_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        let mut out = kernels::alloc_uninit(values.len());
        let _ = gelu_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// `ReLU` activation, written into `out` (allocation-free variant of [`relu`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`relu`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [-3.0_f32, 0.0, 5.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::relu_into(&v, &mut out).unwrap();
    /// assert_eq!(out, [0.0, 0.0, 5.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn relu_into(values: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_relu(backend, values, out);
        Ok(())
    }

    /// `ReLU` activation over a slice.
    ///
    /// Computes `relu(x)_i = max(x_i, 0)` elementwise. Returns a new `Vec` of
    /// the same length; an empty slice yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`relu_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::relu(&[-3.0_f32, 0.0, 5.0]);
    /// assert_eq!(v, [0.0, 0.0, 5.0]);
    /// ```
    #[must_use]
    pub fn relu(values: &[f32]) -> Vec<f32> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = relu_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// RMS norm (`x_i / sqrt(mean(x²) + eps)`), written into `out` (allocation-free variant of [`rms_norm`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`rms_norm`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [3.0_f32, 4.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::rms_norm_into(&v, 0.0, &mut out).unwrap();
    /// let r = 12.5_f32.sqrt(); // sqrt(mean(9, 16))
    /// assert!((out[0] - 3.0 / r).abs() < 1e-6);
    /// assert!((out[1] - 4.0 / r).abs() < 1e-6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn rms_norm_into(values: &[f32], eps: f32, out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_rms_norm(backend, values, eps, out);
        Ok(())
    }

    /// RMS norm over a slice: `x_i / sqrt(mean(x²) + eps)`.
    ///
    /// The standard LLM normalization (Llama, Qwen). `eps` guards against
    /// division by zero for all-zero input; a typical value is `1e-5`.
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`.
    ///
    /// Convenience wrapper around [`rms_norm_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        let mut out = kernels::alloc_uninit(values.len());
        let _ = rms_norm_into(values, eps, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Cosine similarity between two equal-length slices:
    /// `dot(a, b) / (|a|·|b|)`.
    ///
    /// Returns an error if the slices have different lengths or are empty
    /// (the angle between empty vectors is undefined). If either vector has
    /// zero norm the result is `0.0` (a zero vector has no direction, so it
    /// shares none with the other — the same convention as scikit-learn).
    /// Otherwise the result is in `[-1, 1]` up to rounding.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`, and
    /// [`Error::EmptyInput`] if both slices are empty.
    ///
    /// # Example
    /// ```
    /// let s = lanes::ml::f32::cosine_similarity(&[1.0_f32, 0.0], &[1.0_f32, 0.0]);
    /// assert_eq!(s, Ok(1.0));
    /// ```
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        if a.is_empty() {
            return Err(Error::EmptyInput);
        }
        // Cache dispatch once: same backend for dot + both norms (OnceLock load, but 3x less).
        let backend = Backend::detect();
        let dot = kernels::dispatch_dot(backend, a, b);
        let na = kernels::sqrt::sqrt(kernels::dispatch_sum_sq(backend, a));
        let nb = kernels::sqrt::sqrt(kernels::dispatch_sum_sq(backend, b));
        if na == 0.0 || nb == 0.0 {
            return Ok(0.0);
        }
        Ok(dot / (na * nb))
    }

    /// Numerically-stable log-sum-exp: `ln(sum_i exp(x_i))`, the denominator
    /// of the log-softmax and the log-normalizer of cross-entropy losses.
    ///
    /// Computes `max(x) + ln(sum_i exp(x_i - max(x)))`. The max subtraction
    /// prevents overflow for large inputs. An empty slice yields
    /// `-infinity`. Returns a scalar `f32`.
    ///
    /// # Example
    /// ```
    /// let s = lanes::ml::f32::logsumexp(&[1.0_f32, 2.0, 3.0]);
    /// assert!((s - 3.407_606).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn logsumexp(values: &[f32]) -> f32 {
        let backend = Backend::detect();
        kernels::dispatch_logsumexp(backend, values)
    }

    /// Layer normalization: `(x_i - mean(x)) / sqrt(variance(x) + eps)`,
    /// written into `out`.
    ///
    /// The standard pre-activation norm (complement to [`rms_norm`], which
    /// drops the mean). Allocation-free: `out` must have the same length as
    /// `values` (a length mismatch returns [`Error::LengthMismatch`]); reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. NaNs propagate.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f32, 2.0, 3.0];
    /// let mut out = vec![0.0_f32; v.len()];
    /// lanes::ml::f32::layer_norm_into(&v, 1e-5, &mut out).unwrap();
    /// let m: f32 = out.iter().sum::<f32>() / 3.0;
    /// assert!(m.abs() < 1e-6);
    /// // sum of squares after norm is n·var/(var+eps) ≈ 3 for this input.
    /// let s: f32 = out.iter().map(|x| x * x).sum();
    /// assert!((s - 3.0).abs() < 1e-3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    #[allow(clippy::cast_precision_loss)] // `len as f32` is inherent to the mean
    pub fn layer_norm_into(values: &[f32], eps: f32, out: &mut [f32]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_layer_norm(backend, values, eps, out);
        Ok(())
    }

    /// Layer normalization, returning a new `Vec`. Convenience wrapper
    /// around [`layer_norm_into`]; prefer the `_into` form in hot loops.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f32::layer_norm(&[1.0_f32, 2.0, 3.0], 1e-5);
    /// let m: f32 = v.iter().sum::<f32>() / 3.0;
    /// assert!(m.abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn layer_norm(values: &[f32], eps: f32) -> Vec<f32> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = layer_norm_into(values, eps, &mut out); // infallible: out.len() == values.len() by construction
        out
    }
}

pub mod f64 {
    //! Double-precision (`f64`) ML kernels.

    use crate::dispatch::Backend;
    use crate::error::Error;
    use crate::kernels;
    use alloc::vec::Vec;

    /// Numerically-stable softmax, written into `out` (allocation-free variant of [`softmax`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`softmax`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f64, 2.0, 3.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::softmax_into(&v, &mut out).unwrap();
    /// let s: f64 = out.iter().sum();
    /// assert!((s - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn softmax_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_softmax_f64(backend, values, out);
        Ok(())
    }

    /// Numerically-stable softmax over a slice.
    ///
    /// Computes `softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
    /// NaN propagates. `softmax([+inf, x...]) = [1, 0, ...]`.
    /// The max subtraction prevents overflow for large inputs. Returns a new
    /// `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`softmax_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::softmax(&[1.0_f64, 2.0, 3.0]);
    /// let s: f64 = v.iter().sum();
    /// assert!((s - 1.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn softmax(values: &[f64]) -> Vec<f64> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = softmax_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Sigmoid activation, written into `out` (allocation-free variant of [`sigmoid`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`sigmoid`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f64, 1.0, -1.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::sigmoid_into(&v, &mut out).unwrap();
    /// assert!((out[0] - 0.5).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn sigmoid_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_sigmoid_f64(backend, values, out);
        Ok(())
    }

    /// Sigmoid activation over a slice.
    ///
    /// Computes `sigmoid(x)_i = 1 / (1 + exp(-x_i))` elementwise. Outputs are
    /// in `(0, 1)`, monotone, with `sigmoid(0) = 0.5`. Large positive inputs
    /// saturate to 1.0, large negative to 0.0 (via the exp saturation — no
    /// overflow). Returns a new `Vec` of the same length; an empty slice
    /// yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`sigmoid_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        let mut out = kernels::alloc_uninit(values.len());
        let _ = sigmoid_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Softplus activation (`ln(1 + e^x)`), written into `out` (allocation-free variant of [`softplus`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`softplus`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f64, 1.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::softplus_into(&v, &mut out).unwrap();
    /// assert!((out[0] - std::f64::consts::LN_2).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn softplus_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_softplus_f64(backend, values, out);
        Ok(())
    }

    /// Softplus activation over a slice: `ln(1 + e^x)` elementwise.
    ///
    /// Computed with the overflow-free form `max(x, 0) + ln1p(e^-|x|)` so
    /// large `x` cannot overflow `exp` and the result is exact to ~1 ulp
    /// across the full range. Returns a new `Vec` of the same length; an
    /// empty slice yields an empty `Vec`.
    ///
    /// Reference: the `log1pexp` formulation and the `ln1p` identity from
    /// musl libc's `s_log1p.c` (<https://musl.libc.org>) and fdlibm's
    /// `s_log1p.c` (<https://www.netlib.org/fdlibm>); see also the CUDA
    /// softplus optimization guide
    /// (<https://www.rightnowai.co/guides/cuda-operations/softplus>).
    ///
    /// Convenience wrapper around [`softplus_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::softplus(&[0.0_f64, 1.0, -1.0, 1000.0]);
    /// assert!((v[0] - std::f64::consts::LN_2).abs() < 1e-12);
    /// assert!((v[1] - 1.313_261_687_518_222_8).abs() < 1e-12);
    /// assert!((v[2] - 0.313_261_687_518_222_8).abs() < 1e-12);
    /// assert!((v[3] - 1000.0).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn softplus(values: &[f64]) -> Vec<f64> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = softplus_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Log-softmax over a slice: `x_i − logsumexp(x)` elementwise, written
    /// into `out`.
    ///
    /// Numerically stable via the max-shift: `x_i − max(x) − ln(Σ_j exp(x_j −
    /// max(x)))`. This is the primitive `PyTorch`'s [`nn.LogSoftmax`] computes
    /// (paired with [`nn.NLLLoss`] it forms [`nn.CrossEntropyLoss`]).
    /// Allocation-free: `out` must have the same length as `values` (a length
    /// mismatch returns [`Error::LengthMismatch`]); reuse it across calls to
    /// avoid per-call allocation in hot loops. An empty slice leaves `out`
    /// untouched.
    ///
    /// Reference: the max-subtraction trick and the fused
    /// log-softmax/NLL/cross-entropy decomposition as documented in
    /// `PyTorch`'s [`nn.LogSoftmax`]
    /// (<https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html>).
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f64, 2.0, 3.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::log_softmax_into(&v, &mut out).unwrap();
    /// let s: f64 = out.iter().map(|x| x.exp()).sum();
    /// assert!((s - 1.0).abs() < 1e-12);
    /// assert!(out[2] > out[1] && out[1] > out[0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn log_softmax_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_log_softmax_f64(backend, values, out);
        Ok(())
    }

    /// Log-softmax over a slice, returning a new `Vec`. Convenience wrapper
    /// around [`log_softmax_into`]; prefer the `_into` form in hot loops.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::log_softmax(&[1.0_f64, 2.0, 3.0]);
    /// let s: f64 = v.iter().map(|x| x.exp()).sum();
    /// assert!((s - 1.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn log_softmax(values: &[f64]) -> Vec<f64> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = log_softmax_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// `SiLU` (`Swish`) activation, written into `out` (allocation-free variant of [`silu`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`silu`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f64, 1.0, -1.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::silu_into(&v, &mut out).unwrap();
    /// assert!(out[0].abs() < 1e-12);
    /// assert!((out[1] - 0.731_058_6).abs() < 1e-6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn silu_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_silu_f64(backend, values, out);
        Ok(())
    }

    /// `SiLU` (`Swish`) activation over a slice.
    ///
    /// Computes `silu(x)_i = x_i / (1 + exp(-x_i))` elementwise — the smooth
    /// LLM activation (Llama, Qwen, etc.). Saturates to `x` for large positive
    /// x, to 0 for large negative x; minimum ≈ −0.278 at x ≈ −1.28. Returns a
    /// new `Vec` of the same length; an empty slice yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`silu_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        let mut out = kernels::alloc_uninit(values.len());
        let _ = silu_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// GELU activation (tanh approximation), written into `out` (allocation-free variant of [`gelu`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`gelu`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [0.0_f64, 1.0, -1.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::gelu_into(&v, &mut out).unwrap();
    /// assert!(out[0].abs() < 1e-12);
    /// assert!((out[1] - 0.84119).abs() < 2e-4);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn gelu_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_gelu_f64(backend, values, out);
        Ok(())
    }

    /// GELU activation over a slice.
    ///
    /// Computes the tanh approximation
    /// `gelu(x)_i = 0.5·x_i·(1 + tanh(√(2/π)·(x_i + 0.044715·x_i³)))` — the
    /// production LLM activation (GPT-2 etc.). `tanh` is derived from `exp`,
    /// so no extra transcendental. Returns a new `Vec` of the same length; an
    /// empty slice yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`gelu_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        let mut out = kernels::alloc_uninit(values.len());
        let _ = gelu_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// `ReLU` activation, written into `out` (allocation-free variant of [`relu`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`relu`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [-3.0_f64, 0.0, 5.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::relu_into(&v, &mut out).unwrap();
    /// assert_eq!(out, [0.0, 0.0, 5.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn relu_into(values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_relu_f64(backend, values, out);
        Ok(())
    }

    /// `ReLU` activation over a slice.
    ///
    /// Computes `relu(x)_i = max(x_i, 0)` elementwise. Returns a new `Vec` of
    /// the same length; an empty slice yields an empty `Vec`.
    ///
    /// Convenience wrapper around [`relu_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::relu(&[-3.0_f64, 0.0, 5.0]);
    /// assert_eq!(v, [0.0, 0.0, 5.0]);
    /// ```
    #[must_use]
    pub fn relu(values: &[f64]) -> Vec<f64> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = relu_into(values, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// RMS norm (`x_i / sqrt(mean(x²) + eps)`), written into `out` (allocation-free variant of [`rms_norm`]).
    ///
    /// Allocation-free: `out` must have the same length as `values`; reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. See [`rms_norm`] for semantics and
    /// numerical properties.
    ///
    /// # Example
    /// ```
    /// let v = [3.0_f64, 4.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::rms_norm_into(&v, 0.0, &mut out).unwrap();
    /// let r = 12.5_f64.sqrt(); // sqrt(mean(9, 16))
    /// assert!((out[0] - 3.0 / r).abs() < 1e-12);
    /// assert!((out[1] - 4.0 / r).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    pub fn rms_norm_into(values: &[f64], eps: f64, out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_rms_norm_f64(backend, values, eps, out);
        Ok(())
    }

    /// RMS norm over a slice: `x_i / sqrt(mean(x²) + eps)`.
    ///
    /// The standard LLM normalization (Llama, Qwen). `eps` guards against
    /// division by zero for all-zero input; a typical value is `1e-5`.
    /// Returns a new `Vec` of the same length; an empty slice yields an empty
    /// `Vec`.
    ///
    /// Convenience wrapper around [`rms_norm_into`]; prefer the `_into`
    /// form in hot loops to avoid per-call allocation.
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
        let mut out = kernels::alloc_uninit(values.len());
        let _ = rms_norm_into(values, eps, &mut out); // infallible: out.len() == values.len() by construction
        out
    }

    /// Cosine similarity between two equal-length slices:
    /// `dot(a, b) / (|a|·|b|)`.
    ///
    /// Returns an error if the slices have different lengths or are empty
    /// (the angle between empty vectors is undefined). If either vector has
    /// zero norm the result is `0.0` (a zero vector has no direction, so it
    /// shares none with the other — the same convention as scikit-learn).
    /// Otherwise the result is in `[-1, 1]` up to rounding.
    ///
    /// # Errors
    /// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`, and
    /// [`Error::EmptyInput`] if both slices are empty.
    ///
    /// # Example
    /// ```
    /// let s = lanes::ml::f64::cosine_similarity(&[1.0_f64, 0.0], &[1.0_f64, 0.0]);
    /// assert_eq!(s, Ok(1.0));
    /// ```
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<f64, Error> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        if a.is_empty() {
            return Err(Error::EmptyInput);
        }
        let backend = Backend::detect();
        let dot = kernels::dispatch_dot_f64(backend, a, b);
        let na = kernels::sqrt::sqrt_f64(kernels::dispatch_sum_sq_f64(backend, a));
        let nb = kernels::sqrt::sqrt_f64(kernels::dispatch_sum_sq_f64(backend, b));
        if na == 0.0 || nb == 0.0 {
            return Ok(0.0);
        }
        Ok(dot / (na * nb))
    }

    /// Numerically-stable log-sum-exp: `ln(sum_i exp(x_i))`, the denominator
    /// of the log-softmax and the log-normalizer of cross-entropy losses.
    ///
    /// Computes `max(x) + ln(sum_i exp(x_i - max(x)))`. The max subtraction
    /// prevents overflow for large inputs. An empty slice yields
    /// `-infinity`.
    ///
    /// # Example
    /// ```
    /// let s = lanes::ml::f64::logsumexp(&[1.0_f64, 2.0, 3.0]);
    /// assert!((s - 3.407_605_964_444_385).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn logsumexp(values: &[f64]) -> f64 {
        let backend = Backend::detect();
        kernels::dispatch_logsumexp_f64(backend, values)
    }

    /// Layer normalization: `(x_i - mean(x)) / sqrt(variance(x) + eps)`,
    /// written into `out`.
    ///
    /// The standard pre-activation norm (complement to [`rms_norm`], which
    /// drops the mean). Allocation-free: `out` must have the same length as
    /// `values` (a length mismatch returns [`Error::LengthMismatch`]); reuse
    /// it across calls to avoid per-call allocation in hot loops. An empty
    /// slice leaves `out` untouched. NaNs propagate.
    ///
    /// # Example
    /// ```
    /// let v = [1.0_f64, 2.0, 3.0];
    /// let mut out = vec![0.0_f64; v.len()];
    /// lanes::ml::f64::layer_norm_into(&v, 1e-10, &mut out).unwrap();
    /// let m: f64 = out.iter().sum::<f64>() / 3.0;
    /// assert!(m.abs() < 1e-12);
    /// // sum of squares after norm is n·var/(var+eps) ≈ 3 for this input.
    /// let s: f64 = out.iter().map(|x| x * x).sum();
    /// assert!((s - 3.0).abs() < 1e-9);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if `out.len() != values.len()`.
    #[allow(clippy::cast_precision_loss)] // `len as f64` is inherent to the mean
    pub fn layer_norm_into(values: &[f64], eps: f64, out: &mut [f64]) -> Result<(), Error> {
        if values.len() != out.len() {
            return Err(Error::LengthMismatch {
                expected: values.len(),
                actual: out.len(),
            });
        }
        let backend = Backend::detect();
        kernels::dispatch_layer_norm_f64(backend, values, eps, out);
        Ok(())
    }

    /// Layer normalization, returning a new `Vec`. Convenience wrapper
    /// around [`layer_norm_into`]; prefer the `_into` form in hot loops.
    ///
    /// # Example
    /// ```
    /// let v = lanes::ml::f64::layer_norm(&[1.0_f64, 2.0, 3.0], 1e-10);
    /// let m: f64 = v.iter().sum::<f64>() / 3.0;
    /// assert!(m.abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn layer_norm(values: &[f64], eps: f64) -> Vec<f64> {
        let mut out = kernels::alloc_uninit(values.len());
        let _ = layer_norm_into(values, eps, &mut out); // infallible: out.len() == values.len() by construction
        out
    }
}
