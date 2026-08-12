//! Machine-learning kernels built on the `lanes` core.
//!
//! These functions compose the core reductions (`sum`, `max`, `dot`) into
//! higher-level ML ops. Each is dispatched to the best SIMD backend at
//! runtime, exactly like the core functions.

use crate::dispatch::Backend;
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
/// let v = lanes::ml::softmax(&[1.0_f32, 2.0, 3.0]);
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
/// Computes `sigmoid(x)_i = 1 / (1 + exp(-x_i))` elementwise. Outputs are in
/// `(0, 1)`, monotone, with `sigmoid(0) = 0.5`. Large positive inputs
/// saturate to 1.0, large negative to 0.0 (via the exp saturation — no
/// overflow). Returns a new `Vec` of the same length; an empty slice yields
/// an empty `Vec`.
///
/// # Example
/// ```
/// let v = lanes::ml::sigmoid(&[0.0_f32, 1.0, -1.0]);
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
/// let v = lanes::ml::silu(&[0.0_f32, 1.0, -1.0]);
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
/// production LLM activation (GPT-2 etc.), accurate to ~1e-3 of exact GELU.
/// `tanh` is derived from `exp`, so no extra transcendental. Returns a new
/// `Vec` of the same length; an empty slice yields an empty `Vec`.
///
/// # Example
/// ```
/// let v = lanes::ml::gelu(&[0.0_f32, 1.0, -1.0]);
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
/// let v = lanes::ml::relu(&[-3.0_f32, 0.0, 5.0]);
/// assert_eq!(v, [0.0, 0.0, 5.0]);
/// ```
#[must_use]
pub fn relu(values: &[f32]) -> Vec<f32> {
    let mut out = alloc::vec![0.0_f32; values.len()];
    let backend = Backend::detect();
    kernels::dispatch_relu(backend, values, &mut out);
    out
}
