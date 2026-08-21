//! WASM SIMD128 kernel stubs.
//!
//! On `wasm32` this module will house SIMD128-accelerated kernels (via
//! `core::arch::wasm32`). For now every kernel delegates to the portable
//! scalar implementation so the crate compiles and passes tests on both
//! native and `wasm32-unknown-unknown` targets.
//!
//! The `#[cfg(target_arch = "wasm32")]` gate in `kernels/mod.rs` ensures
//! this module is only compiled on WASM; native builds are unaffected.
#![allow(clippy::all, clippy::pedantic)]
#![allow(unused_imports)]

// Re-export scalar kernels that have identical signatures on SIMD backends.
// Split by feature so `cargo check --no-default-features --target wasm32`
// does not try to import alloc-gated symbols that don't exist.
pub(crate) use crate::kernels::scalar::{
    argmax, argmax_f64, argmin, argmin_f64, bf16_to_f32, center_f32, center_f64, count_infinite,
    count_infinite_f64, count_nan, count_nan_f64, count_zero, count_zero_f64, count_zero_i8, dot,
    dot_bf16, dot_f16, dot_f64, dot_i8, f16_to_f32, f32_to_bf16, f32_to_f16, hamming_popcount,
    jaccard, jaccard_counts, js_divergence, js_divergence_f64, kl_divergence, kl_divergence_f64,
    l1_norm, l1_norm_f64, l1_norm_i8, prod, prod_f64, squared_distance, squared_distance_f64,
    squared_distance_i8, sum, sum_f64, sum_i8, sum_sq, sum_sq_f64, variance_fused_f32,
    variance_fused_f64,
};

#[cfg(feature = "alloc")]
pub(crate) use crate::kernels::scalar::{
    abs_sub, abs_sub_f64, clip, clip_f64, erf, erf_f64, erfc, erfc_f64, exp, exp_f64, gelu,
    gelu_f64, hypot, hypot_f64, layer_norm, layer_norm_f64, ln, ln_f64, log_softmax,
    log_softmax_f64, log1p, log1p_f64, logsumexp, logsumexp_f64, powi, powi_f64, relu, relu_f64,
    rms_norm, rms_norm_f64, rsqrt, rsqrt_f64, sigmoid, sigmoid_f64, silu, silu_f64, softmax,
    softmax_f64, softplus, softplus_f64, sqrt, sqrt_f64, tanh, tanh_f64,
};

// ---- Option-returning kernels -------------------------------------------
// Scalar returns `Option<T>` but every SIMD backend returns the raw `T`
// (the non-empty invariant is upheld by the dispatch layer via `Option`
// wrapping). The WASM stubs match the SIMD signature so the shared
// `dispatch!` macro's `Some`/`jaccard_similarity` wrappers apply correctly.
// For scalar fallback on WASM we still delegate via the scalar impl;
// the `unwrap` is safe here because the dispatch wrapper only calls these
// on non-empty slices (caller checks emptiness before dispatch), matching
// the x86/NEON invariant.

#[inline]
pub fn min(values: &[f32]) -> f32 {
    crate::kernels::scalar::min(values).unwrap()
}
#[inline]
pub fn max(values: &[f32]) -> f32 {
    crate::kernels::scalar::max(values).unwrap()
}
#[inline]
pub fn max_norm(values: &[f32]) -> f32 {
    crate::kernels::scalar::max_norm(values).unwrap()
}
#[inline]
pub fn min_f64(values: &[f64]) -> f64 {
    crate::kernels::scalar::min_f64(values).unwrap()
}
#[inline]
pub fn max_f64(values: &[f64]) -> f64 {
    crate::kernels::scalar::max_f64(values).unwrap()
}
#[inline]
pub fn max_norm_f64(values: &[f64]) -> f64 {
    crate::kernels::scalar::max_norm_f64(values).unwrap()
}
#[inline]
pub fn min_i8(values: &[i8]) -> i8 {
    crate::kernels::scalar::min_i8(values).unwrap()
}
#[inline]
pub fn max_i8(values: &[i8]) -> i8 {
    crate::kernels::scalar::max_i8(values).unwrap()
}
#[inline]
pub fn max_abs_i8(values: &[i8]) -> u8 {
    crate::kernels::scalar::max_abs_i8(values).unwrap()
}
