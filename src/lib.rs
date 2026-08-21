//! # lanes
//!
//! High-performance computational algorithm kernels with runtime SIMD dispatch.
//!
//! `lanes` provides a small collection of optimized numerical algorithms that
//! automatically select the best available SIMD instruction set at runtime.
//! Write your code once and get near-optimal performance on every platform —
//! from servers with AVX-512 to targets with only scalar arithmetic.
//!
//! ## Quick Start
//!
//! ```rust
//! use lanes::stats::f32 as stats;
//!
//! let a = vec![1.0_f32; 1024];
//! let b = vec![2.0_f32; 1024];
//!
//! let dot_product = stats::dot(&a, &b).unwrap();
//! let total = stats::sum(&a);
//!
//! assert_eq!(total, 1024.0);
//! assert_eq!(dot_product, 2048.0);
//! ```
//!
//! ## Architecture
//!
//! The crate is layered as public API → algorithm layer → kernel layer →
//! backend layer. The public entry points ([`stats::f32::sum`],
//! [`stats::f64::sum`], [`stats::f32::min`], [`stats::f32::max`],
//! [`stats::f32::dot`]) validate their inputs, resolve the execution backend
//! once (cached in a `OnceLock`), and dispatch to the matching optimized
//! kernel. Every operation has a portable scalar fallback.
//!
//! ## Precision families
//!
//! Every family (`stats`, `distance`, `math`, `special`, `ml`) is split
//! into an `f32` (single-precision) and an `f64` (double-precision)
//! submodule. Pick the precision at the call site:
//!
//! ```rust
//! use lanes::stats::{f32, f64};
//!
//! let s32 = f32::sum(&[1.0_f32, 2.0, 3.0]);
//! let s64 = f64::sum(&[1.0_f64, 2.0, 3.0]);
//! assert_eq!(s32, 6.0_f32);
//! assert_eq!(s64, 6.0_f64);
//! ```
//!
//! The `binary` family is the exception: it operates on packed `&[u8]`
//! bitmaps with bit-level semantics and has no precision submodules.
//! Integer reductions live in `stats::i8` (exact `i64` accumulation).
//!
//! ## Supported backends
//!
//! | Architecture | Backend | Selection |
//! |---|---|---|
//! | `x86_64` | AVX-512F (+ VPOPCNTDQ/VNNI when present) | runtime detection (`avx512f`); sub-features via `platform::Avx512Caps` |
//! | `x86_64` | AVX2 + FMA (+ F16C for half) | runtime detection (`avx2` + `fma`); half converts use F16C (`cvtph`) when `f16c` present |
//! | `x86_64` | SSE2 | mandatory on x86-64; runtime detection (`sse2`) |
//! | aarch64 | NEON | mandatory on ARMv8-A |
//! | `wasm32` | WASM | scalar fallthrough now; `wasm32-unknown-unknown` builds wire the WASM backend (SIMD128 intrinsics are a drop-in next step) |
//! | any | Scalar | always available |
//!
//! ## Floating-point semantics
//!
//! All kernels operate on `f32` or `f64` (chosen via the family submodule).
//! The following is documented precisely so results are predictable across
//! backends:
//!
//! * **Reduction order** is backend-dependent. Scalar kernels reduce strictly
//!   left-to-right; SIMD kernels reduce in fixed-width chunks and then
//!   combine the chunk results. For inputs whose intermediate values exceed
//!   the range of exact representation, results may differ in the last ulp.
//! * **`sum`/`dot` propagate NaN** — any NaN input yields a NaN result.
//! * **`min`/`max` have identical NaN semantics on every backend** (IEEE 754
//!   `minNum`/`maxNum`, matching [`f32::min`]/[`f32::max`]): a NaN input is
//!   ignored unless every input is NaN, in which case the result is NaN.
//! * **`max_norm` returns NaN if any input is NaN** on every backend
//!   (matching the scalar `total_cmp` reference, where NaN sorts above all).
//! * Signed zero: for `min`/`max` inputs containing both `-0.0` and `+0.0`
//!   as the extremum, the sign of the result is backend-dependent (the
//!   values compare equal; the sign follows the backend's combine order).
//!
//! Do not assume bit-identical results across backends for arbitrary
//! floating-point input; assume determinism *within* a backend for the
//! same input.
//!
//! ## Error handling
//!
//! Fallible kernels report failures through [`Error`] instead of panicking,
//! so callers can branch on the exact failure mode:
//!
//! * Two-input operations (`dot`, `squared_distance`, `abs_sub`, `hypot`,
//!   `cosine_similarity`, `kl_divergence`, `js_divergence`, `hamming`,
//!   `jaccard`) return
//!   `Err(`[`Error::LengthMismatch`]`)` when their operands disagree in
//!   length.
//! * Every `_into` variant returns `Err(`[`Error::LengthMismatch`]`)` when
//!   the caller-provided output buffer has the wrong length.
//! * `geometric_mean` returns `Err(`[`Error::EmptyInput`]`)` for an empty
//!   slice and `Err(`[`Error::NonPositiveInput`]`)` (with the offending
//!   index) when any value is ≤ 0. NaN inputs are *not* an error: they
//!   propagate to a NaN result.
//! * `clip` returns `Err(`[`Error::InvalidBounds`]`)` when `lo > hi` or a
//!   bound is NaN (mirroring the [`f32::clamp`] precondition).
//!
//! Infallible kernels (reductions like `sum`, single-input maps like `exp`)
//! never fail; NaN inputs propagate to NaN results rather than becoming
//! errors.
//!
//! [`Error`] and [`Backend`] are both `#[non_exhaustive]`: new error
//! variants (as new kernel families land) and new backends (e.g. WASM
//! SIMD128) may be added in minor releases, so `match`es on them must
//! keep a wildcard arm.
//!
//! ## Safety policy
//!
//! All `unsafe` code is confined to the kernel layer, behind `pub(crate)`
//! visibility, and each function documents the invariant that makes it safe
//! (the enclosing `#[target_feature]` gate plus the caller's runtime feature
//! check). The crate forbids `unsafe_op_in_unsafe_fn`: every unsafe
//! operation inside an unsafe function is an explicit, reviewed block.

#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![cfg_attr(not(feature = "std"), no_std)]

// Unit tests always need `std` (test harness), even when the crate itself
// is built without the `std` feature.
#[cfg(test)]
extern crate std;

// `alloc` is always available (with an allocator on the target); the `ml`
// layer needs it for `Vec`, and `std` implies it.
extern crate alloc;

mod algorithms;
mod dispatch;
mod error;
mod kernels;
#[cfg(feature = "std")]
mod platform;

// Public API re-exports.
pub use dispatch::{Backend, available_backends};
pub use error::Error;

/// Statistical reductions (aggregates over slices): `sum`, `prod`, `min`,
/// `max`, `argmax`, `argmin`, `sum_sq`, `mean`, `variance`, `std_dev`,
/// `geometric_mean`, `dot`, `count_zero`, `count_nan`, `count_infinite`.
///
/// Precision is selected via the [`f32`] or [`f64`] submodule; the [`i8`]
/// submodule offers exact integer reductions with `i64` accumulation.
pub mod stats {
    pub use crate::algorithms::stats::{f32, f64, i8};
}

/// Distance and norm functions: `l1_norm`, `l2_norm`, `max_norm`,
/// `squared_distance`, `kl_divergence`, `js_divergence`.
/// All are `no_std`-clean (the sqrt for `l2_norm` is the std-free kernel).
/// Precision is selected via the [`f32`] or [`f64`] submodule; the [`i8`]
/// submodule offers exact integer norms with `i64` accumulation.
pub mod distance {
    pub use crate::algorithms::distance::{f32, f64, i8};
}

/// Binary (bit-packed) vector distances: [`binary::hamming`],
/// [`binary::jaccard`]. Both operate on `&[u8]` bitmaps with bit-level
/// semantics (a slice of `n` bytes is a binary vector of `8n` dimensions)
/// and are `no_std`-clean.
pub mod binary {
    pub use crate::algorithms::binary::{hamming, jaccard};
}

/// Elementwise math functions (per-element maps): `sqrt`, `clip`, `rsqrt`,
/// `exp`, `ln`, `tanh`, `hypot`, `powi`, `abs_sub`. Each also has an
/// allocation-free `_into` variant
/// (e.g. [`math::f32::exp_into`]) that writes into a caller-provided
/// buffer — prefer those in hot loops. Precision is selected via the
/// [`f32`] or [`f64`] submodule.
#[cfg(feature = "alloc")]
pub mod math {
    pub use crate::algorithms::math::{f32, f64};
}

/// Special functions (`erf`, `erfc`) — elementwise maps over slices.
/// Select precision via the [`f32`] or [`f64`] submodule.
#[cfg(feature = "alloc")]
pub mod special {
    pub use crate::algorithms::special::{f32, f64};
}

/// ML kernels built on the `lanes` core (`softmax`, `log_softmax`,
/// `sigmoid`, `silu`, `gelu`, `relu`, `softplus`, `rms_norm`, `layer_norm`,
/// `cosine_similarity`, `logsumexp`). Every map-style op also has an
/// allocation-free `_into` variant (e.g. [`ml::f32::softmax_into`],
/// [`ml::f32::layer_norm_into`]) that writes into a caller-provided buffer
/// — prefer those in hot loops. Available on any target with an allocator:
/// built with `std`, or with `no_std` + the `alloc` feature. Precision is
/// selected via the [`f32`] or [`f64`] submodule.
#[cfg(feature = "alloc")]
pub mod ml {
    pub use crate::algorithms::ml::{f32, f64};
}

/// Half-precision (f16) and brain floating-point (bf16) conversions and
/// mixed-precision dot products (`f16_to_f32`, `f32_to_f16`, `bf16_to_f32`,
/// `f32_to_bf16`, `dot_f16`, `dot_bf16`). All narrowing conversions use
/// round-to-nearest-even; the module is `no_std`-compatible (writes into
/// caller-provided buffers, no `alloc`). F16 paths on `x86_64` use F16C
/// hardware when present and fall back to scalar; bf16 paths are vectorized
/// integer shifts on every SIMD tier.
pub mod convert {
    pub use crate::algorithms::convert::{
        bf16_to_f32, dot_bf16, dot_f16, f16_to_f32, f32_to_bf16, f32_to_f16,
    };
}
