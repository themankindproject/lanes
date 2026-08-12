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
//! use lanes::{dot, sum};
//!
//! let a = vec![1.0_f32; 1024];
//! let b = vec![2.0_f32; 1024];
//!
//! let dot_product = dot(&a, &b).unwrap();
//! let total = sum(&a);
//!
//! assert_eq!(total, 1024.0);
//! assert_eq!(dot_product, 2048.0);
//! ```
//!
//! ## Architecture
//!
//! The crate is layered as public API → algorithm layer → kernel layer →
//! backend layer. The public entry points (`sum`, `prod`, `min`, `max`, `dot`)
//! validate their inputs, resolve the execution backend once (cached in a
//! `OnceLock`), and dispatch to the matching optimized kernel. Every
//! operation has a portable scalar fallback.
//!
//! See the [architecture document](https://github.com/themankindproject/lanes/blob/main/docs/architecture.md)
//! for the full design, dispatch model, and extension roadmap.
//!
//! ## Supported backends
//!
//! | Architecture | Backend | Selection |
//! |---|---|---|
//! | `x86_64` | AVX-512F | runtime detection (`avx512f`) |
//! | `x86_64` | AVX2 + FMA | runtime detection (`avx2` + `fma`) |
//! | `x86_64` | SSE2 | mandatory on x86-64; runtime detection (`sse2`) |
//! | aarch64 | NEON | mandatory on ARMv8-A |
//! | any | Scalar | always available |
//!
//! WASM is a future target: it currently uses the scalar backend, and the
//! code is kept free of OS-specific dependencies so a SIMD128 backend can
//! be added later.
//!
//! ## Floating-point semantics
//!
//! All kernels operate on `f32`. The following is documented precisely so
//! results are predictable across backends:
//!
//! * **Reduction order** is backend-dependent. Scalar kernels reduce strictly
//!   left-to-right; SIMD kernels reduce in fixed-width chunks and then
//!   combine the chunk results. For inputs whose intermediate values exceed
//!   the range of exact representation, results may differ in the last ulp.
//! * **`sum`/`dot` propagate NaN** — any NaN input yields a NaN result.
//! * **`min`/`max` use IEEE 754 `minNum`/`maxNum` semantics in the scalar
//!   kernel** (NaN inputs are ignored except when every input is NaN; read
//!   [`f32::min`] for the exact rules). SIMD kernels follow the corresponding
//!   hardware instruction semantics, which may propagate a NaN present in a
//!   vector. For NaN-free inputs all backends agree exactly.
//! * Signed zero: `min` follows `minNum` semantics for `-0.0`/`+0.0`.
//!
//! Do not assume bit-identical results across backends for arbitrary
//! floating-point input; assume determinism *within* a backend for the
//! same input.
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
pub use algorithms::{dot, max, min, prod, sum};
pub use dispatch::Backend;
pub use error::Error;

/// Statistical reductions (aggregates over slices): `sum`, `prod`, `min`,
/// `max`, `sum_sq`, `mean`, `variance`, `dot`.
pub mod stats {
    #[cfg(feature = "alloc")]
    pub use crate::algorithms::stats::variance;
    pub use crate::algorithms::stats::{dot, max, mean, min, prod, sum, sum_sq};
}

/// Distance and norm functions: `l1_norm`, `l2_norm`, `max_norm`.
/// All are `no_std`-clean (the sqrt for `l2_norm` is the std-free kernel).
pub mod distance {
    pub use crate::algorithms::distance::{l1_norm, l2_norm, max_norm};
}

/// Elementwise math functions (per-element maps): `sqrt`, `clip`, `rsqrt`,
/// `exp`.
#[cfg(feature = "alloc")]
pub mod math {
    pub use crate::algorithms::math::{clip, exp, rsqrt, sqrt};
}

/// ML kernels built on the `lanes` core (softmax, and future layer-norm,
/// quantize, argmax, cosine-sim). Available on any target with an allocator:
/// built with `std`, or with `no_std` + the `alloc` feature.
#[cfg(feature = "alloc")]
pub mod ml {
    pub use crate::algorithms::ml::{gelu, relu, sigmoid, silu, softmax};
}
