//! Public algorithm functions with runtime SIMD dispatch.
//!
//! Each function automatically selects the best available backend
//! for the current CPU. The dispatch decision is cached after the
//! first call (with the `std` feature enabled).
//!
//! This layer is safe code only: all `unsafe` lives in the kernel layer.
//! The attribute below makes that boundary compiler-enforced.
#![forbid(unsafe_code)]

// The `ml` family returns heap-allocated `Vec`s, so it needs `alloc`.
#[cfg(feature = "alloc")]
pub mod ml;

// Statistical reductions (aggregates over slices).
pub mod stats;

// Distance and norm functions.
pub mod distance;

// Elementwise math functions (per-element maps).
#[cfg(feature = "alloc")]
pub mod math;
