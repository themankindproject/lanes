//! Public algorithm functions with runtime SIMD dispatch.
//!
//! Each function automatically selects the best available backend
//! for the current CPU. The dispatch decision is cached after the
//! first call (with the `std` feature enabled).

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
