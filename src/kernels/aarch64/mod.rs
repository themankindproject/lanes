//! AArch64 SIMD kernel implementations.
//!
//! This module provides optimized kernels using ARM NEON (128-bit SIMD),
//! which is mandatory on all ARMv8-A processors.

pub(crate) mod neon;
