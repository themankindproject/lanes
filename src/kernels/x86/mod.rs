//! x86-64 SIMD kernel implementations.
//!
//! This module provides optimized kernels using SSE2 (the mandatory x86-64
//! 128-bit baseline), AVX2, and AVX-512 instruction sets.

pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod sse2;
