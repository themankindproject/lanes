//! Runtime CPU feature detection and backend dispatch.
//!
//! The [`Backend::detect`] function determines the best available SIMD
//! backend for the current CPU at runtime and caches the result for the
//! lifetime of the process.
//!
//! # Diagnostics override
//!
//! With the `std` feature, the environment variable `LANES_BACKEND` can
//! force a specific backend for benchmarking or debugging. Accepted values:
//! `scalar`, `sse2`, `avx2`, `avx512`, `neon`. The request is honoured only
//! if the backend is both compiled in and actually supported by the host CPU;
//! otherwise detection proceeds as usual (a requested backend is never
//! invoked on hardware that does not support it).

/// Available SIMD backends for computation.
///
/// The appropriate backend is selected at runtime based on CPU feature
/// detection. Variants are target-dependent: only backends compiled for
/// the current architecture are listed.
///
/// This enum is marked `#[non_exhaustive]`: new backends (e.g. WASM
/// SIMD128, SVE) may be added in a minor release, and the variant set
/// already differs per target architecture — so downstream `match`es must
/// keep a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Backend {
    /// Portable scalar implementation — always available.
    Scalar,
    /// x86-64 SSE2 (128-bit SIMD) — mandatory on all x86-64 CPUs, so this is
    /// the guaranteed 128-bit tier (same role as NEON on aarch64).
    #[cfg(target_arch = "x86_64")]
    Sse2,
    /// x86-64 AVX2 + FMA (256-bit SIMD).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// x86-64 AVX-512F (512-bit SIMD).
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// ARM NEON (128-bit SIMD).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Backend {
    /// Detect the best available SIMD backend for the current CPU.
    ///
    /// On the first call (with the `std` feature), the result is cached in a
    /// `OnceLock` so subsequent calls are essentially free.
    ///
    /// Without the `std` feature there is no runtime CPU probing, but the
    /// architecture baseline still guarantees a SIMD tier on the supported
    /// targets: SSE2 is mandatory on x86-64 and NEON is mandatory on
    /// ARMv8-A (aarch64), so those backends are selected statically. All
    /// other targets fall back to [`Backend::Scalar`].
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(feature = "std")]
        {
            use std::sync::OnceLock;
            static DETECTED: OnceLock<Backend> = OnceLock::new();
            *DETECTED.get_or_init(crate::platform::detect_best_backend)
        }
        #[cfg(not(feature = "std"))]
        {
            static_backend()
        }
    }
}

/// Best backend guaranteed to be available without runtime CPU probing.
///
/// SSE2 is part of the x86-64 baseline and NEON is part of the ARMv8-A
/// baseline, so on those targets the SIMD tier is a compile-time guarantee
/// even in `no_std` builds (where `is_x86_feature_detected!` is
/// unavailable). Everywhere else only the scalar fallback is guaranteed.
#[cfg(not(feature = "std"))]
fn static_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    let backend = Backend::Sse2;
    #[cfg(target_arch = "aarch64")]
    let backend = Backend::Neon;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let backend = Backend::Scalar;
    backend
}
