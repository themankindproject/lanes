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
//! `scalar`, `avx2`, `avx512`, `neon`. The request is honoured only if the
//! backend is both compiled in and actually supported by the host CPU;
//! otherwise detection proceeds as usual (a requested backend is never
//! invoked on hardware that does not support it).

/// Available SIMD backends for computation.
///
/// The appropriate backend is selected at runtime based on CPU feature
/// detection. Variants are target-dependent: only backends compiled for
/// the current architecture are listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Without the `std` feature, this always returns [`Backend::Scalar`].
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
            Backend::Scalar
        }
    }

    /// Stable, human-readable name of this backend.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Self::Sse2 => "sse2",
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => "avx512",
            #[cfg(target_arch = "aarch64")]
            Self::Neon => "neon",
        }
    }
}
