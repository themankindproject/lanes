//! Platform-specific CPU feature detection.
//!
//! This module abstracts hardware capability probing behind a uniform
//! interface used by the dispatch layer. It is only compiled when the
//! `std` feature is enabled (runtime CPU probing requires `std`); in
//! `no_std` builds [`Backend::detect`] always returns the scalar backend.

use crate::dispatch::Backend;

/// The `LANES_BACKEND` environment variable, if set to a known backend name.
#[must_use]
fn forced_backend_from_env() -> Option<Backend> {
    let raw = std::env::var("LANES_BACKEND").ok()?;
    forced_backend(&raw)
}

/// Map a backend name (as accepted by `LANES_BACKEND`) to a [`Backend`].
///
/// Unknown or unavailable names return `None` so that callers fall back
/// to auto-detection. The pure-string form keeps this testable without a
/// process environment.
#[must_use]
fn forced_backend(raw: &str) -> Option<Backend> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "scalar" => Some(Backend::Scalar),
        #[cfg(target_arch = "x86_64")]
        "sse2" => Some(Backend::Sse2),
        #[cfg(target_arch = "x86_64")]
        "avx2" => Some(Backend::Avx2),
        #[cfg(target_arch = "x86_64")]
        "avx512" => Some(Backend::Avx512),
        #[cfg(target_arch = "aarch64")]
        "neon" => Some(Backend::Neon),
        _ => None,
    }
}

/// Detect the best available SIMD backend for the running CPU, honoring the
/// `LANES_BACKEND` diagnostic override when it names a supported backend.
///
/// This performs actual hardware probing (e.g., `cpuid` on x86-64) and the
/// result is cached by the dispatch layer (`Backend::detect`).
#[must_use]
pub(crate) fn detect_best_backend() -> Backend {
    let detected = auto_detect();
    match forced_backend_from_env() {
        Some(requested) if supports(requested) => requested,
        _ => detected,
    }
}

/// Whether `backend` is compiled in and actually supported by this CPU.
///
/// A backend must pass this gate before any of its (unsafe) kernels may be
/// invoked. `Scalar` is always supported.
#[must_use]
pub(crate) fn supports(backend: Backend) -> bool {
    match backend {
        Backend::Scalar => true,
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => is_x86_feature_detected!("sse2"),
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => is_x86_feature_detected!("avx512f"),
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // NEON is mandatory in the ARMv8-A baseline, so it is always
            // available on aarch64. (`is_aarch64_feature_detected!` is
            // unavailable when cross-compiling, and the runtime check would
            // be redundant anyway.)
            true
        }
    }
}

/// Detect the best available SIMD backend for the running CPU.
#[cfg(target_arch = "x86_64")]
fn auto_detect() -> Backend {
    // AVX-512F first (most capable).
    if is_x86_feature_detected!("avx512f") {
        return Backend::Avx512;
    }
    // Then AVX2 + FMA.
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return Backend::Avx2;
    }
    // Then SSE2 (mandatory on x86-64, but the runtime check keeps the
    // detection logic uniform across tiers).
    if is_x86_feature_detected!("sse2") {
        return Backend::Sse2;
    }
    Backend::Scalar
}

/// On aarch64, NEON is mandatory (part of the ARMv8-A baseline).
#[cfg(target_arch = "aarch64")]
fn auto_detect() -> Backend {
    Backend::Neon
}

/// Fallback for all other architectures.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn auto_detect() -> Backend {
    Backend::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_backend_parses_known_names() {
        assert_eq!(forced_backend("scalar"), Some(Backend::Scalar));
        assert_eq!(forced_backend(" SCALAR "), Some(Backend::Scalar));
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(forced_backend("sse2"), Some(Backend::Sse2));
            assert_eq!(forced_backend("avx2"), Some(Backend::Avx2));
            assert_eq!(forced_backend("AVX512"), Some(Backend::Avx512));
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(forced_backend("neon"), Some(Backend::Neon));
    }

    #[test]
    fn forced_backend_ignores_unknown_names() {
        assert_eq!(forced_backend("cuda"), None);
        assert_eq!(forced_backend(""), None);
    }

    #[test]
    fn forced_backend_stable_across_calls() {
        let b1 = Backend::detect();
        let b2 = Backend::detect();
        assert_eq!(b1, b2);
    }
}
