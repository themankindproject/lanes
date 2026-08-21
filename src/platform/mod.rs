//! Platform-specific CPU feature detection.
//!
//! This module abstracts hardware capability probing behind a uniform
//! interface used by the dispatch layer. It is only compiled when the
//! `std` feature is enabled (runtime CPU probing requires `std`); in
//! `no_std` builds [`Backend::detect`] always returns the scalar backend.

use crate::dispatch::Backend;

/// AVX-512 sub-feature capabilities detected at runtime.
///
/// These enhance specific operations within the AVX-512F backend. Each CPU
/// that has AVX-512F may or may not have these extensions:
/// - `vpopcntdq`: Native 512-bit popcount for binary distances (hamming/jaccard)
/// - `vnni`: VPDPBUSD for fused byte dot products (i8 family)
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // infrastructure for per-kernel sub-feature dispatch
pub(crate) struct Avx512Caps {
    /// AVX-512 VPOPCNTDQ: native 64-bit lane popcount instruction.
    /// Accelerates hamming/jaccard binary distance computations.
    pub(crate) vpopcntdq: bool,
    /// AVX-512 VNNI: Vector Neural Network Instructions (VPDPBUSD).
    /// Accelerates i8 dot products and sum-of-absolute-differences.
    pub(crate) vnni: bool,
}

#[cfg(target_arch = "x86_64")]
impl Avx512Caps {
    /// Detect AVX-512 sub-feature capabilities on the running CPU.
    #[allow(dead_code)] // used by dispatch_info example and future per-kernel dispatch
    pub(crate) fn detect() -> Self {
        use std::sync::OnceLock;
        static CAPS: OnceLock<Avx512Caps> = OnceLock::new();
        *CAPS.get_or_init(|| Avx512Caps {
            vpopcntdq: is_x86_feature_detected!("avx512vpopcntdq"),
            vnni: is_x86_feature_detected!("avx512vnni"),
        })
    }
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
        #[cfg(target_arch = "wasm32")]
        "wasm" => Some(Backend::Wasm),
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
    if let Ok(raw) = std::env::var("LANES_BACKEND") {
        if let Some(requested) = forced_backend(&raw) {
            if supports(requested) {
                return requested;
            }
            #[cfg(debug_assertions)]
            eprintln!("[lanes] LANES_BACKEND='{raw}' ignored: backend not supported on this CPU");
        } else {
            #[cfg(debug_assertions)]
            eprintln!("[lanes] LANES_BACKEND='{raw}' ignored: unknown backend name");
        }
    }
    detected
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
        #[cfg(target_arch = "wasm32")]
        Backend::Wasm => {
            // WASM SIMD128 is either available at compile time or detectable
            // at runtime where the toolchain provides `is_wasm_feature_detected`.
            // Fall back to `true` — if SIMD128 is not available the scalar
            // fallback remains correct, and on wasm32 SIMD128 is the baseline
            // when compiled with `+simd128`.
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

#[cfg(target_arch = "wasm32")]
fn auto_detect() -> Backend {
    Backend::Wasm
}

/// Fallback for all other architectures.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
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
