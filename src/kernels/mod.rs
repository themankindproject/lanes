//! Kernel implementations for each backend.
//!
//! Each sub-module provides optimized implementations for its respective
//! SIMD instruction set. The dispatch functions in this module select the
//! correct implementation based on the detected [`Backend`].
//!
//! The scalar module is always available and serves as the reference
//! implementation and universal fallback.

pub(crate) mod exp;
pub(crate) mod macros;
pub(crate) mod scalar;
pub(crate) mod sqrt;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

use crate::dispatch::Backend;

/// Identity pass-through used by `dispatch_unary!` for non-Option returns.
#[inline]
fn id<T>(v: T) -> T {
    v
}

/// Shared skeleton for unary-reduction dispatch fns: one `match` on `Backend`
/// with the per-backend kernel paths as arguments. The return type is
/// parameterized (`f32` or `Option<f32>` — pass `$wrap` as `()` or `Some`).
macro_rules! dispatch_unary {
    ($name:ident, $ret:ty, $wrap:tt, $scalar:path, $sse2:path, $avx2:path, $avx512:path, $neon:path) => {
        #[inline]
        pub(crate) fn $name(backend: Backend, values: &[f32]) -> $ret {
            match backend {
                Backend::Scalar => $scalar(values),

                #[cfg(target_arch = "x86_64")]
                Backend::Sse2 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    $wrap(unsafe { $sse2(values) })
                }

                #[cfg(target_arch = "x86_64")]
                Backend::Avx2 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    $wrap(unsafe { $avx2(values) })
                }

                #[cfg(target_arch = "x86_64")]
                Backend::Avx512 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    $wrap(unsafe { $avx512(values) })
                }

                #[cfg(target_arch = "aarch64")]
                Backend::Neon => {
                    // SAFETY: NEON is mandatory on all aarch64 targets.
                    $wrap(unsafe { $neon(values) })
                }
            }
        }
    };
}

/// Shared skeleton for map dispatch fns: one `match` on `Backend` with the
/// per-backend kernel paths as arguments. Writes results into `out`.
///
/// Gated on `alloc`: every map's output is a heap-allocated buffer.
macro_rules! dispatch_map {
    ($name:ident, $scalar:path, $sse2:path, $avx2:path, $avx512:path, $neon:path) => {
        #[cfg(feature = "alloc")]
        #[inline]
        pub(crate) fn $name(backend: Backend, values: &[f32], out: &mut [f32]) {
            match backend {
                Backend::Scalar => $scalar(values, out),

                #[cfg(target_arch = "x86_64")]
                Backend::Sse2 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    unsafe { $sse2(values, out) }
                }

                #[cfg(target_arch = "x86_64")]
                Backend::Avx2 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    unsafe { $avx2(values, out) }
                }

                #[cfg(target_arch = "x86_64")]
                Backend::Avx512 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    unsafe { $avx512(values, out) }
                }

                #[cfg(target_arch = "aarch64")]
                Backend::Neon => {
                    // SAFETY: NEON is mandatory on all aarch64 targets.
                    unsafe { $neon(values, out) }
                }
            }
        }
    };
}

dispatch_unary!(
    dispatch_sum,
    f32,
    id,
    scalar::sum,
    x86::sse2::sum,
    x86::avx2::sum,
    x86::avx512::sum,
    aarch64::neon::sum
);

dispatch_unary!(
    dispatch_prod,
    f32,
    id,
    scalar::prod,
    x86::sse2::prod,
    x86::avx2::prod,
    x86::avx512::prod,
    aarch64::neon::prod
);

dispatch_unary!(
    dispatch_min,
    Option<f32>,
    Some,
    scalar::min,
    x86::sse2::min,
    x86::avx2::min,
    x86::avx512::min,
    aarch64::neon::min
);

dispatch_unary!(
    dispatch_max,
    Option<f32>,
    Some,
    scalar::max,
    x86::sse2::max,
    x86::avx2::max,
    x86::avx512::max,
    aarch64::neon::max
);

dispatch_unary!(
    dispatch_sum_sq,
    f32,
    id,
    scalar::sum_sq,
    x86::sse2::sum_sq,
    x86::avx2::sum_sq,
    x86::avx512::sum_sq,
    aarch64::neon::sum_sq
);

dispatch_unary!(
    dispatch_l1_norm,
    f32,
    id,
    scalar::l1_norm,
    x86::sse2::l1_norm,
    x86::avx2::l1_norm,
    x86::avx512::l1_norm,
    aarch64::neon::l1_norm
);

dispatch_unary!(
    dispatch_max_norm,
    Option<f32>,
    Some,
    scalar::max_norm,
    x86::sse2::max_norm,
    x86::avx2::max_norm,
    x86::avx512::max_norm,
    aarch64::neon::max_norm
);

/// Dispatch the dot product operation to the appropriate backend.
///
/// Falls through to scalar for backends that are not available on the
/// current compilation target.
#[inline]
pub(crate) fn dispatch_dot(backend: Backend, a: &[f32], b: &[f32]) -> f32 {
    match backend {
        Backend::Scalar => scalar::dot(a, b),

        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => {
            // SAFETY: The dispatch layer has verified SSE2 is available.
            unsafe { x86::sse2::dot(a, b) }
        }

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            // SAFETY: The dispatch layer has verified AVX2 + FMA are available.
            unsafe { x86::avx2::dot(a, b) }
        }

        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => {
            // SAFETY: The dispatch layer has verified AVX-512F is available.
            unsafe { x86::avx512::dot(a, b) }
        }

        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // SAFETY: NEON is mandatory on all aarch64 targets.
            unsafe { aarch64::neon::dot(a, b) }
        }
    }
}

dispatch_map!(
    dispatch_softmax,
    scalar::softmax,
    x86::sse2::softmax,
    x86::avx2::softmax,
    x86::avx512::softmax,
    aarch64::neon::softmax
);

dispatch_map!(
    dispatch_sigmoid,
    scalar::sigmoid,
    x86::sse2::sigmoid,
    x86::avx2::sigmoid,
    x86::avx512::sigmoid,
    aarch64::neon::sigmoid
);

dispatch_map!(
    dispatch_silu,
    scalar::silu,
    x86::sse2::silu,
    x86::avx2::silu,
    x86::avx512::silu,
    aarch64::neon::silu
);

dispatch_map!(
    dispatch_gelu,
    scalar::gelu,
    x86::sse2::gelu,
    x86::avx2::gelu,
    x86::avx512::gelu,
    aarch64::neon::gelu
);

dispatch_map!(
    dispatch_relu,
    scalar::relu,
    x86::sse2::relu,
    x86::avx2::relu,
    x86::avx512::relu,
    aarch64::neon::relu
);

dispatch_map!(
    dispatch_sqrt,
    scalar::sqrt,
    x86::sse2::sqrt,
    x86::avx2::sqrt,
    x86::avx512::sqrt,
    aarch64::neon::sqrt
);

/// Dispatch the elementwise `clip` map (`clamp(x, lo, hi)`) to the
/// appropriate backend.
///
/// Gated on `alloc`: its only public caller (`lanes::math::clip`) returns a
/// `Vec`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn dispatch_clip(backend: Backend, values: &[f32], lo: f32, hi: f32, out: &mut [f32]) {
    match backend {
        Backend::Scalar => scalar::clip(values, lo, hi, out),

        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => {
            // SAFETY: The dispatch layer has verified SSE2 is available.
            unsafe { x86::sse2::clip(values, lo, hi, out) }
        }

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            // SAFETY: The dispatch layer has verified AVX2 is available.
            unsafe { x86::avx2::clip(values, lo, hi, out) }
        }

        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => {
            // SAFETY: The dispatch layer has verified AVX-512F is available.
            unsafe { x86::avx512::clip(values, lo, hi, out) }
        }

        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // SAFETY: NEON is mandatory on all aarch64 targets.
            unsafe { aarch64::neon::clip(values, lo, hi, out) }
        }
    }
}

dispatch_map!(
    dispatch_rsqrt,
    scalar::rsqrt,
    x86::sse2::rsqrt,
    x86::avx2::rsqrt,
    x86::avx512::rsqrt,
    aarch64::neon::rsqrt
);

dispatch_map!(
    dispatch_exp,
    scalar::exp,
    x86::sse2::exp,
    x86::avx2::exp,
    x86::avx512::exp,
    aarch64::neon::exp
);
