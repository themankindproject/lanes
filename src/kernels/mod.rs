//! Kernel implementations for each backend.
//!
//! Each sub-module provides optimized implementations for its respective
//! SIMD instruction set. The dispatch functions in this module select the
//! correct implementation based on the detected [`Backend`].
//!
//! The scalar module is always available and serves as the reference
//! implementation and universal fallback.

#[cfg(feature = "alloc")]
pub(crate) mod erf;
pub(crate) mod exp;
pub(crate) mod hypot;
pub(crate) mod ln;
pub(crate) mod macros;
pub(crate) mod powi;
pub(crate) mod scalar;
pub(crate) mod sqrt;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm;

use crate::dispatch::Backend;

/// Identity pass-through used by `dispatch!` for non-Option returns.
/// Only needed where SIMD match arms exist (the scalar arm passes through
/// directly); on other targets every `$wrap` call site is `#[cfg]`'d out.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline]
fn id<T>(v: T) -> T {
    v
}

/// Convert jaccard counts `(intersection, union)` to the similarity
/// `intersection / union`, or `None` when the union is empty. Shared by
/// the scalar kernel and the `dispatch_jaccard` wrapper for SIMD backends
/// (which reduce to counts).
#[inline]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // counts ≤ 8·len; f32 is the documented precision; final f64→f32 is intentional
pub(crate) fn jaccard_similarity(counts: (usize, usize)) -> Option<f32> {
    let (intersection, union) = counts;
    if union == 0 {
        None
    } else {
        Some((intersection as f64 / union as f64) as f32)
    }
}

/// Allocate a `Vec<T>` of `len` elements without zero-initializing them.
///
/// The allocating wrappers (`math`, `ml`) build an output buffer and then
/// hand it to a kernel that writes every element. Building the buffer with
/// `vec![0.0; n]` first zero-fills the whole region, which the kernel then
/// immediately overwrites — pure wasted store traffic on memory-bound maps.
/// This helper skips that zero-fill: it reserves capacity for `len` elements
/// and sets the length, returning the buffer uninitialized so the caller's
/// kernel can write it directly.
///
/// # Safety invariant (upheld by every call site)
///
/// The returned buffer holds uninitialized memory. It is sound to expose
/// uninitialized `f32`/`f64` (no invalid bit patterns, no `Drop`), and every
/// caller immediately passes the buffer to a kernel that writes all `len`
/// elements before any element is read or the buffer is returned. The
/// single-pass maps (`simd_map!`/`simd_clip!`) and `rms_norm`/`log_softmax`
/// only store into `out`; the multi-pass kernels (`softmax`, `layer_norm`)
/// write every element in their first output pass and only read `out` in a
/// later pass. The single `set_len` below is the only `unsafe` operation and
/// is confined to this kernel-layer helper, keeping the algorithm layer
/// `#![forbid(unsafe_code)]`.
#[cfg(feature = "alloc")]
#[inline]
// `clippy::uninit_vec` flags `with_capacity` + `set_len` as a general hazard
// (exposing uninitialized memory). Here it is sound and intentional: the
// element type is `f32`/`f64` at every call site (no invalid bit patterns,
// no `Drop`), and each caller fully writes the buffer via a map kernel before
// it is read or returned. Verified under Miri with
// `-Zmiri-strict-provenance` (the CI flag). See the SAFETY note below.
#[allow(clippy::uninit_vec)]
pub(crate) fn alloc_uninit<T>(len: usize) -> alloc::vec::Vec<T> {
    let mut out = alloc::vec::Vec::with_capacity(len);
    // SAFETY: `with_capacity(len)` reserved room for `len` elements, so
    // `set_len(len)` stays within capacity. `T` is `f32`/`f64` at every call
    // site (no invalid bit patterns, no `Drop`), and each caller fully writes
    // the buffer before reading it, so no uninitialized value is ever
    // observed.
    unsafe {
        out.set_len(len);
    }
    out
}

/// Shared skeleton for every dispatch fn: one `match` on `Backend` with the
/// per-backend kernel paths as arguments.
///
/// The unified form absorbs the three old shapes (unary reduce, map, index
/// tracking) plus the two-input and parameterized kernels:
///
/// * `$name` — dispatch fn name.
/// * `[$( $pname:ident: $ptype:ty ),*]` — typed parameters declared on the
///   generated fn and passed positionally to every backend kernel, e.g.
///   `[values: &[f32]]`, `[values: &[f32], out: &mut [f32]]`,
///   `[a: &[f32], b: &[f32]]`, `[values: &[f32], lo: f32, hi: f32, out: &mut [f32]]`.
/// * `$ret` — return type.
/// * `$wrap` — a unary wrapper applied to the SIMD results (pass `id` for
///   `f32`, `Some` for `Option<f32>`, `()` for `()`). The scalar arm returns
///   `$scalar` directly, so scalar kernels that already return `Option<f32>`
///   pass through; `()` is the identity for `()` returns.
/// * `$alloc` — pass `alloc` to gate the generated fn on
///   `#[cfg(feature = "alloc")]` (maps returning a buffer), or omit for
///   always-compiled reductions.
macro_rules! dispatch {
    (
        $name:ident, [$( $pname:ident: $ptype:ty ),*], $ret:ty,
        $scalar:path, $sse2:path, $avx2:path, $avx512:path, $neon:path, $wasm:path,
        $wrap:tt, alloc
    ) => {
        #[cfg(feature = "alloc")]
        dispatch!(inner $name, [$( $pname: $ptype ),*], $ret,
            $scalar, $sse2, $avx2, $avx512, $neon, $wasm, $wrap);
    };

    (
        $name:ident, [$( $pname:ident: $ptype:ty ),*], $ret:ty,
        $scalar:path, $sse2:path, $avx2:path, $avx512:path, $neon:path, $wasm:path,
        $wrap:tt
    ) => {
        dispatch!(inner $name, [$( $pname: $ptype ),*], $ret,
            $scalar, $sse2, $avx2, $avx512, $neon, $wasm, $wrap);
    };

    (inner $name:ident, [$( $pname:ident: $ptype:ty ),*], $ret:ty,
        $scalar:path, $sse2:path, $avx2:path, $avx512:path, $neon:path, $wasm:path,
        $wrap:tt) => {
        #[inline]
        pub(crate) fn $name(backend: Backend, $( $pname: $ptype ),* ) -> $ret {
            match backend {
                Backend::Scalar => $scalar($( $pname ),*),

                #[cfg(target_arch = "x86_64")]
                Backend::Sse2 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    $wrap(unsafe { $sse2($( $pname ),*) })
                }

                #[cfg(target_arch = "x86_64")]
                Backend::Avx2 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    $wrap(unsafe { $avx2($( $pname ),*) })
                }

                #[cfg(target_arch = "x86_64")]
                Backend::Avx512 => {
                    // SAFETY: The dispatch layer (detect_best_backend) has
                    // verified this ISA is available on the current CPU.
                    $wrap(unsafe { $avx512($( $pname ),*) })
                }

                #[cfg(target_arch = "aarch64")]
                Backend::Neon => {
                    // SAFETY: NEON is mandatory on all aarch64 targets.
                    $wrap(unsafe { $neon($( $pname ),*) })
                }

                #[cfg(target_arch = "wasm32")]
                Backend::Wasm => {
                    // SAFETY: WASM SIMD128 is available when this backend is selected.
                    // WASM re-exports are safe fns (no target_feature), so keep the
                    // dispatch uniform without spurious unused_unsafe.
                    #[allow(unused_unsafe)]
                    {
                        $wrap(unsafe { $wasm($( $pname ),*) })
                    }
                }
            }
        }
    };
}

dispatch!(
    dispatch_sum,
    [values: &[f32]],
    f32,
    scalar::sum,
    x86::sse2::sum,
    x86::avx2::sum,
    x86::avx512::sum,
    aarch64::neon::sum,
    wasm::sum,
    id
);

dispatch!(
    dispatch_prod,
    [values: &[f32]],
    f32,
    scalar::prod,
    x86::sse2::prod,
    x86::avx2::prod,
    x86::avx512::prod,
    aarch64::neon::prod,
    wasm::prod,
    id
);

dispatch!(
    dispatch_min,
    [values: &[f32]],
    Option<f32>,
    scalar::min,
    x86::sse2::min,
    x86::avx2::min,
    x86::avx512::min,
    aarch64::neon::min,
    wasm::min,
    Some
);

dispatch!(
    dispatch_max,
    [values: &[f32]],
    Option<f32>,
    scalar::max,
    x86::sse2::max,
    x86::avx2::max,
    x86::avx512::max,
    aarch64::neon::max,
    wasm::max,
    Some
);

dispatch!(
    dispatch_sum_sq,
    [values: &[f32]],
    f32,
    scalar::sum_sq,
    x86::sse2::sum_sq,
    x86::avx2::sum_sq,
    x86::avx512::sum_sq,
    aarch64::neon::sum_sq,
    wasm::sum_sq,
    id
);

dispatch!(
    dispatch_l1_norm,
    [values: &[f32]],
    f32,
    scalar::l1_norm,
    x86::sse2::l1_norm,
    x86::avx2::l1_norm,
    x86::avx512::l1_norm,
    aarch64::neon::l1_norm,
    wasm::l1_norm,
    id
);

dispatch!(
    dispatch_max_norm,
    [values: &[f32]],
    Option<f32>,
    scalar::max_norm,
    x86::sse2::max_norm,
    x86::avx2::max_norm,
    x86::avx512::max_norm,
    aarch64::neon::max_norm,
    wasm::max_norm,
    Some
);

dispatch!(
    dispatch_argmax,
    [values: &[f32]],
    (f32, usize),
    scalar::argmax,
    x86::sse2::argmax,
    x86::avx2::argmax,
    x86::avx512::argmax,
    aarch64::neon::argmax,
    wasm::argmax,
    id
);

dispatch!(
    dispatch_argmin,
    [values: &[f32]],
    (f32, usize),
    scalar::argmin,
    x86::sse2::argmin,
    x86::avx2::argmin,
    x86::avx512::argmin,
    aarch64::neon::argmin,
    wasm::argmin,
    id
);

dispatch!(
    dispatch_count_zero,
    [values: &[f32]],
    usize,
    scalar::count_zero,
    x86::sse2::count_zero,
    x86::avx2::count_zero,
    x86::avx512::count_zero,
    aarch64::neon::count_zero,
    wasm::count_zero,
    id
);

dispatch!(
    dispatch_count_nan,
    [values: &[f32]],
    usize,
    scalar::count_nan,
    x86::sse2::count_nan,
    x86::avx2::count_nan,
    x86::avx512::count_nan,
    aarch64::neon::count_nan,
    wasm::count_nan,
    id
);

dispatch!(
    dispatch_count_infinite,
    [values: &[f32]],
    usize,
    scalar::count_infinite,
    x86::sse2::count_infinite,
    x86::avx2::count_infinite,
    x86::avx512::count_infinite,
    aarch64::neon::count_infinite,
    wasm::count_infinite,
    id
);

// Dispatch the dot product operation to the appropriate backend.
// Falls through to scalar for backends that are not available on the
// current compilation target.
dispatch!(
    dispatch_dot,
    [a: &[f32], b: &[f32]],
    f32,
    scalar::dot,
    x86::sse2::dot,
    x86::avx2::dot,
    x86::avx512::dot,
    aarch64::neon::dot,
    wasm::dot,
    id
);

dispatch!(
    dispatch_squared_distance,
    [a: &[f32], b: &[f32]],
    f32,
    scalar::squared_distance,
    x86::sse2::squared_distance,
    x86::avx2::squared_distance,
    x86::avx512::squared_distance,
    aarch64::neon::squared_distance,
    wasm::squared_distance,
    id
);

dispatch!(
    dispatch_kl_divergence,
    [p: &[f32], q: &[f32]],
    f32,
    scalar::kl_divergence,
    x86::sse2::kl_divergence,
    x86::avx2::kl_divergence,
    x86::avx512::kl_divergence,
    aarch64::neon::kl_divergence,
    wasm::kl_divergence,
    id
);

dispatch!(
    dispatch_js_divergence,
    [p: &[f32], q: &[f32]],
    f32,
    scalar::js_divergence,
    x86::sse2::js_divergence,
    x86::avx2::js_divergence,
    x86::avx512::js_divergence,
    aarch64::neon::js_divergence,
    wasm::js_divergence,
    id
);

dispatch!(
    dispatch_hamming,
    [a: &[u8], b: &[u8]],
    usize,
    scalar::hamming_popcount,
    x86::sse2::hamming_popcount,
    x86::avx2::hamming_popcount,
    x86::avx512::hamming_popcount,
    aarch64::neon::hamming_popcount,
    wasm::hamming_popcount,
    id
);

dispatch!(
    dispatch_jaccard,
    [a: &[u8], b: &[u8]],
    Option<f32>,
    scalar::jaccard,
    x86::sse2::jaccard_counts,
    x86::avx2::jaccard_counts,
    x86::avx512::jaccard_counts,
    aarch64::neon::jaccard_counts,
    wasm::jaccard_counts,
    jaccard_similarity
);

dispatch!(
    dispatch_dot_i8,
    [a: &[i8], b: &[i8]],
    i64,
    scalar::dot_i8,
    x86::sse2::dot_i8,
    x86::avx2::dot_i8,
    x86::avx512::dot_i8,
    aarch64::neon::dot_i8,
    wasm::dot_i8,
    id
);

dispatch!(
    dispatch_sum_i8,
    [values: &[i8]],
    i64,
    scalar::sum_i8,
    x86::sse2::sum_i8,
    x86::avx2::sum_i8,
    x86::avx512::sum_i8,
    aarch64::neon::sum_i8,
    wasm::sum_i8,
    id
);

dispatch!(
    dispatch_min_i8,
    [values: &[i8]],
    Option<i8>,
    scalar::min_i8,
    x86::sse2::min_i8,
    x86::avx2::min_i8,
    x86::avx512::min_i8,
    aarch64::neon::min_i8,
    wasm::min_i8,
    Some
);

dispatch!(
    dispatch_max_i8,
    [values: &[i8]],
    Option<i8>,
    scalar::max_i8,
    x86::sse2::max_i8,
    x86::avx2::max_i8,
    x86::avx512::max_i8,
    aarch64::neon::max_i8,
    wasm::max_i8,
    Some
);

dispatch!(
    dispatch_max_abs_i8,
    [values: &[i8]],
    Option<u8>,
    scalar::max_abs_i8,
    x86::sse2::max_abs_i8,
    x86::avx2::max_abs_i8,
    x86::avx512::max_abs_i8,
    aarch64::neon::max_abs_i8,
    wasm::max_abs_i8,
    Some
);

dispatch!(
    dispatch_count_zero_i8,
    [values: &[i8]],
    usize,
    scalar::count_zero_i8,
    x86::sse2::count_zero_i8,
    x86::avx2::count_zero_i8,
    x86::avx512::count_zero_i8,
    aarch64::neon::count_zero_i8,
    wasm::count_zero_i8,
    id
);

dispatch!(
    dispatch_l1_norm_i8,
    [values: &[i8]],
    i64,
    scalar::l1_norm_i8,
    x86::sse2::l1_norm_i8,
    x86::avx2::l1_norm_i8,
    x86::avx512::l1_norm_i8,
    aarch64::neon::l1_norm_i8,
    wasm::l1_norm_i8,
    id
);

dispatch!(
    dispatch_squared_distance_i8,
    [a: &[i8], b: &[i8]],
    i64,
    scalar::squared_distance_i8,
    x86::sse2::squared_distance_i8,
    x86::avx2::squared_distance_i8,
    x86::avx512::squared_distance_i8,
    aarch64::neon::squared_distance_i8,
    wasm::squared_distance_i8,
    id
);

dispatch!(
    dispatch_softmax,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::softmax,
    x86::sse2::softmax,
    x86::avx2::softmax,
    x86::avx512::softmax,
    aarch64::neon::softmax,
    wasm::softmax,
    id,
    alloc
);

dispatch!(
    dispatch_logsumexp,
    [values: &[f32]],
    f32,
    scalar::logsumexp,
    x86::sse2::logsumexp,
    x86::avx2::logsumexp,
    x86::avx512::logsumexp,
    aarch64::neon::logsumexp,
    wasm::logsumexp,
    id,
    alloc
);

dispatch!(
    dispatch_log_softmax,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::log_softmax,
    x86::sse2::log_softmax,
    x86::avx2::log_softmax,
    x86::avx512::log_softmax,
    aarch64::neon::log_softmax,
    wasm::log_softmax,
    id,
    alloc
);

dispatch!(
    dispatch_layer_norm,
    [values: &[f32], eps: f32, out: &mut [f32]],
    (),
    scalar::layer_norm,
    x86::sse2::layer_norm,
    x86::avx2::layer_norm,
    x86::avx512::layer_norm,
    aarch64::neon::layer_norm,
    wasm::layer_norm,
    id,
    alloc
);

dispatch!(
    dispatch_softplus,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::softplus,
    x86::sse2::softplus,
    x86::avx2::softplus,
    x86::avx512::softplus,
    aarch64::neon::softplus,
    wasm::softplus,
    id,
    alloc
);

dispatch!(
    dispatch_sigmoid,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::sigmoid,
    x86::sse2::sigmoid,
    x86::avx2::sigmoid,
    x86::avx512::sigmoid,
    aarch64::neon::sigmoid,
    wasm::sigmoid,
    id,
    alloc
);

dispatch!(
    dispatch_silu,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::silu,
    x86::sse2::silu,
    x86::avx2::silu,
    x86::avx512::silu,
    aarch64::neon::silu,
    wasm::silu,
    id,
    alloc
);

dispatch!(
    dispatch_gelu,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::gelu,
    x86::sse2::gelu,
    x86::avx2::gelu,
    x86::avx512::gelu,
    aarch64::neon::gelu,
    wasm::gelu,
    id,
    alloc
);

dispatch!(
    dispatch_relu,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::relu,
    x86::sse2::relu,
    x86::avx2::relu,
    x86::avx512::relu,
    aarch64::neon::relu,
    wasm::relu,
    id,
    alloc
);

dispatch!(
    dispatch_tanh,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::tanh,
    x86::sse2::tanh,
    x86::avx2::tanh,
    x86::avx512::tanh,
    aarch64::neon::tanh,
    wasm::tanh,
    id,
    alloc
);

dispatch!(
    dispatch_rms_norm,
    [values: &[f32], eps: f32, out: &mut [f32]],
    (),
    scalar::rms_norm,
    x86::sse2::rms_norm,
    x86::avx2::rms_norm,
    x86::avx512::rms_norm,
    aarch64::neon::rms_norm,
    wasm::rms_norm,
    id,
    alloc
);

dispatch!(
    dispatch_sqrt,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::sqrt,
    x86::sse2::sqrt,
    x86::avx2::sqrt,
    x86::avx512::sqrt,
    aarch64::neon::sqrt,
    wasm::sqrt,
    id,
    alloc
);

// Dispatch the elementwise `clip` map (`clamp(x, lo, hi)`) to the
// appropriate backend. Gated on `alloc`: its only public caller
// (`lanes::math::clip`) returns a `Vec`.
dispatch!(
    dispatch_clip,
    [values: &[f32], lo: f32, hi: f32, out: &mut [f32]],
    (),
    scalar::clip,
    x86::sse2::clip,
    x86::avx2::clip,
    x86::avx512::clip,
    aarch64::neon::clip,
    wasm::clip,
    id,
    alloc
);

dispatch!(
    dispatch_abs_sub,
    [a: &[f32], b: &[f32], out: &mut [f32]],
    (),
    scalar::abs_sub,
    x86::sse2::abs_sub,
    x86::avx2::abs_sub,
    x86::avx512::abs_sub,
    aarch64::neon::abs_sub,
    wasm::abs_sub,
    id,
    alloc
);

dispatch!(
    dispatch_hypot,
    [a: &[f32], b: &[f32], out: &mut [f32]],
    (),
    scalar::hypot,
    x86::sse2::hypot,
    x86::avx2::hypot,
    x86::avx512::hypot,
    aarch64::neon::hypot,
    wasm::hypot,
    id,
    alloc
);

dispatch!(
    dispatch_powi,
    [values: &[f32], n: i32, out: &mut [f32]],
    (),
    scalar::powi,
    x86::sse2::powi,
    x86::avx2::powi,
    x86::avx512::powi,
    aarch64::neon::powi,
    wasm::powi,
    id,
    alloc
);

dispatch!(
    dispatch_rsqrt,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::rsqrt,
    x86::sse2::rsqrt,
    x86::avx2::rsqrt,
    x86::avx512::rsqrt,
    aarch64::neon::rsqrt,
    wasm::rsqrt,
    id,
    alloc
);

dispatch!(
    dispatch_exp,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::exp,
    x86::sse2::exp,
    x86::avx2::exp,
    x86::avx512::exp,
    aarch64::neon::exp,
    wasm::exp,
    id,
    alloc
);

dispatch!(
    dispatch_ln,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::ln,
    x86::sse2::ln,
    x86::avx2::ln,
    x86::avx512::ln,
    aarch64::neon::ln,
    wasm::ln,
    id,
    alloc
);

// ===========================================================================
// f64 (double-precision) dispatch. Same `dispatch!` skeleton — the macro is
// already type-generic, so each entry just wires the f64 kernels.
// ===========================================================================

dispatch!(
    dispatch_center_f32,
    [values: &[f32], mean: f32, out: &mut [f32]],
    (),
    scalar::center_f32,
    x86::sse2::center_f32,
    x86::avx2::center_f32,
    x86::avx512::center_f32,
    aarch64::neon::center_f32,
    wasm::center_f32,
    id
);

dispatch!(
    dispatch_center_f64,
    [values: &[f64], mean: f64, out: &mut [f64]],
    (),
    scalar::center_f64,
    x86::sse2::center_f64,
    x86::avx2::center_f64,
    x86::avx512::center_f64,
    aarch64::neon::center_f64,
    wasm::center_f64,
    id
);

dispatch!(
    dispatch_variance_fused_f32,
    [values: &[f32], mean: f32],
    f32,
    scalar::variance_fused_f32,
    x86::sse2::variance_fused_f32,
    x86::avx2::variance_fused_f32,
    x86::avx512::variance_fused_f32,
    aarch64::neon::variance_fused_f32,
    wasm::variance_fused_f32,
    id
);

dispatch!(
    dispatch_variance_fused_f64,
    [values: &[f64], mean: f64],
    f64,
    scalar::variance_fused_f64,
    x86::sse2::variance_fused_f64,
    x86::avx2::variance_fused_f64,
    x86::avx512::variance_fused_f64,
    aarch64::neon::variance_fused_f64,
    wasm::variance_fused_f64,
    id
);

dispatch!(
dispatch_sum_f64,
    [values: &[f64]],
    f64,
    scalar::sum_f64,
    x86::sse2::sum_f64,
    x86::avx2::sum_f64,
    x86::avx512::sum_f64,
    aarch64::neon::sum_f64,
    wasm::sum_f64,
    id
);

dispatch!(
    dispatch_prod_f64,
    [values: &[f64]],
    f64,
    scalar::prod_f64,
    x86::sse2::prod_f64,
    x86::avx2::prod_f64,
    x86::avx512::prod_f64,
    aarch64::neon::prod_f64,
    wasm::prod_f64,
    id
);

dispatch!(
    dispatch_min_f64,
    [values: &[f64]],
    Option<f64>,
    scalar::min_f64,
    x86::sse2::min_f64,
    x86::avx2::min_f64,
    x86::avx512::min_f64,
    aarch64::neon::min_f64,
    wasm::min_f64,
    Some
);

dispatch!(
    dispatch_max_f64,
    [values: &[f64]],
    Option<f64>,
    scalar::max_f64,
    x86::sse2::max_f64,
    x86::avx2::max_f64,
    x86::avx512::max_f64,
    aarch64::neon::max_f64,
    wasm::max_f64,
    Some
);

dispatch!(
    dispatch_sum_sq_f64,
    [values: &[f64]],
    f64,
    scalar::sum_sq_f64,
    x86::sse2::sum_sq_f64,
    x86::avx2::sum_sq_f64,
    x86::avx512::sum_sq_f64,
    aarch64::neon::sum_sq_f64,
    wasm::sum_sq_f64,
    id
);

dispatch!(
    dispatch_l1_norm_f64,
    [values: &[f64]],
    f64,
    scalar::l1_norm_f64,
    x86::sse2::l1_norm_f64,
    x86::avx2::l1_norm_f64,
    x86::avx512::l1_norm_f64,
    aarch64::neon::l1_norm_f64,
    wasm::l1_norm_f64,
    id
);

dispatch!(
    dispatch_max_norm_f64,
    [values: &[f64]],
    Option<f64>,
    scalar::max_norm_f64,
    x86::sse2::max_norm_f64,
    x86::avx2::max_norm_f64,
    x86::avx512::max_norm_f64,
    aarch64::neon::max_norm_f64,
    wasm::max_norm_f64,
    Some
);

dispatch!(
    dispatch_argmax_f64,
    [values: &[f64]],
    (f64, usize),
    scalar::argmax_f64,
    x86::sse2::argmax_f64,
    x86::avx2::argmax_f64,
    x86::avx512::argmax_f64,
    aarch64::neon::argmax_f64,
    wasm::argmax_f64,
    id
);

dispatch!(
    dispatch_argmin_f64,
    [values: &[f64]],
    (f64, usize),
    scalar::argmin_f64,
    x86::sse2::argmin_f64,
    x86::avx2::argmin_f64,
    x86::avx512::argmin_f64,
    aarch64::neon::argmin_f64,
    wasm::argmin_f64,
    id
);

dispatch!(
    dispatch_count_zero_f64,
    [values: &[f64]],
    usize,
    scalar::count_zero_f64,
    x86::sse2::count_zero_f64,
    x86::avx2::count_zero_f64,
    x86::avx512::count_zero_f64,
    aarch64::neon::count_zero_f64,
    wasm::count_zero_f64,
    id
);

dispatch!(
    dispatch_count_nan_f64,
    [values: &[f64]],
    usize,
    scalar::count_nan_f64,
    x86::sse2::count_nan_f64,
    x86::avx2::count_nan_f64,
    x86::avx512::count_nan_f64,
    aarch64::neon::count_nan_f64,
    wasm::count_nan_f64,
    id
);

dispatch!(
    dispatch_count_infinite_f64,
    [values: &[f64]],
    usize,
    scalar::count_infinite_f64,
    x86::sse2::count_infinite_f64,
    x86::avx2::count_infinite_f64,
    x86::avx512::count_infinite_f64,
    aarch64::neon::count_infinite_f64,
    wasm::count_infinite_f64,
    id
);

dispatch!(
    dispatch_dot_f64,
    [a: &[f64], b: &[f64]],
    f64,
    scalar::dot_f64,
    x86::sse2::dot_f64,
    x86::avx2::dot_f64,
    x86::avx512::dot_f64,
    aarch64::neon::dot_f64,
    wasm::dot_f64,
    id
);

dispatch!(
    dispatch_squared_distance_f64,
    [a: &[f64], b: &[f64]],
    f64,
    scalar::squared_distance_f64,
    x86::sse2::squared_distance_f64,
    x86::avx2::squared_distance_f64,
    x86::avx512::squared_distance_f64,
    aarch64::neon::squared_distance_f64,
    wasm::squared_distance_f64,
    id
);

dispatch!(
    dispatch_kl_divergence_f64,
    [p: &[f64], q: &[f64]],
    f64,
    scalar::kl_divergence_f64,
    x86::sse2::kl_divergence_f64,
    x86::avx2::kl_divergence_f64,
    x86::avx512::kl_divergence_f64,
    aarch64::neon::kl_divergence_f64,
    wasm::kl_divergence_f64,
    id
);

dispatch!(
    dispatch_js_divergence_f64,
    [p: &[f64], q: &[f64]],
    f64,
    scalar::js_divergence_f64,
    x86::sse2::js_divergence_f64,
    x86::avx2::js_divergence_f64,
    x86::avx512::js_divergence_f64,
    aarch64::neon::js_divergence_f64,
    wasm::js_divergence_f64,
    id
);

dispatch!(
    dispatch_sqrt_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::sqrt_f64,
    x86::sse2::sqrt_f64,
    x86::avx2::sqrt_f64,
    x86::avx512::sqrt_f64,
    aarch64::neon::sqrt_f64,
    wasm::sqrt_f64,
    id,
    alloc
);

dispatch!(
    dispatch_rsqrt_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::rsqrt_f64,
    x86::sse2::rsqrt_f64,
    x86::avx2::rsqrt_f64,
    x86::avx512::rsqrt_f64,
    aarch64::neon::rsqrt_f64,
    wasm::rsqrt_f64,
    id,
    alloc
);

dispatch!(
    dispatch_clip_f64,
    [values: &[f64], lo: f64, hi: f64, out: &mut [f64]],
    (),
    scalar::clip_f64,
    x86::sse2::clip_f64,
    x86::avx2::clip_f64,
    x86::avx512::clip_f64,
    aarch64::neon::clip_f64,
    wasm::clip_f64,
    id,
    alloc
);

dispatch!(
    dispatch_abs_sub_f64,
    [a: &[f64], b: &[f64], out: &mut [f64]],
    (),
    scalar::abs_sub_f64,
    x86::sse2::abs_sub_f64,
    x86::avx2::abs_sub_f64,
    x86::avx512::abs_sub_f64,
    aarch64::neon::abs_sub_f64,
    wasm::abs_sub_f64,
    id,
    alloc
);

dispatch!(
    dispatch_hypot_f64,
    [a: &[f64], b: &[f64], out: &mut [f64]],
    (),
    scalar::hypot_f64,
    x86::sse2::hypot_f64,
    x86::avx2::hypot_f64,
    x86::avx512::hypot_f64,
    aarch64::neon::hypot_f64,
    wasm::hypot_f64,
    id,
    alloc
);

dispatch!(
    dispatch_powi_f64,
    [values: &[f64], n: i32, out: &mut [f64]],
    (),
    scalar::powi_f64,
    x86::sse2::powi_f64,
    x86::avx2::powi_f64,
    x86::avx512::powi_f64,
    aarch64::neon::powi_f64,
    wasm::powi_f64,
    id,
    alloc
);

dispatch!(
    dispatch_exp_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::exp_f64,
    x86::sse2::exp_f64,
    x86::avx2::exp_f64,
    x86::avx512::exp_f64,
    aarch64::neon::exp_f64,
    wasm::exp_f64,
    id,
    alloc
);

dispatch!(
    dispatch_erf,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::erf,
    x86::sse2::erf,
    x86::avx2::erf,
    x86::avx512::erf,
    aarch64::neon::erf,
    wasm::erf,
    id,
    alloc
);

dispatch!(
    dispatch_erfc,
    [values: &[f32], out: &mut [f32]],
    (),
    scalar::erfc,
    x86::sse2::erfc,
    x86::avx2::erfc,
    x86::avx512::erfc,
    aarch64::neon::erfc,
    wasm::erfc,
    id,
    alloc
);

dispatch!(
    dispatch_erf_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::erf_f64,
    x86::sse2::erf_f64,
    x86::avx2::erf_f64,
    x86::avx512::erf_f64,
    aarch64::neon::erf_f64,
    wasm::erf_f64,
    id,
    alloc
);

dispatch!(
    dispatch_erfc_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::erfc_f64,
    x86::sse2::erfc_f64,
    x86::avx2::erfc_f64,
    x86::avx512::erfc_f64,
    aarch64::neon::erfc_f64,
    wasm::erfc_f64,
    id,
    alloc
);

dispatch!(
    dispatch_ln_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::ln_f64,
    x86::sse2::ln_f64,
    x86::avx2::ln_f64,
    x86::avx512::ln_f64,
    aarch64::neon::ln_f64,
    wasm::ln_f64,
    id,
    alloc
);

dispatch!(
    dispatch_softmax_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::softmax_f64,
    x86::sse2::softmax_f64,
    x86::avx2::softmax_f64,
    x86::avx512::softmax_f64,
    aarch64::neon::softmax_f64,
    wasm::softmax_f64,
    id,
    alloc
);

dispatch!(
    dispatch_logsumexp_f64,
    [values: &[f64]],
    f64,
    scalar::logsumexp_f64,
    x86::sse2::logsumexp_f64,
    x86::avx2::logsumexp_f64,
    x86::avx512::logsumexp_f64,
    aarch64::neon::logsumexp_f64,
    wasm::logsumexp_f64,
    id,
    alloc
);

dispatch!(
    dispatch_log_softmax_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::log_softmax_f64,
    x86::sse2::log_softmax_f64,
    x86::avx2::log_softmax_f64,
    x86::avx512::log_softmax_f64,
    aarch64::neon::log_softmax_f64,
    wasm::log_softmax_f64,
    id,
    alloc
);

dispatch!(
    dispatch_layer_norm_f64,
    [values: &[f64], eps: f64, out: &mut [f64]],
    (),
    scalar::layer_norm_f64,
    x86::sse2::layer_norm_f64,
    x86::avx2::layer_norm_f64,
    x86::avx512::layer_norm_f64,
    aarch64::neon::layer_norm_f64,
    wasm::layer_norm_f64,
    id,
    alloc
);

dispatch!(
    dispatch_softplus_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::softplus_f64,
    x86::sse2::softplus_f64,
    x86::avx2::softplus_f64,
    x86::avx512::softplus_f64,
    aarch64::neon::softplus_f64,
    wasm::softplus_f64,
    id,
    alloc
);

dispatch!(
    dispatch_sigmoid_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::sigmoid_f64,
    x86::sse2::sigmoid_f64,
    x86::avx2::sigmoid_f64,
    x86::avx512::sigmoid_f64,
    aarch64::neon::sigmoid_f64,
    wasm::sigmoid_f64,
    id,
    alloc
);

dispatch!(
    dispatch_silu_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::silu_f64,
    x86::sse2::silu_f64,
    x86::avx2::silu_f64,
    x86::avx512::silu_f64,
    aarch64::neon::silu_f64,
    wasm::silu_f64,
    id,
    alloc
);

dispatch!(
    dispatch_gelu_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::gelu_f64,
    x86::sse2::gelu_f64,
    x86::avx2::gelu_f64,
    x86::avx512::gelu_f64,
    aarch64::neon::gelu_f64,
    wasm::gelu_f64,
    id,
    alloc
);

dispatch!(
    dispatch_relu_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::relu_f64,
    x86::sse2::relu_f64,
    x86::avx2::relu_f64,
    x86::avx512::relu_f64,
    aarch64::neon::relu_f64,
    wasm::relu_f64,
    id,
    alloc
);

dispatch!(
    dispatch_tanh_f64,
    [values: &[f64], out: &mut [f64]],
    (),
    scalar::tanh_f64,
    x86::sse2::tanh_f64,
    x86::avx2::tanh_f64,
    x86::avx512::tanh_f64,
    aarch64::neon::tanh_f64,
    wasm::tanh_f64,
    id,
    alloc
);

dispatch!(
    dispatch_rms_norm_f64,
    [values: &[f64], eps: f64, out: &mut [f64]],
    (),
    scalar::rms_norm_f64,
    x86::sse2::rms_norm_f64,
    x86::avx2::rms_norm_f64,
    x86::avx512::rms_norm_f64,
    aarch64::neon::rms_norm_f64,
    wasm::rms_norm_f64,
    id,
    alloc
);

// For now all backends fall through to the scalar implementation.
// SIMD-accelerated convert kernels will be added later.

/// Dispatch f16 → f32 conversion.
#[inline]
pub(crate) fn dispatch_f16_to_f32(backend: Backend, input: &[u16], output: &mut [f32]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => unsafe { x86::avx512::f16_to_f32(input, output) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { x86::avx2::f16_to_f32(input, output) },
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => unsafe { x86::sse2::f16_to_f32(input, output) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { aarch64::neon::f16_to_f32(input, output) },
        _ => scalar::f16_to_f32(input, output),
    }
}

/// Dispatch f32 → f16 conversion.
#[inline]
pub(crate) fn dispatch_f32_to_f16(backend: Backend, input: &[f32], output: &mut [u16]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => unsafe { x86::avx512::f32_to_f16(input, output) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { x86::avx2::f32_to_f16(input, output) },
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => unsafe { x86::sse2::f32_to_f16(input, output) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { aarch64::neon::f32_to_f16(input, output) },
        _ => scalar::f32_to_f16(input, output),
    }
}

/// Dispatch bf16 → f32 conversion.
#[inline]
pub(crate) fn dispatch_bf16_to_f32(backend: Backend, input: &[u16], output: &mut [f32]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => unsafe { x86::avx512::bf16_to_f32(input, output) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { x86::avx2::bf16_to_f32(input, output) },
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => unsafe { x86::sse2::bf16_to_f32(input, output) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { aarch64::neon::bf16_to_f32(input, output) },
        _ => scalar::bf16_to_f32(input, output),
    }
}

/// Dispatch f32 → bf16 conversion.
#[inline]
pub(crate) fn dispatch_f32_to_bf16(backend: Backend, input: &[f32], output: &mut [u16]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => unsafe { x86::avx512::f32_to_bf16(input, output) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { x86::avx2::f32_to_bf16(input, output) },
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => unsafe { x86::sse2::f32_to_bf16(input, output) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { aarch64::neon::f32_to_bf16(input, output) },
        _ => scalar::f32_to_bf16(input, output),
    }
}

/// Dispatch f16 dot product (computed in f32).
#[inline]
pub(crate) fn dispatch_dot_f16(backend: Backend, a: &[u16], b: &[u16]) -> f32 {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => unsafe { x86::avx512::dot_f16(a, b) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { x86::avx2::dot_f16(a, b) },
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => unsafe { x86::sse2::dot_f16(a, b) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { aarch64::neon::dot_f16(a, b) },
        _ => scalar::dot_f16(a, b),
    }
}

/// Dispatch bf16 dot product (computed in f32).
#[inline]
pub(crate) fn dispatch_dot_bf16(backend: Backend, a: &[u16], b: &[u16]) -> f32 {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => unsafe { x86::avx512::dot_bf16(a, b) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { x86::avx2::dot_bf16(a, b) },
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => unsafe { x86::sse2::dot_bf16(a, b) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { aarch64::neon::dot_bf16(a, b) },
        _ => scalar::dot_bf16(a, b),
    }
}

/// In-place bitonic sort by total order (no alloc).
///
/// `n ∈ {8,16,32}` dispatches the optimal network (Batcher [1],
/// Dobbelaere [2] via Intel [3] `xss-optimal-networks.hpp`); other lengths
/// fall back to `sort_unstable_by(total_cmp)`. Deterministic ascending
/// `total_cmp` (NaN last, `-0 < +0`). No heap allocation.
///
/// References: see `crate::algorithms::sort` (added with citations).
#[inline]
pub(crate) fn dispatch_bitonic_sort_f32(backend: Backend, values: &mut [f32]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => {
            #[allow(unused_unsafe)]
            unsafe {
                x86::avx512::bitonic_sort_f32(values);
            }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            #[allow(unused_unsafe)]
            unsafe {
                x86::avx2::bitonic_sort_f32(values);
            }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => {
            #[allow(unused_unsafe)]
            unsafe {
                x86::sse2::bitonic_sort_f32(values);
            }
        }
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            #[allow(unused_unsafe)]
            unsafe {
                aarch64::neon::bitonic_sort_f32(values);
            }
        }
        _ => scalar::bitonic_sort_f32(values),
    }
}

#[inline]
pub(crate) fn dispatch_bitonic_sort_f64(backend: Backend, values: &mut [f64]) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => {
            #[allow(unused_unsafe)]
            unsafe {
                x86::avx512::bitonic_sort_f64(values);
            }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            #[allow(unused_unsafe)]
            unsafe {
                x86::avx2::bitonic_sort_f64(values);
            }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => {
            #[allow(unused_unsafe)]
            unsafe {
                x86::sse2::bitonic_sort_f64(values);
            }
        }
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            #[allow(unused_unsafe)]
            unsafe {
                aarch64::neon::bitonic_sort_f64(values);
            }
        }
        _ => scalar::bitonic_sort_f64(values),
    }
}
