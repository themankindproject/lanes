//! AVX-512F (512-bit) SIMD kernel implementations for x86-64.
//!
//! Processes 16 `f32` values per iteration. Requires the `avx512f` CPU
//! feature; the caller (dispatch layer) must verify availability before
//! invoking any function here — see `platform::supports`.
//!
//! Floating-point semantics: sums/dot products accumulate in 16-lane
//! vectors and combine via a horizontal reduction, so the reduction order
//! differs from the scalar kernels. `min`/`max` follow the AVX-512
//! `vminps`/`vmaxps` hardware semantics (a NaN present in the data
//! propagates), unlike the scalar `f32::min`/`f32::max` semantics. For
//! NaN-free inputs the SIMD and scalar results agree exactly.
//!
//! Note: newer stdarch releases declare most intrinsics `safe fn`; older
//! ones (the MSRV toolchain) declare them `unsafe fn`. The explicit
//! `unsafe {}` blocks keep this file compiling on both, hence the allow.

#![allow(
    clippy::many_single_char_names, // intrinsic style: v, n, r, p are conventional
    clippy::excessive_precision,    // ln2/log2e split constants are full-precision
    clippy::approx_constant,
    clippy::cast_lossless,
)]
#![allow(unused_unsafe)]

#[allow(clippy::wildcard_imports)]
use core::arch::x86_64::*;

// Bitwise ops on float vectors, routed through the integer domain:
// `_mm512_and_ps` / `_mm512_or_ps` / `_mm512_xor_ps` / `_mm512_andnot_ps`
// (and the `_pd` twins) are AVX-512DQ, not AVX-512F. Dispatch gates this
// backend on `avx512f` alone (see `platform::supports`), so the DQ forms
// would SIGILL on F-only parts (e.g. Knights Landing). The `_si512`
// integer versions are AVX-512F.
#[cfg(feature = "alloc")]
#[inline]
fn and_ps(a: __m512, b: __m512) -> __m512 {
    unsafe {
        _mm512_castsi512_ps(_mm512_and_si512(
            _mm512_castps_si512(a),
            _mm512_castps_si512(b),
        ))
    }
}
#[cfg(feature = "alloc")]
#[inline]
fn or_ps(a: __m512, b: __m512) -> __m512 {
    unsafe {
        _mm512_castsi512_ps(_mm512_or_si512(
            _mm512_castps_si512(a),
            _mm512_castps_si512(b),
        ))
    }
}
#[cfg(feature = "alloc")]
#[inline]
fn xor_ps(a: __m512, b: __m512) -> __m512 {
    unsafe {
        _mm512_castsi512_ps(_mm512_xor_si512(
            _mm512_castps_si512(a),
            _mm512_castps_si512(b),
        ))
    }
}
#[cfg(feature = "alloc")]
#[inline]
fn andnot_ps(a: __m512, b: __m512) -> __m512 {
    unsafe {
        _mm512_castsi512_ps(_mm512_andnot_si512(
            _mm512_castps_si512(a),
            _mm512_castps_si512(b),
        ))
    }
}
#[cfg(feature = "alloc")]
#[inline]
fn and_pd(a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        _mm512_castsi512_pd(_mm512_and_si512(
            _mm512_castpd_si512(a),
            _mm512_castpd_si512(b),
        ))
    }
}
#[cfg(feature = "alloc")]
#[inline]
fn or_pd(a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        _mm512_castsi512_pd(_mm512_or_si512(
            _mm512_castpd_si512(a),
            _mm512_castpd_si512(b),
        ))
    }
}
#[cfg(feature = "alloc")]
#[inline]
fn xor_pd(a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        _mm512_castsi512_pd(_mm512_xor_si512(
            _mm512_castpd_si512(a),
            _mm512_castpd_si512(b),
        ))
    }
}
#[cfg(feature = "alloc")]
#[inline]
fn andnot_pd(a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        _mm512_castsi512_pd(_mm512_andnot_si512(
            _mm512_castpd_si512(a),
            _mm512_castpd_si512(b),
        ))
    }
}

// Sum reduction: accumulate 16-wide, horizontal-sum, scalar tail.
crate::simd_reduce!(
    sum,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setzero_ps(),
    _mm512_add_ps,
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |r, v| r + v,
    _mm512_add_ps
);

// Product reduction: 16-wide multiply, scalar-multiply tail.
crate::simd_reduce!(
    prod,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_set1_ps(1.0),
    _mm512_mul_ps,
    |v| unsafe { _mm512_reduce_mul_ps(v) },
    |r, v| r * v
);

crate::simd_minmax!(
    min,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_set1_ps(f32::INFINITY),
    _mm512_min_ps,
    |v| unsafe { _mm512_reduce_min_ps(v) },
    f32::min,
    |v: __m512| unsafe { _mm512_cmp_ps_mask(v, v, _CMP_ORD_Q) != 0 },
    |v: __m512| unsafe {
        let nan = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
        _mm512_mask_blend_ps(nan, v, _mm512_set1_ps(f32::INFINITY))
    },
    |v: f32| !v.is_nan(),
    |r: f32, saw_real: bool| if saw_real { r } else { f32::NAN }
);

crate::simd_minmax!(
    max,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_set1_ps(f32::NEG_INFINITY),
    _mm512_max_ps,
    |v| unsafe { _mm512_reduce_max_ps(v) },
    f32::max,
    |v: __m512| unsafe { _mm512_cmp_ps_mask(v, v, _CMP_ORD_Q) != 0 },
    |v: __m512| unsafe {
        let nan = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
        _mm512_mask_blend_ps(nan, v, _mm512_set1_ps(f32::NEG_INFINITY))
    },
    |v: f32| !v.is_nan(),
    |r: f32, saw_real: bool| if saw_real { r } else { f32::NAN }
);
// Sum of squares: 16-wide multiply-accumulate (acc += v*v).
crate::simd_reduce!(
    sum_sq,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setzero_ps(),
    |acc: __m512, v: __m512| _mm512_add_ps(acc, _mm512_mul_ps(v, v)),
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |r: f32, v: f32| r + v * v,
    _mm512_add_ps
);

// L1 norm: sum of absolute values.
crate::simd_reduce!(
    l1_norm,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setzero_ps(),
    |acc: __m512, v: __m512| unsafe { _mm512_add_ps(acc, _mm512_abs_ps(v)) },
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |r: f32, v: f32| r + v.abs(),
    _mm512_add_ps
);

crate::simd_minmax!(
    max_norm,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_set1_ps(0.0),
    |acc: __m512, v: __m512| unsafe { _mm512_max_ps(acc, _mm512_abs_ps(v)) },
    |v| unsafe { _mm512_reduce_max_ps(v) },
    |r: f32, v: f32| f32::max(r, v.abs()),
    |v: __m512| unsafe { _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) != 0 },
    |v: __m512| unsafe {
        let nan = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
        _mm512_mask_blend_ps(nan, v, _mm512_setzero_ps())
    },
    |v: f32| v.is_nan(),
    |r: f32, saw_nan: bool| if saw_nan { f32::NAN } else { r }
);

// Dot product: 16-wide multiply-accumulate (mul+add; AVX-512F has no FMA).
crate::simd_reduce2!(
    dot,
    f32,
    ["avx512f"],
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setzero_ps(),
    |acc, va, vb| _mm512_add_ps(acc, _mm512_mul_ps(va, vb)),
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |r, a, b| r + a * b,
    _mm512_add_ps
);

// Softmax: 3-pass map (max → exp+sum → scale). exp is per-lane scalar.
// Uses the crate's `no_std` `exp`, so available in all builds.
crate::simd_softmax!(
    softmax,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    _mm512_max_ps,
    _mm512_sub_ps,
    |v| unsafe { vexp_512(v) },
    _mm512_add_ps,
    _mm512_mul_ps,
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |v| unsafe { _mm512_reduce_max_ps(v) },
    |s| unsafe { _mm512_set1_ps(s) },
    |x: f32| crate::kernels::exp::exp(x)
);

// Logsumexp: two-pass scalar-returning reduction (max → Σexp → max+ln).
crate::simd_logsumexp!(
    logsumexp,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_max_ps,
    _mm512_sub_ps,
    |v| unsafe { vexp_512(v) },
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |v| unsafe { _mm512_reduce_max_ps(v) },
    |s| unsafe { _mm512_set1_ps(s) },
    |x: f32| crate::kernels::exp::exp(x),
    crate::kernels::ln::ln
);

// Log-softmax: three-pass map (max → Σexp → (x-m)-ln(sum)), 0-alloc.
crate::simd_log_softmax!(
    log_softmax,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    _mm512_max_ps,
    _mm512_sub_ps,
    |v| unsafe { vexp_512(v) },
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |v| unsafe { _mm512_reduce_max_ps(v) },
    |s| unsafe { _mm512_set1_ps(s) },
    |x: f32| crate::kernels::exp::exp(x),
    crate::kernels::ln::ln
);

// Layer norm: three-pass (mean → center+Σsq → scale), 0-alloc.
crate::simd_layer_norm!(
    layer_norm,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    _mm512_add_ps,
    _mm512_sub_ps,
    _mm512_setzero_ps(),
    |acc: __m512, v: __m512| _mm512_add_ps(acc, _mm512_mul_ps(v, v)),
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |s| unsafe { _mm512_set1_ps(s) },
    |v, inv| _mm512_mul_ps(v, _mm512_set1_ps(inv)),
    crate::kernels::sqrt::sqrt
);

crate::simd_map!(
    sigmoid,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe {
        // Saturated fast path: all lanes already at 0/1 (skip the exp).
        let pos = _mm512_cmp_ps_mask(v, _mm512_set1_ps(16.64), _CMP_GT_OQ);
        let neg = _mm512_cmp_ps_mask(v, _mm512_set1_ps(-88.73), _CMP_LT_OQ);
        if pos | neg == 0xFFFF {
            return _mm512_maskz_mov_ps(pos, _mm512_set1_ps(1.0));
        }
        _mm512_div_ps(
            _mm512_set1_ps(1.0),
            _mm512_add_ps(
                _mm512_set1_ps(1.0),
                vexp_512(xor_ps(v, _mm512_castsi512_ps(_mm512_set1_epi32(i32::MIN)))),
            ),
        )
    },
    |x: f32| {
        if x > 16.64 {
            1.0
        } else if x < -88.73 {
            0.0
        } else {
            1.0 / (1.0 + crate::kernels::exp::exp(-x))
        }
    }
);
crate::simd_map!(
    silu,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe {
        // Saturated fast path: silu(x) = x for x > 16.64, 0 for x < -88.
        let pos = _mm512_cmp_ps_mask(v, _mm512_set1_ps(16.64), _CMP_GT_OQ);
        let neg = _mm512_cmp_ps_mask(v, _mm512_set1_ps(-88.73), _CMP_LT_OQ);
        if pos | neg == 0xFFFF {
            return _mm512_maskz_mov_ps(pos, v);
        }
        _mm512_div_ps(
            v,
            _mm512_add_ps(
                _mm512_set1_ps(1.0),
                vexp_512(xor_ps(v, _mm512_castsi512_ps(_mm512_set1_epi32(i32::MIN)))),
            ),
        )
    },
    |x: f32| {
        if x > 16.64 {
            x
        } else if x < -88.73 {
            0.0
        } else {
            x / (1.0 + crate::kernels::exp::exp(-x))
        }
    }
);
crate::simd_map!(
    gelu,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe {
        // Saturated fast path: gelu(x) = x for x > 7.0, 0 for x < -7.0.
        let pos = _mm512_cmp_ps_mask(v, _mm512_set1_ps(7.0), _CMP_GT_OQ);
        let neg = _mm512_cmp_ps_mask(v, _mm512_set1_ps(-7.0), _CMP_LT_OQ);
        if pos | neg == 0xFFFF {
            return _mm512_maskz_mov_ps(pos, v);
        }
        let x2 = _mm512_mul_ps(v, v);
        let x3 = _mm512_mul_ps(x2, v);
        let z = _mm512_mul_ps(
            _mm512_set1_ps(0.797_884_6),
            _mm512_add_ps(v, _mm512_mul_ps(_mm512_set1_ps(0.044_715), x3)),
        );
        let e = vexp_512(_mm512_add_ps(z, z));
        let tanh_z = _mm512_sub_ps(
            _mm512_set1_ps(1.0),
            _mm512_div_ps(_mm512_set1_ps(2.0), _mm512_add_ps(e, _mm512_set1_ps(1.0))),
        );
        _mm512_mul_ps(
            _mm512_set1_ps(0.5),
            _mm512_mul_ps(v, _mm512_add_ps(_mm512_set1_ps(1.0), tanh_z)),
        )
    },
    |x: f32| {
        if x > 7.0 {
            x
        } else if x < -7.0 {
            0.0
        } else {
            let z = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            let tanh_z = 1.0 - 2.0 / (crate::kernels::exp::exp(2.0 * z) + 1.0);
            0.5 * x * (1.0 + tanh_z)
        }
    }
);
crate::simd_map!(
    relu,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe { _mm512_max_ps(v, _mm512_set1_ps(0.0)) },
    |x: f32| x.max(0.0)
);

// Tanh map: tanh(x) = 1 - 2/(exp(2x)+1) from the vector vexp kernel.
crate::simd_map!(
    tanh,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe {
        let a = andnot_ps(_mm512_set1_ps(-0.0), v);
        // ±1 for |x| > 9.011, x for |x| < 2e-4, series for |x| < 0.1,
        // ratio (e-1)/(e+1) beyond (Sterbenz-exact, clamped for overflow).
        let big_mask = _mm512_cmp_ps_mask(a, _mm512_set1_ps(9.011), _CMP_GT_OQ);
        if big_mask == 0xFFFF {
            return or_ps(_mm512_set1_ps(1.0), and_ps(_mm512_set1_ps(-0.0), v));
        }
        let x2 = _mm512_mul_ps(v, v);
        let x4 = _mm512_mul_ps(x2, x2);
        let series = _mm512_add_ps(
            _mm512_sub_ps(v, _mm512_div_ps(_mm512_mul_ps(v, x2), _mm512_set1_ps(3.0))),
            _mm512_div_ps(_mm512_mul_ps(v, x4), _mm512_set1_ps(7.5)),
        );
        let e = vexp_512(_mm512_add_ps(v, v));
        let em = _mm512_min_ps(e, _mm512_set1_ps(f32::MAX));
        let ratio = _mm512_div_ps(
            _mm512_sub_ps(em, _mm512_set1_ps(1.0)),
            _mm512_add_ps(em, _mm512_set1_ps(1.0)),
        );
        let big = or_ps(ratio, and_ps(_mm512_set1_ps(-0.0), v));
        let ser_mask = _mm512_cmp_ps_mask(a, _mm512_set1_ps(0.1), _CMP_LT_OQ);
        let small_mask = _mm512_cmp_ps_mask(a, _mm512_set1_ps(2e-4), _CMP_LT_OQ);
        let mid = _mm512_mask_blend_ps(ser_mask, big, series);
        let result = _mm512_mask_blend_ps(small_mask, mid, v);
        _mm512_mask_blend_ps(
            big_mask,
            result,
            or_ps(_mm512_set1_ps(1.0), and_ps(_mm512_set1_ps(-0.0), v)),
        )
    },
    |x: f32| {
        let a = x.abs();
        if a > 9.011 {
            x.signum()
        } else if a < 2e-4 {
            x
        } else if a < 0.1 {
            let x2 = x * x;
            x * (1.0 - x2 / 3.0 + 2.0 * x2 * x2 / 15.0)
        } else {
            let e = crate::kernels::exp::exp(2.0 * x);
            (e - 1.0) / (e + 1.0)
        }
    }
);

// RMS norm: two-pass (sum of squares, then scale by rsqrt).
crate::simd_rms_norm!(
    rms_norm,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    _mm512_setzero_ps(),
    |acc: __m512, v: __m512| _mm512_add_ps(acc, _mm512_mul_ps(v, v)),
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |v, inv| _mm512_mul_ps(v, _mm512_set1_ps(inv)),
    crate::kernels::sqrt::sqrt
);

// Exp map: per-element exp, vector vexp for chunks + scalar exp for tails.
crate::simd_map!(
    exp,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v: __m512| unsafe { vexp_512(v) },
    |x: f32| crate::kernels::exp::exp(x)
);
// Sqrt: one-pass map, native hardware sqrt (correctly rounded, IEEE).
crate::simd_map!(
    sqrt,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe { _mm512_sqrt_ps(v) },
    |x: f32| crate::kernels::sqrt::sqrt(x)
);

// Clip: one-pass map with lo/hi params, min(max(v, lo), hi).
crate::simd_clip!(
    clip,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v: __m512, lo: f32, hi: f32| _mm512_min_ps(
        _mm512_max_ps(v, _mm512_set1_ps(lo)),
        _mm512_set1_ps(hi)
    ),
    |x: f32, lo: f32, hi: f32| x.clamp(lo, hi)
);
// abs_sub: |a - b| per lane (native abs after sub).
crate::simd_map2!(
    abs_sub,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |a: __m512, b: __m512| unsafe { _mm512_abs_ps(_mm512_sub_ps(a, b)) },
    |x: f32, y: f32| (x - y).abs()
);
// hypot: overflow-safe sqrt(a²+b²) via scale-by-max (SLEEF u35 strategy).
// Special-case order: min==0 → max, then NaN, then inf last (inf overrides
// NaN: hypot(inf, nan) == inf per IEEE).
crate::simd_map2!(
    hypot,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |a: __m512, b: __m512| unsafe {
        let ax = _mm512_abs_ps(a);
        let ay = _mm512_abs_ps(b);
        let mx = _mm512_max_ps(ax, ay);
        let mn = _mm512_min_ps(ax, ay);
        let t = _mm512_div_ps(mn, mx);
        let one = _mm512_set1_ps(1.0);
        let r = _mm512_mul_ps(mx, _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(t, t), one)));
        // min==0 → max (covers hypot(x,0)=|x| and hypot(0,0)=0).
        let zero_m = _mm512_cmp_ps_mask(mn, _mm512_setzero_ps(), _CMP_EQ_OQ);
        let r = _mm512_mask_blend_ps(zero_m, r, mx);
        // any NaN → NaN.
        let nan_m = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q) | _mm512_cmp_ps_mask(b, b, _CMP_UNORD_Q);
        let r = _mm512_mask_blend_ps(nan_m, r, _mm512_set1_ps(f32::NAN));
        // any inf → inf (overrides NaN; IEEE hypot(inf, nan) == inf).
        let inf = _mm512_set1_ps(f32::INFINITY);
        let inf_m =
            _mm512_cmp_ps_mask(ax, inf, _CMP_EQ_OQ) | _mm512_cmp_ps_mask(ay, inf, _CMP_EQ_OQ);
        _mm512_mask_blend_ps(inf_m, r, inf)
    },
    |x: f32, y: f32| crate::kernels::hypot::hypot(x, y)
);
// powi: bit-exact exponentiation by squaring (shared scalar exponent ⇒
// identical multiply sequence per lane; see simd_powi!).
crate::simd_powi!(
    powi,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |a, b| unsafe { _mm512_mul_ps(a, b) },
    |a, b| unsafe { _mm512_div_ps(a, b) },
    unsafe { _mm512_set1_ps(1.0) },
    |x: f32, n: i32| crate::kernels::powi::powi(x, n)
);

// Vector ln (f32): fdlibm e_log reduction, see simd_ln! in macros.rs.
crate::simd_ln!(
    vln_512,
    "avx512f",
    __m512,
    __m512i,
    |s| unsafe { _mm512_set1_ps(s) },
    |i| unsafe { _mm512_set1_epi32(i) },
    |a, b| unsafe { _mm512_add_ps(a, b) },
    |a, b| unsafe { _mm512_sub_ps(a, b) },
    |a, b| unsafe { _mm512_mul_ps(a, b) },
    |v| unsafe { _mm512_cvtepi32_ps(v) },
    |v| unsafe { _mm512_castsi512_ps(v) },
    |v| unsafe { _mm512_castps_si512(v) },
    |a, b| unsafe { _mm512_and_si512(a, b) },
    |a, b| unsafe { _mm512_or_si512(a, b) },
    |v| unsafe { _mm512_srli_epi32(v, 23) },
    |a, b| unsafe {
        _mm512_castsi512_ps(_mm512_maskz_mov_epi32(
            _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ),
            _mm512_set1_epi32(-1),
        ))
    },
    |a, b| unsafe {
        _mm512_castsi512_ps(_mm512_maskz_mov_epi32(
            _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ),
            _mm512_set1_epi32(-1),
        ))
    },
    |a, b| unsafe {
        _mm512_castsi512_ps(_mm512_maskz_mov_epi32(
            _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ),
            _mm512_set1_epi32(-1),
        ))
    },
    |a, b| unsafe { and_ps(a, b) },
    |a, b| unsafe { andnot_ps(a, b) },
    |a, b| unsafe { or_ps(a, b) }
);
// Ln: one-pass map; the register kernel handles normal x, the scalar tail
// covers special cases (x <= 0, inf, NaN, subnormal).

// Ln: one-pass map; the register kernel handles normal x, the scalar tail
// covers special cases (x <= 0, inf, NaN, subnormal).
crate::simd_map!(
    ln,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v: __m512| unsafe { vln_512(v) },
    |x: f32| crate::kernels::ln::ln(x)
);
// Softplus: overflow-free `max(x,0) + ln1p(e^-|x|)`. Reference: the identity
// ln1p(z) = z·ln(1+z)/((1+z)-1) from musl s_log1pf.c / fdlibm s_log1p.c
// (https://musl.libc.org, https://www.netlib.org/fdlibm).
crate::simd_map!(
    softplus,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe {
        let zero = _mm512_setzero_ps();
        let a = andnot_ps(_mm512_castsi512_ps(_mm512_set1_epi32(i32::MIN)), v);
        let z = vexp_512(_mm512_sub_ps(zero, a));
        let u = _mm512_add_ps(_mm512_set1_ps(1.0), z);
        let ln_u = vln_512(u);
        let lp = _mm512_div_ps(
            _mm512_mul_ps(ln_u, z),
            _mm512_sub_ps(u, _mm512_set1_ps(1.0)),
        );
        let one = _mm512_castsi512_ps(_mm512_maskz_mov_epi32(
            _mm512_cmp_ps_mask(u, _mm512_set1_ps(1.0), _CMP_EQ_OQ),
            _mm512_set1_epi32(-1),
        ));
        let lp = or_ps(and_ps(one, z), andnot_ps(one, lp));
        _mm512_add_ps(_mm512_max_ps(v, zero), lp)
    },
    |x: f32| {
        let a = x.abs();
        let z = crate::kernels::exp::exp(-a);
        x.max(0.0) + crate::kernels::scalar::log1p(z)
    }
);

// Rsqrt: one-pass map, 1/sqrt(v) (exact via div+sqrt, not the ~12-bit
// hardware approximation — correctness-first).
crate::simd_map!(
    rsqrt,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v: __m512| _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_sqrt_ps(v)),
    |x: f32| 1.0 / crate::kernels::sqrt::sqrt(x)
);
crate::simd_exp!(
    vexp_512,
    f32,
    "avx512f",
    __m512,
    __m512i,
    |s| unsafe { _mm512_set1_ps(s) },
    |i| unsafe { _mm512_set1_epi32(i) },
    |a, b| unsafe { _mm512_mul_ps(a, b) },
    |a, b| unsafe { _mm512_add_ps(a, b) },
    |a, b| unsafe { _mm512_sub_ps(a, b) },
    |a, b| unsafe { and_ps(a, b) },
    |a, b| unsafe { andnot_ps(a, b) },
    |a, b| unsafe { or_ps(a, b) },
    |a, b| unsafe {
        // cmp returns a u16 mask; expand to a full-width float mask vector.
        _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(a, b, _CMP_GT_OQ), _mm512_set1_ps(-1.0))
    },
    |v| unsafe { _mm512_castsi512_ps(v) },
    |v| unsafe { _mm512_castps_si512(v) },
    |v| unsafe { _mm512_cvttps_epi32(v) },
    |v| unsafe { _mm512_slli_epi32(v, 23) },
    |a, b| unsafe { _mm512_add_epi32(a, b) },
    |a, b| unsafe { _mm512_maskz_mov_epi32(_mm512_cmpgt_epi32_mask(a, b), _mm512_set1_epi32(-1)) },
    |a, b| unsafe { _mm512_maskz_mov_epi32(_mm512_cmplt_epi32_mask(a, b), _mm512_set1_epi32(-1)) },
    |a, b| unsafe { _mm512_and_si512(a, b) },
    |a, b| unsafe { _mm512_andnot_si512(a, b) },
    |a, b| unsafe { _mm512_or_si512(a, b) }
);

/// Horizontal argmax of the 16 lanes: `(max value, its index)`.
///
/// Ties resolve to the lowest lane. The index is read from `idx` at the
/// first lane equal to the max.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn argmax_pair_512(v: __m512, idx: __m512i) -> (f32, usize) {
    // NaN lanes must never win: mask them to -inf before the reduction.
    let nan = unsafe { _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) };
    let clean = unsafe { _mm512_mask_blend_ps(nan, v, _mm512_set1_ps(f32::NEG_INFINITY)) };
    let m = unsafe { _mm512_reduce_max_ps(clean) };
    let eq = unsafe { _mm512_cmp_ps_mask(v, _mm512_set1_ps(m), _CMP_EQ_OQ) };
    if eq == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f32::NAN, 0);
    }
    let mut idxs = [0_i32; 16];
    unsafe { _mm512_storeu_si512(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().enumerate() {
        if eq & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

/// Horizontal argmin of the 16 lanes: `(min value, its index)`.
///
/// Ties resolve to the lowest lane. The index is read from `idx` at the
/// first lane equal to the min.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn argmin_pair_512(v: __m512, idx: __m512i) -> (f32, usize) {
    // NaN lanes must never win: mask them to +inf before the reduction.
    let nan = unsafe { _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) };
    let clean = unsafe { _mm512_mask_blend_ps(nan, v, _mm512_set1_ps(f32::INFINITY)) };
    let m = unsafe { _mm512_reduce_min_ps(clean) };
    let eq = unsafe { _mm512_cmp_ps_mask(v, _mm512_set1_ps(m), _CMP_EQ_OQ) };
    if eq == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f32::NAN, 0);
    }
    let mut idxs = [0_i32; 16];
    unsafe { _mm512_storeu_si512(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().enumerate() {
        if eq & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

// Argmax: index of the first occurrence of the maximum.
crate::simd_argminmax!(
    argmax,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    _mm512_set1_epi32,
    _mm512_add_epi32,
    // NaN-aware dethrone (see scalar `argmax`).
    |a, b| unsafe {
        let gt = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
        let nan_b = _mm512_cmp_ps_mask(b, b, _CMP_UNORD_Q);
        let nan_a = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q);
        !nan_a & (gt | nan_b)
    },
    |mask: __mmask16, a: __m512, b: __m512| unsafe { _mm512_mask_blend_ps(mask, b, a) },
    |mask: __mmask16, a: __m512i, b: __m512i| unsafe { _mm512_mask_blend_epi32(mask, b, a) },
    |cand: f32, cur: f32| cand > cur,
    |v, iv| unsafe { argmax_pair_512(v, iv) }
);

// Argmin: index of the first occurrence of the minimum.
crate::simd_argminmax!(
    argmin,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    _mm512_set1_epi32,
    _mm512_add_epi32,
    // NaN-aware dethrone (see argmax above).
    |a, b| unsafe {
        let lt = _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
        let nan_b = _mm512_cmp_ps_mask(b, b, _CMP_UNORD_Q);
        let nan_a = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q);
        !nan_a & (lt | nan_b)
    },
    |mask: __mmask16, a: __m512, b: __m512| unsafe { _mm512_mask_blend_ps(mask, b, a) },
    |mask: __mmask16, a: __m512i, b: __m512i| unsafe { _mm512_mask_blend_epi32(mask, b, a) },
    |cand: f32, cur: f32| cand < cur,
    |v, iv| unsafe { argmin_pair_512(v, iv) }
);

// count_nan: lanes where v != v (mask is already a bitmask).
crate::simd_count!(
    count_nan,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |v: __m512| unsafe { _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) },
    |m: __mmask16| m.count_ones() as usize,
    |x: f32| x.is_nan()
);

// count_zero: lanes equal to +/-0.0 (they compare equal).
crate::simd_count!(
    count_zero,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |v: __m512| unsafe { _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_EQ_OQ) },
    |m: __mmask16| m.count_ones() as usize,
    |x: f32| x == 0.0
);

// count_infinite: lanes whose |v| == +inf.
crate::simd_count!(
    count_infinite,
    f32,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |v: __m512| unsafe {
        _mm512_cmp_ps_mask(_mm512_abs_ps(v), _mm512_set1_ps(f32::INFINITY), _CMP_EQ_OQ)
    },
    |m: __mmask16| m.count_ones() as usize,
    |x: f32| x.is_infinite()
);

// ===========================================================================
// f64 (double-precision) kernels. AVX-512F `__m512d` = 8 lanes. Horizontal
// reductions use the built-in `_mm512_reduce_*_pd`; masks are `__mmask8`.
// ===========================================================================

/// Horizontal argmax of the 8 f64 lanes: `(max value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn argmax_pair_512d(v: __m512d, idx: __m512i) -> (f64, usize) {
    // NaN lanes must never win: mask them to -inf before the reduction.
    let nan = unsafe { _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q) };
    let clean = unsafe { _mm512_mask_blend_pd(nan, v, _mm512_set1_pd(f64::NEG_INFINITY)) };
    let m = unsafe { _mm512_reduce_max_pd(clean) };
    let eq = unsafe { _mm512_cmp_pd_mask(v, _mm512_set1_pd(m), _CMP_EQ_OQ) };
    if eq == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f64::NAN, 0);
    }
    // 16 i32: the 512-bit store covers 16 lanes; each f64 lane's index
    // occupies an i32 pair (see the invocation's duplicated `$vidx`).
    let mut idxs = [0_i32; 16];
    unsafe { _mm512_storeu_si512(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().step_by(2).enumerate() {
        if eq & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

/// Horizontal argmin of the 8 f64 lanes: `(min value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn argmin_pair_512d(v: __m512d, idx: __m512i) -> (f64, usize) {
    // NaN lanes must never win: mask them to +inf before the reduction.
    let nan = unsafe { _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q) };
    let clean = unsafe { _mm512_mask_blend_pd(nan, v, _mm512_set1_pd(f64::INFINITY)) };
    let m = unsafe { _mm512_reduce_min_pd(clean) };
    let eq = unsafe { _mm512_cmp_pd_mask(v, _mm512_set1_pd(m), _CMP_EQ_OQ) };
    if eq == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f64::NAN, 0);
    }
    // 16 i32: the 512-bit store covers 16 lanes; each f64 lane's index
    // occupies an i32 pair (see the invocation's duplicated `$vidx`).
    let mut idxs = [0_i32; 16];
    unsafe { _mm512_storeu_si512(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().step_by(2).enumerate() {
        if eq & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

// f64 reductions for AVX-512F (8 lanes).
crate::simd_reduce!(
    sum_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    _mm512_setzero_pd(),
    _mm512_add_pd,
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |r, v| r + v,
    _mm512_add_pd
);

crate::simd_reduce!(
    prod_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    _mm512_set1_pd(1.0),
    _mm512_mul_pd,
    |v| unsafe { _mm512_reduce_mul_pd(v) },
    |r, v| r * v
);

crate::simd_minmax!(
    min_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    _mm512_set1_pd(f64::INFINITY),
    _mm512_min_pd,
    |v| unsafe { _mm512_reduce_min_pd(v) },
    f64::min,
    |v: __m512d| unsafe { _mm512_cmp_pd_mask(v, v, _CMP_ORD_Q) != 0 },
    |v: __m512d| unsafe {
        let nan = _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q);
        _mm512_mask_blend_pd(nan, v, _mm512_set1_pd(f64::INFINITY))
    },
    |v: f64| !v.is_nan(),
    |r: f64, saw_real: bool| if saw_real { r } else { f64::NAN }
);

crate::simd_minmax!(
    max_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    _mm512_set1_pd(f64::NEG_INFINITY),
    _mm512_max_pd,
    |v| unsafe { _mm512_reduce_max_pd(v) },
    f64::max,
    |v: __m512d| unsafe { _mm512_cmp_pd_mask(v, v, _CMP_ORD_Q) != 0 },
    |v: __m512d| unsafe {
        let nan = _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q);
        _mm512_mask_blend_pd(nan, v, _mm512_set1_pd(f64::NEG_INFINITY))
    },
    |v: f64| !v.is_nan(),
    |r: f64, saw_real: bool| if saw_real { r } else { f64::NAN }
);

crate::simd_reduce!(
    sum_sq_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    _mm512_setzero_pd(),
    |acc: __m512d, v: __m512d| _mm512_add_pd(acc, _mm512_mul_pd(v, v)),
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |r: f64, v: f64| r + v * v,
    _mm512_add_pd
);

crate::simd_reduce!(
    l1_norm_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    _mm512_setzero_pd(),
    |acc: __m512d, v: __m512d| unsafe { _mm512_add_pd(acc, _mm512_abs_pd(v)) },
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |r: f64, v: f64| r + v.abs(),
    _mm512_add_pd
);

crate::simd_minmax!(
    max_norm_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    _mm512_set1_pd(0.0),
    |acc: __m512d, v: __m512d| unsafe { _mm512_max_pd(acc, _mm512_abs_pd(v)) },
    |v| unsafe { _mm512_reduce_max_pd(v) },
    |r: f64, v: f64| f64::max(r, v.abs()),
    |v: __m512d| unsafe { _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q) != 0 },
    |v: __m512d| unsafe {
        let nan = _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q);
        _mm512_mask_blend_pd(nan, v, _mm512_setzero_pd())
    },
    |v: f64| v.is_nan(),
    |r: f64, saw_nan: bool| if saw_nan { f64::NAN } else { r }
);

crate::simd_reduce2!(
    dot_f64,
    f64,
    ["avx512f"],
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    _mm512_setzero_pd(),
    |acc: __m512d, a: __m512d, b: __m512d| _mm512_fmadd_pd(a, b, acc),
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |r, a, b| r + a * b,
    _mm512_add_pd
);

crate::simd_argminmax!(
    argmax_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    // i32-pair duplicated indices: the f64 mask blend covers 64-bit lanes.
    _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7),
    _mm512_set1_epi32,
    _mm512_add_epi32,
    // NaN-aware dethrone (see scalar `argmax`).
    |a: __m512d, b: __m512d| unsafe {
        let gt = _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
        let nan_b = _mm512_cmp_pd_mask(b, b, _CMP_UNORD_Q);
        let nan_a = _mm512_cmp_pd_mask(a, a, _CMP_UNORD_Q);
        !nan_a & (gt | nan_b)
    },
    |mask: __mmask8, a: __m512d, b: __m512d| unsafe { _mm512_mask_blend_pd(mask, b, a) },
    // Index vector holds one i32 index pair per f64 lane: blend i64
    // lanes so the 8-bit f64 mask maps 1:1 onto the pairs.
    |mask: __mmask8, a: __m512i, b: __m512i| unsafe {
        // f64 mask bits map 1:1 to i64 lanes (each holds an i32 index
        // pair); blending i64 lanes keeps the pairs intact.
        _mm512_mask_blend_epi64(mask, b, a)
    },
    |cand: f64, cur: f64| cand > cur,
    |v, iv| unsafe { argmax_pair_512d(v, iv) }
);

crate::simd_argminmax!(
    argmin_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    // i32-pair duplicated indices: the f64 mask blend covers 64-bit lanes.
    _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7),
    _mm512_set1_epi32,
    _mm512_add_epi32,
    // NaN-aware dethrone (see argmax_f64 above).
    |a: __m512d, b: __m512d| unsafe {
        let lt = _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
        let nan_b = _mm512_cmp_pd_mask(b, b, _CMP_UNORD_Q);
        let nan_a = _mm512_cmp_pd_mask(a, a, _CMP_UNORD_Q);
        !nan_a & (lt | nan_b)
    },
    |mask: __mmask8, a: __m512d, b: __m512d| unsafe { _mm512_mask_blend_pd(mask, b, a) },
    // Index vector holds one i32 index pair per f64 lane: blend i64
    // lanes so the 8-bit f64 mask maps 1:1 onto the pairs.
    |mask: __mmask8, a: __m512i, b: __m512i| unsafe {
        // Same widening rationale as argmax_f64: blend i64 lanes.
        _mm512_mask_blend_epi64(mask, b, a)
    },
    |cand: f64, cur: f64| cand < cur,
    |v, iv| unsafe { argmin_pair_512d(v, iv) }
);

// count_nan_f64: lanes where v != v (mask is already a bitmask).
crate::simd_count!(
    count_nan_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |v: __m512d| unsafe { _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q) },
    |m: __mmask8| m.count_ones() as usize,
    |x: f64| x.is_nan()
);

// count_zero_f64: lanes equal to +/-0.0 (they compare equal).
crate::simd_count!(
    count_zero_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |v: __m512d| unsafe { _mm512_cmp_pd_mask(v, _mm512_setzero_pd(), _CMP_EQ_OQ) },
    |m: __mmask8| m.count_ones() as usize,
    |x: f64| x == 0.0
);

// count_infinite_f64: lanes whose |v| == +inf.
crate::simd_count!(
    count_infinite_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |v: __m512d| unsafe {
        _mm512_cmp_pd_mask(_mm512_abs_pd(v), _mm512_set1_pd(f64::INFINITY), _CMP_EQ_OQ)
    },
    |m: __mmask8| m.count_ones() as usize,
    |x: f64| x.is_infinite()
);

// f64 elementwise maps for AVX-512F (8 lanes).
crate::simd_map!(
    sqrt_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v| unsafe { _mm512_sqrt_pd(v) },
    |x: f64| crate::kernels::sqrt::sqrt_f64(x)
);

crate::simd_map!(
    rsqrt_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d| unsafe { _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(v)) },
    |x: f64| 1.0 / crate::kernels::sqrt::sqrt_f64(x)
);

crate::simd_clip!(
    clip_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d, lo: f64, hi: f64| unsafe {
        _mm512_min_pd(_mm512_max_pd(v, _mm512_set1_pd(lo)), _mm512_set1_pd(hi))
    },
    |x: f64, lo: f64, hi: f64| x.clamp(lo, hi)
);
// abs_sub: |a - b| per lane (native abs after sub).
crate::simd_map2!(
    abs_sub_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |a: __m512d, b: __m512d| unsafe { _mm512_abs_pd(_mm512_sub_pd(a, b)) },
    |x: f64, y: f64| (x - y).abs()
);
// hypot_f64: overflow-safe sqrt(a²+b²) via scale-by-max (see f32 hypot).
crate::simd_map2!(
    hypot_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |a: __m512d, b: __m512d| unsafe {
        let ax = _mm512_abs_pd(a);
        let ay = _mm512_abs_pd(b);
        let mx = _mm512_max_pd(ax, ay);
        let mn = _mm512_min_pd(ax, ay);
        let t = _mm512_div_pd(mn, mx);
        let one = _mm512_set1_pd(1.0);
        let r = _mm512_mul_pd(mx, _mm512_sqrt_pd(_mm512_add_pd(_mm512_mul_pd(t, t), one)));
        let zero_m = _mm512_cmp_pd_mask(mn, _mm512_setzero_pd(), _CMP_EQ_OQ);
        let r = _mm512_mask_blend_pd(zero_m, r, mx);
        let nan_m = _mm512_cmp_pd_mask(a, a, _CMP_UNORD_Q) | _mm512_cmp_pd_mask(b, b, _CMP_UNORD_Q);
        let r = _mm512_mask_blend_pd(nan_m, r, _mm512_set1_pd(f64::NAN));
        let inf = _mm512_set1_pd(f64::INFINITY);
        let inf_m =
            _mm512_cmp_pd_mask(ax, inf, _CMP_EQ_OQ) | _mm512_cmp_pd_mask(ay, inf, _CMP_EQ_OQ);
        _mm512_mask_blend_pd(inf_m, r, inf)
    },
    |x: f64, y: f64| crate::kernels::hypot::hypot_f64(x, y)
);
// powi_f64: bit-exact exponentiation by squaring (see f32 powi).
crate::simd_powi!(
    powi_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |a, b| unsafe { _mm512_mul_pd(a, b) },
    |a, b| unsafe { _mm512_div_pd(a, b) },
    unsafe { _mm512_set1_pd(1.0) },
    |x: f64, n: i32| crate::kernels::powi::powi_f64(x, n)
);

// f64 vector exp for AVX-512F (8 lanes).
crate::simd_exp_f64!(
    vexp_512d,
    "avx512f",
    __m512d,
    __m512i,
    |s| unsafe { _mm512_set1_pd(s) },
    |i| unsafe { _mm512_set1_epi64(i) },
    |a, b| unsafe { _mm512_mul_pd(a, b) },
    |a, b| unsafe { _mm512_add_pd(a, b) },
    |a, b| unsafe { _mm512_sub_pd(a, b) },
    |v| unsafe { _mm512_castsi512_pd(v) },
    |v| unsafe { _mm512_castpd_si512(v) },
    // Round-to-nearest: trunc(v + copysign(0.5, v)) in f64, converted to
    // i32 (|n| ≤ 1024 fits i32), then sign-extended to i64.
    // `_mm512_cvttpd_epi64` is AVX-512DQ, not AVX-512F — using it here
    // SIGILLs on AVX-512F CPUs without DQ (e.g. Knights Landing).
    |v| unsafe {
        let sign = and_pd(v, _mm512_castsi512_pd(_mm512_set1_epi64(i64::MIN)));
        let half = or_pd(sign, _mm512_set1_pd(0.5));
        let n32 = _mm512_cvttpd_epi32(_mm512_add_pd(v, half));
        _mm512_cvtepi32_epi64(n32)
    },
    |v| unsafe {
        // Reverse: pack the low i32 of each i64 lane (values fit) and
        // convert i32 → f64. `_mm512_cvtepi64_pd` is AVX-512DQ, not F.
        let packed = _mm512_cvtepi64_epi32(v);
        _mm512_cvtepi32_pd(packed)
    },
    |v| unsafe { _mm512_slli_epi64(v, 52) },
    |a, b| unsafe { _mm512_add_epi64(a, b) },
    |a, b| unsafe { _mm512_maskz_mov_epi64(_mm512_cmpgt_epi64_mask(a, b), _mm512_set1_epi64(-1)) },
    |a, b| unsafe { _mm512_maskz_mov_epi64(_mm512_cmpgt_epi64_mask(b, a), _mm512_set1_epi64(-1)) },
    |a, b| unsafe { _mm512_and_si512(a, b) },
    |a, b| unsafe { _mm512_andnot_si512(a, b) },
    |a, b| unsafe { _mm512_or_si512(a, b) }
);
// Vector ln (f64): fdlibm e_log, see simd_ln_f64! in macros.rs.
crate::simd_ln_f64!(
    vln_512d,
    "avx512f",
    __m512d,
    __m512i,
    |s| unsafe { _mm512_set1_pd(s) },
    |i| unsafe { _mm512_set1_epi64(i) },
    |a, b| unsafe { _mm512_add_pd(a, b) },
    |a, b| unsafe { _mm512_sub_pd(a, b) },
    |a, b| unsafe { _mm512_mul_pd(a, b) },
    |a, b| unsafe { _mm512_div_pd(a, b) },
    |v| unsafe { _mm512_castsi512_pd(v) },
    |v| unsafe { _mm512_castsi512_pd(v) },
    |v| unsafe { _mm512_castpd_si512(v) },
    |a, b| unsafe { _mm512_and_si512(a, b) },
    |a, b| unsafe { _mm512_or_si512(a, b) },
    |v| unsafe { _mm512_srli_epi64(v, 52) },
    |a, b| unsafe {
        _mm512_castsi512_pd(_mm512_maskz_mov_epi64(
            _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ),
            _mm512_set1_epi64(-1),
        ))
    },
    |a, b| unsafe {
        _mm512_castsi512_pd(_mm512_maskz_mov_epi64(
            _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ),
            _mm512_set1_epi64(-1),
        ))
    },
    |a, b| unsafe {
        _mm512_castsi512_pd(_mm512_maskz_mov_epi64(
            _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ),
            _mm512_set1_epi64(-1),
        ))
    },
    |a, b| unsafe { and_pd(a, b) },
    |a, b| unsafe { andnot_pd(a, b) },
    |a, b| unsafe { or_pd(a, b) }
);
// Ln (f64): one-pass map; the register kernel handles normal x.
crate::simd_map!(
    ln_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d| unsafe { vln_512d(v) },
    |x: f64| crate::kernels::ln::ln_f64(x)
);
// Softplus (f64): overflow-free `max(x,0) + ln1p(e^-|x|)`.
crate::simd_map!(
    softplus_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v| unsafe {
        let zero = _mm512_setzero_pd();
        let a = andnot_pd(_mm512_castsi512_pd(_mm512_set1_epi64(i64::MIN)), v);
        let z = vexp_512d(_mm512_sub_pd(zero, a));
        let u = _mm512_add_pd(_mm512_set1_pd(1.0), z);
        let ln_u = vln_512d(u);
        let lp = _mm512_div_pd(
            _mm512_mul_pd(ln_u, z),
            _mm512_sub_pd(u, _mm512_set1_pd(1.0)),
        );
        let one = _mm512_castsi512_pd(_mm512_maskz_mov_epi64(
            _mm512_cmp_pd_mask(u, _mm512_set1_pd(1.0), _CMP_EQ_OQ),
            _mm512_set1_epi64(-1),
        ));
        let lp = or_pd(and_pd(one, z), andnot_pd(one, lp));
        _mm512_add_pd(_mm512_max_pd(v, zero), lp)
    },
    |x: f64| {
        let a = x.abs();
        let z = crate::kernels::exp::exp_f64(-a);
        x.max(0.0) + crate::kernels::scalar::log1p_f64(z)
    }
);

crate::simd_map!(
    exp_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d| unsafe { vexp_512d(v) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

crate::simd_softmax!(
    softmax_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |a, b| unsafe { _mm512_max_pd(a, b) },
    |a, b| unsafe { _mm512_sub_pd(a, b) },
    |v| unsafe { vexp_512d(v) },
    |a, b| unsafe { _mm512_add_pd(a, b) },
    |a, b| unsafe { _mm512_mul_pd(a, b) },
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |v| unsafe { _mm512_reduce_max_pd(v) },
    |s| unsafe { _mm512_set1_pd(s) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

// Logsumexp (f64): two-pass scalar-returning reduction.
crate::simd_logsumexp!(
    logsumexp_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |a, b| unsafe { _mm512_max_pd(a, b) },
    |a, b| unsafe { _mm512_sub_pd(a, b) },
    |v| unsafe { vexp_512d(v) },
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |v| unsafe { _mm512_reduce_max_pd(v) },
    |s| unsafe { _mm512_set1_pd(s) },
    |x: f64| crate::kernels::exp::exp_f64(x),
    crate::kernels::ln::ln_f64
);

// Log-softmax (f64): three-pass map, 0-alloc.
crate::simd_log_softmax!(
    log_softmax_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |a, b| unsafe { _mm512_max_pd(a, b) },
    |a, b| unsafe { _mm512_sub_pd(a, b) },
    |v| unsafe { vexp_512d(v) },
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |v| unsafe { _mm512_reduce_max_pd(v) },
    |s| unsafe { _mm512_set1_pd(s) },
    |x: f64| crate::kernels::exp::exp_f64(x),
    crate::kernels::ln::ln_f64
);

// Layer norm (f64): three-pass, 0-alloc.
crate::simd_layer_norm!(
    layer_norm_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |a, b| unsafe { _mm512_add_pd(a, b) },
    |a, b| unsafe { _mm512_sub_pd(a, b) },
    _mm512_setzero_pd(),
    |acc: __m512d, v: __m512d| _mm512_add_pd(acc, _mm512_mul_pd(v, v)),
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |s| unsafe { _mm512_set1_pd(s) },
    |v, inv| _mm512_mul_pd(v, _mm512_set1_pd(inv)),
    crate::kernels::sqrt::sqrt_f64
);

crate::simd_map!(
    sigmoid_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d| unsafe {
        // Saturated fast path: all lanes already at 0/1 (skip the exp).
        let pos = _mm512_cmp_pd_mask(v, _mm512_set1_pd(36.74), _CMP_GT_OQ);
        let neg = _mm512_cmp_pd_mask(v, _mm512_set1_pd(-744.0), _CMP_LT_OQ);
        if pos | neg == 0xFF {
            return _mm512_maskz_mov_pd(pos, _mm512_set1_pd(1.0));
        }
        _mm512_div_pd(
            _mm512_set1_pd(1.0),
            _mm512_add_pd(
                _mm512_set1_pd(1.0),
                vexp_512d(xor_pd(v, _mm512_castsi512_pd(_mm512_set1_epi64(i64::MIN)))),
            ),
        )
    },
    |x: f64| {
        if x > 36.74 {
            1.0
        } else if x < -744.0 {
            0.0
        } else {
            1.0 / (1.0 + crate::kernels::exp::exp_f64(-x))
        }
    }
);

crate::simd_map!(
    silu_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d| unsafe {
        // Saturated fast path: silu(x) = x for x > 36.74, 0 for x < -745.
        let pos = _mm512_cmp_pd_mask(v, _mm512_set1_pd(36.74), _CMP_GT_OQ);
        let neg = _mm512_cmp_pd_mask(v, _mm512_set1_pd(-744.0), _CMP_LT_OQ);
        if pos | neg == 0xFF {
            return _mm512_maskz_mov_pd(pos, v);
        }
        _mm512_div_pd(
            v,
            _mm512_add_pd(
                _mm512_set1_pd(1.0),
                vexp_512d(xor_pd(v, _mm512_castsi512_pd(_mm512_set1_epi64(i64::MIN)))),
            ),
        )
    },
    |x: f64| {
        if x > 36.74 {
            x
        } else if x < -744.0 {
            0.0
        } else {
            x / (1.0 + crate::kernels::exp::exp_f64(-x))
        }
    }
);

crate::simd_map!(
    gelu_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d| unsafe {
        // Saturated fast path: gelu(x) = x for x > 7.21, 0 for x < -7.21.
        let pos = _mm512_cmp_pd_mask(v, _mm512_set1_pd(7.21), _CMP_GT_OQ);
        let neg = _mm512_cmp_pd_mask(v, _mm512_set1_pd(-7.21), _CMP_LT_OQ);
        if pos | neg == 0xFF {
            return _mm512_maskz_mov_pd(pos, v);
        }
        let x2 = _mm512_mul_pd(v, v);
        let x3 = _mm512_mul_pd(x2, v);
        let z = _mm512_mul_pd(
            _mm512_set1_pd(0.797_884_560_802_865_4),
            _mm512_add_pd(v, _mm512_mul_pd(_mm512_set1_pd(0.044_715), x3)),
        );
        let e = vexp_512d(_mm512_add_pd(z, z));
        let tanh_z = _mm512_sub_pd(
            _mm512_set1_pd(1.0),
            _mm512_div_pd(_mm512_set1_pd(2.0), _mm512_add_pd(e, _mm512_set1_pd(1.0))),
        );
        _mm512_mul_pd(
            _mm512_set1_pd(0.5),
            _mm512_mul_pd(v, _mm512_add_pd(_mm512_set1_pd(1.0), tanh_z)),
        )
    },
    |x: f64| {
        if x > 7.21 {
            x
        } else if x < -7.21 {
            0.0
        } else {
            let z = 0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x);
            let tanh_z = 1.0 - 2.0 / (crate::kernels::exp::exp_f64(2.0 * z) + 1.0);
            0.5 * x * (1.0 + tanh_z)
        }
    }
);

crate::simd_map!(
    relu_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d| unsafe { _mm512_max_pd(v, _mm512_set1_pd(0.0)) },
    |x: f64| x.max(0.0)
);

// Tanh map (f64): tanh(x) = 1 - 2/(exp(2x)+1).
crate::simd_map!(
    tanh_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    |v: __m512d| unsafe {
        let a = andnot_pd(_mm512_set1_pd(-0.0), v);
        // ±1 for |x| > 19.062, x for |x| < 5e-8, series for |x| < 0.1,
        // ratio (e-1)/(e+1) beyond (Sterbenz-exact, clamped for overflow).
        let big_mask = _mm512_cmp_pd_mask(a, _mm512_set1_pd(19.062), _CMP_GT_OQ);
        if big_mask == 0xFF {
            return or_pd(_mm512_set1_pd(1.0), and_pd(_mm512_set1_pd(-0.0), v));
        }
        let y = _mm512_mul_pd(v, v);
        let p = _mm512_set1_pd(0.003_592_128_572_437_055);
        let p = _mm512_add_pd(
            _mm512_mul_pd(p, y),
            _mm512_set1_pd(-0.008_863_235_529_902_197),
        );
        let p = _mm512_add_pd(_mm512_mul_pd(p, y), _mm512_set1_pd(0.021_869_488_536_155_2));
        let p = _mm512_add_pd(
            _mm512_mul_pd(p, y),
            _mm512_set1_pd(-0.053_968_253_968_253_97),
        );
        let p = _mm512_add_pd(
            _mm512_mul_pd(p, y),
            _mm512_set1_pd(0.133_333_333_333_333_33),
        );
        let p = _mm512_add_pd(
            _mm512_mul_pd(p, y),
            _mm512_set1_pd(-0.333_333_333_333_333_3),
        );
        let series = _mm512_mul_pd(v, _mm512_add_pd(_mm512_mul_pd(p, y), _mm512_set1_pd(1.0)));
        let e = vexp_512d(_mm512_add_pd(v, v));
        let em = _mm512_min_pd(e, _mm512_set1_pd(f64::MAX));
        let ratio = _mm512_div_pd(
            _mm512_sub_pd(em, _mm512_set1_pd(1.0)),
            _mm512_add_pd(em, _mm512_set1_pd(1.0)),
        );
        let big = or_pd(ratio, and_pd(_mm512_set1_pd(-0.0), v));
        let ser_mask = _mm512_cmp_pd_mask(a, _mm512_set1_pd(0.1), _CMP_LT_OQ);
        let small_mask = _mm512_cmp_pd_mask(a, _mm512_set1_pd(2e-8), _CMP_LT_OQ);
        let mid = _mm512_mask_blend_pd(ser_mask, big, series);
        let result = _mm512_mask_blend_pd(small_mask, mid, v);
        _mm512_mask_blend_pd(
            big_mask,
            result,
            or_pd(_mm512_set1_pd(1.0), and_pd(_mm512_set1_pd(-0.0), v)),
        )
    },
    |x: f64| {
        let a = x.abs();
        if a > 19.062 {
            x.signum()
        } else if a < 2e-8 {
            x
        } else if a < 0.1 {
            let y = x * x;
            let p = 0.003_592_128_572_437_055_f64;
            let p = p * y - 0.008_863_235_529_902_197;
            let p = p * y + 0.021_869_488_536_155_2;
            let p = p * y - 0.053_968_253_968_253_97;
            let p = p * y + 0.133_333_333_333_333_33;
            let p = p * y - 0.333_333_333_333_333_3;
            x * (p * y + 1.0)
        } else {
            let e = crate::kernels::exp::exp_f64(2.0 * x);
            (e - 1.0) / (e + 1.0)
        }
    }
);

// RMS norm (f64).
crate::simd_rms_norm!(
    rms_norm_f64,
    f64,
    "avx512f",
    8,
    |p| unsafe { _mm512_loadu_pd(p) },
    |p, v| unsafe { _mm512_storeu_pd(p, v) },
    _mm512_setzero_pd(),
    |acc: __m512d, v: __m512d| _mm512_add_pd(acc, _mm512_mul_pd(v, v)),
    |v| unsafe { _mm512_reduce_add_pd(v) },
    |v, inv| _mm512_mul_pd(v, _mm512_set1_pd(inv)),
    crate::kernels::sqrt::sqrt_f64
);

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::vec::Vec;

    /// Integer-valued inputs in [-scale, scale]; sums of up to 1024 such
    /// values are exactly representable in f32, so backends must agree exactly.
    fn exact_data(len: usize, seed: u64, scale: i32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 32) % 8_192) as i32 - 4_096
            })
            .map(|v| unsafe { (v % scale) as f32 })
            .collect()
    }

    #[test]
    fn sum_matches_scalar_when_avx512_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [1, 2, 3, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1024] {
            let data = exact_data(len, 23, 4_096);
            // Products of 2^n overflow f32 quickly; cap prod at small
            // lengths so the result stays exactly representable.
            let prod_len = len.min(64);
            let prod_data = exact_data(prod_len, 24, 2);
            let a = exact_data(len, 29, 64);
            let b = exact_data(len, 31, 64);

            // SAFETY: tested inside the avx512f detection guard.
            unsafe {
                assert_eq!(sum(&data), exact_sum(&data), "sum mismatch for len {len}");
                assert_eq!(
                    prod(&prod_data),
                    exact_prod(&prod_data),
                    "prod mismatch for len {prod_len}"
                );
                assert_eq!(dot(&a, &b), exact_dot(&a, &b), "dot mismatch for len {len}");
            }

            if len > 0 {
                // SAFETY: tested inside the avx512f detection guard.
                unsafe {
                    assert_eq!(min(&data), exact_min(&data), "min mismatch for len {len}");
                    assert_eq!(max(&data), exact_max(&data), "max mismatch for len {len}");
                }
            }
        }
    }

    #[test]
    fn sum_sq_matches_scalar_when_avx512_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Small scale so squares stay exactly representable (≤ 4096²).
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 21, 64);
            // SAFETY: tested inside the avx2 detection guard.
            let simd = unsafe { sum_sq(&data) };
            let scalar: f32 = data.iter().map(|x| x * x).sum();
            assert_eq!(simd, scalar, "sum_sq mismatch for len {len}");
        }
    }

    #[test]
    fn l1_norm_matches_scalar_when_avx512_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 23, 64);
            // SAFETY: tested inside the avx2 detection guard.
            let simd = unsafe { l1_norm(&data) };
            let scalar: f32 = data.iter().copied().map(f32::abs).sum();
            assert_eq!(simd, scalar, "l1_norm mismatch for len {len}");
        }
    }

    #[test]
    fn max_norm_matches_scalar_when_avx512_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 25, 64);
            // SAFETY: tested inside the avx2 detection guard.
            let simd = unsafe { max_norm(&data) };
            let scalar: f32 = data
                .iter()
                .copied()
                .map(f32::abs)
                .max_by(f32::total_cmp)
                .unwrap_or(0.0);
            assert_eq!(simd, scalar, "max_norm mismatch for len {len}");
        }
    }

    #[test]
    fn argmax_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 31, 4_096);
            // SAFETY: tested inside the feature guard.
            let (v, i) = unsafe { argmax(&data) };
            assert_eq!(v, data[i], "argmax value mismatch for len {len}");
            assert_eq!(
                i,
                data.iter()
                    .enumerate()
                    .fold(0, |bi, (i, &x)| { if x > data[bi] { i } else { bi } }),
                "argmax index mismatch for len {len}"
            );
        }
        // Tie-break: first occurrence wins.
        let tied = [1.0_f32, 5.0, 3.0, 5.0, 2.0];
        // SAFETY: tested inside the feature guard.
        assert_eq!(unsafe { argmax(&tied) }, (5.0, 1));
    }

    #[test]
    fn argmin_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 37, 4_096);
            // SAFETY: tested inside the feature guard.
            let (v, i) = unsafe { argmin(&data) };
            assert_eq!(v, data[i], "argmin value mismatch for len {len}");
            assert_eq!(
                i,
                data.iter()
                    .enumerate()
                    .fold(0, |bi, (i, &x)| { if x < data[bi] { i } else { bi } }),
                "argmin index mismatch for len {len}"
            );
        }
        // Tie-break: first occurrence wins.
        let tied = [3.0_f32, 1.0, 2.0, 1.0, 4.0];
        // SAFETY: tested inside the feature guard.
        assert_eq!(unsafe { argmin(&tied) }, (1.0, 1));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn softmax_matches_scalar_when_avx512_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 15, 16, 17, 31, 32, 33] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 2.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the avx512f detection guard.
            unsafe { softmax(&data, &mut simd_out) };
            crate::kernels::scalar::softmax(&data, &mut ref_out);
            for i in 0..len {
                assert!(
                    (simd_out[i] - ref_out[i]).abs() < 1e-6,
                    "softmax mismatch for len {len} lane {i}"
                );
            }
        }
    }

    fn exact_sum(values: &[f32]) -> f32 {
        values.iter().sum()
    }

    fn exact_prod(values: &[f32]) -> f32 {
        values.iter().product()
    }

    fn exact_min(values: &[f32]) -> f32 {
        values.iter().copied().reduce(f32::min).unwrap()
    }

    fn exact_max(values: &[f32]) -> f32 {
        values.iter().copied().reduce(f32::max).unwrap()
    }

    fn exact_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn sigmoid_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.7 - 12.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the detection guard.
            unsafe { sigmoid(&data, &mut simd_out) };
            crate::kernels::scalar::sigmoid(&data, &mut ref_out);
            for i in 0..len {
                assert!(
                    (simd_out[i] - ref_out[i]).abs() < 1e-6,
                    "sigmoid mismatch for len {len} lane {i}"
                );
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn silu_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.7 - 12.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the detection guard.
            unsafe { silu(&data, &mut simd_out) };
            crate::kernels::scalar::silu(&data, &mut ref_out);
            for i in 0..len {
                assert!(
                    (simd_out[i] - ref_out[i]).abs() < 1e-6,
                    "silu mismatch for len {len} lane {i}"
                );
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn gelu_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.7 - 12.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the detection guard.
            unsafe { gelu(&data, &mut simd_out) };
            crate::kernels::scalar::gelu(&data, &mut ref_out);
            for i in 0..len {
                assert!(
                    (simd_out[i] - ref_out[i]).abs() < 1e-5,
                    "gelu mismatch for len {len} lane {i}"
                );
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn relu_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.7 - 12.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the detection guard.
            unsafe { relu(&data, &mut simd_out) };
            crate::kernels::scalar::relu(&data, &mut ref_out);
            for i in 0..len {
                assert!(
                    (simd_out[i] - ref_out[i]).abs() < 1e-6,
                    "relu mismatch for len {len} lane {i}"
                );
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn sqrt_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 1.5 + 0.25).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the detection guard.
            unsafe { sqrt(&data, &mut simd_out) };
            crate::kernels::scalar::sqrt(&data, &mut ref_out);
            for i in 0..len {
                // Hardware sqrt is correctly rounded; scalar is ≤ 1 ulp —
                // so allow 1 ulp of slack.
                assert!(
                    (simd_out[i] - ref_out[i]).abs() <= ref_out[i].abs() * 2e-7 + 1e-6,
                    "sqrt mismatch for len {len} lane {i}"
                );
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn clip_rsqrt_match_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 2.5 - 17.0).collect();
            let mut c_out = vec![0.0_f32; len];
            let mut c_ref = vec![0.0_f32; len];
            let mut r_out = vec![0.0_f32; len];
            let mut r_ref = vec![0.0_f32; len];
            // SAFETY: tested inside the detection guard.
            unsafe { clip(&data, -7.0, 9.0, &mut c_out) };
            crate::kernels::scalar::clip(&data, -7.0, 9.0, &mut c_ref);
            // SAFETY: tested inside the detection guard.
            unsafe { rsqrt(&data, &mut r_out) };
            crate::kernels::scalar::rsqrt(&data, &mut r_ref);
            for i in 0..len {
                assert!(
                    (c_out[i] - c_ref[i]).abs() < 1e-6,
                    "clip mismatch for len {len} lane {i}"
                );
                let tol = r_ref[i].abs() * 2e-7 + 1e-6;
                assert!(
                    (r_out[i] - r_ref[i]).abs() <= tol || (r_out[i].is_nan() && r_ref[i].is_nan()),
                    "rsqrt mismatch for len {len} lane {i}"
                );
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn exp_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.7 - 8.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the detection guard.
            unsafe { exp(&data, &mut simd_out) };
            crate::kernels::scalar::exp(&data, &mut ref_out);
            for i in 0..len {
                let tol = ref_out[i].abs() * 2e-7 + 1e-6;
                assert!(
                    (simd_out[i] - ref_out[i]).abs() <= tol,
                    "exp mismatch for len {len} lane {i}"
                );
            }
        }
    }
}
