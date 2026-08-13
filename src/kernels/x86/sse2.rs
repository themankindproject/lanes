//! SSE2 (128-bit) SIMD kernel implementations for x86-64.
//!
//! SSE2 is mandatory on all x86-64 CPUs, so these kernels are always
//! available on x86-64 targets (still verified via
//! `std::arch::is_x86_feature_detected!("sse2")` before dispatch).
//!
//! Floating-point semantics: sums/dot products accumulate in 4-lane vectors
//! and combine via a horizontal reduction, so the reduction order differs
//! from the scalar kernels. `min`/`max` follow the SSE `minps`/`maxps`
//! hardware semantics (a NaN present in the data propagates), unlike the
//! scalar `f32::min`/`f32::max` semantics. For NaN-free inputs the SIMD and
//! scalar results agree exactly.
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

// Sum reduction: accumulate 4-wide, horizontal-sum, scalar tail.
crate::simd_reduce!(
    sum,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_setzero_ps(),
    _mm_add_ps,
    |v| unsafe { hsum_128(v) },
    |r, v| r + v
);

// Product reduction: 4-wide multiply, scalar-multiply tail.
crate::simd_reduce!(
    prod,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_set1_ps(1.0),
    _mm_mul_ps,
    |v| unsafe { hprod_128(v) },
    |r, v| r * v
);

// Minimum reduction: `minps` semantics, `minf` tail.
crate::simd_reduce!(
    min,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_set1_ps(f32::INFINITY),
    _mm_min_ps,
    |v| unsafe { hmin_128(v) },
    f32::min
);

// Maximum reduction: `maxps` semantics, `maxf` tail.
crate::simd_reduce!(
    max,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_set1_ps(f32::NEG_INFINITY),
    _mm_max_ps,
    |v| unsafe { hmax_128(v) },
    f32::max
);

// Sum of squares: 4-wide multiply-accumulate (acc += v*v).
crate::simd_reduce!(
    sum_sq,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_setzero_ps(),
    |acc: __m128, v: __m128| _mm_add_ps(acc, _mm_mul_ps(v, v)),
    |v| unsafe { hsum_128(v) },
    |r: f32, v: f32| r + v * v
);

// L1 norm: sum of absolute values.
crate::simd_reduce!(
    l1_norm,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_setzero_ps(),
    |acc: __m128, v: __m128| _mm_add_ps(acc, _mm_andnot_ps(_mm_set1_ps(-0.0), v)),
    |v| unsafe { hsum_128(v) },
    |r: f32, v: f32| r + v.abs()
);

// Max norm: maximum absolute value.
crate::simd_reduce!(
    max_norm,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_set1_ps(0.0),
    |acc: __m128, v: __m128| _mm_max_ps(acc, _mm_andnot_ps(_mm_set1_ps(-0.0), v)),
    |v| unsafe { hmax_128(v) },
    |r: f32, v: f32| f32::max(r, v.abs())
);

// Argmax: index of the first occurrence of the maximum.
crate::simd_argminmax!(
    argmax,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_setr_epi32(0, 1, 2, 3),
    _mm_set1_epi32,
    _mm_add_epi32,
    // NaN-aware dethrone: a candidate wins iff it is non-NaN and (the
    // current seed is NaN or it compares strictly better). See scalar.
    |a: __m128, b: __m128| unsafe {
        let gt = _mm_cmpgt_ps(a, b);
        let nan_b = _mm_cmpunord_ps(b, b);
        let nan_a = _mm_cmpunord_ps(a, a);
        _mm_andnot_ps(nan_a, _mm_or_ps(gt, nan_b))
    },
    |mask: __m128, a: __m128, b: __m128| {
        // SAFETY: caller guarantees SSE2.
        unsafe { _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b)) }
    },
    |mask: __m128, a: __m128i, b: __m128i| {
        let m = unsafe { _mm_castps_si128(mask) };
        // SAFETY: caller guarantees SSE2.
        unsafe { _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b)) }
    },
    |cand: f32, cur: f32| cand > cur,
    |v, iv| unsafe { argmax_pair_128(v, iv) }
);

// Argmin: index of the first occurrence of the minimum.
crate::simd_argminmax!(
    argmin,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_setr_epi32(0, 1, 2, 3),
    _mm_set1_epi32,
    _mm_add_epi32,
    // NaN-aware dethrone (see argmax above).
    |a: __m128, b: __m128| unsafe {
        let lt = _mm_cmplt_ps(a, b);
        let nan_b = _mm_cmpunord_ps(b, b);
        let nan_a = _mm_cmpunord_ps(a, a);
        _mm_andnot_ps(nan_a, _mm_or_ps(lt, nan_b))
    },
    |mask: __m128, a: __m128, b: __m128| {
        // SAFETY: caller guarantees SSE2.
        unsafe { _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b)) }
    },
    |mask: __m128, a: __m128i, b: __m128i| {
        let m = unsafe { _mm_castps_si128(mask) };
        // SAFETY: caller guarantees SSE2.
        unsafe { _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b)) }
    },
    |cand: f32, cur: f32| cand < cur,
    |v, iv| unsafe { argmin_pair_128(v, iv) }
);

// Dot product: 4-wide multiply-accumulate (mul+add; SSE2 has no FMA).
crate::simd_reduce2!(
    dot,
    f32,
    ["sse2"],
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_setzero_ps(),
    |acc, va, vb| _mm_add_ps(acc, _mm_mul_ps(va, vb)),
    |v| unsafe { hsum_128(v) },
    |r, a, b| r + a * b
);

// Softmax: 3-pass map (max → exp+sum → scale). exp is per-lane scalar
// (no vector exp intrinsic); the macro handles the chunk loop.
// Uses the crate's `no_std` `exp`, so available in all builds.
#[cfg(feature = "alloc")]
crate::simd_softmax!(
    softmax,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    _mm_max_ps,
    _mm_sub_ps,
    |v| unsafe { vexp_128(v) },
    _mm_add_ps,
    _mm_mul_ps,
    |v| unsafe { hsum_128(v) },
    |v| unsafe { hmax_128(v) },
    |s| unsafe { _mm_set1_ps(s) },
    |x: f32| crate::kernels::exp::exp(x)
);

crate::simd_map!(
    sigmoid,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v| unsafe {
        _mm_div_ps(
            _mm_set1_ps(1.0),
            _mm_add_ps(
                _mm_set1_ps(1.0),
                vexp_128(_mm_xor_ps(v, _mm_castsi128_ps(_mm_set1_epi32(i32::MIN)))),
            ),
        )
    },
    |x: f32| 1.0 / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    silu,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v| unsafe {
        _mm_div_ps(
            v,
            _mm_add_ps(
                _mm_set1_ps(1.0),
                vexp_128(_mm_xor_ps(v, _mm_castsi128_ps(_mm_set1_epi32(i32::MIN)))),
            ),
        )
    },
    |x: f32| x / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    gelu,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v| unsafe {
        let x2 = _mm_mul_ps(v, v);
        let x3 = _mm_mul_ps(x2, v);
        let z = _mm_mul_ps(
            _mm_set1_ps(0.797_884_6),
            _mm_add_ps(v, _mm_mul_ps(_mm_set1_ps(0.044_715), x3)),
        );
        let e = vexp_128(_mm_add_ps(z, z));
        let tanh_z = _mm_sub_ps(
            _mm_set1_ps(1.0),
            _mm_div_ps(_mm_set1_ps(2.0), _mm_add_ps(e, _mm_set1_ps(1.0))),
        );
        _mm_mul_ps(
            _mm_set1_ps(0.5),
            _mm_mul_ps(v, _mm_add_ps(_mm_set1_ps(1.0), tanh_z)),
        )
    },
    |x: f32| {
        let z = 0.797_884_6 * (x + 0.044_715 * x * x * x);
        let tanh_z = 1.0 - 2.0 / (crate::kernels::exp::exp(2.0 * z) + 1.0);
        0.5 * x * (1.0 + tanh_z)
    }
);
crate::simd_map!(
    relu,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v| unsafe { _mm_max_ps(v, _mm_set1_ps(0.0)) },
    |x: f32| x.max(0.0)
);

// Tanh map: piecewise — Taylor series for |x| < 0.1 (the exp form cancels
// to 0 there), 1 - 2/(exp(2x)+1) beyond.
crate::simd_map!(
    tanh,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v| unsafe {
        let series = {
            let x2 = _mm_mul_ps(v, v);
            let x4 = _mm_mul_ps(x2, x2);
            // x - x³/3 + 2x⁵/15
            _mm_add_ps(
                _mm_sub_ps(v, _mm_div_ps(_mm_mul_ps(v, x2), _mm_set1_ps(3.0))),
                _mm_div_ps(_mm_mul_ps(v, x4), _mm_set1_ps(7.5)),
            )
        };
        let e = vexp_128(_mm_add_ps(v, v));
        // clamp e to FLT_MAX: (max-1)/(max+1) rounds to 1.0, so the ratio
        // saturates to ±1 on exp overflow; copysign restores the sign.
        let em = _mm_min_ps(e, _mm_set1_ps(f32::MAX));
        let ratio = _mm_div_ps(
            _mm_sub_ps(em, _mm_set1_ps(1.0)),
            _mm_add_ps(em, _mm_set1_ps(1.0)),
        );
        let big = _mm_or_ps(ratio, _mm_and_ps(_mm_set1_ps(-0.0), v));
        let small = _mm_cmplt_ps(_mm_andnot_ps(_mm_set1_ps(-0.0), v), _mm_set1_ps(0.1));
        _mm_or_ps(_mm_and_ps(small, series), _mm_andnot_ps(small, big))
    },
    |x: f32| {
        if x.abs() < 0.1 {
            let x2 = x * x;
            x * (1.0 - x2 / 3.0 + 2.0 * x2 * x2 / 15.0)
        } else {
            let e = crate::kernels::exp::exp(2.0 * x);
            if e.is_infinite() {
                x.signum()
            } else {
                (e - 1.0) / (e + 1.0)
            }
        }
    }
);

// RMS norm: two-pass (sum of squares, then scale by rsqrt).
crate::simd_rms_norm!(
    rms_norm,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    _mm_setzero_ps(),
    |acc: __m128, v: __m128| _mm_add_ps(acc, _mm_mul_ps(v, v)),
    |v| unsafe { hsum_128(v) },
    |v, inv| _mm_mul_ps(v, _mm_set1_ps(inv)),
    crate::kernels::sqrt::sqrt
);

// Exp map: per-element exp, vector vexp for chunks + scalar exp for tails.
crate::simd_map!(
    exp,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v: __m128| unsafe { vexp_128(v) },
    |x: f32| crate::kernels::exp::exp(x)
);
// Sqrt: one-pass map, native hardware sqrt (correctly rounded, IEEE).
crate::simd_map!(
    sqrt,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v| unsafe { _mm_sqrt_ps(v) },
    |x: f32| crate::kernels::sqrt::sqrt(x)
);

// Clip: one-pass map with lo/hi params, min(max(v, lo), hi).
crate::simd_map_param!(
    clip,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v: __m128, lo: f32, hi: f32| _mm_min_ps(_mm_max_ps(v, _mm_set1_ps(lo)), _mm_set1_ps(hi)),
    |x: f32, lo: f32, hi: f32| x.clamp(lo, hi)
);
// Rsqrt: one-pass map, 1/sqrt(v) (exact via div+sqrt, not the ~12-bit
// hardware approximation — correctness-first).
crate::simd_map!(
    rsqrt,
    f32,
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v: __m128| _mm_div_ps(_mm_set1_ps(1.0), _mm_sqrt_ps(v)),
    |x: f32| 1.0 / crate::kernels::sqrt::sqrt(x)
);
crate::simd_exp!(
    vexp_128,
    f32,
    "sse2",
    __m128,
    __m128i,
    |s| unsafe { _mm_set1_ps(s) },
    |i| unsafe { _mm_set1_epi32(i) },
    |a, b| unsafe { _mm_mul_ps(a, b) },
    |a, b| unsafe { _mm_add_ps(a, b) },
    |a, b| unsafe { _mm_sub_ps(a, b) },
    |a, b| unsafe { _mm_and_ps(a, b) },
    |a, b| unsafe { _mm_andnot_ps(a, b) },
    |a, b| unsafe { _mm_or_ps(a, b) },
    |a, b| unsafe { _mm_cmpgt_ps(a, b) },
    |v| unsafe { _mm_castsi128_ps(v) },
    |v| unsafe { _mm_cvttps_epi32(v) },
    |v| unsafe { _mm_slli_epi32(v, 23) },
    |a, b| unsafe { _mm_add_epi32(a, b) },
    |a, b| unsafe { _mm_cmpgt_epi32(a, b) },
    |a, b| unsafe { _mm_cmplt_epi32(a, b) },
    |a, b| unsafe { _mm_and_si128(a, b) },
    |a, b| unsafe { _mm_andnot_si128(a, b) },
    |a, b| unsafe { _mm_or_si128(a, b) }
);

/// Horizontal sum of the 4 lanes in a `__m128` register.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2 (implied by the `target_feature`
/// gate on all public functions in this module).
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hsum_128(v: __m128) -> f32 {
    // SAFETY: caller guarantees SSE2; all intrinsics below require SSE2.
    let shuf = unsafe { _mm_movehdup_ps(v) }; // [1,1,3,3]
    let sums = unsafe { _mm_add_ps(v, shuf) }; // [0+1, _, 2+3, _]
    let hi64 = unsafe { _mm_movehl_ps(sums, sums) }; // [2+3, _, _, _]
    let result = unsafe { _mm_add_ss(sums, hi64) }; // [0+1+2+3, _, _, _]
    unsafe { _mm_cvtss_f32(result) }
}

/// Horizontal product of the 4 lanes in a `__m128` register.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hprod_128(v: __m128) -> f32 {
    // SAFETY: caller guarantees SSE2.
    let s = unsafe { _mm_shuffle_ps(v, v, 0b0100_1110) }; // [a2, a3, a0, a1]
    let m = unsafe { _mm_mul_ps(v, s) }; // [a0*a2, a1*a3, a0*a2, a1*a3]
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0000_0001) }; // [a1*a3, a0*a2, ...]
    let m = unsafe { _mm_mul_ps(m, s) }; // all lanes = a0*a1*a2*a3
    unsafe { _mm_cvtss_f32(m) }
}

/// Horizontal minimum of the 4 lanes in a `__m128` register.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hmin_128(v: __m128) -> f32 {
    // SAFETY: caller guarantees SSE2.
    let s = unsafe { _mm_shuffle_ps(v, v, 0b0100_1110) }; // [a2, a3, a0, a1]
    let m = unsafe { _mm_min_ps(v, s) }; // [min(a0,a2), min(a1,a3), ...]
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0000_0001) }; // [a1, a0, a1, a0]
    let m = unsafe { _mm_min_ps(m, s) }; // all lanes = min(a0..a3)
    unsafe { _mm_cvtss_f32(m) }
}

/// Horizontal maximum of the 4 lanes in a `__m128` register.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hmax_128(v: __m128) -> f32 {
    // SAFETY: caller guarantees SSE2.
    let s = unsafe { _mm_shuffle_ps(v, v, 0b0100_1110) }; // [a2, a3, a0, a1]
    let m = unsafe { _mm_max_ps(v, s) }; // [max(a0,a2), max(a1,a3), ...]
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0000_0001) }; // [a1, a0, a1, a0]
    let m = unsafe { _mm_max_ps(m, s) }; // all lanes = max(a0..a3)
    unsafe { _mm_cvtss_f32(m) }
}

/// Horizontal argmax of the 4 lanes: `(max value, its index)`.
///
/// Ties resolve to the lowest lane. The index is read from `idx` at the
/// first lane equal to the max.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn argmax_pair_128(v: __m128, idx: __m128i) -> (f32, usize) {
    // NaN lanes must never win: mask them to -inf before the reduction.
    let nan = unsafe { _mm_cmpunord_ps(v, v) };
    let clean = unsafe {
        _mm_or_ps(
            _mm_and_ps(nan, _mm_set1_ps(f32::NEG_INFINITY)),
            _mm_andnot_ps(nan, v),
        )
    };
    let m = unsafe { hmax_128(clean) };
    let eq = unsafe { _mm_cmpeq_ps(v, _mm_set1_ps(m)) };
    let mask = unsafe { _mm_movemask_ps(eq) };
    if mask == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f32::NAN, 0);
    }
    let mut idxs = [0_i32; 4];
    unsafe { _mm_storeu_si128(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes (the chunk
    // loop can hold a later chunk's tie in a lower lane).
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().enumerate() {
        if mask & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

/// Horizontal argmin of the 4 lanes: `(min value, its index)`.
///
/// Ties resolve to the first occurrence: the lowest global index among the
/// lanes equal to the min.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn argmin_pair_128(v: __m128, idx: __m128i) -> (f32, usize) {
    // NaN lanes must never win: mask them to +inf before the reduction.
    let nan = unsafe { _mm_cmpunord_ps(v, v) };
    let clean = unsafe {
        _mm_or_ps(
            _mm_and_ps(nan, _mm_set1_ps(f32::INFINITY)),
            _mm_andnot_ps(nan, v),
        )
    };
    let m = unsafe { hmin_128(clean) };
    let eq = unsafe { _mm_cmpeq_ps(v, _mm_set1_ps(m)) };
    let mask = unsafe { _mm_movemask_ps(eq) };
    if mask == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f32::NAN, 0);
    }
    let mut idxs = [0_i32; 4];
    unsafe { _mm_storeu_si128(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().enumerate() {
        if mask & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

// ===========================================================================
// f64 (double-precision) kernels. SSE2 `__m128d` = 2 lanes. Same contracts
// as the f32 versions; the horizontal helpers are explicit because SSE2 has
// no `hadd_pd` (shuffle + add instead).
// ===========================================================================

/// Horizontal sum of the 2 lanes in a `__m128d` register.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hsum_128d(v: __m128d) -> f64 {
    // SAFETY: caller guarantees SSE2.
    let hi = unsafe { _mm_unpackhi_pd(v, v) }; // [a1, a1]
    let s = unsafe { _mm_add_sd(v, hi) }; // [a0+a1, a1]
    unsafe { _mm_cvtsd_f64(s) }
}

/// Horizontal product of the 2 lanes in a `__m128d` register.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hprod_128d(v: __m128d) -> f64 {
    // SAFETY: caller guarantees SSE2.
    let hi = unsafe { _mm_unpackhi_pd(v, v) }; // [a1, a1]
    let m = unsafe { _mm_mul_sd(v, hi) }; // [a0*a1, a1]
    unsafe { _mm_cvtsd_f64(m) }
}

/// Horizontal minimum of the 2 lanes in a `__m128d` register.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hmin_128d(v: __m128d) -> f64 {
    // SAFETY: caller guarantees SSE2.
    let hi = unsafe { _mm_unpackhi_pd(v, v) }; // [a1, a1]
    let m = unsafe { _mm_min_sd(v, hi) }; // [min(a0,a1), a1]
    unsafe { _mm_cvtsd_f64(m) }
}

/// Horizontal maximum of the 2 lanes in a `__m128d` register.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hmax_128d(v: __m128d) -> f64 {
    // SAFETY: caller guarantees SSE2.
    let hi = unsafe { _mm_unpackhi_pd(v, v) }; // [a1, a1]
    let m = unsafe { _mm_max_sd(v, hi) }; // [max(a0,a1), a1]
    unsafe { _mm_cvtsd_f64(m) }
}

/// Horizontal argmax of the 2 f64 lanes: `(max value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn argmax_pair_128d(v: __m128d, idx: __m128i) -> (f64, usize) {
    // NaN lanes must never win: mask them to -inf before the reduction.
    let nan = unsafe { _mm_cmpunord_pd(v, v) };
    let clean = unsafe {
        _mm_or_pd(
            _mm_and_pd(nan, _mm_set1_pd(f64::NEG_INFINITY)),
            _mm_andnot_pd(nan, v),
        )
    };
    let m = unsafe { hmax_128d(clean) };
    let eq = unsafe { _mm_cmpeq_pd(v, _mm_set1_pd(m)) };
    let mask = unsafe { _mm_movemask_pd(eq) };
    if mask == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f64::NAN, 0);
    }
    // 4 i32: the store is 16 bytes; each f64 lane's index occupies an i32
    // pair (see the invocation's `$vidx` = [0,0,1,1]).
    let mut idxs = [0_i32; 4];
    unsafe { _mm_storeu_si128(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().step_by(2).enumerate() {
        if mask & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

/// Horizontal argmin of the 2 f64 lanes: `(min value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports SSE2.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn argmin_pair_128d(v: __m128d, idx: __m128i) -> (f64, usize) {
    // NaN lanes must never win: mask them to +inf before the reduction.
    let nan = unsafe { _mm_cmpunord_pd(v, v) };
    let clean = unsafe {
        _mm_or_pd(
            _mm_and_pd(nan, _mm_set1_pd(f64::INFINITY)),
            _mm_andnot_pd(nan, v),
        )
    };
    let m = unsafe { hmin_128d(clean) };
    let eq = unsafe { _mm_cmpeq_pd(v, _mm_set1_pd(m)) };
    let mask = unsafe { _mm_movemask_pd(eq) };
    if mask == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f64::NAN, 0);
    }
    // 4 i32: the store is 16 bytes; each f64 lane's index occupies an i32
    // pair (see the invocation's `$vidx` = [0,0,1,1]).
    let mut idxs = [0_i32; 4];
    unsafe { _mm_storeu_si128(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().step_by(2).enumerate() {
        if mask & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

// f64 reductions and maps for SSE2 (2 lanes).
crate::simd_reduce!(
    sum_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    _mm_setzero_pd(),
    _mm_add_pd,
    |v| unsafe { hsum_128d(v) },
    |r, v| r + v
);

crate::simd_reduce!(
    prod_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    _mm_set1_pd(1.0),
    _mm_mul_pd,
    |v| unsafe { hprod_128d(v) },
    |r, v| r * v
);

crate::simd_reduce!(
    min_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    _mm_set1_pd(f64::INFINITY),
    _mm_min_pd,
    |v| unsafe { hmin_128d(v) },
    f64::min
);

crate::simd_reduce!(
    max_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    _mm_set1_pd(f64::NEG_INFINITY),
    _mm_max_pd,
    |v| unsafe { hmax_128d(v) },
    f64::max
);

crate::simd_reduce!(
    sum_sq_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    _mm_setzero_pd(),
    |acc: __m128d, v: __m128d| _mm_add_pd(acc, _mm_mul_pd(v, v)),
    |v| unsafe { hsum_128d(v) },
    |r: f64, v: f64| r + v * v
);

crate::simd_reduce!(
    l1_norm_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    _mm_setzero_pd(),
    |acc: __m128d, v: __m128d| _mm_add_pd(acc, _mm_andnot_pd(_mm_set1_pd(-0.0), v)),
    |v| unsafe { hsum_128d(v) },
    |r: f64, v: f64| r + v.abs()
);

crate::simd_reduce!(
    max_norm_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    _mm_set1_pd(0.0),
    |acc: __m128d, v: __m128d| _mm_max_pd(acc, _mm_andnot_pd(_mm_set1_pd(-0.0), v)),
    |v| unsafe { hmax_128d(v) },
    |r: f64, v: f64| f64::max(r, v.abs())
);

crate::simd_reduce2!(
    dot_f64,
    f64,
    ["sse2"],
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    _mm_setzero_pd(),
    |acc: __m128d, a: __m128d, b: __m128d| _mm_add_pd(acc, _mm_mul_pd(a, b)),
    |v| unsafe { hsum_128d(v) },
    |r, a, b| r + a * b
);

crate::simd_argminmax!(
    argmax_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    // i32-pair duplicated indices: the f64 mask blend covers 64-bit lanes.
    _mm_setr_epi32(0, 0, 1, 1),
    _mm_set1_epi32,
    _mm_add_epi32,
    // NaN-aware dethrone: non-NaN candidate wins over NaN seed or a
    // strictly greater value (see scalar `argmax`).
    |a: __m128d, b: __m128d| unsafe {
        let gt = _mm_cmpgt_pd(a, b);
        let nan_b = _mm_cmpunord_pd(b, b);
        let nan_a = _mm_cmpunord_pd(a, a);
        _mm_andnot_pd(nan_a, _mm_or_pd(gt, nan_b))
    },
    |mask: __m128d, a: __m128d, b: __m128d| unsafe {
        _mm_or_pd(_mm_and_pd(mask, a), _mm_andnot_pd(mask, b))
    },
    |mask: __m128d, a: __m128i, b: __m128i| unsafe {
        let m = _mm_castpd_si128(mask);
        _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b))
    },
    |a: f64, b: f64| a > b,
    |v, idx| unsafe { argmax_pair_128d(v, idx) }
);

crate::simd_argminmax!(
    argmin_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    // i32-pair duplicated indices: the f64 mask blend covers 64-bit lanes.
    _mm_setr_epi32(0, 0, 1, 1),
    _mm_set1_epi32,
    _mm_add_epi32,
    // NaN-aware dethrone (see argmax_f64 above).
    |a: __m128d, b: __m128d| unsafe {
        let lt = _mm_cmplt_pd(a, b);
        let nan_b = _mm_cmpunord_pd(b, b);
        let nan_a = _mm_cmpunord_pd(a, a);
        _mm_andnot_pd(nan_a, _mm_or_pd(lt, nan_b))
    },
    |mask: __m128d, a: __m128d, b: __m128d| unsafe {
        _mm_or_pd(_mm_and_pd(mask, a), _mm_andnot_pd(mask, b))
    },
    |mask: __m128d, a: __m128i, b: __m128i| unsafe {
        let m = _mm_castpd_si128(mask);
        _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b))
    },
    |a: f64, b: f64| a < b,
    |v, idx| unsafe { argmin_pair_128d(v, idx) }
);

// f64 elementwise maps for SSE2 (2 lanes).
#[cfg(feature = "alloc")]
crate::simd_map!(
    sqrt_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v| unsafe { _mm_sqrt_pd(v) },
    |x: f64| crate::kernels::sqrt::sqrt_f64(x)
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    rsqrt_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v: __m128d| unsafe { _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(v)) },
    |x: f64| 1.0 / crate::kernels::sqrt::sqrt_f64(x)
);

#[cfg(feature = "alloc")]
crate::simd_map_param!(
    clip_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v: __m128d, lo: f64, hi: f64| unsafe {
        _mm_min_pd(_mm_max_pd(v, _mm_set1_pd(lo)), _mm_set1_pd(hi))
    },
    |x: f64, lo: f64, hi: f64| x.clamp(lo, hi)
);

// f64 vector exp for SSE2 (2 lanes).
#[cfg(feature = "alloc")]
crate::simd_exp_f64!(
    vexp_128d,
    "sse2",
    __m128d,
    __m128i,
    |s| unsafe { _mm_set1_pd(s) },
    |i| unsafe { _mm_set1_epi64x(i) },
    |a, b| unsafe { _mm_mul_pd(a, b) },
    |a, b| unsafe { _mm_add_pd(a, b) },
    |a, b| unsafe { _mm_sub_pd(a, b) },
    |v| unsafe { _mm_castsi128_pd(v) },
    // Round-to-nearest without SSE4.1: trunc(v + copysign(0.5, v))
    // (round-half-away-from-zero; ≤1 ulp difference from ties-even at the
    // exp boundary, within tolerance). Conversion goes through i32 (|n| ≤
    // 1024 fits): `_mm_cvttpd_epi64` is AVX-512DQ, not SSE2 — using it
    // here SIGILLs on non-AVX-512 CPUs.
    |v| unsafe {
        let sign = _mm_and_pd(v, _mm_castsi128_pd(_mm_set1_epi64x(i64::MIN)));
        let half = _mm_or_pd(sign, _mm_set1_pd(0.5));
        let n32 = _mm_cvttpd_epi32(_mm_add_pd(v, half));
        // Sign-extend i32 → i64 with SSE2-only ops (|n| ≤ 1024 fits i32).
        let sign_bits = _mm_srai_epi32(n32, 31);
        _mm_unpacklo_epi32(n32, sign_bits)
    },
    |v| unsafe {
        // Reverse: take the low i32 of each i64 lane (values fit), pack
        // them into the low qword, and convert i32 → f64 (SSE2).
        // `_mm_cvtepi64_pd` is AVX-512DQ, not SSE2.
        // v = [n0, s0, n1, s1] as i32; pick src[0], src[2] → [n0, n1, ..].
        let lo = _mm_shuffle_epi32(v, 0b00_00_10_00);
        _mm_cvtepi32_pd(lo)
    },
    |v| unsafe { _mm_slli_epi64(v, 52) },
    |a, b| unsafe { _mm_add_epi64(a, b) },
    // Signed 64-bit compare in the float domain: n_int and the clamp
    // constants (±1024) are exactly representable in f64, and SSE2 has no
    // `_mm_cmpgt_epi64` (that's SSE4.2).
    |a, b| unsafe { _mm_castpd_si128(_mm_cmpgt_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b))) },
    |a, b| unsafe { _mm_castpd_si128(_mm_cmplt_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b))) },
    |a, b| unsafe { _mm_and_si128(a, b) },
    |a, b| unsafe { _mm_andnot_si128(a, b) },
    |a, b| unsafe { _mm_or_si128(a, b) }
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    exp_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v: __m128d| unsafe { vexp_128d(v) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

#[cfg(feature = "alloc")]
crate::simd_softmax!(
    softmax_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |a, b| unsafe { _mm_max_pd(a, b) },
    |a, b| unsafe { _mm_sub_pd(a, b) },
    |v| unsafe { vexp_128d(v) },
    |a, b| unsafe { _mm_add_pd(a, b) },
    |a, b| unsafe { _mm_mul_pd(a, b) },
    |v| unsafe { hsum_128d(v) },
    |v| unsafe { hmax_128d(v) },
    |s| unsafe { _mm_set1_pd(s) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    sigmoid_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v: __m128d| unsafe {
        _mm_div_pd(
            _mm_set1_pd(1.0),
            _mm_add_pd(
                _mm_set1_pd(1.0),
                vexp_128d(_mm_xor_pd(v, _mm_castsi128_pd(_mm_set1_epi64x(i64::MIN)))),
            ),
        )
    },
    |x: f64| 1.0 / (1.0 + crate::kernels::exp::exp_f64(-x))
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    silu_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v: __m128d| unsafe {
        _mm_div_pd(
            v,
            _mm_add_pd(
                _mm_set1_pd(1.0),
                vexp_128d(_mm_xor_pd(v, _mm_castsi128_pd(_mm_set1_epi64x(i64::MIN)))),
            ),
        )
    },
    |x: f64| x / (1.0 + crate::kernels::exp::exp_f64(-x))
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    gelu_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v: __m128d| unsafe {
        let x2 = _mm_mul_pd(v, v);
        let x3 = _mm_mul_pd(x2, v);
        let z = _mm_mul_pd(
            _mm_set1_pd(0.797_884_560_802_865_4),
            _mm_add_pd(v, _mm_mul_pd(_mm_set1_pd(0.044_715), x3)),
        );
        let e = vexp_128d(_mm_add_pd(z, z));
        let tanh_z = _mm_sub_pd(
            _mm_set1_pd(1.0),
            _mm_div_pd(_mm_set1_pd(2.0), _mm_add_pd(e, _mm_set1_pd(1.0))),
        );
        _mm_mul_pd(
            _mm_set1_pd(0.5),
            _mm_mul_pd(v, _mm_add_pd(_mm_set1_pd(1.0), tanh_z)),
        )
    },
    |x: f64| {
        let z = 0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x);
        let tanh_z = 1.0 - 2.0 / (crate::kernels::exp::exp_f64(2.0 * z) + 1.0);
        0.5 * x * (1.0 + tanh_z)
    }
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    relu_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v: __m128d| unsafe { _mm_max_pd(v, _mm_set1_pd(0.0)) },
    |x: f64| x.max(0.0)
);

// Tanh map (f64): tanh(x) = 1 - 2/(exp(2x)+1).
#[cfg(feature = "alloc")]
crate::simd_map!(
    tanh_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    |v: __m128d| unsafe {
        // Piecewise: Horner series through x¹³ for |x| < 0.1 (truncation
        // < 0.1 ulp there; the exp form cancels to 0), exp form beyond.
        let y = _mm_mul_pd(v, v);
        let p = _mm_set1_pd(0.003_592_128_572_437_055);
        let p = _mm_add_pd(_mm_mul_pd(p, y), _mm_set1_pd(-0.008_863_235_529_902_197));
        let p = _mm_add_pd(_mm_mul_pd(p, y), _mm_set1_pd(0.021_869_488_536_155_2));
        let p = _mm_add_pd(_mm_mul_pd(p, y), _mm_set1_pd(-0.053_968_253_968_253_97));
        let p = _mm_add_pd(_mm_mul_pd(p, y), _mm_set1_pd(0.133_333_333_333_333_33));
        let p = _mm_add_pd(_mm_mul_pd(p, y), _mm_set1_pd(-0.333_333_333_333_333_3));
        let series = _mm_mul_pd(v, _mm_add_pd(_mm_mul_pd(p, y), _mm_set1_pd(1.0)));
        let e = vexp_128d(_mm_add_pd(v, v));
        let em = _mm_min_pd(e, _mm_set1_pd(f64::MAX));
        let ratio = _mm_div_pd(
            _mm_sub_pd(em, _mm_set1_pd(1.0)),
            _mm_add_pd(em, _mm_set1_pd(1.0)),
        );
        let big = _mm_or_pd(ratio, _mm_and_pd(_mm_set1_pd(-0.0), v));
        let small = _mm_cmplt_pd(_mm_andnot_pd(_mm_set1_pd(-0.0), v), _mm_set1_pd(0.1));
        _mm_or_pd(_mm_and_pd(small, series), _mm_andnot_pd(small, big))
    },
    |x: f64| {
        if x.abs() < 0.1 {
            let y = x * x;
            let p = 0.003_592_128_572_437_055_f64;
            let p = p.mul_add(y, -0.008_863_235_529_902_197);
            let p = p.mul_add(y, 0.021_869_488_536_155_2);
            let p = p.mul_add(y, -0.053_968_253_968_253_97);
            let p = p.mul_add(y, 0.133_333_333_333_333_33);
            let p = p.mul_add(y, -0.333_333_333_333_333_3);
            x * p.mul_add(y, 1.0)
        } else {
            let e = crate::kernels::exp::exp_f64(2.0 * x);
            if e.is_infinite() {
                x.signum()
            } else {
                (e - 1.0) / (e + 1.0)
            }
        }
    }
);

// RMS norm (f64).
#[cfg(feature = "alloc")]
crate::simd_rms_norm!(
    rms_norm_f64,
    f64,
    "sse2",
    2,
    |p| unsafe { _mm_loadu_pd(p) },
    |p, v| unsafe { _mm_storeu_pd(p, v) },
    _mm_setzero_pd(),
    |acc: __m128d, v: __m128d| _mm_add_pd(acc, _mm_mul_pd(v, v)),
    |v| unsafe { hsum_128d(v) },
    |v, inv| _mm_mul_pd(v, _mm_set1_pd(inv)),
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
    fn sum_matches_scalar_when_sse2_available() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 47, 4_096);
            // Products of 2^n overflow f32 quickly; cap prod at small
            // lengths so the result stays exactly representable.
            let prod_len = len.min(64);
            let prod_data = exact_data(prod_len, 48, 2);
            let a = exact_data(len, 53, 64);
            let b = exact_data(len, 59, 64);

            // SAFETY: tested inside the sse2 detection guard.
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
                // SAFETY: tested inside the sse2 detection guard.
                unsafe {
                    assert_eq!(min(&data), exact_min(&data), "min mismatch for len {len}");
                    assert_eq!(max(&data), exact_max(&data), "max mismatch for len {len}");
                }
            }
        }
    }

    #[test]
    fn sum_sq_matches_scalar_when_sse2_available() {
        if !std::arch::is_x86_feature_detected!("sse2") {
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
    fn l1_norm_matches_scalar_when_sse2_available() {
        if !std::arch::is_x86_feature_detected!("sse2") {
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
    fn max_norm_matches_scalar_when_sse2_available() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
    fn softmax_matches_scalar_when_sse2_available() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 2.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the sse2 detection guard.
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

    /// Vector exp must match the scalar (f64-based) exp to ≤ 2 ulp across
    /// the softmax-relevant normal range, both signs. The scalar exp also
    /// returns f32 denormals below x < -87.3; the vector saturates those to
    /// 0 (documented difference — denormals contribute < 1e-38 to a softmax
    /// sum and are normalized away). Saturation itself is checked separately.
    #[cfg(feature = "alloc")]
    #[test]
    fn vexp_matches_scalar_exp_full_range() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        for i in -8700..8800 {
            let x = i as f32 * 0.01;
            let v = unsafe { _mm_set1_ps(x) };
            let r = unsafe { vexp_128(v) };
            let mut arr = [0.0_f32; 4];
            // SAFETY: storeu needs no alignment.
            unsafe { _mm_storeu_ps(arr.as_mut_ptr(), r) };
            let scalar = crate::kernels::exp::exp(x);
            let ours = arr[0];
            let ulps = ulps_f32(ours, scalar);
            assert!(ulps <= 2, "x={x}: vexp={ours} scalar={scalar} ulps={ulps}");
        }
        // Saturation: below the normal range, the vector gives 0 (scalar may
        // give a denormal — documented); above it, inf.
        let v = unsafe { _mm_set1_ps(-100.0) };
        let r = unsafe { vexp_128(v) };
        let mut arr = [0.0_f32; 4];
        // SAFETY: storeu needs no alignment.
        unsafe { _mm_storeu_ps(arr.as_mut_ptr(), r) };
        assert_eq!(arr[0], 0.0);
        let v = unsafe { _mm_set1_ps(100.0) };
        let r = unsafe { vexp_128(v) };
        // SAFETY: storeu needs no alignment.
        unsafe { _mm_storeu_ps(arr.as_mut_ptr(), r) };
        assert_eq!(arr[0], f32::INFINITY);
    }

    #[cfg(feature = "alloc")]
    fn ulps_f32(a: f32, b: f32) -> i64 {
        if a == b {
            return 0;
        }
        let ia = i64::from(a.to_bits());
        let ib = i64::from(b.to_bits());
        let (sa, sb) = (ia < 0, ib < 0);
        let ia = if sa { 0x8000_0000 - ia } else { ia };
        let ib = if sb { 0x8000_0000 - ib } else { ib };
        (ia - ib).unsigned_abs().cast_signed()
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
        if !std::arch::is_x86_feature_detected!("sse2") {
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
