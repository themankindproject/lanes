//! AVX2 (256-bit) SIMD kernel implementations for x86-64.
//!
//! These functions require the `avx2` and `fma` CPU features. The caller
//! (dispatch layer) must verify these features are available before
//! invoking any function in this module — see `platform::supports`.
//!
//! Floating-point semantics: sums/dot products accumulate in 8-lane
//! vectors and then combine via a horizontal reduction, so the reduction
//! order differs from the scalar kernels. `min`/`max` follow the AVX
//! `vminps`/`vmaxps` hardware semantics, which propagate a NaN present in
//! the data, unlike the scalar `f32::min`/`f32::max`. For NaN-free inputs
//! the SIMD and scalar results agree exactly.
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

// Sum reduction: accumulate 8-wide, horizontal-sum, scalar tail.
crate::simd_reduce!(
    sum,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_setzero_ps(),
    _mm256_add_ps,
    |v| unsafe { hsum_256(v) },
    |r, v| r + v
);

// Product reduction: 8-wide multiply, scalar-multiply tail.
crate::simd_reduce!(
    prod,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_set1_ps(1.0),
    _mm256_mul_ps,
    |v| unsafe { hprod_256(v) },
    |r, v| r * v
);

// Minimum reduction: `vminps` semantics, `minf` tail.
crate::simd_reduce!(
    min,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_set1_ps(f32::INFINITY),
    _mm256_min_ps,
    |v| unsafe { hmin_256(v) },
    f32::min
);

// Maximum reduction: `vmaxps` semantics, `maxf` tail.
crate::simd_reduce!(
    max,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_set1_ps(f32::NEG_INFINITY),
    _mm256_max_ps,
    |v| unsafe { hmax_256(v) },
    f32::max
);
// Sum of squares: 8-wide multiply-accumulate (acc += v*v).
crate::simd_reduce!(
    sum_sq,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_setzero_ps(),
    |acc: __m256, v: __m256| _mm256_add_ps(acc, _mm256_mul_ps(v, v)),
    |v| unsafe { hsum_256(v) },
    |r: f32, v: f32| r + v * v
);

// L1 norm: sum of absolute values.
crate::simd_reduce!(
    l1_norm,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_setzero_ps(),
    |acc: __m256, v: __m256| _mm256_add_ps(acc, _mm256_andnot_ps(_mm256_set1_ps(-0.0), v)),
    |v| unsafe { hsum_256(v) },
    |r: f32, v: f32| r + v.abs()
);

// Max norm: maximum absolute value.
crate::simd_reduce!(
    max_norm,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_set1_ps(0.0),
    |acc: __m256, v: __m256| _mm256_max_ps(acc, _mm256_andnot_ps(_mm256_set1_ps(-0.0), v)),
    |v| unsafe { hmax_256(v) },
    |r: f32, v: f32| f32::max(r, v.abs())
);

// Argmax: index of the first occurrence of the maximum.
crate::simd_argminmax!(
    argmax,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    _mm256_set1_epi32,
    _mm256_add_epi32,
    // NaN-aware dethrone (see scalar `argmax`).
    |a, b| unsafe {
        let gt = _mm256_cmp_ps(a, b, _CMP_GT_OQ);
        let nan_b = _mm256_cmp_ps(b, b, _CMP_UNORD_Q);
        let nan_a = _mm256_cmp_ps(a, a, _CMP_UNORD_Q);
        _mm256_andnot_ps(nan_a, _mm256_or_ps(gt, nan_b))
    },
    |mask: __m256, a: __m256, b: __m256| unsafe { _mm256_blendv_ps(b, a, mask) },
    |mask: __m256, a: __m256i, b: __m256i| {
        let m = unsafe { _mm256_castps_si256(mask) };
        // SAFETY: caller guarantees AVX2; blendv_epi8 selects per byte and
        // the mask is all-ones per dword lane, so whole indices are picked.
        unsafe { _mm256_blendv_epi8(b, a, m) }
    },
    |cand: f32, cur: f32| cand > cur,
    |v, iv| unsafe { argmax_pair_256(v, iv) }
);

// Argmin: index of the first occurrence of the minimum.
crate::simd_argminmax!(
    argmin,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    _mm256_set1_epi32,
    _mm256_add_epi32,
    // NaN-aware dethrone (see argmax above).
    |a, b| unsafe {
        let lt = _mm256_cmp_ps(a, b, _CMP_LT_OQ);
        let nan_b = _mm256_cmp_ps(b, b, _CMP_UNORD_Q);
        let nan_a = _mm256_cmp_ps(a, a, _CMP_UNORD_Q);
        _mm256_andnot_ps(nan_a, _mm256_or_ps(lt, nan_b))
    },
    |mask: __m256, a: __m256, b: __m256| unsafe { _mm256_blendv_ps(b, a, mask) },
    |mask: __m256, a: __m256i, b: __m256i| {
        let m = unsafe { _mm256_castps_si256(mask) };
        // SAFETY: caller guarantees AVX2; blendv_epi8 selects per byte and
        // the mask is all-ones per dword lane, so whole indices are picked.
        unsafe { _mm256_blendv_epi8(b, a, m) }
    },
    |cand: f32, cur: f32| cand < cur,
    |v, iv| unsafe { argmin_pair_256(v, iv) }
);

// Dot product: 8-wide fused multiply-accumulate (AVX2 + FMA).
crate::simd_reduce2!(
    dot,
    f32,
    ["avx2", "fma"],
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_setzero_ps(),
    |acc, va, vb| _mm256_fmadd_ps(va, vb, acc),
    |v| unsafe { hsum_256(v) },
    |r, a, b| r + a * b
);

// Softmax: 3-pass map (max → exp+sum → scale). exp is per-lane scalar.
// Uses the crate's `no_std` `exp`, so available in all builds.
crate::simd_softmax!(
    softmax,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    _mm256_max_ps,
    _mm256_sub_ps,
    |v| unsafe { vexp_256(v) },
    _mm256_add_ps,
    _mm256_mul_ps,
    |v| unsafe { hsum_256(v) },
    |v| unsafe { hmax_256(v) },
    |s| unsafe { _mm256_set1_ps(s) },
    |x: f32| crate::kernels::exp::exp(x)
);

crate::simd_map!(
    sigmoid,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe {
        // Saturated fast path: all lanes already at 0/1 (skip the exp).
        let pos = _mm256_cmp_ps(v, _mm256_set1_ps(16.64), _CMP_GT_OQ);
        let neg = _mm256_cmp_ps(v, _mm256_set1_ps(-88.73), _CMP_LT_OQ);
        if _mm256_movemask_ps(_mm256_or_ps(pos, neg)) == 0xFF {
            return _mm256_and_ps(pos, _mm256_set1_ps(1.0));
        }
        _mm256_div_ps(
            _mm256_set1_ps(1.0),
            _mm256_add_ps(
                _mm256_set1_ps(1.0),
                vexp_256(_mm256_xor_ps(
                    v,
                    _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN)),
                )),
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
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe {
        // Saturated fast path: silu(x) = x for x > 16.64, 0 for x < -88.
        let pos = _mm256_cmp_ps(v, _mm256_set1_ps(16.64), _CMP_GT_OQ);
        let neg = _mm256_cmp_ps(v, _mm256_set1_ps(-88.73), _CMP_LT_OQ);
        if _mm256_movemask_ps(_mm256_or_ps(pos, neg)) == 0xFF {
            return _mm256_and_ps(pos, v);
        }
        _mm256_div_ps(
            v,
            _mm256_add_ps(
                _mm256_set1_ps(1.0),
                vexp_256(_mm256_xor_ps(
                    v,
                    _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN)),
                )),
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
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe {
        // Saturated fast path: gelu(x) = x for x > 7.0, 0 for x < -7.0.
        let pos = _mm256_cmp_ps(v, _mm256_set1_ps(7.0), _CMP_GT_OQ);
        let neg = _mm256_cmp_ps(v, _mm256_set1_ps(-7.0), _CMP_LT_OQ);
        if _mm256_movemask_ps(_mm256_or_ps(pos, neg)) == 0xFF {
            return _mm256_and_ps(pos, v);
        }
        let x2 = _mm256_mul_ps(v, v);
        let x3 = _mm256_mul_ps(x2, v);
        let z = _mm256_mul_ps(
            _mm256_set1_ps(0.797_884_6),
            _mm256_add_ps(v, _mm256_mul_ps(_mm256_set1_ps(0.044_715), x3)),
        );
        let e = vexp_256(_mm256_add_ps(z, z));
        let tanh_z = _mm256_sub_ps(
            _mm256_set1_ps(1.0),
            _mm256_div_ps(_mm256_set1_ps(2.0), _mm256_add_ps(e, _mm256_set1_ps(1.0))),
        );
        _mm256_mul_ps(
            _mm256_set1_ps(0.5),
            _mm256_mul_ps(v, _mm256_add_ps(_mm256_set1_ps(1.0), tanh_z)),
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
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe { _mm256_max_ps(v, _mm256_set1_ps(0.0)) },
    |x: f32| x.max(0.0)
);

// Tanh map: tanh(x) = 1 - 2/(exp(2x)+1) from the vector vexp kernel.
crate::simd_map!(
    tanh,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe {
        let a = _mm256_andnot_ps(_mm256_set1_ps(-0.0), v);
        // ±1 for |x| > 9.011, x for |x| < 2e-4, series for |x| < 0.1,
        // ratio (e-1)/(e+1) beyond (Sterbenz-exact, clamped for overflow).
        let big_mask = _mm256_cmp_ps(a, _mm256_set1_ps(9.011), _CMP_GT_OQ);
        if _mm256_movemask_ps(big_mask) == 0xFF {
            return _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(_mm256_set1_ps(-0.0), v));
        }
        let x2 = _mm256_mul_ps(v, v);
        let x4 = _mm256_mul_ps(x2, x2);
        let series = _mm256_add_ps(
            _mm256_sub_ps(v, _mm256_div_ps(_mm256_mul_ps(v, x2), _mm256_set1_ps(3.0))),
            _mm256_div_ps(_mm256_mul_ps(v, x4), _mm256_set1_ps(7.5)),
        );
        let e = vexp_256(_mm256_add_ps(v, v));
        let em = _mm256_min_ps(e, _mm256_set1_ps(f32::MAX));
        let ratio = _mm256_div_ps(
            _mm256_sub_ps(em, _mm256_set1_ps(1.0)),
            _mm256_add_ps(em, _mm256_set1_ps(1.0)),
        );
        let big = _mm256_or_ps(ratio, _mm256_and_ps(_mm256_set1_ps(-0.0), v));
        let ser_mask = _mm256_cmp_ps(a, _mm256_set1_ps(0.1), _CMP_LT_OQ);
        let small = _mm256_cmp_ps(a, _mm256_set1_ps(2e-4), _CMP_LT_OQ);
        let mid = _mm256_or_ps(
            _mm256_and_ps(ser_mask, series),
            _mm256_andnot_ps(ser_mask, big),
        );
        let result = _mm256_or_ps(_mm256_and_ps(small, v), _mm256_andnot_ps(small, mid));
        _mm256_or_ps(
            _mm256_and_ps(
                big_mask,
                _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(_mm256_set1_ps(-0.0), v)),
            ),
            _mm256_andnot_ps(big_mask, result),
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
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    _mm256_setzero_ps(),
    |acc: __m256, v: __m256| _mm256_add_ps(acc, _mm256_mul_ps(v, v)),
    |v| unsafe { hsum_256(v) },
    |v, inv| _mm256_mul_ps(v, _mm256_set1_ps(inv)),
    crate::kernels::sqrt::sqrt
);

// Exp map: per-element exp, vector vexp for chunks + scalar exp for tails.
crate::simd_map!(
    exp,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v: __m256| unsafe { vexp_256(v) },
    |x: f32| crate::kernels::exp::exp(x)
);
// Sqrt: one-pass map, native hardware sqrt (correctly rounded, IEEE).
crate::simd_map!(
    sqrt,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe { _mm256_sqrt_ps(v) },
    |x: f32| crate::kernels::sqrt::sqrt(x)
);

// Clip: one-pass map with lo/hi params, min(max(v, lo), hi).
crate::simd_map_param!(
    clip,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v: __m256, lo: f32, hi: f32| _mm256_min_ps(
        _mm256_max_ps(v, _mm256_set1_ps(lo)),
        _mm256_set1_ps(hi)
    ),
    |x: f32, lo: f32, hi: f32| x.clamp(lo, hi)
);
// Rsqrt: one-pass map, 1/sqrt(v) (exact via div+sqrt, not the ~12-bit
// hardware approximation — correctness-first).
crate::simd_map!(
    rsqrt,
    f32,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v: __m256| _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_sqrt_ps(v)),
    |x: f32| 1.0 / crate::kernels::sqrt::sqrt(x)
);
crate::simd_exp!(
    vexp_256,
    f32,
    "avx2",
    __m256,
    __m256i,
    |s| unsafe { _mm256_set1_ps(s) },
    |i| unsafe { _mm256_set1_epi32(i) },
    |a, b| unsafe { _mm256_mul_ps(a, b) },
    |a, b| unsafe { _mm256_add_ps(a, b) },
    |a, b| unsafe { _mm256_sub_ps(a, b) },
    |a, b| unsafe { _mm256_and_ps(a, b) },
    |a, b| unsafe { _mm256_andnot_ps(a, b) },
    |a, b| unsafe { _mm256_or_ps(a, b) },
    |a, b| unsafe { _mm256_cmp_ps(a, b, _CMP_GT_OQ) },
    |v| unsafe { _mm256_castsi256_ps(v) },
    |v| unsafe { _mm256_cvttps_epi32(v) },
    |v| unsafe { _mm256_slli_epi32(v, 23) },
    |a, b| unsafe { _mm256_add_epi32(a, b) },
    |a, b| unsafe { _mm256_cmpgt_epi32(a, b) },
    |a, b| unsafe { _mm256_cmpgt_epi32(b, a) },
    |a, b| unsafe { _mm256_and_si256(a, b) },
    |a, b| unsafe { _mm256_andnot_si256(a, b) },
    |a, b| unsafe { _mm256_or_si256(a, b) }
);

/// Horizontal sum of all 8 lanes in a `__m256` register.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (implied by the `target_feature`
/// gate on all public functions in this module).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_256(v: __m256) -> f32 {
    // SAFETY: caller guarantees AVX2; all intrinsics below require AVX2 or
    // the always-present SSE baseline.
    let hi128 = unsafe { _mm256_extractf128_ps(v, 1) };
    let lo128 = unsafe { _mm256_castps256_ps128(v) };
    let sum128 = unsafe { _mm_add_ps(lo128, hi128) };
    let shuf = unsafe { _mm_movehdup_ps(sum128) }; // [1,1,3,3]
    let sums = unsafe { _mm_add_ps(sum128, shuf) }; // [0+1, _, 2+3, _]
    let hi64 = unsafe { _mm_movehl_ps(sums, sums) }; // [2+3, _, _, _]
    let result = unsafe { _mm_add_ss(sums, hi64) }; // [0+1+2+3, _, _, _]
    unsafe { _mm_cvtss_f32(result) }
}

/// Horizontal product of all 8 lanes in a `__m256` register.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hprod_256(v: __m256) -> f32 {
    // SAFETY: caller guarantees AVX2; the 128-bit ops are SSE baseline.
    let hi128 = unsafe { _mm256_extractf128_ps(v, 1) };
    let lo128 = unsafe { _mm256_castps256_ps128(v) };
    let m = unsafe { _mm_mul_ps(lo128, hi128) }; // [p0*p4, p1*p5, p2*p6, p3*p7]
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0100_1110) }; // [a2, a3, a0, a1]
    let m = unsafe { _mm_mul_ps(m, s) }; // [a0*a2, a1*a3, ...]
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0000_0001) }; // [a1*a3, a0*a2, ...]
    let m = unsafe { _mm_mul_ps(m, s) }; // all lanes = p0*p1*p2*p3*p4*p5*p6*p7
    unsafe { _mm_cvtss_f32(m) }
}

/// Horizontal minimum of the 8 lanes in a `__m256` register.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hmin_256(v: __m256) -> f32 {
    // SAFETY: caller guarantees AVX2; the 128-bit ops are SSE baseline.
    let hi128 = unsafe { _mm256_extractf128_ps(v, 1) };
    let lo128 = unsafe { _mm256_castps256_ps128(v) };
    let m = unsafe { _mm_min_ps(lo128, hi128) }; // [min(l0,l4), min(l1,l5), min(l2,l6), min(l3,l7)]
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0100_1110) }; // [a2, a3, a0, a1]
    let m = unsafe { _mm_min_ps(m, s) }; // [min0124.., min(l1,l3,l5,l7), ...]
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0000_0001) }; // [a1, a0, a1, a0]
    let m = unsafe { _mm_min_ps(m, s) }; // all lanes = min(l0..l7)
    unsafe { _mm_cvtss_f32(m) }
}

/// Horizontal maximum of the 8 lanes in a `__m256` register.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hmax_256(v: __m256) -> f32 {
    // SAFETY: caller guarantees AVX2; the 128-bit ops are SSE baseline.
    let hi128 = unsafe { _mm256_extractf128_ps(v, 1) };
    let lo128 = unsafe { _mm256_castps256_ps128(v) };
    let m = unsafe { _mm_max_ps(lo128, hi128) };
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0100_1110) };
    let m = unsafe { _mm_max_ps(m, s) };
    let s = unsafe { _mm_shuffle_ps(m, m, 0b0000_0001) };
    let m = unsafe { _mm_max_ps(m, s) };
    unsafe { _mm_cvtss_f32(m) }
}

/// Horizontal argmax of the 8 lanes: `(max value, its index)`.
///
/// Ties resolve to the lowest lane. The index is read from `idx` at the
/// first lane equal to the max.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn argmax_pair_256(v: __m256, idx: __m256i) -> (f32, usize) {
    // NaN lanes must never win: mask them to -inf before the reduction.
    let nan = unsafe { _mm256_cmp_ps(v, v, _CMP_UNORD_Q) };
    let clean = unsafe {
        _mm256_or_ps(
            _mm256_and_ps(nan, _mm256_set1_ps(f32::NEG_INFINITY)),
            _mm256_andnot_ps(nan, v),
        )
    };
    let m = unsafe { hmax_256(clean) };
    let eq = unsafe { _mm256_cmp_ps(v, _mm256_set1_ps(m), _CMP_EQ_OQ) };
    let mask = unsafe { _mm256_movemask_ps(eq) };
    if mask == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f32::NAN, 0);
    }
    let mut idxs = [0_i32; 8];
    unsafe { _mm256_storeu_si256(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().enumerate() {
        if mask & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

/// Horizontal argmin of the 8 lanes: `(min value, its index)`.
///
/// Ties resolve to the lowest lane. The index is read from `idx` at the
/// first lane equal to the min.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn argmin_pair_256(v: __m256, idx: __m256i) -> (f32, usize) {
    // NaN lanes must never win: mask them to +inf before the reduction.
    let nan = unsafe { _mm256_cmp_ps(v, v, _CMP_UNORD_Q) };
    let clean = unsafe {
        _mm256_or_ps(
            _mm256_and_ps(nan, _mm256_set1_ps(f32::INFINITY)),
            _mm256_andnot_ps(nan, v),
        )
    };
    let m = unsafe { hmin_256(clean) };
    let eq = unsafe { _mm256_cmp_ps(v, _mm256_set1_ps(m), _CMP_EQ_OQ) };
    let mask = unsafe { _mm256_movemask_ps(eq) };
    if mask == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f32::NAN, 0);
    }
    let mut idxs = [0_i32; 8];
    unsafe { _mm256_storeu_si256(idxs.as_mut_ptr().cast(), idx) };
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
// f64 (double-precision) kernels. AVX2 `__m256d` = 4 lanes. Same contracts
// as the f32 versions; horizontals use `hadd_pd` + cross-128 extract.
// ===========================================================================

/// Horizontal sum of the 4 f64 lanes in a `__m256d` register.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_256d(v: __m256d) -> f64 {
    // SAFETY: caller guarantees AVX2.
    let hi128 = unsafe { _mm256_extractf128_pd(v, 1) };
    let lo128 = unsafe { _mm256_castpd256_pd128(v) };
    let sum128 = unsafe { _mm_add_pd(lo128, hi128) }; // [l0+l2, l1+l3]
    let hi = unsafe { _mm_unpackhi_pd(sum128, sum128) }; // [l1+l3, l1+l3]
    let s = unsafe { _mm_add_sd(sum128, hi) }; // [sum, _]
    unsafe { _mm_cvtsd_f64(s) }
}

/// Horizontal product of the 4 f64 lanes in a `__m256d` register.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hprod_256d(v: __m256d) -> f64 {
    // SAFETY: caller guarantees AVX2.
    let hi128 = unsafe { _mm256_extractf128_pd(v, 1) };
    let lo128 = unsafe { _mm256_castpd256_pd128(v) };
    let m = unsafe { _mm_mul_pd(lo128, hi128) }; // [l0*l2, l1*l3]
    let hi = unsafe { _mm_unpackhi_pd(m, m) }; // [l1*l3, l1*l3]
    let m = unsafe { _mm_mul_sd(m, hi) }; // [l0*l1*l2*l3, _]
    unsafe { _mm_cvtsd_f64(m) }
}

/// Horizontal minimum of the 4 f64 lanes in a `__m256d` register.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hmin_256d(v: __m256d) -> f64 {
    // SAFETY: caller guarantees AVX2.
    let hi128 = unsafe { _mm256_extractf128_pd(v, 1) };
    let lo128 = unsafe { _mm256_castpd256_pd128(v) };
    let m = unsafe { _mm_min_pd(lo128, hi128) }; // [min(l0,l2), min(l1,l3)]
    let hi = unsafe { _mm_unpackhi_pd(m, m) }; // [min(l1,l3), min(l1,l3)]
    let m = unsafe { _mm_min_sd(m, hi) }; // [min all, _]
    unsafe { _mm_cvtsd_f64(m) }
}

/// Horizontal maximum of the 4 f64 lanes in a `__m256d` register.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hmax_256d(v: __m256d) -> f64 {
    // SAFETY: caller guarantees AVX2.
    let hi128 = unsafe { _mm256_extractf128_pd(v, 1) };
    let lo128 = unsafe { _mm256_castpd256_pd128(v) };
    let m = unsafe { _mm_max_pd(lo128, hi128) }; // [max(l0,l2), max(l1,l3)]
    let hi = unsafe { _mm_unpackhi_pd(m, m) }; // [max(l1,l3), max(l1,l3)]
    let m = unsafe { _mm_max_sd(m, hi) }; // [max all, _]
    unsafe { _mm_cvtsd_f64(m) }
}

/// Horizontal argmax of the 4 f64 lanes: `(max value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn argmax_pair_256d(v: __m256d, idx: __m256i) -> (f64, usize) {
    // NaN lanes must never win: mask them to -inf before the reduction.
    let nan = unsafe { _mm256_cmp_pd(v, v, _CMP_UNORD_Q) };
    let clean = unsafe {
        _mm256_or_pd(
            _mm256_and_pd(nan, _mm256_set1_pd(f64::NEG_INFINITY)),
            _mm256_andnot_pd(nan, v),
        )
    };
    let m = unsafe { hmax_256d(clean) };
    let eq = unsafe { _mm256_cmp_pd(v, _mm256_set1_pd(m), _CMP_EQ_OQ) };
    let mask = unsafe { _mm256_movemask_pd(eq) };
    if mask == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f64::NAN, 0);
    }
    // 8 i32: the 256-bit store covers 8 lanes; each f64 lane's index
    // occupies an i32 pair (see the invocation's duplicated `$vidx`).
    let mut idxs = [0_i32; 8];
    unsafe { _mm256_storeu_si256(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().step_by(2).enumerate() {
        if mask & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

/// Horizontal argmin of the 4 f64 lanes: `(min value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn argmin_pair_256d(v: __m256d, idx: __m256i) -> (f64, usize) {
    // NaN lanes must never win: mask them to +inf before the reduction.
    let nan = unsafe { _mm256_cmp_pd(v, v, _CMP_UNORD_Q) };
    let clean = unsafe {
        _mm256_or_pd(
            _mm256_and_pd(nan, _mm256_set1_pd(f64::INFINITY)),
            _mm256_andnot_pd(nan, v),
        )
    };
    let m = unsafe { hmin_256d(clean) };
    let eq = unsafe { _mm256_cmp_pd(v, _mm256_set1_pd(m), _CMP_EQ_OQ) };
    let mask = unsafe { _mm256_movemask_pd(eq) };
    if mask == 0 {
        // All-NaN chunk: match the scalar seed (NaN value, index 0).
        return (f64::NAN, 0);
    }
    // 8 i32: the 256-bit store covers 8 lanes; each f64 lane's index
    // occupies an i32 pair (see the invocation's duplicated `$vidx`).
    let mut idxs = [0_i32; 8];
    unsafe { _mm256_storeu_si256(idxs.as_mut_ptr().cast(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i32::MAX;
    for (l, idxv) in idxs.iter().step_by(2).enumerate() {
        if mask & (1 << l) != 0 {
            best = best.min(*idxv);
        }
    }
    (m, best as usize)
}

// f64 reductions for AVX2 (4 lanes).
crate::simd_reduce!(
    sum_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    _mm256_setzero_pd(),
    _mm256_add_pd,
    |v| unsafe { hsum_256d(v) },
    |r, v| r + v
);

crate::simd_reduce!(
    prod_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    _mm256_set1_pd(1.0),
    _mm256_mul_pd,
    |v| unsafe { hprod_256d(v) },
    |r, v| r * v
);

crate::simd_reduce!(
    min_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    _mm256_set1_pd(f64::INFINITY),
    _mm256_min_pd,
    |v| unsafe { hmin_256d(v) },
    f64::min
);

crate::simd_reduce!(
    max_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    _mm256_set1_pd(f64::NEG_INFINITY),
    _mm256_max_pd,
    |v| unsafe { hmax_256d(v) },
    f64::max
);

crate::simd_reduce!(
    sum_sq_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    _mm256_setzero_pd(),
    |acc: __m256d, v: __m256d| _mm256_add_pd(acc, _mm256_mul_pd(v, v)),
    |v| unsafe { hsum_256d(v) },
    |r: f64, v: f64| r + v * v
);

crate::simd_reduce!(
    l1_norm_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    _mm256_setzero_pd(),
    |acc: __m256d, v: __m256d| _mm256_add_pd(acc, _mm256_andnot_pd(_mm256_set1_pd(-0.0), v)),
    |v| unsafe { hsum_256d(v) },
    |r: f64, v: f64| r + v.abs()
);

crate::simd_reduce!(
    max_norm_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    _mm256_set1_pd(0.0),
    |acc: __m256d, v: __m256d| _mm256_max_pd(acc, _mm256_andnot_pd(_mm256_set1_pd(-0.0), v)),
    |v| unsafe { hmax_256d(v) },
    |r: f64, v: f64| f64::max(r, v.abs())
);

crate::simd_reduce2!(
    dot_f64,
    f64,
    ["avx2", "fma"],
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    _mm256_setzero_pd(),
    |acc: __m256d, a: __m256d, b: __m256d| _mm256_fmadd_pd(a, b, acc),
    |v| unsafe { hsum_256d(v) },
    |r, a, b| r + a * b
);

crate::simd_argminmax!(
    argmax_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    // i32-pair duplicated indices: the f64 mask blend covers 64-bit lanes.
    _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3),
    _mm256_set1_epi32,
    _mm256_add_epi32,
    // NaN-aware dethrone (see scalar `argmax`).
    |a: __m256d, b: __m256d| unsafe {
        let gt = _mm256_cmp_pd(a, b, _CMP_GT_OQ);
        let nan_b = _mm256_cmp_pd(b, b, _CMP_UNORD_Q);
        let nan_a = _mm256_cmp_pd(a, a, _CMP_UNORD_Q);
        _mm256_andnot_pd(nan_a, _mm256_or_pd(gt, nan_b))
    },
    |mask: __m256d, a: __m256d, b: __m256d| unsafe { _mm256_blendv_pd(b, a, mask) },
    |mask: __m256d, a: __m256i, b: __m256i| unsafe {
        let m = _mm256_castpd_si256(mask);
        _mm256_or_si256(_mm256_and_si256(m, a), _mm256_andnot_si256(m, b))
    },
    |a: f64, b: f64| a > b,
    |v, idx| unsafe { argmax_pair_256d(v, idx) }
);

crate::simd_argminmax!(
    argmin_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    // i32-pair duplicated indices: the f64 mask blend covers 64-bit lanes.
    _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3),
    _mm256_set1_epi32,
    _mm256_add_epi32,
    // NaN-aware dethrone (see argmax_f64 above).
    |a: __m256d, b: __m256d| unsafe {
        let lt = _mm256_cmp_pd(a, b, _CMP_LT_OQ);
        let nan_b = _mm256_cmp_pd(b, b, _CMP_UNORD_Q);
        let nan_a = _mm256_cmp_pd(a, a, _CMP_UNORD_Q);
        _mm256_andnot_pd(nan_a, _mm256_or_pd(lt, nan_b))
    },
    |mask: __m256d, a: __m256d, b: __m256d| unsafe { _mm256_blendv_pd(b, a, mask) },
    |mask: __m256d, a: __m256i, b: __m256i| unsafe {
        let m = _mm256_castpd_si256(mask);
        _mm256_or_si256(_mm256_and_si256(m, a), _mm256_andnot_si256(m, b))
    },
    |a: f64, b: f64| a < b,
    |v, idx| unsafe { argmin_pair_256d(v, idx) }
);

// f64 elementwise maps for AVX2 (4 lanes).
crate::simd_map!(
    sqrt_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v| unsafe { _mm256_sqrt_pd(v) },
    |x: f64| crate::kernels::sqrt::sqrt_f64(x)
);

crate::simd_map!(
    rsqrt_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v: __m256d| unsafe { _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(v)) },
    |x: f64| 1.0 / crate::kernels::sqrt::sqrt_f64(x)
);

crate::simd_map_param!(
    clip_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v: __m256d, lo: f64, hi: f64| unsafe {
        _mm256_min_pd(_mm256_max_pd(v, _mm256_set1_pd(lo)), _mm256_set1_pd(hi))
    },
    |x: f64, lo: f64, hi: f64| x.clamp(lo, hi)
);

// f64 vector exp for AVX2 (4 lanes).
crate::simd_exp_f64!(
    vexp_256d,
    "avx2",
    __m256d,
    __m256i,
    |s| unsafe { _mm256_set1_pd(s) },
    |i| unsafe { _mm256_set1_epi64x(i) },
    |a, b| unsafe { _mm256_mul_pd(a, b) },
    |a, b| unsafe { _mm256_add_pd(a, b) },
    |a, b| unsafe { _mm256_sub_pd(a, b) },
    |v| unsafe { _mm256_castsi256_pd(v) },
    // Round-to-nearest: trunc(v + copysign(0.5, v)) in f64, converted to
    // i32 (|n| ≤ 1024 fits i32), then sign-extended to i64. `_mm256_cvttpd_epi64`
    // is AVX-512DQ, not AVX2 — using it here SIGILLs on AVX2-only CPUs.
    |v| unsafe {
        let sign = _mm256_and_pd(v, _mm256_castsi256_pd(_mm256_set1_epi64x(i64::MIN)));
        let half = _mm256_or_pd(sign, _mm256_set1_pd(0.5));
        let n32 = _mm256_cvttpd_epi32(_mm256_add_pd(v, half));
        _mm256_cvtepi32_epi64(n32)
    },
    |v| unsafe {
        // Reverse: extract the low i32 of each i64 lane (the values fit),
        // pack them into the low 128 bits, and convert i32 → f64.
        // `_mm256_cvtepi64_pd` is AVX-512DQ, not AVX2.
        let packed = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
        _mm256_cvtepi32_pd(_mm256_castsi256_si128(packed))
    },
    |v| unsafe { _mm256_slli_epi64(v, 52) },
    |a, b| unsafe { _mm256_add_epi64(a, b) },
    |a, b| unsafe { _mm256_cmpgt_epi64(a, b) },
    |a, b| unsafe { _mm256_cmpgt_epi64(b, a) },
    |a, b| unsafe { _mm256_and_si256(a, b) },
    |a, b| unsafe { _mm256_andnot_si256(a, b) },
    |a, b| unsafe { _mm256_or_si256(a, b) }
);

crate::simd_map!(
    exp_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v: __m256d| unsafe { vexp_256d(v) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

crate::simd_softmax!(
    softmax_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |a, b| unsafe { _mm256_max_pd(a, b) },
    |a, b| unsafe { _mm256_sub_pd(a, b) },
    |v| unsafe { vexp_256d(v) },
    |a, b| unsafe { _mm256_add_pd(a, b) },
    |a, b| unsafe { _mm256_mul_pd(a, b) },
    |v| unsafe { hsum_256d(v) },
    |v| unsafe { hmax_256d(v) },
    |s| unsafe { _mm256_set1_pd(s) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

crate::simd_map!(
    sigmoid_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v: __m256d| unsafe {
        // Saturated fast path: all lanes already at 0/1 (skip the exp).
        let pos = _mm256_cmp_pd(v, _mm256_set1_pd(36.74), _CMP_GT_OQ);
        let neg = _mm256_cmp_pd(v, _mm256_set1_pd(-744.0), _CMP_LT_OQ);
        if _mm256_movemask_pd(_mm256_or_pd(pos, neg)) == 0xF {
            return _mm256_and_pd(pos, _mm256_set1_pd(1.0));
        }
        _mm256_div_pd(
            _mm256_set1_pd(1.0),
            _mm256_add_pd(
                _mm256_set1_pd(1.0),
                vexp_256d(_mm256_xor_pd(
                    v,
                    _mm256_castsi256_pd(_mm256_set1_epi64x(i64::MIN)),
                )),
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
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v: __m256d| unsafe {
        // Saturated fast path: silu(x) = x for x > 36.74, 0 for x < -745.
        let pos = _mm256_cmp_pd(v, _mm256_set1_pd(36.74), _CMP_GT_OQ);
        let neg = _mm256_cmp_pd(v, _mm256_set1_pd(-744.0), _CMP_LT_OQ);
        if _mm256_movemask_pd(_mm256_or_pd(pos, neg)) == 0xF {
            return _mm256_and_pd(pos, v);
        }
        _mm256_div_pd(
            v,
            _mm256_add_pd(
                _mm256_set1_pd(1.0),
                vexp_256d(_mm256_xor_pd(
                    v,
                    _mm256_castsi256_pd(_mm256_set1_epi64x(i64::MIN)),
                )),
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
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v: __m256d| unsafe {
        // Saturated fast path: gelu(x) = x for x > 7.21, 0 for x < -7.21.
        let pos = _mm256_cmp_pd(v, _mm256_set1_pd(7.21), _CMP_GT_OQ);
        let neg = _mm256_cmp_pd(v, _mm256_set1_pd(-7.21), _CMP_LT_OQ);
        if _mm256_movemask_pd(_mm256_or_pd(pos, neg)) == 0xF {
            return _mm256_and_pd(pos, v);
        }
        let x2 = _mm256_mul_pd(v, v);
        let x3 = _mm256_mul_pd(x2, v);
        let z = _mm256_mul_pd(
            _mm256_set1_pd(0.797_884_560_802_865_4),
            _mm256_add_pd(v, _mm256_mul_pd(_mm256_set1_pd(0.044_715), x3)),
        );
        let e = vexp_256d(_mm256_add_pd(z, z));
        let tanh_z = _mm256_sub_pd(
            _mm256_set1_pd(1.0),
            _mm256_div_pd(_mm256_set1_pd(2.0), _mm256_add_pd(e, _mm256_set1_pd(1.0))),
        );
        _mm256_mul_pd(
            _mm256_set1_pd(0.5),
            _mm256_mul_pd(v, _mm256_add_pd(_mm256_set1_pd(1.0), tanh_z)),
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
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v: __m256d| unsafe { _mm256_max_pd(v, _mm256_set1_pd(0.0)) },
    |x: f64| x.max(0.0)
);

// Tanh map (f64): tanh(x) = 1 - 2/(exp(2x)+1).
crate::simd_map!(
    tanh_f64,
    f64,
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    |v: __m256d| unsafe {
        let a = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
        // ±1 for |x| > 19.062, x for |x| < 5e-8, series for |x| < 0.1,
        // ratio (e-1)/(e+1) beyond (Sterbenz-exact, clamped for overflow).
        let big_mask = _mm256_cmp_pd(a, _mm256_set1_pd(19.062), _CMP_GT_OQ);
        if _mm256_movemask_pd(big_mask) == 0xF {
            return _mm256_or_pd(_mm256_set1_pd(1.0), _mm256_and_pd(_mm256_set1_pd(-0.0), v));
        }
        let y = _mm256_mul_pd(v, v);
        let p = _mm256_set1_pd(0.003_592_128_572_437_055);
        let p = _mm256_add_pd(
            _mm256_mul_pd(p, y),
            _mm256_set1_pd(-0.008_863_235_529_902_197),
        );
        let p = _mm256_add_pd(_mm256_mul_pd(p, y), _mm256_set1_pd(0.021_869_488_536_155_2));
        let p = _mm256_add_pd(
            _mm256_mul_pd(p, y),
            _mm256_set1_pd(-0.053_968_253_968_253_97),
        );
        let p = _mm256_add_pd(
            _mm256_mul_pd(p, y),
            _mm256_set1_pd(0.133_333_333_333_333_33),
        );
        let p = _mm256_add_pd(
            _mm256_mul_pd(p, y),
            _mm256_set1_pd(-0.333_333_333_333_333_3),
        );
        let series = _mm256_mul_pd(v, _mm256_add_pd(_mm256_mul_pd(p, y), _mm256_set1_pd(1.0)));
        let e = vexp_256d(_mm256_add_pd(v, v));
        let em = _mm256_min_pd(e, _mm256_set1_pd(f64::MAX));
        let ratio = _mm256_div_pd(
            _mm256_sub_pd(em, _mm256_set1_pd(1.0)),
            _mm256_add_pd(em, _mm256_set1_pd(1.0)),
        );
        let big = _mm256_or_pd(ratio, _mm256_and_pd(_mm256_set1_pd(-0.0), v));
        let ser_mask = _mm256_cmp_pd(a, _mm256_set1_pd(0.1), _CMP_LT_OQ);
        let small = _mm256_cmp_pd(a, _mm256_set1_pd(2e-8), _CMP_LT_OQ);
        let mid = _mm256_or_pd(
            _mm256_and_pd(ser_mask, series),
            _mm256_andnot_pd(ser_mask, big),
        );
        let result = _mm256_or_pd(_mm256_and_pd(small, v), _mm256_andnot_pd(small, mid));
        _mm256_or_pd(
            _mm256_and_pd(
                big_mask,
                _mm256_or_pd(_mm256_set1_pd(1.0), _mm256_and_pd(_mm256_set1_pd(-0.0), v)),
            ),
            _mm256_andnot_pd(big_mask, result),
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
            let p = p.mul_add(y, -0.008_863_235_529_902_197);
            let p = p.mul_add(y, 0.021_869_488_536_155_2);
            let p = p.mul_add(y, -0.053_968_253_968_253_97);
            let p = p.mul_add(y, 0.133_333_333_333_333_33);
            let p = p.mul_add(y, -0.333_333_333_333_333_3);
            x * p.mul_add(y, 1.0)
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
    "avx2",
    4,
    |p| unsafe { _mm256_loadu_pd(p) },
    |p, v| unsafe { _mm256_storeu_pd(p, v) },
    _mm256_setzero_pd(),
    |acc: __m256d, v: __m256d| _mm256_add_pd(acc, _mm256_mul_pd(v, v)),
    |v| unsafe { hsum_256d(v) },
    |v, inv| _mm256_mul_pd(v, _mm256_set1_pd(inv)),
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
    fn sum_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 7, 4_096);
            // SAFETY: tested inside the avx2 detection guard.
            let simd = unsafe { sum(&data) };
            let scalar: f32 = data.iter().sum();
            assert_eq!(simd, scalar, "sum mismatch for len {len}");
        }
    }

    #[test]
    fn prod_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            // Small scale: products of values in [-16, 16] stay exactly
            // representable in f32 across all tested lengths, so backends
            // must agree exactly (large scales would legitimately round
            // differently per reduction order — a documented non-equality).
            let data = exact_data(len, 9, 16);
            // SAFETY: tested inside the avx2 detection guard.
            let simd = unsafe { prod(&data) };
            let scalar: f32 = data.iter().product();
            assert_eq!(simd, scalar, "prod mismatch for len {len}");
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn softmax_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 2.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the avx2 detection guard.
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

    #[test]
    fn min_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 11, 4_096);
            // SAFETY: tested inside the avx2 detection guard.
            let simd = unsafe { min(&data) };
            let scalar = data.iter().copied().reduce(f32::min).unwrap();
            assert_eq!(simd, scalar, "min mismatch for len {len}");
        }
    }

    #[test]
    fn max_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 13, 4_096);
            // SAFETY: tested inside the avx2 detection guard.
            let simd = unsafe { max(&data) };
            let scalar = data.iter().copied().reduce(f32::max).unwrap();
            assert_eq!(simd, scalar, "max mismatch for len {len}");
        }
    }

    #[test]
    fn argmax_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
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
        if !std::arch::is_x86_feature_detected!("avx2") {
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

    #[test]
    fn dot_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 512, 1024] {
            let a = exact_data(len, 17, 64); // products <= 4096 -> sums exact
            let b = exact_data(len, 19, 64);
            // SAFETY: tested inside the avx2 detection guard.
            let simd = unsafe { dot(&a, &b) };
            let scalar: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert_eq!(simd, scalar, "dot mismatch for len {len}");
        }
    }

    #[test]
    fn sum_sq_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
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
    fn l1_norm_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
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
    fn max_norm_matches_scalar_when_avx2_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
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

    #[cfg(feature = "alloc")]
    #[test]
    fn sigmoid_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
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
        if !std::arch::is_x86_feature_detected!("avx2") {
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
        if !std::arch::is_x86_feature_detected!("avx2") {
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
        if !std::arch::is_x86_feature_detected!("avx2") {
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
        if !std::arch::is_x86_feature_detected!("avx2") {
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
        if !std::arch::is_x86_feature_detected!("avx2") {
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
        if !std::arch::is_x86_feature_detected!("avx2") {
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
