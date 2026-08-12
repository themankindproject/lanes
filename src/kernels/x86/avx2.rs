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
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_set1_ps(0.0),
    |acc: __m256, v: __m256| _mm256_max_ps(acc, _mm256_andnot_ps(_mm256_set1_ps(-0.0), v)),
    |v| unsafe { hmax_256(v) },
    |r: f32, v: f32| f32::max(r, v.abs())
);

// Dot product: 8-wide fused multiply-accumulate (AVX2 + FMA).
crate::simd_reduce2_feat!(
    dot,
    "avx2",
    "fma",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    _mm256_setzero_ps(),
    |acc, va, vb| _mm256_fmadd_ps(va, vb, acc),
    |v| unsafe { hsum_256(v) },
    |r, a, b| r + a * b
);

// Softmax: 3-pass map (max → exp+sum → scale). exp is per-lane scalar.
// Uses the crate's `no_std` `exp`, so available in all builds.
#[cfg(feature = "alloc")]
crate::simd_softmax!(
    softmax,
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
    |s| unsafe { _mm256_set1_ps(s) }
);

crate::simd_map!(
    sigmoid,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe {
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
    |x: f32| 1.0 / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    silu,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe {
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
    |x: f32| x / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    gelu,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe {
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
        let z = 0.797_884_6 * (x + 0.044_715 * x * x * x);
        let tanh_z = 1.0 - 2.0 / (crate::kernels::exp::exp(2.0 * z) + 1.0);
        0.5 * x * (1.0 + tanh_z)
    }
);
crate::simd_map!(
    relu,
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v| unsafe { _mm256_max_ps(v, _mm256_set1_ps(0.0)) },
    |x: f32| x.max(0.0)
);

// Exp map: per-element exp, vector vexp for chunks + scalar exp for tails.
crate::simd_map!(
    exp,
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
    "avx2",
    8,
    |p| unsafe { _mm256_loadu_ps(p) },
    |p, v| unsafe { _mm256_storeu_ps(p, v) },
    |v: __m256| _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_sqrt_ps(v)),
    |x: f32| 1.0 / crate::kernels::sqrt::sqrt(x)
);
crate::simd_exp!(
    vexp_256,
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
    |v| unsafe { _mm256_castps_si256(v) },
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
