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
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    _mm_set1_ps(0.0),
    |acc: __m128, v: __m128| _mm_max_ps(acc, _mm_andnot_ps(_mm_set1_ps(-0.0), v)),
    |v| unsafe { hmax_128(v) },
    |r: f32, v: f32| f32::max(r, v.abs())
);

// Dot product: 4-wide multiply-accumulate (mul+add; SSE2 has no FMA).
crate::simd_reduce2!(
    dot,
    "sse2",
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
    |s| unsafe { _mm_set1_ps(s) }
);

crate::simd_map!(
    sigmoid,
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
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v| unsafe { _mm_max_ps(v, _mm_set1_ps(0.0)) },
    |x: f32| x.max(0.0)
);

// Exp map: per-element exp, vector vexp for chunks + scalar exp for tails.
crate::simd_map!(
    exp,
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
    "sse2",
    4,
    |p| unsafe { _mm_loadu_ps(p) },
    |p, v| unsafe { _mm_storeu_ps(p, v) },
    |v: __m128| _mm_div_ps(_mm_set1_ps(1.0), _mm_sqrt_ps(v)),
    |x: f32| 1.0 / crate::kernels::sqrt::sqrt(x)
);
crate::simd_exp!(
    vexp_128,
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
    |v| unsafe { _mm_castps_si128(v) },
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
