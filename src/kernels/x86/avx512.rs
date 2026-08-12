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

// Sum reduction: accumulate 16-wide, horizontal-sum, scalar tail.
crate::simd_reduce!(
    sum,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setzero_ps(),
    _mm512_add_ps,
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |r, v| r + v
);

// Product reduction: 16-wide multiply, scalar-multiply tail.
crate::simd_reduce!(
    prod,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_set1_ps(1.0),
    _mm512_mul_ps,
    |v| unsafe { _mm512_reduce_mul_ps(v) },
    |r, v| r * v
);

// Minimum reduction: `vminps` semantics, `minf` tail.
crate::simd_reduce!(
    min,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_set1_ps(f32::INFINITY),
    _mm512_min_ps,
    |v| unsafe { _mm512_reduce_min_ps(v) },
    f32::min
);

// Maximum reduction: `vmaxps` semantics, `maxf` tail.
crate::simd_reduce!(
    max,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_set1_ps(f32::NEG_INFINITY),
    _mm512_max_ps,
    |v| unsafe { _mm512_reduce_max_ps(v) },
    f32::max
);
// Sum of squares: 16-wide multiply-accumulate (acc += v*v).
crate::simd_reduce!(
    sum_sq,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setzero_ps(),
    |acc: __m512, v: __m512| _mm512_add_ps(acc, _mm512_mul_ps(v, v)),
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |r: f32, v: f32| r + v * v
);

// L1 norm: sum of absolute values.
crate::simd_reduce!(
    l1_norm,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setzero_ps(),
    |acc: __m512, v: __m512| _mm512_add_ps(acc, unsafe {
        _mm512_andnot_ps(_mm512_set1_ps(-0.0), v)
    }),
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |r: f32, v: f32| r + v.abs()
);

// Max norm: maximum absolute value.
crate::simd_reduce!(
    max_norm,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_set1_ps(0.0),
    |acc: __m512, v: __m512| _mm512_max_ps(acc, unsafe {
        _mm512_andnot_ps(_mm512_set1_ps(-0.0), v)
    }),
    |v| unsafe { _mm512_reduce_max_ps(v) },
    |r: f32, v: f32| f32::max(r, v.abs())
);

// Dot product: 16-wide multiply-accumulate (mul+add; AVX-512F has no FMA).
crate::simd_reduce2!(
    dot,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    _mm512_setzero_ps(),
    |acc, va, vb| _mm512_add_ps(acc, _mm512_mul_ps(va, vb)),
    |v| unsafe { _mm512_reduce_add_ps(v) },
    |r, a, b| r + a * b
);

// Softmax: 3-pass map (max → exp+sum → scale). exp is per-lane scalar.
// Uses the crate's `no_std` `exp`, so available in all builds.
#[cfg(feature = "alloc")]
crate::simd_softmax!(
    softmax,
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
    |s| unsafe { _mm512_set1_ps(s) }
);

crate::simd_map!(
    sigmoid,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe {
        _mm512_div_ps(
            _mm512_set1_ps(1.0),
            _mm512_add_ps(
                _mm512_set1_ps(1.0),
                vexp_512(_mm512_xor_ps(
                    v,
                    _mm512_castsi512_ps(_mm512_set1_epi32(i32::MIN)),
                )),
            ),
        )
    },
    |x: f32| 1.0 / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    silu,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe {
        _mm512_div_ps(
            v,
            _mm512_add_ps(
                _mm512_set1_ps(1.0),
                vexp_512(_mm512_xor_ps(
                    v,
                    _mm512_castsi512_ps(_mm512_set1_epi32(i32::MIN)),
                )),
            ),
        )
    },
    |x: f32| x / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    gelu,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe {
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
        let z = 0.797_884_6 * (x + 0.044_715 * x * x * x);
        let tanh_z = 1.0 - 2.0 / (crate::kernels::exp::exp(2.0 * z) + 1.0);
        0.5 * x * (1.0 + tanh_z)
    }
);
crate::simd_map!(
    relu,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe { _mm512_max_ps(v, _mm512_set1_ps(0.0)) },
    |x: f32| x.max(0.0)
);

// Exp map: per-element exp, vector vexp for chunks + scalar exp for tails.
crate::simd_map!(
    exp,
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
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v| unsafe { _mm512_sqrt_ps(v) },
    |x: f32| crate::kernels::sqrt::sqrt(x)
);

// Clip: one-pass map with lo/hi params, min(max(v, lo), hi).
crate::simd_map_param!(
    clip,
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

// Rsqrt: one-pass map, 1/sqrt(v) (exact via div+sqrt, not the ~12-bit
// hardware approximation — correctness-first).
crate::simd_map!(
    rsqrt,
    "avx512f",
    16,
    |p| unsafe { _mm512_loadu_ps(p) },
    |p, v| unsafe { _mm512_storeu_ps(p, v) },
    |v: __m512| _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_sqrt_ps(v)),
    |x: f32| 1.0 / crate::kernels::sqrt::sqrt(x)
);
crate::simd_exp!(
    vexp_512,
    "avx512f",
    __m512,
    __m512i,
    |s| unsafe { _mm512_set1_ps(s) },
    |i| unsafe { _mm512_set1_epi32(i) },
    |a, b| unsafe { _mm512_mul_ps(a, b) },
    |a, b| unsafe { _mm512_add_ps(a, b) },
    |a, b| unsafe { _mm512_sub_ps(a, b) },
    |a, b| unsafe { _mm512_and_ps(a, b) },
    |a, b| unsafe { _mm512_andnot_ps(a, b) },
    |a, b| unsafe { _mm512_or_ps(a, b) },
    |a, b| unsafe {
        // cmp returns a u16 mask; expand to a full-width float mask vector.
        _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(a, b, _CMP_GT_OQ), _mm512_set1_ps(-1.0))
    },
    |v| unsafe { _mm512_castps_si512(v) },
    |v| unsafe { _mm512_castsi512_ps(v) },
    |v| unsafe { _mm512_cvttps_epi32(v) },
    |v| unsafe { _mm512_slli_epi32(v, 23) },
    |a, b| unsafe { _mm512_add_epi32(a, b) },
    |a, b| unsafe { _mm512_maskz_mov_epi32(_mm512_cmpgt_epi32_mask(a, b), _mm512_set1_epi32(-1)) },
    |a, b| unsafe { _mm512_maskz_mov_epi32(_mm512_cmplt_epi32_mask(a, b), _mm512_set1_epi32(-1)) },
    |a, b| unsafe { _mm512_and_si512(a, b) },
    |a, b| unsafe { _mm512_andnot_si512(a, b) },
    |a, b| unsafe { _mm512_or_si512(a, b) }
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
