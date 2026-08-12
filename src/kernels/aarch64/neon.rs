//! ARM NEON (128-bit) SIMD kernel implementations for `AArch64`.
//!
//! NEON is mandatory on all ARMv8-A cores, so these kernels are always
//! available on aarch64 targets (still verified via
//! `std::arch::is_aarch64_feature_detected!("neon")` before dispatch).
//!
//! Floating-point semantics: sums/dot products accumulate in 4-lane vectors
//! and combine via a horizontal reduction, so the reduction order differs
//! from the scalar kernels. `min`/`max` follow the ARM `vminq`/`vmaxq`
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
use core::arch::aarch64::*;

// Sum reduction: accumulate 4-wide, horizontal-sum, scalar tail.
crate::simd_reduce!(
    sum,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(0.0),
    vaddq_f32,
    |v| unsafe { vaddvq_f32(v) },
    |r, v| r + v
);

// Product reduction: 4-wide multiply, scalar-multiply tail.
// NEON has no horizontal-product intrinsic; pair-multiply via vrev64q.
crate::simd_reduce!(
    prod,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(1.0),
    vmulq_f32,
    |v| unsafe {
        let p = vmulq_f32(v, vrev64q_f32(v));
        let q = vmulq_f32(p, vextq_f32(p, p, 2));
        vgetq_lane_f32(q, 0)
    },
    |r, v| r * v
);

// Minimum reduction: `vminq` semantics, `minf` tail.
crate::simd_reduce!(
    min,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(f32::INFINITY),
    vminq_f32,
    |v| unsafe { vminvq_f32(v) },
    f32::min
);

// Maximum reduction: `vmaxq` semantics, `maxf` tail.
crate::simd_reduce!(
    max,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(f32::NEG_INFINITY),
    vmaxq_f32,
    |v| unsafe { vmaxvq_f32(v) },
    f32::max
);
// Sum of squares: 4-wide multiply-accumulate (acc += v*v).
crate::simd_reduce!(
    sum_sq,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(0.0),
    |acc: float32x4_t, v: float32x4_t| vaddq_f32(acc, vmulq_f32(v, v)),
    |v| unsafe { vaddvq_f32(v) },
    |r: f32, v: f32| r + v * v
);

// L1 norm: sum of absolute values.
crate::simd_reduce!(
    l1_norm,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(0.0),
    |acc: float32x4_t, v: float32x4_t| vaddq_f32(
        acc,
        vreinterpretq_f32_s32(vandq_s32(
            vreinterpretq_s32_f32(v),
            vmvnq_s32(vdupq_n_s32(i32::MIN))
        ))
    ),
    |v| unsafe { vaddvq_f32(v) },
    |r: f32, v: f32| r + v.abs()
);

// Max norm: maximum absolute value.
crate::simd_reduce!(
    max_norm,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(0.0),
    |acc: float32x4_t, v: float32x4_t| vmaxq_f32(
        acc,
        vreinterpretq_f32_s32(vandq_s32(
            vreinterpretq_s32_f32(v),
            vmvnq_s32(vdupq_n_s32(i32::MIN))
        ))
    ),
    |v| unsafe { vmaxvq_f32(v) },
    |r: f32, v: f32| f32::max(r, v.abs())
);

// Dot product: 4-wide fused multiply-accumulate (`vfmaq` is `acc + va*vb`).
crate::simd_reduce2!(
    dot,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(0.0),
    vfmaq_f32,
    |v| unsafe { vaddvq_f32(v) },
    |r, a, b| r + a * b
);

// Softmax: 3-pass map (max → exp+sum → scale). exp is per-lane scalar.
// Uses the crate's `no_std` `exp`, so available in all builds.
#[cfg(feature = "alloc")]
crate::simd_softmax!(
    softmax,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    vmaxq_f32,
    vsubq_f32,
    |v| unsafe { vexp_neon(v) },
    vaddq_f32,
    vmulq_f32,
    |v| unsafe { vaddvq_f32(v) },
    |v| unsafe { vmaxvq_f32(v) },
    |s| unsafe { vdupq_n_f32(s) }
);

crate::simd_map!(
    sigmoid,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe {
        vdivq_f32(
            vdupq_n_f32(1.0),
            vaddq_f32(vdupq_n_f32(1.0), vexp_neon(vnegq_f32(v))),
        )
    },
    |x: f32| 1.0 / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    silu,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe { vdivq_f32(v, vaddq_f32(vdupq_n_f32(1.0), vexp_neon(vnegq_f32(v)))) },
    |x: f32| x / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    gelu,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe {
        let x2 = vmulq_f32(v, v);
        let x3 = vmulq_f32(x2, v);
        let z = vmulq_f32(
            vdupq_n_f32(0.797_884_6),
            vaddq_f32(v, vmulq_f32(vdupq_n_f32(0.044_715), x3)),
        );
        let e = vexp_neon(vaddq_f32(z, z));
        let tanh_z = vsubq_f32(
            vdupq_n_f32(1.0),
            vdivq_f32(vdupq_n_f32(2.0), vaddq_f32(e, vdupq_n_f32(1.0))),
        );
        vmulq_f32(
            vdupq_n_f32(0.5),
            vmulq_f32(v, vaddq_f32(vdupq_n_f32(1.0), tanh_z)),
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
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe { vmaxq_f32(v, vdupq_n_f32(0.0)) },
    |x: f32| x.max(0.0)
);

// Exp map: per-element exp, vector vexp for chunks + scalar exp for tails.
crate::simd_map!(
    exp,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v: float32x4_t| unsafe { vexp_neon(v) },
    |x: f32| crate::kernels::exp::exp(x)
);
// Sqrt: one-pass map, native hardware sqrt (correctly rounded, IEEE).
crate::simd_map!(
    sqrt,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe { vsqrtq_f32(v) },
    |x: f32| crate::kernels::sqrt::sqrt(x)
);

// Clip: one-pass map with lo/hi params, min(max(v, lo), hi).
crate::simd_map_param!(
    clip,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v: float32x4_t, lo: f32, hi: f32| vminq_f32(vmaxq_f32(v, vdupq_n_f32(lo)), vdupq_n_f32(hi)),
    |x: f32, lo: f32, hi: f32| x.clamp(lo, hi)
);

// Rsqrt: one-pass map, 1/sqrt(v) (exact via div+sqrt, not the ~12-bit
// hardware approximation — correctness-first).
crate::simd_map!(
    rsqrt,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v: float32x4_t| vdivq_f32(vdupq_n_f32(1.0), vsqrtq_f32(v)),
    |x: f32| 1.0 / crate::kernels::sqrt::sqrt(x)
);
crate::simd_exp!(
    vexp_neon,
    "neon",
    float32x4_t,
    int32x4_t,
    |s| unsafe { vdupq_n_f32(s) },
    |i| unsafe { vdupq_n_s32(i) },
    |a, b| unsafe { vmulq_f32(a, b) },
    |a, b| unsafe { vaddq_f32(a, b) },
    |a, b| unsafe { vsubq_f32(a, b) },
    // NEON float bitwise ops don't exist in stdarch; do them on the int
    // reinterpretation (same bits). `vbicq(a, b) = a & ~b`, so the
    // `~a & b` andnot semantics are `vbicq(b, a)`.
    |a, b| unsafe {
        vreinterpretq_f32_s32(vandq_s32(
            vreinterpretq_s32_f32(a),
            vreinterpretq_s32_f32(b),
        ))
    },
    |a, b| unsafe {
        vreinterpretq_f32_s32(vbicq_s32(
            vreinterpretq_s32_f32(b),
            vreinterpretq_s32_f32(a),
        ))
    },
    |a, b| unsafe {
        vreinterpretq_f32_s32(vorrq_s32(
            vreinterpretq_s32_f32(a),
            vreinterpretq_s32_f32(b),
        ))
    },
    |a, b| unsafe {
        // NEON float compare returns uint32x4_t; reinterp to f32 mask vector.
        vreinterpretq_f32_s32(vreinterpretq_s32_u32(vcgtq_f32(a, b)))
    },
    |v| unsafe { vreinterpretq_s32_f32(v) },
    |v| unsafe { vreinterpretq_f32_s32(v) },
    |v| unsafe { vcvtq_s32_f32(v) },
    |v| unsafe { vshlq_n_s32(v, 23) },
    |a, b| unsafe { vaddq_s32(a, b) },
    |a, b| unsafe { vreinterpretq_s32_u32(vcgtq_s32(a, b)) },
    |a, b| unsafe { vreinterpretq_s32_u32(vcltq_s32(a, b)) },
    |a, b| unsafe { vandq_s32(a, b) },
    // `vbicq(a, b) = a & ~b`; andnot(a, b) = ~a & b = vbicq(b, a).
    |a, b| unsafe { vbicq_s32(b, a) },
    |a, b| unsafe { vorrq_s32(a, b) }
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
    fn sum_matches_scalar_when_neon_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for len in [0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1024] {
            let data = exact_data(len, 37, 4_096);
            // Products of 2^n overflow f32 quickly; cap prod at small
            // lengths so the result stays exactly representable.
            let prod_len = len.min(64);
            let prod_data = exact_data(prod_len, 38, 2);
            let a = exact_data(len, 41, 64);
            let b = exact_data(len, 43, 64);

            // SAFETY: tested inside the neon detection guard.
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
                // SAFETY: tested inside the neon detection guard.
                unsafe {
                    assert_eq!(min(&data), exact_min(&data), "min mismatch for len {len}");
                    assert_eq!(max(&data), exact_max(&data), "max mismatch for len {len}");
                }
            }
        }
    }

    #[test]
    fn sum_sq_matches_scalar_when_neon_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
    fn l1_norm_matches_scalar_when_neon_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
    fn max_norm_matches_scalar_when_neon_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
    fn softmax_matches_scalar_when_neon_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for len in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 2.0).collect();
            let mut simd_out = vec![0.0_f32; len];
            let mut ref_out = vec![0.0_f32; len];
            // SAFETY: tested inside the neon detection guard.
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
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
        if !std::arch::is_aarch64_feature_detected!("neon") {
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
