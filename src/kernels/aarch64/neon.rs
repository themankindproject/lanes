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
    f32,
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
    f32,
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
    f32,
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
    f32,
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
    f32,
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
    f32,
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
    f32,
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

/// Horizontal argmax of the 4 lanes: `(max value, its index)`.
///
/// Ties resolve to the lowest lane. The index is read from `idx` at the
/// first lane equal to the max.
///
/// # Safety
/// Caller must ensure NEON is available (mandatory on aarch64).
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn argmax_pair_neon(v: float32x4_t, idx: int32x4_t) -> (f32, usize) {
    let m = unsafe { vmaxvq_f32(v) };
    let eq = unsafe { vceqq_f32(v, vdupq_n_f32(m)) };
    // Scan the 4 mask lanes; the first non-zero lane is the first max.
    let mut mask = [0_u32; 4];
    unsafe { vst1q_u32(mask.as_mut_ptr(), eq) };
    let lane = mask.iter().position(|&l| l != 0).unwrap_or(0);
    let mut idxs = [0_i32; 4];
    unsafe { vst1q_s32(idxs.as_mut_ptr(), idx) };
    (m, idxs[lane] as usize)
}

/// Horizontal argmin of the 4 lanes: `(min value, its index)`.
///
/// Ties resolve to the lowest lane. The index is read from `idx` at the
/// first lane equal to the min.
///
/// # Safety
/// Caller must ensure NEON is available (mandatory on aarch64).
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn argmin_pair_neon(v: float32x4_t, idx: int32x4_t) -> (f32, usize) {
    let m = unsafe { vminvq_f32(v) };
    let eq = unsafe { vceqq_f32(v, vdupq_n_f32(m)) };
    // Scan the 4 mask lanes; the first non-zero lane is the first min.
    let mut mask = [0_u32; 4];
    unsafe { vst1q_u32(mask.as_mut_ptr(), eq) };
    let lane = mask.iter().position(|&l| l != 0).unwrap_or(0);
    let mut idxs = [0_i32; 4];
    unsafe { vst1q_s32(idxs.as_mut_ptr(), idx) };
    (m, idxs[lane] as usize)
}

// Argmax: index of the first occurrence of the maximum.
crate::simd_argminmax!(
    argmax,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vcombine_s32(
        vcreate_s32(0x0000_0001_0000_0000),
        vcreate_s32(0x0000_0003_0000_0002)
    ),
    |i| unsafe { vdupq_n_s32(i) },
    |a, b| unsafe { vaddq_s32(a, b) },
    |a, b| unsafe { vcgtq_f32(a, b) },
    |mask: uint32x4_t, a: float32x4_t, b: float32x4_t| unsafe { vbslq_f32(mask, a, b) },
    |mask: uint32x4_t, a: int32x4_t, b: int32x4_t| unsafe { vbslq_s32(mask, a, b) },
    |cand: f32, cur: f32| cand > cur,
    |v, iv| unsafe { argmax_pair_neon(v, iv) }
);

// Argmin: index of the first occurrence of the minimum.
crate::simd_argminmax!(
    argmin,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vcombine_s32(
        vcreate_s32(0x0000_0001_0000_0000),
        vcreate_s32(0x0000_0003_0000_0002)
    ),
    |i| unsafe { vdupq_n_s32(i) },
    |a, b| unsafe { vaddq_s32(a, b) },
    |a, b| unsafe { vcltq_f32(a, b) },
    |mask: uint32x4_t, a: float32x4_t, b: float32x4_t| unsafe { vbslq_f32(mask, a, b) },
    |mask: uint32x4_t, a: int32x4_t, b: int32x4_t| unsafe { vbslq_s32(mask, a, b) },
    |cand: f32, cur: f32| cand < cur,
    |v, iv| unsafe { argmin_pair_neon(v, iv) }
);

// Dot product: 4-wide fused multiply-accumulate (`vfmaq` is `acc + va*vb`).
crate::simd_reduce2!(
    dot,
    f32,
    ["neon"],
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
    f32,
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
    |s| unsafe { vdupq_n_f32(s) },
    |x: f32| crate::kernels::exp::exp(x)
);

crate::simd_map!(
    sigmoid,
    f32,
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
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe { vdivq_f32(v, vaddq_f32(vdupq_n_f32(1.0), vexp_neon(vnegq_f32(v)))) },
    |x: f32| x / (1.0 + crate::kernels::exp::exp(-x))
);
crate::simd_map!(
    gelu,
    f32,
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
    f32,
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
    f32,
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
    f32,
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
    f32,
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
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v: float32x4_t| vdivq_f32(vdupq_n_f32(1.0), vsqrtq_f32(v)),
    |x: f32| 1.0 / crate::kernels::sqrt::sqrt(x)
);
crate::simd_exp!(
    vexp_neon,
    f32,
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

// ===========================================================================
// f64 (double-precision) kernels. NEON `float64x2_t` = 2 lanes. Horizontal
// sum uses `vaddvq_f64`; min/max/prod use lane extraction (no vector
// horizontal variant exists for those).
// ===========================================================================

/// Horizontal sum of the 2 f64 lanes in a `float64x2_t`.
///
/// # Safety
/// Caller must ensure the CPU supports NEON (mandatory on aarch64).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hsum_128d(v: float64x2_t) -> f64 {
    // SAFETY: caller guarantees NEON.
    unsafe { vaddvq_f64(v) }
}

/// Horizontal product of the 2 f64 lanes in a `float64x2_t`.
///
/// # Safety
/// Caller must ensure the CPU supports NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hprod_128d(v: float64x2_t) -> f64 {
    // SAFETY: caller guarantees NEON.
    unsafe { vgetq_lane_f64(v, 0) * vgetq_lane_f64(v, 1) }
}

/// Horizontal minimum of the 2 f64 lanes in a `float64x2_t`.
///
/// # Safety
/// Caller must ensure the CPU supports NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hmin_128d(v: float64x2_t) -> f64 {
    // SAFETY: caller guarantees NEON.
    unsafe { vminvq_f64(v) }
}

/// Horizontal maximum of the 2 f64 lanes in a `float64x2_t`.
///
/// # Safety
/// Caller must ensure the CPU supports NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hmax_128d(v: float64x2_t) -> f64 {
    // SAFETY: caller guarantees NEON.
    unsafe { vmaxvq_f64(v) }
}

/// Horizontal argmax of the 2 f64 lanes: `(max value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports NEON.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn argmax_pair_128d(v: float64x2_t, idx: int32x4_t) -> (f64, usize) {
    let m = unsafe { hmax_128d(v) };
    let eq = unsafe { vceqq_f64(v, vdupq_n_f64(m)) }; // uint64x2_t: [m==l0, m==l1]
    // Lane 0 is the max iff its mask bit is set (all-ones); ties → lane 0.
    let lane = usize::from(unsafe { vgetq_lane_u64(eq, 0) } != 0);
    let mut idxs = [0_i32; 4];
    unsafe { vst1q_s32(idxs.as_mut_ptr(), idx) };
    (m, idxs[lane] as usize)
}

/// Horizontal argmin of the 2 f64 lanes: `(min value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports NEON.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn argmin_pair_128d(v: float64x2_t, idx: int32x4_t) -> (f64, usize) {
    let m = unsafe { hmin_128d(v) };
    let eq = unsafe { vceqq_f64(v, vdupq_n_f64(m)) }; // uint64x2_t: [m==l0, m==l1]
    let lane = usize::from(unsafe { vgetq_lane_u64(eq, 0) } != 0);
    let mut idxs = [0_i32; 4];
    unsafe { vst1q_s32(idxs.as_mut_ptr(), idx) };
    (m, idxs[lane] as usize)
}

// f64 reductions for NEON (2 lanes).
crate::simd_reduce!(
    sum_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    vaddq_f64,
    |v| unsafe { hsum_128d(v) },
    |r, v| r + v
);

crate::simd_reduce!(
    prod_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(1.0),
    vmulq_f64,
    |v| unsafe { hprod_128d(v) },
    |r, v| r * v
);

crate::simd_reduce!(
    min_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(f64::INFINITY),
    vminq_f64,
    |v| unsafe { hmin_128d(v) },
    f64::min
);

crate::simd_reduce!(
    max_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(f64::NEG_INFINITY),
    vmaxq_f64,
    |v| unsafe { hmax_128d(v) },
    f64::max
);

crate::simd_reduce!(
    sum_sq_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, v: float64x2_t| vaddq_f64(acc, vmulq_f64(v, v)),
    |v| unsafe { hsum_128d(v) },
    |r: f64, v: f64| r + v * v
);

crate::simd_reduce!(
    l1_norm_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, v: float64x2_t| vaddq_f64(acc, vabsq_f64(v)),
    |v| unsafe { hsum_128d(v) },
    |r: f64, v: f64| r + v.abs()
);

crate::simd_reduce!(
    max_norm_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, v: float64x2_t| vmaxq_f64(acc, vabsq_f64(v)),
    |v| unsafe { hmax_128d(v) },
    |r: f64, v: f64| f64::max(r, v.abs())
);

crate::simd_reduce2!(
    dot_f64,
    f64,
    ["neon"],
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, a: float64x2_t, b: float64x2_t| vfmaq_f64(acc, a, b),
    |v| unsafe { hsum_128d(v) },
    |r, a, b| r + a * b
);

crate::simd_argminmax!(
    argmax_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vcombine_s32(vcreate_s32(0x0000_0001_0000_0000), vcreate_s32(0)),
    |i| unsafe { vdupq_n_s32(i) },
    |a, b| unsafe { vaddq_s32(a, b) },
    |a: float64x2_t, b: float64x2_t| unsafe { vcgtq_f64(a, b) },
    |mask: uint64x2_t, a: float64x2_t, b: float64x2_t| unsafe { vbslq_f64(mask, a, b) },
    // Blend the i32 index vector per 64-bit lane: the mask is uint64x2_t
    // (one bit per f64 lane), so reinterpret the index vector to i64 lanes,
    // blend, and reinterpret back. (A 64→32-bit reinterpret of the mask
    // would wrongly set the upper 32-bit half of each lane's mask.)
    |mask: uint64x2_t, a: int32x4_t, b: int32x4_t| unsafe {
        vreinterpretq_s32_s64(vbslq_s64(
            mask,
            vreinterpretq_s64_s32(a),
            vreinterpretq_s64_s32(b),
        ))
    },
    |a: f64, b: f64| a > b,
    |v, idx| unsafe { argmax_pair_128d(v, idx) }
);

crate::simd_argminmax!(
    argmin_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vcombine_s32(vcreate_s32(0x0000_0001_0000_0000), vcreate_s32(0)),
    |i| unsafe { vdupq_n_s32(i) },
    |a, b| unsafe { vaddq_s32(a, b) },
    |a: float64x2_t, b: float64x2_t| unsafe { vcltq_f64(a, b) },
    |mask: uint64x2_t, a: float64x2_t, b: float64x2_t| unsafe { vbslq_f64(mask, a, b) },
    |mask: uint64x2_t, a: int32x4_t, b: int32x4_t| unsafe {
        vreinterpretq_s32_s64(vbslq_s64(
            mask,
            vreinterpretq_s64_s32(a),
            vreinterpretq_s64_s32(b),
        ))
    },
    |a: f64, b: f64| a < b,
    |v, idx| unsafe { argmin_pair_128d(v, idx) }
);

// f64 elementwise maps for NEON (2 lanes).
#[cfg(feature = "alloc")]
crate::simd_map!(
    sqrt_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v| unsafe { vsqrtq_f64(v) },
    |x: f64| crate::kernels::sqrt::sqrt_f64(x)
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    rsqrt_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe { vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(v)) },
    |x: f64| 1.0 / crate::kernels::sqrt::sqrt_f64(x)
);

#[cfg(feature = "alloc")]
crate::simd_map_param!(
    clip_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t, lo: f64, hi: f64| unsafe {
        vmaxq_f64(vminq_f64(v, vdupq_n_f64(hi)), vdupq_n_f64(lo))
    },
    |x: f64, lo: f64, hi: f64| x.clamp(lo, hi)
);

// f64 vector exp for NEON (2 lanes).
#[cfg(feature = "alloc")]
crate::simd_exp_f64!(
    vexp_128d,
    "neon",
    float64x2_t,
    int64x2_t,
    |s| unsafe { vdupq_n_f64(s) },
    |i| unsafe { vdupq_n_s64(i) },
    |a, b| unsafe { vmulq_f64(a, b) },
    |a, b| unsafe { vaddq_f64(a, b) },
    |a, b| unsafe { vsubq_f64(a, b) },
    |a, b| unsafe { vandq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)) },
    |a, b| unsafe { vbicq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)) },
    |a, b| unsafe { vorrq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)) },
    |a, b| unsafe { vcgtq_f64(a, b) },
    |v| unsafe { vreinterpretq_s64_f64(v) },
    |v| unsafe { vreinterpretq_f64_s64(v) },
    |v| unsafe { vcvtq_s64_f64(v) },
    |v| unsafe { vshlq_n_s64(v, 52) },
    |a, b| unsafe { vaddq_s64(a, b) },
    // Signed compares: n_int can be negative (n < -1022 case), so unsigned
    // compares would mis-handle the underflow clamp. The compare result is
    // uint64x2_t; reinterpret to int64x2_t for the and/andnot ops below.
    |a, b| unsafe { vreinterpretq_s64_u64(vcgtq_s64(a, b)) },
    |a, b| unsafe { vreinterpretq_s64_u64(vcgtq_s64(b, a)) },
    |a, b| unsafe { vandq_s64(a, b) },
    // vbicq(a, b) = a & ~b; the macro calls $andnot_i(under, n_bits) with
    // x86 semantics (~under & n_bits), so the operands must be swapped.
    |a, b| unsafe { vbicq_s64(b, a) },
    |a, b| unsafe { vorrq_s64(a, b) }
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    exp_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe { vexp_128d(v) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

#[cfg(feature = "alloc")]
crate::simd_softmax!(
    softmax_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |a, b| unsafe { vmaxq_f64(a, b) },
    |a, b| unsafe { vsubq_f64(a, b) },
    |v| unsafe { vexp_128d(v) },
    |a, b| unsafe { vaddq_f64(a, b) },
    |a, b| unsafe { vmulq_f64(a, b) },
    |v| unsafe { vaddvq_f64(v) },
    |v| unsafe { vmaxvq_f64(v) },
    |s| unsafe { vdupq_n_f64(s) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    sigmoid_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe {
        vdivq_f64(
            vdupq_n_f64(1.0),
            vaddq_f64(vdupq_n_f64(1.0), vexp_128d(vnegq_f64(v))),
        )
    },
    |x: f64| 1.0 / (1.0 + crate::kernels::exp::exp_f64(-x))
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    silu_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe { vdivq_f64(v, vaddq_f64(vdupq_n_f64(1.0), vexp_128d(vnegq_f64(v))),) },
    |x: f64| x / (1.0 + crate::kernels::exp::exp_f64(-x))
);

#[cfg(feature = "alloc")]
crate::simd_map!(
    gelu_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe {
        let x2 = vmulq_f64(v, v);
        let x3 = vmulq_f64(x2, v);
        let z = vmulq_f64(
            vdupq_n_f64(0.797_884_560_802_865_4),
            vaddq_f64(v, vmulq_f64(vdupq_n_f64(0.044_715), x3)),
        );
        let e = vexp_128d(vaddq_f64(z, z));
        let tanh_z = vsubq_f64(
            vdupq_n_f64(1.0),
            vdivq_f64(vdupq_n_f64(2.0), vaddq_f64(e, vdupq_n_f64(1.0))),
        );
        vmulq_f64(
            vdupq_n_f64(0.5),
            vmulq_f64(v, vaddq_f64(vdupq_n_f64(1.0), tanh_z)),
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
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe { vmaxq_f64(v, vdupq_n_f64(0.0)) },
    |x: f64| x.max(0.0)
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

    #[test]
    fn argmax_matches_scalar_when_neon_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 31, 4_096);
            // SAFETY: tested inside the neon detection guard.
            let (v, i) = unsafe { argmax(&data) };
            assert_eq!(v, data[i], "argmax value mismatch for len {len}");
            assert_eq!(
                i,
                data.iter()
                    .enumerate()
                    .fold(0, |bi, (i, &x)| if x > data[bi] { i } else { bi }),
                "argmax index mismatch for len {len}"
            );
        }
        // Tie-break: first occurrence wins.
        let tied = [1.0_f32, 5.0, 3.0, 5.0, 2.0];
        // SAFETY: tested inside the neon detection guard.
        assert_eq!(unsafe { argmax(&tied) }, (5.0, 1));
    }

    #[test]
    fn argmin_matches_scalar_when_neon_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 63, 64, 65, 512, 1024] {
            let data = exact_data(len, 37, 4_096);
            // SAFETY: tested inside the neon detection guard.
            let (v, i) = unsafe { argmin(&data) };
            assert_eq!(v, data[i], "argmin value mismatch for len {len}");
            assert_eq!(
                i,
                data.iter()
                    .enumerate()
                    .fold(0, |bi, (i, &x)| if x < data[bi] { i } else { bi }),
                "argmin index mismatch for len {len}"
            );
        }
        // Tie-break: first occurrence wins.
        let tied = [3.0_f32, 1.0, 2.0, 1.0, 4.0];
        // SAFETY: tested inside the neon detection guard.
        assert_eq!(unsafe { argmin(&tied) }, (1.0, 1));
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

    #[cfg(feature = "alloc")]
    #[test]
    fn debug_f64_exp_negative_one() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let v = vdupq_n_f64(-1.0);
        let r = unsafe { vexp_128d(v) };
        let mut out = [0.0_f64; 2];
        unsafe { vst1q_f64(out.as_mut_ptr(), r) };
        let expected = crate::kernels::exp::exp_f64(-1.0);
        eprintln!("DEBUG vexp_128d(-1.0) = {:?}, scalar = {expected}", out);
        assert!(
            (out[0] - expected).abs() < 1e-9,
            "vexp_128d(-1.0) = {} vs scalar {expected}",
            out[0]
        );
    }
}
