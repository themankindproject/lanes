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
    |r, v| r + v,
    vaddq_f32
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

crate::simd_minmax!(
    min,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(f32::INFINITY),
    vminq_f32,
    |v| unsafe { vminvq_f32(v) },
    f32::min,
    |v: float32x4_t| unsafe { vmaxvq_u32(vceqq_f32(v, v)) != 0 },
    |v: float32x4_t| unsafe { vbslq_f32(vceqq_f32(v, v), v, vdupq_n_f32(f32::INFINITY)) },
    |v: f32| !v.is_nan(),
    |r: f32, saw_real: bool| if saw_real { r } else { f32::NAN }
);

crate::simd_minmax!(
    max,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(f32::NEG_INFINITY),
    vmaxq_f32,
    |v| unsafe { vmaxvq_f32(v) },
    f32::max,
    |v: float32x4_t| unsafe { vmaxvq_u32(vceqq_f32(v, v)) != 0 },
    |v: float32x4_t| unsafe { vbslq_f32(vceqq_f32(v, v), v, vdupq_n_f32(f32::NEG_INFINITY)) },
    |v: f32| !v.is_nan(),
    |r: f32, saw_real: bool| if saw_real { r } else { f32::NAN }
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
    |r: f32, v: f32| r + v * v,
    vaddq_f32
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
    |r: f32, v: f32| r + v.abs(),
    vaddq_f32
);

crate::simd_minmax!(
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
    |r: f32, v: f32| f32::max(r, v.abs()),
    |v: float32x4_t| unsafe { vminvq_u32(vceqq_f32(v, v)) == 0 },
    |v: float32x4_t| unsafe { vbslq_f32(vceqq_f32(v, v), v, vdupq_n_f32(0.0)) },
    |v: f32| v.is_nan(),
    |r: f32, saw_nan: bool| if saw_nan { f32::NAN } else { r }
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
    // FMAXV propagates NaN on AArch64; mask NaN lanes to -inf first so
    // they can never win (matches the x86 pair reducers).
    let not_nan = unsafe { vceqq_f32(v, v) };
    let clean = unsafe { vbslq_f32(not_nan, v, vdupq_n_f32(f32::NEG_INFINITY)) };
    let m = unsafe { vmaxvq_f32(clean) };
    let eq = unsafe { vceqq_f32(v, vdupq_n_f32(m)) };
    // All-NaN chunk: no lane matches; fall back to lane 0 (index 0).
    let mut mask = [0_u32; 4];
    unsafe { vst1q_u32(mask.as_mut_ptr(), eq) };
    let mut idxs = [0_i32; 4];
    unsafe { vst1q_s32(idxs.as_mut_ptr(), idx) };
    if mask.iter().all(|&l| l == 0) {
        return (f32::NAN, 0);
    }
    let best = mask
        .iter()
        .zip(idxs)
        .filter(|(m, _)| **m != 0)
        .map(|(_, i)| i)
        .min()
        .unwrap_or(0);
    (m, best as usize)
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
    // FMINV propagates NaN on AArch64; mask NaN lanes to +inf first.
    let not_nan = unsafe { vceqq_f32(v, v) };
    let clean = unsafe { vbslq_f32(not_nan, v, vdupq_n_f32(f32::INFINITY)) };
    let m = unsafe { vminvq_f32(clean) };
    let eq = unsafe { vceqq_f32(v, vdupq_n_f32(m)) };
    // All-NaN chunk: no lane matches; fall back to lane 0 (index 0).
    let mut mask = [0_u32; 4];
    unsafe { vst1q_u32(mask.as_mut_ptr(), eq) };
    let mut idxs = [0_i32; 4];
    unsafe { vst1q_s32(idxs.as_mut_ptr(), idx) };
    if mask.iter().all(|&l| l == 0) {
        return (f32::NAN, 0);
    }
    let best = mask
        .iter()
        .zip(idxs)
        .filter(|(m, _)| **m != 0)
        .map(|(_, i)| i)
        .min()
        .unwrap_or(0);
    (m, best as usize)
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
    // NaN-aware dethrone (see scalar `argmax`): win = !NaN(a) && (a > b || NaN(b)).
    // vceqq(x, x) is all-ones for non-NaN lanes, so it is the !NaN mask.
    |a, b| unsafe {
        let gt = vcgtq_f32(a, b);
        let not_nan_b = vceqq_f32(b, b);
        let not_nan_a = vceqq_f32(a, a);
        vandq_u32(not_nan_a, vorrq_u32(gt, vmvnq_u32(not_nan_b)))
    },
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
    // NaN-aware dethrone (see argmax above).
    |a, b| unsafe {
        let lt = vcltq_f32(a, b);
        let not_nan_b = vceqq_f32(b, b);
        let not_nan_a = vceqq_f32(a, a);
        vandq_u32(not_nan_a, vorrq_u32(lt, vmvnq_u32(not_nan_b)))
    },
    |mask: uint32x4_t, a: float32x4_t, b: float32x4_t| unsafe { vbslq_f32(mask, a, b) },
    |mask: uint32x4_t, a: int32x4_t, b: int32x4_t| unsafe { vbslq_s32(mask, a, b) },
    |cand: f32, cur: f32| cand < cur,
    |v, iv| unsafe { argmin_pair_neon(v, iv) }
);

// count_nan: lanes where v != v (vceq is all-ones for ordered, so invert).
crate::simd_count!(
    count_nan,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |v: float32x4_t| unsafe { vmvnq_u32(vceqq_f32(v, v)) },
    |m: uint32x4_t| unsafe { vaddvq_u32(vandq_u32(m, vdupq_n_u32(1))) } as usize,
    |x: f32| x.is_nan()
);

// count_zero: lanes equal to +/-0.0 (they compare equal).
crate::simd_count!(
    count_zero,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |v: float32x4_t| unsafe { vceqq_f32(v, vdupq_n_f32(0.0)) },
    |m: uint32x4_t| unsafe { vaddvq_u32(vandq_u32(m, vdupq_n_u32(1))) } as usize,
    |x: f32| x == 0.0
);

// count_infinite: lanes whose |v| == +inf.
crate::simd_count!(
    count_infinite,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |v: float32x4_t| unsafe { vceqq_f32(vabsq_f32(v), vdupq_n_f32(f32::INFINITY)) },
    |m: uint32x4_t| unsafe { vaddvq_u32(vandq_u32(m, vdupq_n_u32(1))) } as usize,
    |x: f32| x.is_infinite()
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
    |r, a, b| r + a * b,
    vaddq_f32
);
// Squared distance: fused sub + square + accumulate (dot skeleton).
crate::simd_reduce2!(
    squared_distance,
    f32,
    ["neon"],
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(0.0),
    |acc: float32x4_t, va: float32x4_t, vb: float32x4_t| unsafe {
        let d = vsubq_f32(va, vb);
        vfmaq_f32(acc, d, d)
    },
    |v| unsafe { vaddvq_f32(v) },
    |r: f32, a: f32, b: f32| {
        let d = a - b;
        r + d * d
    },
    vaddq_f32
);
// KL divergence: fused div → ln → mul → accumulate (dot skeleton). The
// register-only fdlibm `ln` handles IEEE specials branch-free, so the
// vector path matches the scalar reference term-for-term.
crate::simd_reduce2!(
    kl_divergence,
    f32,
    ["neon"],
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(0.0),
    |acc: float32x4_t, vp: float32x4_t, vq: float32x4_t| unsafe {
        let r = vln_neon(vdivq_f32(vp, vq));
        vaddq_f32(acc, vmulq_f32(vp, r))
    },
    |v| unsafe { vaddvq_f32(v) },
    |r: f32, a: f32, b: f32| r + a * crate::kernels::ln::ln(a / b),
    vaddq_f32
);
// Jensen–Shannon divergence: raw two-sided sum (the wrapper halves it).
crate::simd_reduce2!(
    js_divergence,
    f32,
    ["neon"],
    4,
    |p| unsafe { vld1q_f32(p) },
    vdupq_n_f32(0.0),
    |acc: float32x4_t, vp: float32x4_t, vq: float32x4_t| unsafe {
        let m = vmulq_f32(vaddq_f32(vp, vq), vdupq_n_f32(0.5));
        let tp = vmulq_f32(vp, vln_neon(vdivq_f32(vp, m)));
        let tq = vmulq_f32(vq, vln_neon(vdivq_f32(vq, m)));
        vaddq_f32(acc, vaddq_f32(tp, tq))
    },
    |v| unsafe { vaddvq_f32(v) },
    |r: f32, a: f32, b: f32| {
        let m = (a + b) * 0.5;
        r + a * crate::kernels::ln::ln(a / m) + b * crate::kernels::ln::ln(b / m)
    },
    vaddq_f32
);

// --- binary family: popcount-based two-input reductions -------------------

/// Sum per-byte popcounts into two u64 counter lanes:
/// per-byte popcount (`vcntq_u8`) followed by the pairwise-add chain
/// u8x16 → u16x8 → u32x4 → u64x2.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn popcnt_neon_sum(x: uint8x16_t) -> uint64x2_t {
    unsafe { vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(x)))) }
}

/// Horizontal sum of the two u64 counter lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hsum_neon_u64(v: uint64x2_t) -> usize {
    unsafe { (vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1)) as usize }
}

// Hamming: popcount of XOR, 16 bytes per iteration.
crate::simd_reduce2_count!(
    hamming_popcount,
    ["neon"],
    16,
    |p: *const u8| unsafe { vld1q_u8(p) },
    vdupq_n_u64(0),
    |acc: uint64x2_t, va: uint8x16_t, vb: uint8x16_t| unsafe {
        vaddq_u64(acc, popcnt_neon_sum(veorq_u8(va, vb)))
    },
    |v: uint64x2_t| unsafe { hsum_neon_u64(v) },
    |r: usize, a: u8, b: u8| r + (a ^ b).count_ones() as usize,
    usize
);

// Jaccard counts: (popcount of AND, popcount of OR) per chunk.
crate::simd_reduce2_count!(
    jaccard_counts,
    ["neon"],
    16,
    |p: *const u8| unsafe { vld1q_u8(p) },
    (vdupq_n_u64(0), vdupq_n_u64(0)),
    |acc: (uint64x2_t, uint64x2_t), va: uint8x16_t, vb: uint8x16_t| unsafe {
        (
            vaddq_u64(acc.0, popcnt_neon_sum(vandq_u8(va, vb))),
            vaddq_u64(acc.1, popcnt_neon_sum(vorrq_u8(va, vb))),
        )
    },
    |v: (uint64x2_t, uint64x2_t)| unsafe { (hsum_neon_u64(v.0), hsum_neon_u64(v.1)) },
    |r: (usize, usize), a: u8, b: u8| {
        (
            r.0 + (a & b).count_ones() as usize,
            r.1 + (a | b).count_ones() as usize,
        )
    },
    (usize, usize)
);

// --- i8 family: widening integer reductions (i8 -> i16 -> i32 -> i64) -----

/// Horizontal sum of the two i64 counter lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hsum_neon_i64(v: int64x2_t) -> i64 {
    unsafe { vgetq_lane_s64(v, 0) + vgetq_lane_s64(v, 1) }
}

// i8 dot: widening multiply (`vmull_s8`, 8×i8 -> 8×i16) on each half,
// pairwise-add-long into i32 lanes (`vpadalq_s16`), widen to i64 counters
// every 1024 chunks (per-lane bound 1024 * 2 * 16384 = 33.5M, ~64x below
// i32 overflow).
crate::simd_reduce2_wide!(
    dot_i8,
    i8,
    ["neon"],
    16,
    |p: *const i8| unsafe { vld1q_s8(p) },
    vdupq_n_s64(0),
    vdupq_n_s32(0),
    1024,
    |narrow: int32x4_t, va: int8x16_t, vb: int8x16_t| unsafe {
        let plo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        let phi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        vpadalq_s16(vpadalq_s16(narrow, plo), phi)
    },
    |acc: int64x2_t, narrow: int32x4_t| unsafe { vaddq_s64(acc, vpaddlq_s32(narrow)) },
    |v: int64x2_t| unsafe { hsum_neon_i64(v) },
    |r: i64, a: i8, b: i8| r + i64::from(a) * i64::from(b)
);

// i8 sum: pairwise-add-long into i16 lanes (`vpadalq_s8`), widen to i64
// counters every 64 chunks (per-lane bound 64 * 256 = 16384, 2x below
// i16 overflow).
crate::simd_reduce_wide!(
    sum_i8,
    i8,
    ["neon"],
    16,
    |p: *const i8| unsafe { vld1q_s8(p) },
    vdupq_n_s64(0),
    vdupq_n_s16(0),
    64,
    |narrow: int16x8_t, v: int8x16_t| unsafe { vpadalq_s8(narrow, v) },
    |acc: int64x2_t, narrow: int16x8_t| unsafe { vaddq_s64(acc, vpaddlq_s32(vpaddlq_s16(narrow))) },
    |v: int64x2_t| unsafe { hsum_neon_i64(v) },
    |r: i64, v: i8| r + i64::from(v)
);

// i8 min: native signed-byte min (`vminq_s8`), horizontal `vminvq_s8`.
// Identity i8::MAX.
crate::simd_reduce!(
    min_i8,
    i8,
    "neon",
    16,
    |p| unsafe { vld1q_s8(p) },
    vdupq_n_s8(127),
    vminq_s8,
    |v| unsafe { vminvq_s8(v) },
    i8::min
);

// i8 max: native signed-byte max (`vmaxq_s8`), horizontal `vmaxvq_s8`.
// Identity i8::MIN.
crate::simd_reduce!(
    max_i8,
    i8,
    "neon",
    16,
    |p| unsafe { vld1q_s8(p) },
    vdupq_n_s8(-128),
    vmaxq_s8,
    |v| unsafe { vmaxvq_s8(v) },
    i8::max
);

// i8 count_zero: byte-equality mask (`vceqq_s8`), AND with 1 per lane,
// horizontal sum. Same shape as the f32 `count_zero`.
crate::simd_count!(
    count_zero_i8,
    i8,
    "neon",
    16,
    |p| unsafe { vld1q_s8(p) },
    |v: int8x16_t| unsafe { vceqq_s8(v, vdupq_n_s8(0)) },
    |m: uint8x16_t| unsafe { vaddvq_u8(vandq_u8(m, vdupq_n_u8(1))) } as usize,
    |x: i8| x == 0
);

// Softmax: 3-pass map (max → exp+sum → scale). exp is per-lane scalar.
// Uses the crate's `no_std` `exp`, so available in all builds.
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

// Logsumexp: two-pass scalar-returning reduction (max → Σexp → max+ln).
crate::simd_logsumexp!(
    logsumexp,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    vmaxq_f32,
    vsubq_f32,
    |v| unsafe { vexp_neon(v) },
    |v| unsafe { vaddvq_f32(v) },
    |v| unsafe { vmaxvq_f32(v) },
    |s| unsafe { vdupq_n_f32(s) },
    |x: f32| crate::kernels::exp::exp(x),
    crate::kernels::ln::ln
);

// Log-softmax: three-pass map (max → Σexp → (x-m)-ln(sum)), 0-alloc.
crate::simd_log_softmax!(
    log_softmax,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    vmaxq_f32,
    vsubq_f32,
    |v| unsafe { vexp_neon(v) },
    |v| unsafe { vaddvq_f32(v) },
    |v| unsafe { vmaxvq_f32(v) },
    |s| unsafe { vdupq_n_f32(s) },
    |x: f32| crate::kernels::exp::exp(x),
    crate::kernels::ln::ln
);

// Layer norm: three-pass (mean → center+Σsq → scale), 0-alloc.
crate::simd_layer_norm!(
    layer_norm,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    vaddq_f32,
    vsubq_f32,
    vdupq_n_f32(0.0),
    |acc: float32x4_t, v: float32x4_t| vaddq_f32(acc, vmulq_f32(v, v)),
    |v| unsafe { vaddvq_f32(v) },
    |s| unsafe { vdupq_n_f32(s) },
    |v, inv| vmulq_f32(v, vdupq_n_f32(inv)),
    crate::kernels::sqrt::sqrt
);

crate::simd_map!(
    sigmoid,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe {
        // Saturated fast path: all lanes already at 0/1 (skip the exp).
        let pos = vcgtq_f32(v, vdupq_n_f32(16.64));
        let neg = vcltq_f32(v, vdupq_n_f32(-88.73));
        if vminvq_u32(vorrq_u32(pos, neg)) != 0 {
            return vreinterpretq_f32_u32(vandq_u32(pos, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
        }
        vdivq_f32(
            vdupq_n_f32(1.0),
            vaddq_f32(vdupq_n_f32(1.0), vexp_neon(vnegq_f32(v))),
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
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe {
        // Saturated fast path: silu(x) = x for x > 16.64, 0 for x < -88.
        let pos = vcgtq_f32(v, vdupq_n_f32(16.64));
        let neg = vcltq_f32(v, vdupq_n_f32(-88.73));
        if vminvq_u32(vorrq_u32(pos, neg)) != 0 {
            return vreinterpretq_f32_u32(vandq_u32(pos, vreinterpretq_u32_f32(v)));
        }
        vdivq_f32(v, vaddq_f32(vdupq_n_f32(1.0), vexp_neon(vnegq_f32(v))))
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
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe {
        // Saturated fast path: gelu(x) = x for x > 7.0, 0 for x < -7.0.
        let pos = vcgtq_f32(v, vdupq_n_f32(7.0));
        let neg = vcltq_f32(v, vdupq_n_f32(-7.0));
        if vminvq_u32(vorrq_u32(pos, neg)) != 0 {
            return vreinterpretq_f32_u32(vandq_u32(pos, vreinterpretq_u32_f32(v)));
        }
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
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe { vmaxq_f32(v, vdupq_n_f32(0.0)) },
    |x: f32| x.max(0.0)
);

// Tanh map: tanh(x) = 1 - 2/(exp(2x)+1) from the vector vexp kernel.
crate::simd_map!(
    tanh,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe {
        let a = vabsq_f32(v);
        // ±1 for |x| > 9.011, x for |x| < 2e-4, series for |x| < 0.1,
        // ratio (e-1)/(e+1) beyond (Sterbenz-exact, clamped for overflow).
        let big_mask = vcgtq_f32(a, vdupq_n_f32(9.011));
        if vminvq_u32(big_mask) != 0 {
            let sign = vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x8000_0000));
            return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(vdupq_n_f32(1.0)), sign));
        }
        let x2 = vmulq_f32(v, v);
        let x4 = vmulq_f32(x2, x2);
        let series = vaddq_f32(
            vsubq_f32(v, vdivq_f32(vmulq_f32(v, x2), vdupq_n_f32(3.0))),
            vdivq_f32(vmulq_f32(v, x4), vdupq_n_f32(7.5)),
        );
        let e = vexp_neon(vaddq_f32(v, v));
        let em = vminq_f32(e, vdupq_n_f32(f32::MAX));
        let ratio = vdivq_f32(
            vsubq_f32(em, vdupq_n_f32(1.0)),
            vaddq_f32(em, vdupq_n_f32(1.0)),
        );
        let sign = vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x8000_0000));
        let big = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(ratio), sign));
        let ser_mask = vcltq_f32(a, vdupq_n_f32(0.1));
        let small = vcltq_f32(a, vdupq_n_f32(2e-4));
        let mid = vbslq_f32(ser_mask, series, big);
        let result = vbslq_f32(small, v, mid);
        let one = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(vdupq_n_f32(1.0)), sign));
        vbslq_f32(big_mask, one, result)
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
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    vdupq_n_f32(0.0),
    |acc: float32x4_t, v: float32x4_t| vaddq_f32(acc, vmulq_f32(v, v)),
    |v| unsafe { vaddvq_f32(v) },
    |v, inv| vmulq_f32(v, vdupq_n_f32(inv)),
    crate::kernels::sqrt::sqrt
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
crate::simd_clip!(
    clip,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v: float32x4_t, lo: f32, hi: f32| vminq_f32(vmaxq_f32(v, vdupq_n_f32(lo)), vdupq_n_f32(hi)),
    |x: f32, lo: f32, hi: f32| x.clamp(lo, hi)
);
// abs_sub: |a - b| per lane (native vabs after sub).
crate::simd_map2!(
    abs_sub,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |a: float32x4_t, b: float32x4_t| unsafe { vabsq_f32(vsubq_f32(a, b)) },
    |x: f32, y: f32| (x - y).abs()
);
// hypot: overflow-safe sqrt(a²+b²) via scale-by-max (SLEEF u35 strategy).
// Special-case order: min==0 → max, then NaN, then inf last (inf overrides
// NaN: hypot(inf, nan) == inf per IEEE).
crate::simd_map2!(
    hypot,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |a: float32x4_t, b: float32x4_t| unsafe {
        let ax = vabsq_f32(a);
        let ay = vabsq_f32(b);
        let mx = vmaxq_f32(ax, ay);
        let mn = vminq_f32(ax, ay);
        let t = vdivq_f32(mn, mx);
        let one = vdupq_n_f32(1.0);
        let r = vmulq_f32(mx, vsqrtq_f32(vaddq_f32(vmulq_f32(t, t), one)));
        // min==0 → max (covers hypot(x,0)=|x| and hypot(0,0)=0).
        let zero_m = vceqq_f32(mn, vdupq_n_f32(0.0));
        let r = vbslq_f32(zero_m, mx, r);
        // any NaN → NaN (vceq is all-ones for ordered; invert).
        let nan_m = vorrq_u32(vmvnq_u32(vceqq_f32(a, a)), vmvnq_u32(vceqq_f32(b, b)));
        let r = vbslq_f32(nan_m, vdupq_n_f32(f32::NAN), r);
        // any inf → inf (overrides NaN; IEEE hypot(inf, nan) == inf).
        let inf = vdupq_n_f32(f32::INFINITY);
        let inf_m = vorrq_u32(vceqq_f32(ax, inf), vceqq_f32(ay, inf));
        vbslq_f32(inf_m, inf, r)
    },
    |x: f32, y: f32| crate::kernels::hypot::hypot(x, y)
);
// powi: bit-exact exponentiation by squaring (shared scalar exponent ⇒
// identical multiply sequence per lane; see simd_powi!).
crate::simd_powi!(
    powi,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |a, b| unsafe { vmulq_f32(a, b) },
    |a, b| unsafe { vdivq_f32(a, b) },
    unsafe { vdupq_n_f32(1.0) },
    |x: f32, n: i32| crate::kernels::powi::powi(x, n)
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
    |v| unsafe { vreinterpretq_f32_s32(v) },
    |v| unsafe { vreinterpretq_s32_f32(v) },
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

// Vector ln (f32): fdlibm e_log reduction, see simd_ln! in macros.rs.
crate::simd_ln!(
    vln_neon,
    "neon",
    float32x4_t,
    int32x4_t,
    |s| unsafe { vdupq_n_f32(s) },
    |i| unsafe { vdupq_n_s32(i) },
    |a, b| unsafe { vaddq_f32(a, b) },
    |a, b| unsafe { vsubq_f32(a, b) },
    |a, b| unsafe { vmulq_f32(a, b) },
    |v| unsafe { vcvtq_f32_s32(v) },
    |v| unsafe { vreinterpretq_f32_s32(v) },
    |v| unsafe { vreinterpretq_s32_f32(v) },
    |a, b| unsafe { vandq_s32(a, b) },
    |a, b| unsafe { vorrq_s32(a, b) },
    |v| unsafe { vshrq_n_s32(v, 23) },
    |a, b| unsafe { vreinterpretq_f32_s32(vreinterpretq_s32_u32(vcgtq_f32(a, b))) },
    |a, b| unsafe { vreinterpretq_f32_s32(vreinterpretq_s32_u32(vcltq_f32(a, b))) },
    |a, b| unsafe { vreinterpretq_f32_s32(vreinterpretq_s32_u32(vceqq_f32(a, b))) },
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
    }
);
// Ln: one-pass map; the register kernel handles normal x, the scalar tail
// covers special cases (x <= 0, inf, NaN, subnormal).

// Ln: one-pass map; the register kernel handles normal x, the scalar tail
// covers special cases (x <= 0, inf, NaN, subnormal).
crate::simd_map!(
    ln,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v: float32x4_t| unsafe { vln_neon(v) },
    |x: f32| crate::kernels::ln::ln(x)
);
// Softplus: overflow-free `max(x,0) + ln1p(e^-|x|)`. Reference: the identity
// ln1p(z) = z·ln(1+z)/((1+z)-1) from musl s_log1pf.c / fdlibm s_log1p.c
// (https://musl.libc.org, https://www.netlib.org/fdlibm).
crate::simd_map!(
    softplus,
    f32,
    "neon",
    4,
    |p| unsafe { vld1q_f32(p) },
    |p, v| unsafe { vst1q_f32(p, v) },
    |v| unsafe {
        let zero = vdupq_n_f32(0.0);
        let a = vabsq_f32(v);
        let z = vexp_neon(vsubq_f32(zero, a));
        let u = vaddq_f32(vdupq_n_f32(1.0), z);
        let ln_u = vln_neon(u);
        let lp = vdivq_f32(vmulq_f32(ln_u, z), vsubq_f32(u, vdupq_n_f32(1.0)));
        let one = vceqq_f32(u, vdupq_n_f32(1.0));
        let lp = vbslq_f32(one, z, lp);
        vaddq_f32(vmaxq_f32(v, zero), lp)
    },
    |x: f32| {
        let a = x.abs();
        let z = crate::kernels::exp::exp(-a);
        x.max(0.0) + crate::kernels::scalar::log1p(z)
    }
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
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn argmax_pair_128d(v: float64x2_t, idx: int64x2_t) -> (f64, usize) {
    // FMAXV propagates NaN on AArch64; mask NaN lanes to -inf first.
    let not_nan = unsafe { vceqq_f64(v, v) };
    let clean = unsafe { vbslq_f64(not_nan, v, vdupq_n_f64(f64::NEG_INFINITY)) };
    let m = unsafe { hmax_128d(clean) };
    let eq = unsafe { vceqq_f64(v, vdupq_n_f64(m)) }; // uint64x2_t: [m==l0, m==l1]
    // Lane 0 is the max iff its mask bit is set (all-ones); ties → lane 0.
    // All-NaN chunk: neither mask is set; fall back to lane 0 (index 0).
    let l0 = unsafe { vgetq_lane_u64(eq, 0) } != 0;
    let l1 = unsafe { vgetq_lane_u64(eq, 1) } != 0;
    if !l0 && !l1 {
        return (f64::NAN, 0);
    }
    let mut idxs = [0_i64; 2];
    unsafe { vst1q_s64(idxs.as_mut_ptr(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i64::MAX;
    if l0 {
        best = best.min(idxs[0]);
    }
    if l1 {
        best = best.min(idxs[1]);
    }
    (m, best as usize)
}

/// Horizontal argmin of the 2 f64 lanes: `(min value, its index)`.
///
/// Ties resolve to the lowest lane.
///
/// # Safety
/// Caller must ensure the CPU supports NEON.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn argmin_pair_128d(v: float64x2_t, idx: int64x2_t) -> (f64, usize) {
    // FMINV propagates NaN on AArch64; mask NaN lanes to +inf first.
    let not_nan = unsafe { vceqq_f64(v, v) };
    let clean = unsafe { vbslq_f64(not_nan, v, vdupq_n_f64(f64::INFINITY)) };
    let m = unsafe { hmin_128d(clean) };
    let eq = unsafe { vceqq_f64(v, vdupq_n_f64(m)) }; // uint64x2_t: [m==l0, m==l1]
    // Lane 0 is the min iff its mask bit is set (all-ones); ties → lane 0.
    // All-NaN chunk: neither mask is set; fall back to lane 0 (index 0).
    let l0 = unsafe { vgetq_lane_u64(eq, 0) } != 0;
    let l1 = unsafe { vgetq_lane_u64(eq, 1) } != 0;
    if !l0 && !l1 {
        return (f64::NAN, 0);
    }
    let mut idxs = [0_i64; 2];
    unsafe { vst1q_s64(idxs.as_mut_ptr(), idx) };
    // First occurrence = smallest GLOBAL index among tied lanes.
    let mut best = i64::MAX;
    if l0 {
        best = best.min(idxs[0]);
    }
    if l1 {
        best = best.min(idxs[1]);
    }
    (m, best as usize)
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
    |r, v| r + v,
    vaddq_f64
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

crate::simd_minmax!(
    min_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(f64::INFINITY),
    vminq_f64,
    |v| unsafe { hmin_128d(v) },
    f64::min,
    |v: float64x2_t| unsafe { vmaxvq_u32(vreinterpretq_u32_u64(vceqq_f64(v, v))) != 0 },
    |v: float64x2_t| unsafe { vbslq_f64(vceqq_f64(v, v), v, vdupq_n_f64(f64::INFINITY)) },
    |v: f64| !v.is_nan(),
    |r: f64, saw_real: bool| if saw_real { r } else { f64::NAN }
);

crate::simd_minmax!(
    max_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(f64::NEG_INFINITY),
    vmaxq_f64,
    |v| unsafe { hmax_128d(v) },
    f64::max,
    |v: float64x2_t| unsafe { vmaxvq_u32(vreinterpretq_u32_u64(vceqq_f64(v, v))) != 0 },
    |v: float64x2_t| unsafe { vbslq_f64(vceqq_f64(v, v), v, vdupq_n_f64(f64::NEG_INFINITY)) },
    |v: f64| !v.is_nan(),
    |r: f64, saw_real: bool| if saw_real { r } else { f64::NAN }
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
    |r: f64, v: f64| r + v * v,
    vaddq_f64
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
    |r: f64, v: f64| r + v.abs(),
    vaddq_f64
);

crate::simd_minmax!(
    max_norm_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, v: float64x2_t| vmaxq_f64(acc, vabsq_f64(v)),
    |v| unsafe { hmax_128d(v) },
    |r: f64, v: f64| f64::max(r, v.abs()),
    |v: float64x2_t| unsafe { vminvq_u32(vreinterpretq_u32_u64(vceqq_f64(v, v))) == 0 },
    |v: float64x2_t| unsafe { vbslq_f64(vceqq_f64(v, v), v, vdupq_n_f64(0.0)) },
    |v: f64| v.is_nan(),
    |r: f64, saw_nan: bool| if saw_nan { f64::NAN } else { r }
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
    |r, a, b| r + a * b,
    vaddq_f64
);
// Squared distance (f64): fused sub + square + accumulate (dot skeleton).
crate::simd_reduce2!(
    squared_distance_f64,
    f64,
    ["neon"],
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, a: float64x2_t, b: float64x2_t| unsafe {
        let d = vsubq_f64(a, b);
        vfmaq_f64(acc, d, d)
    },
    |v| unsafe { hsum_128d(v) },
    |r: f64, a: f64, b: f64| {
        let d = a - b;
        r + d * d
    },
    vaddq_f64
);
// KL divergence (f64): fused div → ln → mul → accumulate (dot skeleton).
crate::simd_reduce2!(
    kl_divergence_f64,
    f64,
    ["neon"],
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, vp: float64x2_t, vq: float64x2_t| unsafe {
        let r = vln_128d(vdivq_f64(vp, vq));
        vaddq_f64(acc, vmulq_f64(vp, r))
    },
    |v| unsafe { hsum_128d(v) },
    |r: f64, a: f64, b: f64| r + a * crate::kernels::ln::ln_f64(a / b),
    vaddq_f64
);
// Jensen–Shannon divergence (f64): raw two-sided sum (wrapper halves it).
crate::simd_reduce2!(
    js_divergence_f64,
    f64,
    ["neon"],
    2,
    |p| unsafe { vld1q_f64(p) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, vp: float64x2_t, vq: float64x2_t| unsafe {
        let m = vmulq_f64(vaddq_f64(vp, vq), vdupq_n_f64(0.5));
        let tp = vmulq_f64(vp, vln_128d(vdivq_f64(vp, m)));
        let tq = vmulq_f64(vq, vln_128d(vdivq_f64(vq, m)));
        vaddq_f64(acc, vaddq_f64(tp, tq))
    },
    |v| unsafe { hsum_128d(v) },
    |r: f64, a: f64, b: f64| {
        let m = (a + b) * 0.5;
        r + a * crate::kernels::ln::ln_f64(a / m) + b * crate::kernels::ln::ln_f64(b / m)
    },
    vaddq_f64
);

crate::simd_argminmax!(
    argmax_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    // i64 index lanes: [0, 1] — one index per f64 lane, so the 64-bit
    // mask blends the index vector correctly (an i32 index vector would
    // need a 64→32 mask expansion that can't track the two halves of an
    // f64 lane independently). vcombine_s64(lo, hi): lane 0 = lo = 0.
    unsafe { vcombine_s64(vcreate_s64(0), vcreate_s64(1)) },
    |i: i64| unsafe { vdupq_n_s64(i) },
    |a, b| unsafe { vaddq_s64(a, b) },
    // NaN-aware dethrone (see scalar `argmax`).
    |a: float64x2_t, b: float64x2_t| unsafe {
        let gt = vcgtq_f64(a, b);
        let not_nan_b = vceqq_f64(b, b);
        let not_nan_a = vceqq_f64(a, a);
        // vmvnq has no 64-bit form; not_nan is all-ones/all-zeros, so the
        // NaN mask is an equality with zero.
        let nan_b = vceqq_u64(not_nan_b, vdupq_n_u64(0));
        vandq_u64(not_nan_a, vorrq_u64(gt, nan_b))
    },
    |mask: uint64x2_t, a: float64x2_t, b: float64x2_t| unsafe { vbslq_f64(mask, a, b) },
    |mask: uint64x2_t, a: int64x2_t, b: int64x2_t| unsafe { vbslq_s64(mask, a, b) },
    |a: f64, b: f64| a > b,
    |v, idx| unsafe { argmax_pair_128d(v, idx) }
);

crate::simd_argminmax!(
    argmin_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    unsafe { vcombine_s64(vcreate_s64(0), vcreate_s64(1)) },
    |i: i64| unsafe { vdupq_n_s64(i) },
    |a, b| unsafe { vaddq_s64(a, b) },
    // NaN-aware dethrone (see argmax_f64 above).
    |a: float64x2_t, b: float64x2_t| unsafe {
        let lt = vcltq_f64(a, b);
        let not_nan_b = vceqq_f64(b, b);
        let not_nan_a = vceqq_f64(a, a);
        // vmvnq has no 64-bit form; not_nan is all-ones/all-zeros, so the
        // NaN mask is an equality with zero.
        let nan_b = vceqq_u64(not_nan_b, vdupq_n_u64(0));
        vandq_u64(not_nan_a, vorrq_u64(lt, nan_b))
    },
    |mask: uint64x2_t, a: float64x2_t, b: float64x2_t| unsafe { vbslq_f64(mask, a, b) },
    |mask: uint64x2_t, a: int64x2_t, b: int64x2_t| unsafe { vbslq_s64(mask, a, b) },
    |a: f64, b: f64| a < b,
    |v, idx| unsafe { argmin_pair_128d(v, idx) }
);

// count_nan_f64: lanes where v != v (vceq is all-ones for ordered; invert
// via XOR with all-ones — NEON has no 64-bit `vmvn`).
crate::simd_count!(
    count_nan_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |v: float64x2_t| unsafe { veorq_u64(vceqq_f64(v, v), vdupq_n_u64(u64::MAX)) },
    |m: uint64x2_t| unsafe { vaddvq_u64(vandq_u64(m, vdupq_n_u64(1))) } as usize,
    |x: f64| x.is_nan()
);

// count_zero_f64: lanes equal to +/-0.0 (they compare equal).
crate::simd_count!(
    count_zero_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |v: float64x2_t| unsafe { vceqq_f64(v, vdupq_n_f64(0.0)) },
    |m: uint64x2_t| unsafe { vaddvq_u64(vandq_u64(m, vdupq_n_u64(1))) } as usize,
    |x: f64| x == 0.0
);

// count_infinite_f64: lanes whose |v| == +inf.
crate::simd_count!(
    count_infinite_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |v: float64x2_t| unsafe { vceqq_f64(vabsq_f64(v), vdupq_n_f64(f64::INFINITY)) },
    |m: uint64x2_t| unsafe { vaddvq_u64(vandq_u64(m, vdupq_n_u64(1))) } as usize,
    |x: f64| x.is_infinite()
);

// f64 elementwise maps for NEON (2 lanes).
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

crate::simd_clip!(
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
// abs_sub: |a - b| per lane (native vabs after sub).
crate::simd_map2!(
    abs_sub_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |a: float64x2_t, b: float64x2_t| unsafe { vabsq_f64(vsubq_f64(a, b)) },
    |x: f64, y: f64| (x - y).abs()
);
// hypot_f64: overflow-safe sqrt(a²+b²) via scale-by-max (see f32 hypot).
crate::simd_map2!(
    hypot_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |a: float64x2_t, b: float64x2_t| unsafe {
        let ax = vabsq_f64(a);
        let ay = vabsq_f64(b);
        let mx = vmaxq_f64(ax, ay);
        let mn = vminq_f64(ax, ay);
        let t = vdivq_f64(mn, mx);
        let one = vdupq_n_f64(1.0);
        let r = vmulq_f64(mx, vsqrtq_f64(vaddq_f64(vmulq_f64(t, t), one)));
        let zero_m = vceqq_f64(mn, vdupq_n_f64(0.0));
        let r = vbslq_f64(zero_m, mx, r);
        // any NaN → NaN (vceq is all-ones for ordered; invert via XOR).
        let nan_m = vorrq_u64(
            veorq_u64(vceqq_f64(a, a), vdupq_n_u64(u64::MAX)),
            veorq_u64(vceqq_f64(b, b), vdupq_n_u64(u64::MAX)),
        );
        let r = vbslq_f64(nan_m, vdupq_n_f64(f64::NAN), r);
        let inf = vdupq_n_f64(f64::INFINITY);
        let inf_m = vorrq_u64(vceqq_f64(ax, inf), vceqq_f64(ay, inf));
        vbslq_f64(inf_m, inf, r)
    },
    |x: f64, y: f64| crate::kernels::hypot::hypot_f64(x, y)
);
// powi_f64: bit-exact exponentiation by squaring (see f32 powi).
crate::simd_powi!(
    powi_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |a, b| unsafe { vmulq_f64(a, b) },
    |a, b| unsafe { vdivq_f64(a, b) },
    unsafe { vdupq_n_f64(1.0) },
    |x: f64, n: i32| crate::kernels::powi::powi_f64(x, n)
);

// f64 vector exp for NEON (2 lanes).
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
    |v| unsafe { vreinterpretq_f64_s64(v) },
    |v| unsafe { vreinterpretq_s64_f64(v) },
    // Round-to-nearest (ties-even) float→int: the aarch64 FCVTNS instruction.
    |v| unsafe { vcvtaq_s64_f64(v) },
    // int→float for the reduction (exact for |n| < 2^52).
    |v| unsafe { vcvtq_f64_s64(v) },
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
// Vector ln (f64): fdlibm e_log, see simd_ln_f64! in macros.rs.
crate::simd_ln_f64!(
    vln_128d,
    "neon",
    float64x2_t,
    int64x2_t,
    |s| unsafe { vdupq_n_f64(s) },
    |i| unsafe { vdupq_n_s64(i) },
    |a, b| unsafe { vaddq_f64(a, b) },
    |a, b| unsafe { vsubq_f64(a, b) },
    |a, b| unsafe { vmulq_f64(a, b) },
    |a, b| unsafe { vdivq_f64(a, b) },
    |v| unsafe { vreinterpretq_f64_s64(v) },
    |v| unsafe { vreinterpretq_f64_s64(v) },
    |v| unsafe { vreinterpretq_s64_f64(v) },
    |a, b| unsafe { vandq_s64(a, b) },
    |a, b| unsafe { vorrq_s64(a, b) },
    |v| unsafe { vshrq_n_s64(v, 52) },
    |a, b| unsafe { vreinterpretq_f64_s64(vreinterpretq_s64_u64(vcgtq_f64(a, b))) },
    |a, b| unsafe { vreinterpretq_f64_s64(vreinterpretq_s64_u64(vcltq_f64(a, b))) },
    |a, b| unsafe { vreinterpretq_f64_s64(vreinterpretq_s64_u64(vceqq_f64(a, b))) },
    |a, b| unsafe {
        vreinterpretq_f64_s64(vandq_s64(
            vreinterpretq_s64_f64(a),
            vreinterpretq_s64_f64(b),
        ))
    },
    |a, b| unsafe {
        vreinterpretq_f64_s64(vbicq_s64(
            vreinterpretq_s64_f64(b),
            vreinterpretq_s64_f64(a),
        ))
    },
    |a, b| unsafe {
        vreinterpretq_f64_s64(vorrq_s64(
            vreinterpretq_s64_f64(a),
            vreinterpretq_s64_f64(b),
        ))
    }
);
// Ln (f64): one-pass map; the register kernel handles normal x.
crate::simd_map!(
    ln_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe { vln_128d(v) },
    |x: f64| crate::kernels::ln::ln_f64(x)
);
// Softplus (f64): overflow-free `max(x,0) + ln1p(e^-|x|)`.
crate::simd_map!(
    softplus_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v| unsafe {
        let zero = vdupq_n_f64(0.0);
        let a = vabsq_f64(v);
        let z = vexp_128d(vsubq_f64(zero, a));
        let u = vaddq_f64(vdupq_n_f64(1.0), z);
        let ln_u = vln_128d(u);
        let lp = vdivq_f64(vmulq_f64(ln_u, z), vsubq_f64(u, vdupq_n_f64(1.0)));
        let one = vceqq_f64(u, vdupq_n_f64(1.0));
        let lp = vbslq_f64(one, z, lp);
        vaddq_f64(vmaxq_f64(v, zero), lp)
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
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe { vexp_128d(v) },
    |x: f64| crate::kernels::exp::exp_f64(x)
);

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

// Logsumexp (f64): two-pass scalar-returning reduction.
crate::simd_logsumexp!(
    logsumexp_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |a, b| unsafe { vmaxq_f64(a, b) },
    |a, b| unsafe { vsubq_f64(a, b) },
    |v| unsafe { vexp_128d(v) },
    |v| unsafe { vaddvq_f64(v) },
    |v| unsafe { vmaxvq_f64(v) },
    |s| unsafe { vdupq_n_f64(s) },
    |x: f64| crate::kernels::exp::exp_f64(x),
    crate::kernels::ln::ln_f64
);

// Log-softmax (f64): three-pass map, 0-alloc.
crate::simd_log_softmax!(
    log_softmax_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |a, b| unsafe { vmaxq_f64(a, b) },
    |a, b| unsafe { vsubq_f64(a, b) },
    |v| unsafe { vexp_128d(v) },
    |v| unsafe { vaddvq_f64(v) },
    |v| unsafe { vmaxvq_f64(v) },
    |s| unsafe { vdupq_n_f64(s) },
    |x: f64| crate::kernels::exp::exp_f64(x),
    crate::kernels::ln::ln_f64
);

// Layer norm (f64): three-pass, 0-alloc.
crate::simd_layer_norm!(
    layer_norm_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |a, b| unsafe { vaddq_f64(a, b) },
    |a, b| unsafe { vsubq_f64(a, b) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, v: float64x2_t| vaddq_f64(acc, vmulq_f64(v, v)),
    |v| unsafe { hsum_128d(v) },
    |s| unsafe { vdupq_n_f64(s) },
    |v, inv| vmulq_f64(v, vdupq_n_f64(inv)),
    crate::kernels::sqrt::sqrt_f64
);

crate::simd_map!(
    sigmoid_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe {
        // Saturated fast path: all lanes already at 0/1 (skip the exp).
        let pos = vcgtq_f64(v, vdupq_n_f64(36.74));
        let neg = vcltq_f64(v, vdupq_n_f64(-744.0));
        if vgetq_lane_u64(vorrq_u64(pos, neg), 0) != 0
            && vgetq_lane_u64(vorrq_u64(pos, neg), 1) != 0
        {
            return vreinterpretq_f64_u64(vandq_u64(pos, vreinterpretq_u64_f64(vdupq_n_f64(1.0))));
        }
        vdivq_f64(
            vdupq_n_f64(1.0),
            vaddq_f64(vdupq_n_f64(1.0), vexp_128d(vnegq_f64(v))),
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
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe {
        // Saturated fast path: silu(x) = x for x > 36.74, 0 for x < -745.
        let pos = vcgtq_f64(v, vdupq_n_f64(36.74));
        let neg = vcltq_f64(v, vdupq_n_f64(-744.0));
        if vgetq_lane_u64(vorrq_u64(pos, neg), 0) != 0
            && vgetq_lane_u64(vorrq_u64(pos, neg), 1) != 0
        {
            return vreinterpretq_f64_u64(vandq_u64(pos, vreinterpretq_u64_f64(v)));
        }
        vdivq_f64(v, vaddq_f64(vdupq_n_f64(1.0), vexp_128d(vnegq_f64(v))))
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
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe {
        // Saturated fast path: gelu(x) = x for x > 7.21, 0 for x < -7.21.
        let pos = vcgtq_f64(v, vdupq_n_f64(7.21));
        let neg = vcltq_f64(v, vdupq_n_f64(-7.21));
        if vgetq_lane_u64(vorrq_u64(pos, neg), 0) != 0
            && vgetq_lane_u64(vorrq_u64(pos, neg), 1) != 0
        {
            return vreinterpretq_f64_u64(vandq_u64(pos, vreinterpretq_u64_f64(v)));
        }
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
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe { vmaxq_f64(v, vdupq_n_f64(0.0)) },
    |x: f64| x.max(0.0)
);

// Tanh map (f64): tanh(x) = 1 - 2/(exp(2x)+1).
crate::simd_map!(
    tanh_f64,
    f64,
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    |v: float64x2_t| unsafe {
        let a = vabsq_f64(v);
        // ±1 for |x| > 19.062, x for |x| < 5e-8, series for |x| < 0.1,
        // ratio (e-1)/(e+1) beyond (Sterbenz-exact, clamped for overflow).
        let big_mask = vcgtq_f64(a, vdupq_n_f64(19.062));
        if vgetq_lane_u64(big_mask, 0) != 0 && vgetq_lane_u64(big_mask, 1) != 0 {
            let sign = vandq_u64(vreinterpretq_u64_f64(v), vdupq_n_u64(0x8000_0000_0000_0000));
            return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(vdupq_n_f64(1.0)), sign));
        }
        let y = vmulq_f64(v, v);
        let p = vdupq_n_f64(0.003_592_128_572_437_055);
        let p = vaddq_f64(vmulq_f64(p, y), vdupq_n_f64(-0.008_863_235_529_902_197));
        let p = vaddq_f64(vmulq_f64(p, y), vdupq_n_f64(0.021_869_488_536_155_2));
        let p = vaddq_f64(vmulq_f64(p, y), vdupq_n_f64(-0.053_968_253_968_253_97));
        let p = vaddq_f64(vmulq_f64(p, y), vdupq_n_f64(0.133_333_333_333_333_33));
        let p = vaddq_f64(vmulq_f64(p, y), vdupq_n_f64(-0.333_333_333_333_333_3));
        let series = vmulq_f64(v, vaddq_f64(vmulq_f64(p, y), vdupq_n_f64(1.0)));
        let e = vexp_128d(vaddq_f64(v, v));
        let em = vminq_f64(e, vdupq_n_f64(f64::MAX));
        let ratio = vdivq_f64(
            vsubq_f64(em, vdupq_n_f64(1.0)),
            vaddq_f64(em, vdupq_n_f64(1.0)),
        );
        let sign = vandq_u64(vreinterpretq_u64_f64(v), vdupq_n_u64(0x8000_0000_0000_0000));
        let big = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(ratio), sign));
        let ser_mask = vcltq_f64(a, vdupq_n_f64(0.1));
        let small = vcltq_f64(a, vdupq_n_f64(2e-8));
        let mid = vbslq_f64(ser_mask, series, big);
        let result = vbslq_f64(small, v, mid);
        let one = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(vdupq_n_f64(1.0)), sign));
        vbslq_f64(big_mask, one, result)
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
    "neon",
    2,
    |p| unsafe { vld1q_f64(p) },
    |p, v| unsafe { vst1q_f64(p, v) },
    vdupq_n_f64(0.0),
    |acc: float64x2_t, v: float64x2_t| vaddq_f64(acc, vmulq_f64(v, v)),
    |v| unsafe { hsum_128d(v) },
    |v, inv| vmulq_f64(v, vdupq_n_f64(inv)),
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
    fn f64_exp_matches_scalar_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        // Regression: the 2^52 add-magic mis-rounded negative inputs on
        // NEON (n = -1.5 for x·log2e = -1.44), producing exp(-1) ≈ 0.520.
        // The round-to-nearest FCVTNS path must give exp(-1) = 0.36788.
        for &x in &[-1.0_f64, -0.5, 0.0, 0.5, 1.0, -100.0, 100.0] {
            let v = unsafe { vdupq_n_f64(x) };
            let r = unsafe { vexp_128d(v) };
            let mut out = [0.0_f64; 2];
            unsafe { vst1q_f64(out.as_mut_ptr(), r) };
            let expected = crate::kernels::exp::exp_f64(x);
            let tol = expected.abs() * 2e-12 + 1e-14;
            assert!(
                (out[0] - expected).abs() <= tol,
                "vexp_128d({x}) = {} vs scalar {expected}",
                out[0]
            );
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn f64_argmax_argmin_matches_scalar_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        // Regression: the index vector must track per-f64-lane (i64 lanes,
        // one index per value lane); an i32 index vector with a 64-bit mask
        // blend corrupted the tracked index on multi-chunk inputs.
        let cases: &[&[f64]] = &[
            &[5.0, 3.0, 8.0, 1.0, 8.0], // argmax → idx 2, argmin → idx 3
            &[1.0, 2.0],                // single chunk
            &[9.0, 1.0, 9.0],           // tie across chunk boundary
            &[-3.0, -1.0, -2.0],        // negatives
        ];
        for data in cases {
            let (m, idx) = unsafe { argmax_f64(data) };
            let ref_idx = data
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(idx, ref_idx, "argmax {data:?}: m={m}");
            let (m, idx) = unsafe { argmin_f64(data) };
            let ref_idx = data
                .iter()
                .enumerate()
                .min_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(idx, ref_idx, "argmin {data:?}: m={m}");
        }
    }
}
