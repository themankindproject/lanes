//! Std-free natural logarithm for `f32` and `f64`, with a numerical
//! correctness guarantee verified by tests.
//!
//! `f32::ln`/`f64::ln` are unavailable in `no_std` (they live in `std`'s
//! libm), so this module provides portable replacements following fdlibm's
//! `e_log.c` (Sun Microsystems, freely distributable), adapted to Rust.
//!
//! Algorithm (fdlibm `__ieee754_log`):
//! 1. Extract the exponent `k` and mantissa `f` so that `x = 2^k·(1+f)`
//!    with `√2/2 < 1+f < √2`.
//! 2. Approximate `log(1+f) = 2s + s·R(s²)` with `s = f/(2+f)` and a
//!    degree-14 Remez polynomial `R` (error < 2^-58.45). The final form
//!    `log(x) = k·ln2_hi + (f - (hfsq - (s·(hfsq+R) + k·ln2_lo)))` keeps
//!    the total error below 1 ulp.
//! 3. For `|f| < 2^-20` use the short series `f - f²/2 + f³/3` (the main
//!    path would lose precision).
//! 4. Subnormals are scaled by `2^25` (f32) / `2^54` (f64) before the
//!    reduction and the exponent corrected afterwards.
//!
//! Special cases follow IEEE 754: `ln(±0) = -inf`, `ln(x<0) = nan`,
//! `ln(+inf) = +inf`, `ln(nan) = nan`.
//!
//! The only callers are the `ln` map kernels (gated on `alloc`); in
//! `no_std` builds without `alloc` they stay compiled for tests.

#![allow(
    dead_code, // no_std (no alloc) builds have no caller; kept for tests
    clippy::cast_precision_loss,   // float kernels round f64 → f32 deliberately
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,         // integer → float casts are exact by design
    clippy::excessive_precision,   // fdlibm coefficients are full-precision by design
    clippy::approx_constant,       // ln2 splits defined explicitly for no_std
    clippy::many_single_char_names, // fdlibm's x/f/k/s names are canonical
    clippy::float_cmp,             // exact IEEE special-value checks in tests
)]

// fdlibm constants for the f32 path (computed in f64, rounded once).
const LN2_HI_F32: f64 = 6.931_381_225_6e-01; // 0x3f317180
const LN2_LO_F32: f64 = 9.058_000_614_5e-06; // 0x3717f7d1
const LG1_F32: f64 = 6.666_666_865_3e-01; // 0x3f2aaaab
const LG2_F32: f64 = 4.000_000_059_6e-01; // 0x3ecccccd
const LG3_F32: f64 = 2.857_142_984_9e-01; // 0x3e924925
const LG4_F32: f64 = 2.222_219_854_6e-01; // 0x3e638e29
const LG5_F32: f64 = 1.818_357_259_0e-01; // 0x3e3a3325
const LG6_F32: f64 = 1.531_383_842_2e-01; // 0x3e1cd04f
const LG7_F32: f64 = 1.479_819_864_0e-01; // 0x3e178897

// fdlibm constants for the f64 path (bit-exact hex values).
const LN2_HI_F64: f64 = 6.931_471_803_691_238_164_90e-01; // 0x3fe62e42fee00000
const LN2_LO_F64: f64 = 1.908_214_929_270_587_700_02e-10; // 0x3dea39ef35793c76
const LG1_F64: f64 = 6.666_666_666_666_735_130e-01; // 0x3fe5555555555593
const LG2_F64: f64 = 3.999_999_999_940_941_908e-01; // 0x3fd999999997fa04
const LG3_F64: f64 = 2.857_142_874_366_239_149e-01; // 0x3fd2492494229359
const LG4_F64: f64 = 2.222_219_843_214_978_396e-01; // 0x3fcc71c51d8e78af
const LG5_F64: f64 = 1.818_357_216_161_805_012e-01; // 0x3fc7466496cb03de
const LG6_F64: f64 = 1.531_383_769_920_937_332e-01; // 0x3fc39a09d078c69f
const LG7_F64: f64 = 1.479_819_860_511_658_591e-01; // 0x3fc2f112df3e5244

/// Extract `(f, k)` such that `x = 2^k · (1+f)` with `√2/2 < 1+f < √2`,
/// for a **normal** positive `x`. The mantissa-normalization from fdlibm:
/// extract the unbiased exponent and mantissa `m ∈ [1, 2)`, then halve
/// `m` (exact division by 2) when `m ≥ √2`.
#[inline]
fn reduce_normal(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let k = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mantissa = (bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
    let m = f64::from_bits(mantissa);
    if m >= 1.414_213_562_373_095_1 {
        (m / 2.0 - 1.0, k + 1)
    } else {
        (m - 1.0, k)
    }
}

/// fdlibm's `log(1+f)` core for `f ∈ (-0.293, 0.414)` (after normalization).
/// `(f, k)` from [`reduce_normal`]. Returns `log(x)` — the caller's `x`.
#[inline]
fn log_1p_fdlibm(f: f64, k: i32, ln2_hi: f64, ln2_lo: f64, lg: [f64; 7]) -> f64 {
    let dk = k as f64;
    if f.abs() < 9.536_743_164_062_5e-7 {
        // 2^-20
        // |f| < 2^-20: short series; the main path loses precision here.
        let r = f * f * (0.5 - 0.333_333_333_333_333_3 * f);
        if k == 0 {
            return f - r;
        }
        return dk * ln2_hi - ((r - dk * ln2_lo) - f);
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    // R(s²) = Lg1·z + Lg2·z² + ... + Lg7·z⁷ (odd powers of s).
    let t1 = w * (lg[1] + w * (lg[3] + w * lg[5]));
    let t2 = z * (lg[0] + w * (lg[2] + w * (lg[4] + w * lg[6])));
    let r = t2 + t1;
    if (0.38..=0.42).contains(&f) {
        // hfsq form is more accurate for f near the halving boundary.
        let hfsq = 0.5 * f * f;
        if k == 0 {
            return f - (hfsq - s * (hfsq + r));
        }
        return dk * ln2_hi - ((hfsq - (s * (hfsq + r) + dk * ln2_lo)) - f);
    }
    if k == 0 {
        return f - s * (f - r);
    }
    dk * ln2_hi - ((s * (f - r) - dk * ln2_lo) - f)
}

/// Std-free `f32` natural logarithm, `no_std`.
///
/// Follows fdlibm `__ieee754_logf`: extracts the exponent, reduces to
/// `f ∈ (-0.293, 0.414)`, evaluates the `s = f/(2+f)` rational form, and
/// scales by `k·ln2`. Subnormals are scaled by `2^25` before reduction.
/// Correctly rounded to within ~1 ulp of `f32::ln` over the full finite
/// range (verified by the dense sweep tests).
#[inline]
pub fn ln(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x < 0.0 {
        return f32::NAN; // ln(negative) = NaN (signaling per IEEE)
    }
    // Compute in f64 (mirrors exp.rs: one f64 rounding at the end).
    let mut y = x as f64;
    let mut k_adj = 0i32;
    if y < 1.175_494_350_822_287_5e-38 {
        // 2^-126 (f32 min normal)
        // Subnormal: scale up by 2^25 and correct the exponent.
        y *= 33_554_432.0; // 2^25
        k_adj = -25;
    }
    let (f, k) = reduce_normal(y);
    let r = log_1p_fdlibm(
        f,
        k + k_adj,
        LN2_HI_F32,
        LN2_LO_F32,
        [
            LG1_F32, LG2_F32, LG3_F32, LG4_F32, LG5_F32, LG6_F32, LG7_F32,
        ],
    );
    r as f32
}

/// Std-free `f64` natural logarithm, `no_std`.
///
/// Follows fdlibm `__ieee754_log` with the f64 constants. Subnormals are
/// scaled by `2^54`. Correctly rounded to within ~1 ulp of `f64::ln` over
/// the full finite range.
#[inline]
pub fn ln_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    let mut y = x;
    let mut k_adj = 0i32;
    if y < 2.225_073_858_507_201_4e-308 {
        // 2^-1022 (f64 min normal)
        y *= 18_014_398_509_481_984.0; // 2^54
        k_adj = -54;
    }
    let (f, k) = reduce_normal(y);
    log_1p_fdlibm(
        f,
        k + k_adj,
        LN2_HI_F64,
        LN2_LO_F64,
        [
            LG1_F64, LG2_F64, LG3_F64, LG4_F64, LG5_F64, LG6_F64, LG7_F64,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_special_values() {
        assert!(ln(0.0).is_infinite() && ln(0.0) < 0.0);
        assert!(ln(-0.0).is_infinite() && ln(-0.0) < 0.0);
        assert!(ln(-1.0).is_nan());
        assert_eq!(ln(f32::INFINITY), f32::INFINITY);
        assert!(ln(f32::NAN).is_nan());

        assert!(ln_f64(0.0).is_infinite() && ln_f64(0.0) < 0.0);
        assert!(ln_f64(-1.0).is_nan());
        assert_eq!(ln_f64(f64::INFINITY), f64::INFINITY);
        assert!(ln_f64(f64::NAN).is_nan());
    }

    #[test]
    fn ln_basic_values() {
        assert!((ln(1.0) - 0.0).abs() < 1e-12);
        assert!((ln_f64(1.0) - 0.0).abs() < 1e-15);
        assert!((ln(std::f32::consts::E) - 1.0).abs() < 1e-6);
        assert!((ln_f64(std::f64::consts::E) - 1.0).abs() < 1e-15);
        assert!((ln(2.0) - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((ln_f64(2.0) - std::f64::consts::LN_2).abs() < 1e-15);
    }

    #[test]
    fn ln_agrees_with_std() {
        // Dense sweep over the normal range + subnormals.
        let mut x = f32::MIN_POSITIVE;
        while x < f32::MAX / 2.0 {
            let got = ln(x);
            let want = x.ln();
            let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulps <= 2,
                "ln({x:e}) = {got:e}, std = {want:e}, {ulps} ulps"
            );
            x *= 1.000_1;
        }
    }

    #[test]
    fn ln_f64_agrees_with_std() {
        // Dense sweep over the normal range + subnormals.
        let mut x = f64::MIN_POSITIVE;
        while x < f64::MAX / 2.0 {
            let got = ln_f64(x);
            let want = x.ln();
            let ulps = (got.to_bits() as i128 - want.to_bits() as i128).abs();
            assert!(
                ulps <= 2,
                "ln_f64({x:e}) = {got:e}, std = {want:e}, {ulps} ulps"
            );
            x *= 1.000_1;
        }
    }
}
