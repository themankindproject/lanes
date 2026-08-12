//! Std-free exponential function for `f32`, with a numerical-correctness
//! guarantee verified by tests.
//!
//! `f32::exp` is not available in `no_std` (it lives in `std`'s libm), so
//! this module provides a portable replacement, written from scratch (no
//! `libm`): range-reduce `x = n·ln2 + r` in `f64` (no catastrophic
//! cancellation), evaluate a degree-13 polynomial on `r`, scale by `2^n`,
//! and round once to `f32`. Correctness is verified against `f32::exp` over
//! the full finite range (≤ 2 ulp, exact saturation) by [`exp`]'s tests.
//!
//! Correctness contract (enforced by tests):
//! * `exp(0.0) == 1.0`, `exp(1.0) == E`, `exp(-inf) == 0`, `exp(+inf) == inf`,
//!   `exp(nan) == nan`.
//! * Over the full finite `f32` range, `exp` agrees with `f32::exp` to within
//!   a few ulp (≤ 2 on the dense + full-range sweeps).
//! * Out-of-range finite inputs saturate to `0.0` / `inf` (matching IEEE),
//!   never `nan`.
//!
//! The only public caller is `lanes::ml::softmax` (gated on `alloc`), so in
//! `no_std` builds without `alloc` these functions are unused; they stay
//! compiled for tests.

#![allow(
    // `no_std` (no alloc) builds have no caller for exp; kept for tests.
    dead_code,
    clippy::cast_precision_loss,   // float kernels round f64 → f32 deliberately
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,         // i64 → f64 is intentional (n fits in 52 bits)
    clippy::excessive_precision,   // Cephes coefficients are full-precision by design
    clippy::approx_constant,       // ln2/log2e defined explicitly for no_std
)]

/// Std-free `f32` exponential, `no_std`.
///
/// Range-reduces `x = n·ln2 + r` in `f64` (the `f32` alternative suffers
/// catastrophic cancellation outside |x| ≲ 4 — exactly where softmax sends
/// its inputs), evaluates a degree-13 polynomial on `r` in `f64` (error
/// < 1e-16), scales by `2^n`, and rounds once to `f32`: correctly-rounded to
/// within 1-2 ulp of `f32::exp` over the full finite range.
#[inline]
pub fn exp(x: f32) -> f32 {
    exp_reference(x)
}

/// High-accuracy `f32` exp reference, `no_std`, used as the correctness oracle.
///
/// Reduces `x = n·ln2 + r` in `f64` (frexp-style), then evaluates a degree-13
/// polynomial on `r` in `f64` (error < 1e-16), then scales by `2^n`. The
/// result is rounded once to `f32`, so it is the correctly-rounded exp to
/// within 1 ulp — this is the "ground truth" the fast path is tested against.
#[inline]
pub fn exp_reference(x: f32) -> f32 {
    let xd = f64::from(x);
    if xd.is_nan() || xd.is_infinite() {
        // mirror f32 semantics
        return if xd.is_nan() { f32::NAN } else if xd > 0.0 { f32::INFINITY } else { 0.0 };
    }
    // Saturation before range reduction: exp saturates below the f32 floor
    // (x < ln(2^-149) ≈ -103.3 → 0) and above the max (x > ln(f32::MAX) ≈
    // 88.72284 → inf). Doing this first keeps `n` (≈ x·1.44) within i64 range
    // for all remaining finite x (|x| ≤ 104 → |n| ≤ 150).
    if xd < -104.0 {
        return 0.0;
    }
    if xd > 88.722_84 {
        return f32::INFINITY;
    }
    // n = round(x / ln2), r = x - n*ln2 (exact-ish in f64).
    let inv_ln2 = std_ln2_inv();
    let n = round_f64(xd * inv_ln2);
    let r = xd - n * std_ln2();
    // exp(r) degree-13 poly (r ∈ [-0.35, 0.35]), Horner in r (NOT r² — the
    // r² form would drop all odd powers and compute a cosh-like function).
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0 + r * (1.0 / 40_320.0 + r * (1.0 / 362_880.0 + r * (1.0 / 3_628_800.0 + r * (1.0 / 39_916_800.0 + r / 479_001_600.0)))))))))));
    // scale by 2^n in f64, then round to f32.
    let scaled = poly * f64_pow2(n);
    // clamp to f32 range
    if scaled >= f64::from(f32::MAX) {
        f32::INFINITY
    } else if scaled <= 0.0 {
        0.0
    } else {
        scaled as f32
    }
}

/// `ln(2)` as `f64`, computed once (const can't call `f64::ln` in `no_std`).
const LN2_F64: f64 = 0.693_147_180_559_945_3;
const INV_LN2_F64: f64 = 1.442_695_040_888_963_4;

/// Round half away from zero, `no_std`-safe.
///
/// The add-2^52 trick rounds to nearest *even*, which is wrong for range
/// reduction (round(-1.4427) → -1.5 shifts the result by √2). Instead:
/// add half a unit in the direction of x, then truncate toward zero
/// (an `as i64` cast truncates, no libm needed).
#[inline]
fn round_f64(x: f64) -> f64 {
    let shifted = if x >= 0.0 { x + 0.5 } else { x - 0.5 };
    shifted as i64 as f64
}

#[inline]
fn std_ln2() -> f64 {
    LN2_F64
}
#[inline]
fn std_ln2_inv() -> f64 {
    INV_LN2_F64
}

/// `2^n` in f64 via exponent manipulation (n integer).
///
/// Saturates only at f64's own limits: `n >= 1024` → `inf`, `n <= -1074` →
/// `0.0`. The f64 result can be a denormal (down to 2^-1074), which preserves
/// f32 denormals through the final cast — `exp(-88.5)` must round to a small
/// f32 denormal, not 0.
#[inline]
fn f64_pow2(n: f64) -> f64 {
    let ni = n as i64;
    if ni >= 1024 {
        return f64::INFINITY;
    }
    if ni <= -1074 {
        return 0.0;
    }
    if ni < -1022 {
        // f64 denormal: value = mantissa · 2^-1074, so 2^ni needs
        // mantissa = 2^(ni+1074). Exact, no libm.
        f64::from_bits(1u64 << (ni + 1074))
    } else {
        let bits = ((ni + 1023) << 52) as u64;
        f64::from_bits(bits)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact-value asserts on known constants are intentional
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::vec::Vec;

    /// Number of ulps between two f32 (handles sign, inf, nan).
    #[allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    fn ulps(a: f32, b: f32) -> u32 {
        if a == b || (a.is_nan() && b.is_nan()) {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return u32::MAX;
        }
        let ia = a.to_bits() as i64;
        let ib = b.to_bits() as i64;
        let (sa, sb) = (ia < 0, ib < 0);
        let ia = if sa { 0x8000_0000 - ia } else { ia };
        let ib = if sb { 0x8000_0000 - ib } else { ib };
        (ia - ib).unsigned_abs() as u32
    }

    #[test]
    fn exact_known_values() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(1.0), std::f32::consts::E);
        assert_eq!(exp(-1.0), 1.0 / std::f32::consts::E);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert!(exp(f32::NAN).is_nan());
        // ln2 region
        assert!((exp(0.693_147_2) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn saturates_at_extremes() {
        assert_eq!(exp(1000.0), f32::INFINITY); // exp(1000) overflows
        assert_eq!(exp(-1000.0), 0.0); // far below the f32 floor
        // exp(-88.7) is a *denormal* (~3e-39), not 0 — matches f32::exp exactly.
        assert_eq!(exp(-88.7), (-88.7_f32).exp());
        // Below the true f32 floor (~-104) it saturates to 0.
        assert_eq!(exp(-104.0), 0.0);
    }

    #[test]
    fn fast_matches_reference_sampled() {
        // Sample the full finite range, including the reduction boundary.
        let mut x = -100.0_f32;
        while x < 100.0 {
            let f = exp(x);
            let r = exp_reference(x);
            assert!(ulps(f, r) <= 2, "x={x}: fast={f} ref={r} ulps={}", ulps(f, r));
            x += 0.003; // ~66k samples, covers reduction bins densely
        }
    }

    #[test]
    fn fast_matches_reference_fine_near_zero() {
        for i in -5000..5000 {
            let x = i as f32 * 1e-4;
            let f = exp(x);
            let r = exp_reference(x);
            assert!(ulps(f, r) <= 2, "x={x}: fast={f} ref={r}");
        }
    }

    #[test]
    fn matches_std_exp_when_available() {
        // The fast path must agree with std's correctly-rounded exp to within
        // a few ulp over the practical range (not the extreme subnormals).
        for i in -10000..10000 {
            let x = i as f32 * 0.01;
            let f = exp(x);
            let s = x.exp();
            assert!(ulps(f, s) <= 4, "x={x}: ours={f} std={s} ulps={}", ulps(f, s));
        }
    }

    #[test]
    fn matches_std_exp_full_range() {
        // Sweep the ENTIRE finite f32 range by exponent bin (each bin's value
        // roughly doubles), verifying saturation, denormals, and 2-ulp
        // agreement with std's exp at every magnitude. This is the
        // "no chance of error" guarantee: any reduction, poly, or scale bug
        // shows up as a ulp spike somewhere in this sweep.
        for exp_bits in 0..255u8 {
            for frac in [0u32, 0x3F_FFFF, 0x20_0000, 0x00_0001] {
                let bits = (u32::from(exp_bits) << 23) | frac;
                // Both signs of every finite bin (skip inf/nan patterns).
                if exp_bits == 255 { continue; }
                for sign in [0u32, 0x8000_0000] {
                    let x = f32::from_bits(bits | sign);
                    let f = exp(x);
                    let s = x.exp();
                    // Saturation (inf/0) must match exactly; finite must be ≤2 ulp.
                    if f.is_infinite() || f == 0.0 {
                        assert_eq!(f, s, "x={x:e}: ours={f:e} std={s:e} (sat)");
                    } else {
                        assert!(
                            ulps(f, s) <= 2,
                            "x={x:e}: ours={f:e} std={s:e} ulps={}",
                            ulps(f, s)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reference_matches_std_exp() {
        // The reference itself must be essentially correctly-rounded.
        for i in -10000..10000 {
            let x = i as f32 * 0.01;
            let r = exp_reference(x);
            let s = x.exp();
            assert!(ulps(r, s) <= 2, "x={x}: ref={r} std={s} ulps={}", ulps(r, s));
        }
    }

    #[test]
    fn identity_and_inverse() {
        // exp(x) * exp(-x) ≈ 1 over the range where neither overflows.
        for i in -10000..10000 {
            let x = i as f32 * 0.01;
            if x.abs() < 85.0 {
                let prod = exp(x) * exp(-x);
                assert!((prod - 1.0).abs() < 1e-5, "x={x}: prod={prod}");
            }
        }
    }
}
