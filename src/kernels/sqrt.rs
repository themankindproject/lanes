//! Square root for `f32`.
//!
//! In `std` builds, `sqrt` delegates to `f32::sqrt` (correctly rounded,
//! hardware-backed). In `no_std` builds, `f32::sqrt` is unavailable (it
//! lives in `std`'s libm), so a portable Newton-based replacement is used:
//! the classic magic-number initial guess + three Newton iterations in f64,
//! which converges to within ~1 ulp of the correctly rounded IEEE result.
//! Special values follow IEEE 754: `sqrt(±0) = ±0`, `sqrt(neg) = NaN`,
//! `sqrt(inf) = inf`, `sqrt(nan) = nan`.
//!
//! The vector kernels (`simd_map!` with the hardware `sqrt` per backend) are
//! correctly rounded — no approximation needed.

#![allow(
    clippy::cast_lossless,       // bit-trick rounding uses as-casts
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss, // u32 → f32 rounding is inherent to the trick
)]

/// `f32` square root. Delegates to `f32::sqrt` in `std` builds; uses the
/// portable Newton approximation in `no_std`.
///
/// IEEE 754 semantics: `sqrt(±0) = ±0`, `sqrt(x < 0) = NaN`,
/// `sqrt(inf) = inf`, `sqrt(nan) = nan`. For all finite positive inputs the
/// result is correctly rounded (`std`) or within ~1 ulp (`no_std`).
#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        sqrt_no_std(x)
    }
}

/// `f64` square root. Delegates to `f64::sqrt` in `std` builds; uses a
/// portable Newton iteration (seeded from the f32 fast-inverse-sqrt magic)
/// in `no_std`.
///
/// IEEE 754 semantics: `sqrt(±0) = ±0`, `sqrt(x < 0) = NaN`,
/// `sqrt(inf) = inf`, `sqrt(nan) = nan`. For all finite positive inputs the
/// result is correctly rounded (`std`) or within ~1 ulp (`no_std`).
#[inline]
pub(crate) fn sqrt_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        sqrt_f64_no_std(x)
    }
}

/// Std-free `f64` square root: f32 magic guess + Newton in f64.
#[cfg(any(not(feature = "std"), test))]
#[allow(dead_code)] // only reachable on no_std builds; clippy --all-features sees std
#[inline]
fn sqrt_f64_no_std(x: f64) -> f64 {
    if x <= 0.0 {
        // sqrt(±0) = ±0, sqrt(neg) = NaN (IEEE). NaN propagates.
        return if x == 0.0 { x } else { f64::NAN };
    }
    if x.is_infinite() {
        return f64::INFINITY; // sqrt(+inf) = inf
    }
    // Subnormal f64: the bit-trick seed on hi bits is garbage below 2^-1022.
    // Scale by 2^100 (exact — adds 100 to exponent), Newton, scale back by
    // 2^-50 (sqrt scales by half the exponent). Mirrors the f32 path exactly.
    if x < f64::from_bits(0x0010_0000_0000_0000) {
        let up = f64::from_bits(0x4630_0000_0000_0000); // 2^100
        let down = f64::from_bits(0x3CB0_0000_0000_0000); // 2^-50
        return sqrt_f64_no_std(x * up) * down;
    }
    // Seed with the f32 fast-inverse-sqrt magic on the high bits, then
    // Newton in f64: g = 0.5*(g + x/g). From a ~4% guess, four iterations
    // reach full f64 precision (error² per step).
    let hi = (x.to_bits() >> 32) as u32;
    let rsqrt_guess = 0x5F37_59DFu32.wrapping_sub(hi >> 1);
    let mut g = f64::from_bits(u64::from(rsqrt_guess) << 32) * x;
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    g
}

/// Std-free `f32` square root: magic-number guess + Newton in f64.
///
/// Compiled in `no_std` builds (where it is the implementation) and in
/// tests (where it is verified against `f32::sqrt`).
#[cfg(any(not(feature = "std"), test))]
#[inline]
fn sqrt_no_std(x: f32) -> f32 {
    if x <= 0.0 {
        // sqrt(±0) = ±0, sqrt(neg) = NaN (IEEE). NaN propagates.
        return if x == 0.0 { x } else { f32::NAN };
    }
    if x.is_infinite() {
        return f32::INFINITY; // sqrt(+inf) = inf
    }
    // Denormals: the magic-number trick breaks below 2^-126 (the exponent
    // manipulation produces garbage). Scale up by 2^100 (exact: adds 100 to
    // the exponent), compute, scale back by 2^-50 (sqrt scales by half).
    if x < f32::from_bits(0x0080_0000) {
        // 2^100 = exponent 227 = 0xE3; 2^-50 = exponent 77 = 0x4D.
        let up = f32::from_bits(0x7180_0000); // 2^100
        let down = f32::from_bits(0x2680_0000); // 2^-50 (0x4D << 23)
        return sqrt_no_std(x * up) * down;
    }
    // Initial guess via the classic fast-inverse-sqrt magic, then Newton in
    // f64 (not f32 — f32 division rounding would leave a 1-ulp error that
    // accumulates). The magic guess has ~4% error; Newton converges
    // quadratically (error² per step), so from 4e-2: 1.6e-3 → 2.6e-6 →
    // 6.8e-12 — three iterations reach full f64 precision.
    let bits = x.to_bits();
    let rsqrt_guess = 0x5F37_59DFu32.wrapping_sub(bits >> 1);
    let mut g = f32::from_bits(rsqrt_guess) as f64 * x as f64;
    // Newton in f64: g = 0.5 * (g + x/g). Three iterations → full precision.
    g = 0.5 * (g + x as f64 / g);
    g = 0.5 * (g + x as f64 / g);
    g = 0.5 * (g + x as f64 / g);
    g as f32
}

#[cfg(test)]
mod tests {
    use super::{sqrt, sqrt_no_std};
    use std::cmp::Ordering;

    /// ULP distance between two floats (sign-magnitude aware).
    fn ulps(a: f32, b: f32) -> u32 {
        let (a, b) = (a.to_bits(), b.to_bits());
        match (a & 0x8000_0000).cmp(&(b & 0x8000_0000)) {
            Ordering::Equal => a.abs_diff(b),
            _ => a.abs_diff(b) & 0x7FFF_FFFF,
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact: sqrt(±0) = ±0 per IEEE
    fn sqrt_known_values() {
        assert_eq!(sqrt(0.0), 0.0);
        assert_eq!(sqrt(-0.0), -0.0);
        // Perfect squares: ≤ 1 ulp (the f64 Newton may round either way
        // across an exactly-representable value).
        assert!(ulps(sqrt(1.0), 1.0) <= 1, "{}", sqrt(1.0));
        assert!(ulps(sqrt(4.0), 2.0) <= 1, "{}", sqrt(4.0));
        assert!(ulps(sqrt(16.0), 4.0) <= 1, "{}", sqrt(16.0));
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact: ±0, inf, nan are exact per IEEE
    fn sqrt_special_values() {
        assert_eq!(sqrt(f32::INFINITY), f32::INFINITY);
        assert!(sqrt(f32::NAN).is_nan());
        assert!(sqrt(-1.0).is_nan());
    }

    #[test]
    fn sqrt_denormals() {
        // Smallest denormal: sqrt(1.4e-45) ~ 3.7e-23 (still denormal, but
        // positive and finite; just verify it doesn't NaN and is consistent).
        let d = f32::from_bits(1);
        let r = sqrt_no_std(d);
        assert!(r.is_finite() && r > 0.0, "sqrt(denormal) = {r}");
        assert!((r * r - d).abs() <= d * 1e-4, "round-trip failed: {r}");
    }

    /// Full-range accuracy of the `no_std` path: every exponent bin, both
    /// signs, compared against `f32::sqrt` (the oracle, available in tests
    /// via std).
    #[test]
    fn sqrt_no_std_matches_std_full_range() {
        // All positive finite exponent bins × 4 fractions × 2 signs.
        for exp in 0..255 {
            for frac in [0x0000_0000, 0x2000_0000, 0x4000_0000, 0x6000_0000] {
                for sign in [0u32, 0x8000_0000] {
                    let bits = sign | (exp << 23) | frac;
                    let x = f32::from_bits(bits);
                    let got = sqrt_no_std(x);
                    let want = x.sqrt();
                    // NaN vs NaN and ±0 vs ±0 are equal; skip inf/nan inputs.
                    if x.is_nan() || x.is_infinite() || x <= 0.0 {
                        continue;
                    }
                    assert!(
                        ulps(got, want) <= 1,
                        "sqrt({x:e}) = {got:e}, std = {want:e}, ulps={}",
                        ulps(got, want)
                    );
                }
            }
        }
    }

    /// The std path is `f32::sqrt` itself — trivially correct; assert the
    /// delegation actually happens (identical results on a sample).
    #[test]
    #[allow(clippy::float_cmp)] // exact: both sides are f32::sqrt
    fn sqrt_std_delegates() {
        for x in [0.25_f32, 1.0, 2.0, 9.0, 1e10, 1e-30] {
            assert_eq!(sqrt(x), x.sqrt());
        }
    }
}
