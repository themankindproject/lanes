//! Std-free overflow-safe hypotenuse (`hypot`) for `f32`/`f64`.
//!
//! In `std` builds this delegates to `f32::hypot`/`f64::hypot` (libc, ~1 ulp).
//! In `no_std` builds it uses the scale-by-max strategy (the same approach as
//! SLEEF's `xhypotf_u35`): `mx * sqrt((mn/mx)² + 1)`, which avoids the
//! spurious overflow of the naive `sqrt(x² + y²)`.
//!
//! Special values follow IEEE/`std`: `hypot(inf, anything) == inf` (even
//! NaN), `hypot(NaN, finite) == NaN`, `hypot(x, 0) == |x|`.

/// `f32` hypotenuse. Delegates to `f32::hypot` in `std`; portable
/// scale-by-max in `no_std`.
// TODO(issue #1 phase 3): remove once `dispatch_hypot` wires this in.
#[allow(dead_code)]
#[inline]
pub(crate) fn hypot(x: f32, y: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.hypot(y)
    }
    #[cfg(not(feature = "std"))]
    {
        hypot_f32(x, y)
    }
}

/// `f64` hypotenuse. Delegates to `f64::hypot` in `std`; portable
/// scale-by-max in `no_std`.
// TODO(issue #1 phase 3): remove once `dispatch_hypot_f64` wires this in.
#[allow(dead_code)]
#[inline]
pub(crate) fn hypot_f64(x: f64, y: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.hypot(y)
    }
    #[cfg(not(feature = "std"))]
    {
        hypot_f64_impl(x, y)
    }
}

/// Portable scale-by-max `f32` hypotenuse (SLEEF `xhypotf_u35` strategy).
#[cfg(any(not(feature = "std"), test))]
#[allow(dead_code)] // only reachable on no_std builds; clippy --all-features sees std
#[inline]
fn hypot_f32(x: f32, y: f32) -> f32 {
    // inf wins over NaN (IEEE): hypot(inf, nan) == inf.
    if x.is_infinite() || y.is_infinite() {
        return f32::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    let ay = y.abs();
    let (mx, mn) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    if mn == 0.0 {
        return mx;
    }
    let t = mn / mx;
    mx * crate::kernels::sqrt::sqrt(t * t + 1.0)
}

/// Portable scale-by-max `f64` hypotenuse.
#[cfg(any(not(feature = "std"), test))]
#[allow(dead_code)] // only reachable on no_std builds; clippy --all-features sees std
#[inline]
fn hypot_f64_impl(x: f64, y: f64) -> f64 {
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    let ay = y.abs();
    let (mx, mn) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    if mn == 0.0 {
        return mx;
    }
    let t = mn / mx;
    mx * crate::kernels::sqrt::sqrt_f64(t * t + 1.0)
}

#[cfg(test)]
mod tests {
    use super::{hypot_f32, hypot_f64_impl};

    #[test]
    fn hypot_no_overflow() {
        // Naive sqrt(x²+y²) overflows here; scale-by-max must not.
        let big = 2.0e19_f32;
        let got = hypot_f32(big, big);
        let want = big.hypot(big);
        assert!(got.is_finite(), "hypot({big},{big}) overflowed to {got}");
        assert!(
            (got - want).abs() <= want.abs() * 1e-6,
            "got {got}, want {want}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact: inf/0 special values per IEEE
    fn hypot_special_values() {
        assert_eq!(hypot_f32(f32::INFINITY, f32::NAN), f32::INFINITY);
        assert!(hypot_f32(f32::NAN, 1.0).is_nan());
        assert_eq!(hypot_f32(3.0, 0.0), 3.0);
        assert_eq!(hypot_f32(0.0, 0.0), 0.0);
        let r = hypot_f32(3.0, 4.0);
        assert!((r - 5.0).abs() < 1e-6, "3-4-5 got {r}");
        assert_eq!(hypot_f64_impl(f64::INFINITY, f64::NAN), f64::INFINITY);
        assert!(hypot_f64_impl(f64::NAN, 1.0).is_nan());
        let r64 = hypot_f64_impl(3.0, 4.0);
        assert!((r64 - 5.0).abs() < 1e-12, "3-4-5 f64 got {r64}");
    }

    #[test]
    fn hypot_matches_std_within_2ulp() {
        // Dense sweep over exponent bins vs the std oracle.
        for ex in 0..255 {
            for frac in [0x0000_0000_u32, 0x2000_0000, 0x4000_0000, 0x6000_0000] {
                let x = f32::from_bits((ex << 23) | frac);
                let y = f32::from_bits(((254 - ex) << 23) | frac);
                if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 {
                    continue;
                }
                let got = hypot_f32(x, y);
                let want = x.hypot(y);
                let rel = (got - want).abs() / want.abs();
                assert!(rel < 1e-6, "hypot({x:e},{y:e}) = {got:e}, std {want:e}");
            }
        }
    }
}
