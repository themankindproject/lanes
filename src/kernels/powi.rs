//! Std-free integer power (`powi`) for `f32`/`f64`.
//!
//! In `std` builds this delegates to `f32::powi`/`f64::powi`. In `no_std`
//! builds it uses the exponentiation-by-squaring algorithm from
//! `compiler-builtins` (`__powisf2`/`__powidf2`), which is exactly what the
//! `std` intrinsic lowers to — so results are bit-identical.
//!
//! Semantics (matching `std`): `powi(x, 0) == 1` for every `x` (including
//! NaN, ±inf, ±0); `powi(NaN, n>0) == NaN`; `powi(x, i32::MIN)` is
//! `1 / x^(2^31)`; negative `n` takes the reciprocal.

/// `f32` integer power. Delegates to `f32::powi` in `std`; portable squaring
/// loop in `no_std`.
// TODO(issue #1 phase 4): remove once `dispatch_powi` wires this in.
#[allow(dead_code)]
#[inline]
pub(crate) fn powi(x: f32, n: i32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.powi(n)
    }
    #[cfg(not(feature = "std"))]
    {
        powi_f32(x, n)
    }
}

/// `f64` integer power. Delegates to `f64::powi` in `std`; portable squaring
/// loop in `no_std`.
// TODO(issue #1 phase 4): remove once `dispatch_powi_f64` wires this in.
#[allow(dead_code)]
#[inline]
pub(crate) fn powi_f64(x: f64, n: i32) -> f64 {
    #[cfg(feature = "std")]
    {
        x.powi(n)
    }
    #[cfg(not(feature = "std"))]
    {
        powi_f64_impl(x, n)
    }
}

/// Portable squaring loop for `f32` — bit-identical to `compiler-builtins`.
#[cfg(any(not(feature = "std"), test))]
#[allow(dead_code)] // only reachable on no_std builds; clippy --all-features sees std
#[inline]
fn powi_f32(x: f32, n: i32) -> f32 {
    let mut base = x;
    let recip = n < 0;
    let mut e = n.unsigned_abs();
    let mut acc = 1.0_f32;
    loop {
        if (e & 1) != 0 {
            acc *= base;
        }
        e >>= 1;
        if e == 0 {
            break;
        }
        base *= base;
    }
    if recip { 1.0 / acc } else { acc }
}

/// Portable squaring loop for `f64` — bit-identical to `compiler-builtins`.
#[cfg(any(not(feature = "std"), test))]
#[allow(dead_code)] // only reachable on no_std builds; clippy --all-features sees std
#[inline]
fn powi_f64_impl(x: f64, n: i32) -> f64 {
    let mut base = x;
    let recip = n < 0;
    let mut e = n.unsigned_abs();
    let mut acc = 1.0_f64;
    loop {
        if (e & 1) != 0 {
            acc *= base;
        }
        e >>= 1;
        if e == 0 {
            break;
        }
        base *= base;
    }
    if recip { 1.0 / acc } else { acc }
}

#[cfg(test)]
mod tests {
    use super::{powi_f32, powi_f64_impl};

    #[test]
    #[allow(clippy::float_cmp)] // exact: bit-identical to std by construction
    fn powi_f32_matches_std_full_range() {
        // Exponents across the whole i32 range, incl. MIN/MAX and 0.
        for n in [0_i32, 1, 2, 3, 7, 8, 31, -1, -7, i32::MAX, i32::MIN] {
            for x in [
                0.0_f32,
                -0.0,
                1.0,
                -1.0,
                2.0,
                -2.0,
                0.5,
                1.5,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ] {
                assert_eq!(
                    powi_f32(x, n).to_bits(),
                    x.powi(n).to_bits(),
                    "powi({x}, {n}): portable={}, std={}",
                    powi_f32(x, n),
                    x.powi(n)
                );
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact: bit-identical to std by construction
    fn powi_f64_matches_std_full_range() {
        for n in [0_i32, 1, 2, 3, 7, 8, 31, -1, -7, i32::MAX, i32::MIN] {
            for x in [
                0.0_f64,
                -0.0,
                1.0,
                -1.0,
                2.0,
                -2.0,
                0.5,
                1.5,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                assert_eq!(
                    powi_f64_impl(x, n).to_bits(),
                    x.powi(n).to_bits(),
                    "powi({x}, {n}): portable={}, std={}",
                    powi_f64_impl(x, n),
                    x.powi(n)
                );
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact: 1.0, 1024.0, 0.25 are exact results
    fn powi_special_values() {
        assert_eq!(powi_f32(f32::NAN, 0), 1.0);
        assert!(powi_f32(f32::NAN, 3).is_nan());
        assert_eq!(powi_f32(f32::INFINITY, 0), 1.0);
        assert_eq!(powi_f32(0.0, 0), 1.0);
        assert_eq!(powi_f32(2.0, 10), 1024.0);
        assert_eq!(powi_f32(2.0, -2), 0.25);
        assert_eq!(powi_f64_impl(2.0, 10), 1024.0);
        assert_eq!(powi_f64_impl(2.0, -2), 0.25);
    }
}
