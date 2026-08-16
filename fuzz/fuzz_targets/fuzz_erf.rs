//! Fuzz target for erf/erfc: contract properties on arbitrary input.
//!
//! No std oracle exists (`float_erf` is unstable), so this fuzzes the
//! invariants that follow from the kernel construction: IEEE specials,
//! ranges, exact odd symmetry, saturation beyond XMAX, the exact
//! complement `erf + erfc == 1` where erf is computed as `1 − erfc`
//! (x ≥ 0.84375, positive sign), and the near-1 sum identity everywhere.
//!
//! Run with: `cargo +nightly fuzz run fuzz_erf`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct ErfInput {
    values: Vec<f32>,
}

fuzz_target!(|input: ErfInput| {
    for &x in &input.values {
        let e = lanes::special::f32::erf(std::slice::from_ref(&x))[0];
        let c = lanes::special::f32::erfc(std::slice::from_ref(&x))[0];

        if x.is_nan() {
            assert!(e.is_nan() && c.is_nan(), "erf/erfc({x}): NaN propagation");
            continue;
        }

        // Ranges: erf ∈ [-1, 1], erfc ∈ [0, 2].
        assert!(e.abs() <= 1.0, "erf({x}) = {e} out of [-1,1]");
        assert!((0.0..=2.0).contains(&c), "erfc({x}) = {c} out of [0,2]");

        // Odd symmetry is bit-exact (both regions commute with rounding).
        let e_neg = lanes::special::f32::erf(&[-x])[0];
        assert_eq!(
            e_neg.to_bits(),
            (-e).to_bits(),
            "erf({x}) = {e}, erf({}) = {e_neg}: symmetry",
            -x
        );

        // Saturation beyond XMAX (27.23).
        if x > 28.0 {
            assert_eq!(e, 1.0, "erf({x}) = {e}: expected saturation");
            assert_eq!(c, 0.0, "erfc({x}) = {c}: expected saturation");
        }
        if x < -28.0 {
            assert_eq!(e, -1.0, "erf({x}) = {e}: expected saturation");
            assert_eq!(c, 2.0, "erfc({x}) = {c}: expected saturation");
        }

        // Exact complement where erf = round32(1 − erfc_f64): the two
        // roundings stay within half an ulp of 1.0 (see the proptest
        // note — do NOT assert e == 1.0 − c for f32).
        if x >= 0.84375 {
            assert_eq!(e + c, 1.0, "erf({x}) + erfc({x}) = {}", e + c);
        }

        // Sum identity everywhere, with a rounding tolerance.
        assert!(
            (e + c - 1.0).abs() <= 1e-6,
            "erf({x}) + erfc({x}) = {}",
            e + c
        );
    }
});
