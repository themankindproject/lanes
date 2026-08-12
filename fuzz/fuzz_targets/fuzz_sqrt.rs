//! Fuzz target for `lanes::math::f32::sqrt` (per-element map).
//!
//! Verifies sqrt never panics, never returns non-NaN for negative/NaN input
//! beyond the IEEE contract, and that `sqrt(x)²` round-trips x within a
//! reasonable relative bound for finite positive x.
//!
//! Run with: `cargo +nightly fuzz run fuzz_sqrt`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct SqrtInput {
    values: Vec<f32>,
}

fuzz_target!(|input: SqrtInput| {
    let out = lanes::math::f32::sqrt(&input.values);

    assert_eq!(out.len(), input.values.len());

    for (x, r) in input.values.iter().zip(&out) {
        if *x < 0.0 || x.is_nan() {
            assert!(r.is_nan(), "sqrt({x}) = {r} should be NaN");
        } else if x.is_infinite() {
            assert_eq!(*r, f32::INFINITY, "sqrt(inf) = {r}");
        } else {
            // sqrt(0) = 0, else round-trip within relative tolerance.
            if *x == 0.0 {
                assert_eq!(*r, 0.0);
            } else {
                let back = r * r;
                // Relative error of the round-trip; denormals get slack.
                let rel = (back - x).abs() / x.abs().max(f32::MIN_POSITIVE);
                assert!(rel < 1e-4, "sqrt({x}) = {r}, round-trip rel err {rel}");
            }
        }
    }
});
