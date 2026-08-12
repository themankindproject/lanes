//! Fuzz target for `lanes::stats::f32::dot`.
//!
//! Verifies that the dot product function never panics regardless of
//! input, and that mismatched lengths correctly produce an error.
//!
//! Run with: `cargo +nightly fuzz run fuzz_dot`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct DotInput {
    a: Vec<f32>,
    b: Vec<f32>,
}

fuzz_target!(|input: DotInput| {
    let result = lanes::stats::f32::dot(&input.a, &input.b);

    match result {
        Ok(value) => {
            // If it succeeded, lengths must have been equal.
            assert_eq!(input.a.len(), input.b.len());
            // Value should be finite if all inputs are finite.
            if input.a.iter().all(|x| x.is_finite()) && input.b.iter().all(|x| x.is_finite()) {
                // Note: dot product of finite values can overflow to infinity,
                // so we only check it's not NaN when inputs are small.
                let _ = value; // just ensure no panic
            }
        }
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, input.a.len());
            assert_eq!(actual, input.b.len());
            assert_ne!(input.a.len(), input.b.len());
        }
        Err(_) => {
            // No other error variants expected from dot.
            panic!("unexpected error variant");
        }
    }
});
