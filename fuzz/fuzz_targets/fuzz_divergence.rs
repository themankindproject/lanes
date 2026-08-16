//! Fuzz target for `kl_divergence` / `js_divergence` (f32 and f64).
//!
//! Verifies that none of the divergence functions panic on arbitrary input
//! (NaN, inf, denormals, zeros, negative values, arbitrary lengths) and
//! that the length-mismatch contract holds exactly.
//!
//! Run with: `cargo +nightly fuzz run fuzz_divergence`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct DivergenceInput {
    p: Vec<f32>,
    q: Vec<f32>,
}

fuzz_target!(|input: DivergenceInput| {
    let p32 = &input.p;
    let q32 = &input.q;
    let p64: Vec<f64> = p32.iter().map(|&x| f64::from(x)).collect();
    let q64: Vec<f64> = q32.iter().map(|&x| f64::from(x)).collect();

    // f32 KL: Ok iff lengths match; any f32 result (finite/inf/NaN) is fine.
    match lanes::distance::f32::kl_divergence(p32, q32) {
        Ok(_) => assert_eq!(p32.len(), q32.len(), "kl f32 ok on mismatched lengths"),
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, p32.len());
            assert_eq!(actual, q32.len());
            assert_ne!(p32.len(), q32.len());
        }
        Err(_) => unreachable!("kl_divergence only returns LengthMismatch"),
    }

    // f32 JS: same contract.
    match lanes::distance::f32::js_divergence(p32, q32) {
        Ok(_) => assert_eq!(p32.len(), q32.len(), "js f32 ok on mismatched lengths"),
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, p32.len());
            assert_eq!(actual, q32.len());
            assert_ne!(p32.len(), q32.len());
        }
        Err(_) => unreachable!("js_divergence only returns LengthMismatch"),
    }

    // f64 twins (same lengths as the f32 slices by construction).
    match lanes::distance::f64::kl_divergence(&p64, &q64) {
        Ok(_) => assert_eq!(p64.len(), q64.len()),
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, p64.len());
            assert_eq!(actual, q64.len());
        }
        Err(_) => unreachable!("kl_divergence f64 only returns LengthMismatch"),
    }
    match lanes::distance::f64::js_divergence(&p64, &q64) {
        Ok(_) => assert_eq!(p64.len(), q64.len()),
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, p64.len());
            assert_eq!(actual, q64.len());
        }
        Err(_) => unreachable!("js_divergence f64 only returns LengthMismatch"),
    }

    // Empty-input contract: both empty -> Ok(0.0).
    if p32.is_empty() && q32.is_empty() {
        assert_eq!(lanes::distance::f32::kl_divergence(p32, q32), Ok(0.0));
        assert_eq!(lanes::distance::f32::js_divergence(p32, q32), Ok(0.0));
        assert_eq!(lanes::distance::f64::kl_divergence(&p64, &q64), Ok(0.0));
        assert_eq!(lanes::distance::f64::js_divergence(&p64, &q64), Ok(0.0));
    }

    // JS symmetry must hold for any equal-length input where both results
    // are comparable (skip NaN/inf results — NaN != NaN by definition and
    // inf - inf is NaN).
    if p32.len() == q32.len() {
        if let (Ok(a), Ok(b)) = (
            lanes::distance::f32::js_divergence(p32, q32),
            lanes::distance::f32::js_divergence(q32, p32),
        ) {
            if a.is_finite() && b.is_finite() {
                let d = (a - b).abs();
                let tol = a.abs().max(b.abs()) * 1e-4 + 1e-6;
                assert!(d <= tol, "js asymmetry: {a} vs {b} (d={d}, tol={tol})");
            }
        }
    }
});
