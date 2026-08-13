//! Fuzz target for the transcendental maps (`exp`, `tanh`).
//!
//! Verifies that the SIMD-dispatched `exp` and `tanh` agree with the
//! standard-library reference to a documented ulp bound on arbitrary input,
//! and that saturation matches exactly.
//!
//! Run with: `cargo +nightly fuzz run fuzz_exp`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct ExpInput {
    values: Vec<f32>,
}

/// ULP distance between two floats (sign-magnitude aware).
fn ulps(a: f32, b: f32) -> u32 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX; // NaN vs non-NaN is never acceptable
    }
    let (a, b) = (a.to_bits(), b.to_bits());
    match (a & 0x8000_0000).cmp(&(b & 0x8000_0000)) {
        std::cmp::Ordering::Equal => a.abs_diff(b),
        _ => a.abs_diff(b) & 0x7FFF_FFFF,
    }
}

fuzz_target!(|input: ExpInput| {
    for &x in &input.values {
        let e = lanes::math::f32::exp(std::slice::from_ref(&x))[0];
        let want = x.exp();

        // Saturation must match exactly (inf/0); finite values ≤ 3 ulp
        // (the kernel contract is ≤ 2 ulp; one extra for the map plumbing).
        if e.is_infinite() || e == 0.0 {
            assert_eq!(e, want, "exp({x:e}): sat {e:e} vs std {want:e}");
        } else {
            assert!(
                ulps(e, want) <= 3,
                "exp({x:e}) = {e:e}, std = {want:e}, ulps = {}",
                ulps(e, want)
            );
        }

        // tanh is in [-1, 1] and derived from exp; allow ~6 ulp (the
        // 1 - 2/(e^2x+1) form loses a couple of bits near ±1).
        let t = lanes::math::f32::tanh(std::slice::from_ref(&x))[0];
        let twant = x.tanh();
        assert!(t.is_nan() == twant.is_nan(), "tanh({x:e}) NaN mismatch");
        if !t.is_nan() {
            assert!(
                ulps(t, twant) <= 6,
                "tanh({x:e}) = {t:e}, std = {twant:e}, ulps = {}",
                ulps(t, twant)
            );
            assert!(t >= -1.0 && t <= 1.0, "tanh({x:e}) = {t} out of [-1,1]");
        }

        // ln: IEEE special cases must match exactly, finite values ≤ 2 ulp
        // (the kernel contract is ≤ 1 ulp; one extra for the map plumbing).
        let l = lanes::math::f32::ln(std::slice::from_ref(&x))[0];
        let lwant = x.ln();
        assert!(
            l.is_nan() == lwant.is_nan(),
            "ln({x:e}) = {l:e}, std = {lwant:e}: NaN mismatch"
        );
        if l.is_infinite() || lwant.is_infinite() {
            assert_eq!(l, lwant, "ln({x:e}) = {l:e}, std = {lwant:e}: inf mismatch");
        } else if l.is_finite() {
            assert!(
                ulps(l, lwant) <= 2,
                "ln({x:e}) = {l:e}, std = {lwant:e}, ulps = {}",
                ulps(l, lwant)
            );
        }
    }
});
