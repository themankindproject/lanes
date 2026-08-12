//! Fuzz target for the ML activation family (`softmax`, `sigmoid`, `silu`,
//! `gelu`, `relu`).
//!
//! Verifies none panic on arbitrary input, and the invariants: softmax
//! outputs are finite and sum to ~1 (when no input overflows exp to inf),
//! sigmoid is in [0, 1], relu is max(x, 0).
//!
//! Run with: `cargo +nightly fuzz run fuzz_ml`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct MlInput {
    values: Vec<f32>,
}

fuzz_target!(|input: MlInput| {
    let v = &input.values;

    // None may panic on any input.
    let sm = lanes::ml::f32::softmax(v);
    let sg = lanes::ml::f32::sigmoid(v);
    let sl = lanes::ml::f32::silu(v);
    let gl = lanes::ml::f32::gelu(v);
    let rl = lanes::ml::f32::relu(v);

    assert_eq!(sm.len(), v.len());
    assert_eq!(sg.len(), v.len());
    assert_eq!(sl.len(), v.len());
    assert_eq!(gl.len(), v.len());
    assert_eq!(rl.len(), v.len());

    // relu: exactly max(x, 0) (bit-exact for the clamp).
    for (x, r) in v.iter().zip(&rl) {
        assert_eq!(*r, x.max(0.0), "relu({x}) = {r}");
    }

    // sigmoid: in [0, 1] when finite.
    for (x, s) in v.iter().zip(&sg) {
        if s.is_finite() {
            assert!(*s >= 0.0 && *s <= 1.0, "sigmoid({x}) = {s}");
        }
    }

    // softmax: finite + sums to ~1 when no exp overflows (|x - max| < 88).
    if !v.is_empty() {
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_ok = v.iter().all(|x| (x - max).abs() < 80.0);
        if exp_ok {
            let sum: f64 = sm.iter().map(|&x| x as f64).sum();
            assert!((sum - 1.0).abs() < 1e-3, "softmax sum={sum}");
            assert!(sm.iter().all(|x| x.is_finite()));
        }
    }
});
