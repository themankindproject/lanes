//! Fuzz target for the elementwise/contract functions: `dot`, `sqrt`, and
//! the ML activations (`softmax`, `sigmoid`, `silu`, `gelu`, `relu`).
//!
//! Verifies none panic on arbitrary input plus the per-function invariants:
//! `dot` errors exactly on length mismatch, `sqrt` follows the IEEE contract,
//! `relu` is bit-exact `max(x, 0)`, sigmoid stays in [0, 1], and softmax
//! outputs sum to ~1 when no exp overflows.
//!
//! Run with: `cargo +nightly fuzz run fuzz_contracts`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct ContractsInput {
    a: Vec<f32>,
    b: Vec<f32>,
}

fuzz_target!(|input: ContractsInput| {
    let a = &input.a;
    let b = &input.b;

    // dot: Ok iff lengths match; Err(LengthMismatch) with exact lengths.
    match lanes::stats::f32::dot(a, b) {
        Ok(_) => assert_eq!(a.len(), b.len(), "dot succeeded on mismatched lengths"),
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, a.len());
            assert_eq!(actual, b.len());
            assert_ne!(a.len(), b.len());
        }
        Err(lanes::Error::EmptyInput) => {
            unreachable!("dot never returns EmptyInput (empty dot is 0.0)")
        }
    }

    // sqrt: IEEE contract per element.
    let sq = lanes::math::f32::sqrt(a);
    assert_eq!(sq.len(), a.len());
    for (x, r) in a.iter().zip(&sq) {
        if *x < 0.0 || x.is_nan() {
            assert!(r.is_nan(), "sqrt({x}) = {r} should be NaN");
        } else if x.is_infinite() {
            assert_eq!(*r, f32::INFINITY, "sqrt(inf) = {r}");
        } else if *x == 0.0 {
            assert_eq!(*r, 0.0);
        } else {
            let back = r * r;
            let rel = (back - x).abs() / x.abs().max(f32::MIN_POSITIVE);
            assert!(rel < 1e-4, "sqrt({x}) = {r}, round-trip rel err {rel}");
        }
    }

    // Activations: same-length outputs, then per-function invariants.
    let sm = lanes::ml::f32::softmax(a);
    let sg = lanes::ml::f32::sigmoid(a);
    let sl = lanes::ml::f32::silu(a);
    let gl = lanes::ml::f32::gelu(a);
    let rl = lanes::ml::f32::relu(a);
    let sp = lanes::ml::f32::softplus(a);
    let ls = lanes::ml::f32::log_softmax(a);
    assert_eq!(sm.len(), a.len());
    assert_eq!(sg.len(), a.len());
    assert_eq!(sl.len(), a.len());
    assert_eq!(gl.len(), a.len());
    assert_eq!(rl.len(), a.len());
    assert_eq!(sp.len(), a.len());
    assert_eq!(ls.len(), a.len());

    // _into variants must agree with the allocating wrappers.
    let mut lsm = vec![0.0_f32; a.len()];
    lanes::ml::f32::log_softmax_into(a, &mut lsm);
    for (x, y) in lsm.iter().zip(ls.iter()) {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        assert!((x - y).abs() < 1e-4 * y.abs().max(1.0), "log_softmax_into {x} vs {y}");
    }

    // relu: exactly max(x, 0) (bit-exact for the clamp).
    for (x, r) in a.iter().zip(&rl) {
        assert_eq!(*r, x.max(0.0), "relu({x}) = {r}");
    }

    // softplus: ≥ max(x, 0) always; approaches x for large |x|; finite
    // inputs never overflow to +inf (the stable form's whole point).
    for (x, s) in a.iter().zip(&sp) {
        if x.is_finite() && s.is_finite() {
            assert!(*s >= x.max(0.0) - 1e-4, "softplus({x}) = {s}");
        }
    }

    // log_softmax: exp() of the output must sum to ~1 whenever no exp
    // overflows in the underlying softmax (same guard as softmax below).
    if !a.is_empty() {
        let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_ok = a.iter().all(|x| (x - max).abs() < 80.0);
        if exp_ok && a.iter().all(|x| x.is_finite()) {
            let sum: f64 = ls.iter().map(|&x| (x as f64).exp()).sum();
            assert!((sum - 1.0).abs() < 1e-3, "log_softmax exp-sum {sum}, a={a:?}");
            assert!(ls.iter().all(|x| x.is_finite() && *x <= 0.0 + 1e-4));
        }
    }

    // sigmoid: in [0, 1] when finite.
    for (x, s) in a.iter().zip(&sg) {
        if s.is_finite() {
            assert!(*s >= 0.0 && *s <= 1.0, "sigmoid({x}) = {s}");
        }
    }

    // softmax: finite + sums to ~1 when no exp overflows (|x - max| < 80).
    if !a.is_empty() {
        let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_ok = a.iter().all(|x| (x - max).abs() < 80.0);
        if exp_ok {
            let sum: f64 = sm.iter().map(|&x| x as f64).sum();
            assert!((sum - 1.0).abs() < 1e-3, "softmax sum={sum}");
            assert!(sm.iter().all(|x| x.is_finite()));
        }
    }

    // abs_sub / hypot / powi: panic on length mismatch by design, so only
    // exercise them on equal-length inputs.
    if a.len() == b.len() {
        let asub = lanes::math::f32::abs_sub(a, b);
        assert_eq!(asub.len(), a.len());
        for ((x, y), r) in a.iter().zip(b.iter()).zip(asub.iter()) {
            if x.is_nan() || y.is_nan() {
                assert!(r.is_nan(), "abs_sub({x},{y}) = {r}");
            } else {
                assert!(*r >= 0.0, "abs_sub({x},{y}) = {r}");
            }
        }

        let h = lanes::math::f32::hypot(a, b);
        assert_eq!(h.len(), a.len());
        for (i, ((x, y), r)) in a.iter().zip(b.iter()).zip(h.iter()).enumerate() {
            if x.is_infinite() || y.is_infinite() {
                assert_eq!(*r, f32::INFINITY, "hypot({x},{y}) lane {i}");
            } else if x.is_nan() || y.is_nan() {
                assert!(r.is_nan(), "hypot({x},{y}) lane {i}");
            } else {
                assert!(r.is_finite() || r.is_infinite());
            }
        }

        let p = lanes::math::f32::powi(a, 3);
        assert_eq!(p.len(), a.len());
    }

    // squared_distance: Ok iff lengths match; Err(LengthMismatch) otherwise.
    match lanes::distance::f32::squared_distance(a, b) {
        Ok(d) => {
            assert_eq!(a.len(), b.len());
            assert!(d >= 0.0 || d.is_nan(), "squared distance must be non-negative");
        }
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, a.len());
            assert_eq!(actual, b.len());
            assert_ne!(a.len(), b.len());
        }
        Err(lanes::Error::EmptyInput) => {
            unreachable!("squared_distance never returns EmptyInput")
        }
    }
});
