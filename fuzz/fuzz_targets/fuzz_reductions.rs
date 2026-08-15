//! Fuzz target for the reduction family (`sum`, `prod`, `min`, `max`,
//! `sum_sq`, `mean`, `l1_norm`, `l2_norm`, `max_norm`).
//!
//! Verifies that no reduction panics regardless of input (including NaN,
//! inf, denormals, and arbitrary lengths), and that empty inputs return the
//! documented identity/None.
//!
//! Run with: `cargo +nightly fuzz run fuzz_reductions`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct ReductionsInput {
    values: Vec<f32>,
}

fuzz_target!(|input: ReductionsInput| {
    let v = &input.values;

    // None of these may panic on any input.
    let _ = lanes::stats::f32::sum(v);
    let _ = lanes::stats::f32::prod(v);
    let _ = lanes::stats::f32::min(v);
    let _ = lanes::stats::f32::max(v);
    let _ = lanes::stats::f32::sum_sq(v);
    let _ = lanes::stats::f32::mean(v);
    let _ = lanes::distance::f32::l1_norm(v);
    let _ = lanes::distance::f32::l2_norm(v);
    let _ = lanes::distance::f32::max_norm(v);

    // Empty-input contracts.
    if v.is_empty() {
        assert_eq!(lanes::stats::f32::sum(v), 0.0);
        assert_eq!(lanes::stats::f32::prod(v), 1.0);
        assert_eq!(lanes::stats::f32::min(v), None);
        assert_eq!(lanes::stats::f32::max(v), None);
        assert_eq!(lanes::stats::f32::sum_sq(v), 0.0);
        assert_eq!(lanes::stats::f32::mean(v), None);
        assert_eq!(lanes::distance::f32::l1_norm(v), 0.0);
        assert_eq!(lanes::distance::f32::max_norm(v), None);
    }

    // Non-empty min/max must be Some.
    if !v.is_empty() {
        assert!(lanes::stats::f32::min(v).is_some());
        assert!(lanes::stats::f32::max(v).is_some());
        assert!(lanes::stats::f32::mean(v).is_some());
        assert!(lanes::distance::f32::max_norm(v).is_some());
    }

    // min <= max when both are finite (NaN semantics differ per backend and
    // are documented — don't assert on NaN presence).
    if let (Some(lo), Some(hi)) = (lanes::stats::f32::min(v), lanes::stats::f32::max(v)) {
        if lo.is_finite() && hi.is_finite() {
            assert!(lo <= hi, "min {lo} > max {hi}");
        }
    }

    // count_*: never panic; counts are <= len, and NaN/infinite predicates
    // are disjoint.
    let cz = lanes::stats::f32::count_zero(v);
    let cn = lanes::stats::f32::count_nan(v);
    let ci = lanes::stats::f32::count_infinite(v);
    assert!(cz <= v.len() && cn <= v.len() && ci <= v.len());
    assert!(cn + ci <= v.len());
    if v.is_empty() {
        assert_eq!(cz, 0);
        assert_eq!(cn, 0);
        assert_eq!(ci, 0);
    }
});
