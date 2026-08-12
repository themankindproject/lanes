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
    let _ = lanes::stats::sum(v);
    let _ = lanes::stats::prod(v);
    let _ = lanes::stats::min(v);
    let _ = lanes::stats::max(v);
    let _ = lanes::stats::sum_sq(v);
    let _ = lanes::stats::mean(v);
    let _ = lanes::distance::l1_norm(v);
    let _ = lanes::distance::l2_norm(v);
    let _ = lanes::distance::max_norm(v);

    // Empty-input contracts.
    if v.is_empty() {
        assert_eq!(lanes::stats::sum(v), 0.0);
        assert_eq!(lanes::stats::prod(v), 1.0);
        assert_eq!(lanes::stats::min(v), None);
        assert_eq!(lanes::stats::max(v), None);
        assert_eq!(lanes::stats::sum_sq(v), 0.0);
        assert_eq!(lanes::stats::mean(v), None);
        assert_eq!(lanes::distance::l1_norm(v), 0.0);
        assert_eq!(lanes::distance::max_norm(v), None);
    }

    // Non-empty min/max must be Some.
    if !v.is_empty() {
        assert!(lanes::stats::min(v).is_some());
        assert!(lanes::stats::max(v).is_some());
        assert!(lanes::stats::mean(v).is_some());
        assert!(lanes::distance::max_norm(v).is_some());
    }

    // min <= max when both are finite (NaN semantics differ per backend and
    // are documented — don't assert on NaN presence).
    if let (Some(lo), Some(hi)) = (lanes::stats::min(v), lanes::stats::max(v)) {
        if lo.is_finite() && hi.is_finite() {
            assert!(lo <= hi, "min {lo} > max {hi}");
        }
    }
});
