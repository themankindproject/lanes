//! Fuzz target for `stats::i8::dot` / `stats::i8::sum`.
//!
//! Verifies that neither function panics on arbitrary input (any bytes,
//! any lengths), that the length-mismatch contract holds, and that
//! results match a naive i64 oracle exactly (integer kernels are exact).
//!
//! Run with: `cargo +nightly fuzz run fuzz_i8`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct I8Input {
    a: Vec<i8>,
    b: Vec<i8>,
}

fuzz_target!(|input: I8Input| {
    let a = &input.a;
    let b = &input.b;

    // dot: Ok iff lengths match; exact against the naive i64 oracle.
    match lanes::stats::i8::dot(a, b) {
        Ok(d) => {
            assert_eq!(a.len(), b.len(), "dot ok on mismatched lengths");
            let naive: i64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| i64::from(x) * i64::from(y))
                .sum();
            assert_eq!(d, naive, "dot disagrees with naive oracle");
        }
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, a.len());
            assert_eq!(actual, b.len());
            assert_ne!(a.len(), b.len());
        }
        Err(_) => unreachable!("dot only returns LengthMismatch"),
    }

    // sum: infallible; exact against the naive i64 oracle.
    let s = lanes::stats::i8::sum(a);
    let naive_sum: i64 = a.iter().map(|&x| i64::from(x)).sum();
    assert_eq!(s, naive_sum, "sum disagrees with naive oracle");

    // sum_sq: infallible; exact against the naive i64 oracle.
    let ssq = lanes::stats::i8::sum_sq(a);
    let naive_sq: i64 = a.iter().map(|&x| i64::from(x) * i64::from(x)).sum();
    assert_eq!(ssq, naive_sq, "sum_sq disagrees with naive oracle");

    // min/max: None iff empty; otherwise exact against the naive oracle.
    assert_eq!(lanes::stats::i8::min(a), a.iter().copied().min());
    assert_eq!(lanes::stats::i8::max(a), a.iter().copied().max());

    // count_zero: infallible; exact against the naive oracle.
    let cz = lanes::stats::i8::count_zero(a);
    let naive_cz = a.iter().filter(|&&x| x == 0).count();
    assert_eq!(cz, naive_cz, "count_zero disagrees with naive oracle");

    // l1_norm: infallible; exact against the naive i64 oracle.
    let l1 = lanes::distance::i8::l1_norm(a);
    let naive_l1: i64 = a.iter().map(|&x| i64::from(x.unsigned_abs())).sum();
    assert_eq!(l1, naive_l1, "l1_norm disagrees with naive oracle");

    // max_norm: None iff empty; exact against the naive oracle.
    let mn = lanes::distance::i8::max_norm(a);
    let naive_mn = if a.is_empty() {
        None
    } else {
        Some(a.iter().map(|&x| x.unsigned_abs()).max().unwrap())
    };
    assert_eq!(mn, naive_mn, "max_norm disagrees with naive oracle");

    // squared_distance: length contract + exact against the naive oracle.
    if a.len() != b.len() {
        assert!(matches!(
            lanes::distance::i8::squared_distance(a, b),
            Err(lanes::Error::LengthMismatch { .. })
        ));
    } else {
        let sd = lanes::distance::i8::squared_distance(a, b).unwrap();
        let naive_sd: i64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = i64::from(x) - i64::from(y);
                d * d
            })
            .sum();
        assert_eq!(sd, naive_sd, "squared_distance disagrees with naive oracle");
    }

    // Empty-input contracts.
    if a.is_empty() {
        assert_eq!(lanes::stats::i8::sum(a), 0);
        assert_eq!(lanes::stats::i8::sum_sq(a), 0);
        assert_eq!(lanes::stats::i8::min(a), None);
        assert_eq!(lanes::stats::i8::max(a), None);
        assert_eq!(lanes::stats::i8::count_zero(a), 0);
    }
    if a.is_empty() && b.is_empty() {
        assert_eq!(lanes::stats::i8::dot(a, b), Ok(0));
    }

    // Commutativity on equal-length input.
    if a.len() == b.len() {
        assert_eq!(lanes::stats::i8::dot(a, b), lanes::stats::i8::dot(b, a));
    }
});
