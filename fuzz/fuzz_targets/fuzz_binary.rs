//! Fuzz target for `binary::hamming` / `binary::jaccard`.
//!
//! Verifies that neither function panics on arbitrary input (any bytes,
//! any lengths) and that the length-mismatch, empty-union, bound, and
//! symmetry contracts hold exactly.
//!
//! Run with: `cargo +nightly fuzz run fuzz_binary`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct BinaryInput {
    a: Vec<u8>,
    b: Vec<u8>,
}

fuzz_target!(|input: BinaryInput| {
    let a = &input.a;
    let b = &input.b;

    // hamming: Ok iff lengths match; result bounded by 8 * len.
    match lanes::binary::hamming(a, b) {
        Ok(d) => {
            assert_eq!(a.len(), b.len(), "hamming ok on mismatched lengths");
            assert!(d <= 8 * a.len(), "hamming out of bounds");
        }
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, a.len());
            assert_eq!(actual, b.len());
            assert_ne!(a.len(), b.len());
        }
        Err(_) => unreachable!("hamming only returns LengthMismatch"),
    }

    // jaccard: same length contract; result in [0, 1] or None.
    match lanes::binary::jaccard(a, b) {
        Ok(j) => {
            assert_eq!(a.len(), b.len(), "jaccard ok on mismatched lengths");
            if let Some(v) = j {
                assert!((0.0..=1.0).contains(&v), "jaccard out of range: {v}");
            }
        }
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, a.len());
            assert_eq!(actual, b.len());
            assert_ne!(a.len(), b.len());
        }
        Err(_) => unreachable!("jaccard only returns LengthMismatch"),
    }

    // Empty-input contracts.
    if a.is_empty() && b.is_empty() {
        assert_eq!(lanes::binary::hamming(a, b), Ok(0));
        assert_eq!(lanes::binary::jaccard(a, b), Ok(None));
    }

    // Symmetry on equal-length input.
    if a.len() == b.len() {
        assert_eq!(lanes::binary::hamming(a, b), lanes::binary::hamming(b, a));
        assert_eq!(lanes::binary::jaccard(a, b), lanes::binary::jaccard(b, a));
    }
});
