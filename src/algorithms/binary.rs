//! Distances over binary (bit-packed) vectors.
//!
//! Both functions interpret `&[u8]` slices as **bitmaps**: every byte holds
//! 8 bits of the vector, so a slice of `n` bytes is a binary vector of
//! `8n` dimensions. Semantics are bit-level, not byte-level —
//! `hamming(&[0b01], &[0b11])` is `1` (one differing bit), not one
//! differing byte.
//!
//! These are the first integer kernels in `lanes`; like the float families
//! they never panic on bad input — length mismatches return
//! [`Error::LengthMismatch`].
//!
//! [`Error::LengthMismatch`]: crate::Error::LengthMismatch

use crate::dispatch::Backend;
use crate::error::Error;
use crate::kernels;

/// Hamming distance between two packed bitmaps: the number of bit
/// positions where `a` and `b` differ, i.e. `popcount(a XOR b)`.
///
/// A slice of `n` bytes is treated as a binary vector of `8n` dimensions.
/// Returns `Ok(0)` for two empty slices.
///
/// # Example
/// ```
/// // 0b01 vs 0b11 differ in exactly one bit.
/// assert_eq!(lanes::binary::hamming(&[0b01], &[0b11]), Ok(1));
///
/// let a = [0b1010_1010u8];
/// let b = [0b0110_0110u8];
/// assert_eq!(lanes::binary::hamming(&a, &b), Ok(4));
/// ```
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
///
/// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
#[must_use = "the distance is only computed if you use it"]
pub fn hamming(a: &[u8], b: &[u8]) -> Result<usize, Error> {
    if a.len() != b.len() {
        return Err(Error::LengthMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let backend = Backend::detect();
    Ok(kernels::dispatch_hamming(backend, a, b))
}

/// Jaccard similarity between two packed bitmaps:
/// `|a AND b| / |a OR b|` — the ratio of shared set bits to total set
/// bits. `1.0` means identical bit sets, `0.0` means disjoint.
///
/// This is a **similarity** (higher = closer), unlike pgvector and
/// NumKong/SimSIMD, which report the Jaccard *distance*
/// (`distance = 1.0 - similarity`). When the union is empty (both bitmaps
/// all-zero, including two empty slices) the ratio is undefined and the
/// result is `Ok(None)` — the same convention `stats` uses for undefined
/// reductions.
///
/// A slice of `n` bytes is treated as a binary vector of `8n` dimensions.
///
/// # Example
/// ```
/// let a = [0b1010_1010u8];
/// let b = [0b0110_0110u8];
/// // AND has 2 set bits, OR has 6 -> similarity 2/6.
/// let j = lanes::binary::jaccard(&a, &b).unwrap().unwrap();
/// assert!((j - 1.0 / 3.0).abs() < 1e-6);
///
/// // All-zero union is undefined.
/// assert_eq!(lanes::binary::jaccard(&[0u8], &[0u8]), Ok(None));
/// ```
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
///
/// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
#[must_use = "the similarity is only computed if you use it"]
pub fn jaccard(a: &[u8], b: &[u8]) -> Result<Option<f32>, Error> {
    if a.len() != b.len() {
        return Err(Error::LengthMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let backend = Backend::detect();
    Ok(kernels::dispatch_jaccard(backend, a, b))
}
