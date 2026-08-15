//! Error types for the `lanes` crate.
use core::fmt;

/// Errors that can occur during lane operations.
///
/// Every fallible kernel reports one of these variants instead of panicking,
/// so callers can branch on the exact failure mode. The `_into` family returns
/// [`Error::LengthMismatch`] when the caller-provided output buffer is the wrong
/// length; the two-input maps (`dot`, `abs_sub`, `hypot`, `squared_distance`,
/// `cosine_similarity`) return it when their operands disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Two slices that must share a length did not.
    ///
    /// `expected` is the length of the first operand (or the input slice for an
    /// `_into` call); `actual` is the length of the second operand (or the
    /// output buffer).
    LengthMismatch {
        /// The expected length (from the first operand / input).
        expected: usize,
        /// The actual length (from the second operand / output buffer).
        actual: usize,
    },
    /// An input slice was empty but the operation requires at least one
    /// element (e.g. `cosine_similarity`, which has no defined value for empty
    /// vectors, or `geometric_mean`).
    EmptyInput,
    /// `clip` received bounds with `lo > hi`, or a NaN bound.
    ///
    /// Mirrors the precondition of [`f32::clamp`]/[`f64::clamp`], which require
    /// `lo <= hi`; a NaN bound fails that comparison and is rejected too.
    InvalidBounds,
    /// `geometric_mean` saw a value `<= 0` at `index`.
    ///
    /// The geometric mean is only defined over strictly positive reals
    /// (`ln(x)` of a non-positive is `-inf`/NaN). NaN inputs are *not* reported
    /// here — they propagate to a NaN result, matching the crate's reduction
    /// semantics.
    NonPositiveInput {
        /// Index of the first offending (`<= 0`) element.
        index: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(f, "length mismatch: expected {expected}, got {actual}")
            }
            Self::EmptyInput => {
                write!(f, "empty input: operation requires at least one element")
            }
            Self::InvalidBounds => {
                write!(
                    f,
                    "invalid bounds: clip requires lo <= hi (and no NaN bounds)"
                )
            }
            Self::NonPositiveInput { index } => {
                write!(
                    f,
                    "non-positive input at index {index}: geometric_mean requires all values > 0"
                )
            }
        }
    }
}

// `core::error::Error` is stable since Rust 1.81 (MSRV is 1.89) and is the same
// trait re-exported as `std::error::Error`, so this single unconditional impl
// serves both `std` and `no_std` builds.
impl core::error::Error for Error {}
