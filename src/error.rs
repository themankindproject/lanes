//! Error types for the `lanes` crate.

use core::fmt;

/// Errors that can occur during lane operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Two input slices had different lengths when they must match.
    LengthMismatch {
        /// The expected length (from the first operand).
        expected: usize,
        /// The actual length (from the second operand).
        actual: usize,
    },
    /// An input slice was empty but the operation requires at least one
    /// element (e.g. `cosine_similarity`, which has no defined value for
    /// empty vectors).
    EmptyInput,
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
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
