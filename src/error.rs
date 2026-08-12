//! Error types for the `lanes` crate.

use core::fmt;

/// Errors that can occur during lane operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The input slice was empty when a non-empty slice was required.
    EmptyInput,
    /// Two input slices had different lengths when they must match.
    LengthMismatch {
        /// The expected length (from the first operand).
        expected: usize,
        /// The actual length (from the second operand).
        actual: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input slice must not be empty"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "length mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
