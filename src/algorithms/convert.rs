//! Half-precision (f16) and brain floating-point (bf16) conversions and
//! mixed-precision dot products.
//!
//! All conversion functions operate on raw `u16` bit patterns. The caller is
//! responsible for ensuring the bit patterns are valid f16 or bf16 values;
//! invalid patterns are handled gracefully (NaN propagation, not UB).
//!
//! ## Conversions
//!
//! * `f16_to_f32` / `bf16_to_f32` — widen half → single (lossless).
//! * `f32_to_f16` / `f32_to_bf16` — narrow single → half with
//!   round-to-nearest-even.
//!
//! ## Dot products
//!
//! * `dot_f16` / `dot_bf16` — pairwise products accumulated in f32 for
//!   precision.

use crate::dispatch::Backend;
use crate::error::Error;
use crate::kernels;

/// Convert IEEE 754 binary16 (f16) values to f32 (lossless).
///
/// Each element of `input` is interpreted as an f16 bit pattern and converted
/// to its exact f32 representation. The conversion is lossless: every finite
/// f16 value is exactly representable in f32.
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `input.len() != output.len()`.
///
/// # Example
/// ```
/// let f16_one: u16 = 0x3C00; // 1.0 in f16
/// let mut out = [0.0_f32; 1];
/// lanes::convert::f16_to_f32(&[f16_one], &mut out).unwrap();
/// assert_eq!(out[0], 1.0_f32);
/// ```
pub fn f16_to_f32(input: &[u16], output: &mut [f32]) -> Result<(), Error> {
    if input.len() != output.len() {
        return Err(Error::LengthMismatch {
            expected: input.len(),
            actual: output.len(),
        });
    }
    let backend = Backend::detect();
    kernels::dispatch_f16_to_f32(backend, input, output);
    Ok(())
}

/// Convert f32 values to IEEE 754 binary16 (f16) with round-to-nearest-even.
///
/// Each element of `input` is narrowed to f16 using the IEEE 754
/// round-to-nearest-even rule. Values that overflow f16 range saturate to
/// ±Inf; values that underflow become ±0. NaN inputs produce a quiet NaN
/// output.
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `input.len() != output.len()`.
///
/// # Example
/// ```
/// let mut out = [0u16; 1];
/// lanes::convert::f32_to_f16(&[1.0_f32], &mut out).unwrap();
/// assert_eq!(out[0], 0x3C00); // 1.0 in f16
/// ```
pub fn f32_to_f16(input: &[f32], output: &mut [u16]) -> Result<(), Error> {
    if input.len() != output.len() {
        return Err(Error::LengthMismatch {
            expected: input.len(),
            actual: output.len(),
        });
    }
    let backend = Backend::detect();
    kernels::dispatch_f32_to_f16(backend, input, output);
    Ok(())
}

/// Convert bf16 (brain float 16) values to f32 (lossless).
///
/// Each element of `input` is interpreted as a bf16 bit pattern and converted
/// to its exact f32 representation by shifting left by 16 bits. The
/// conversion is lossless: bf16 is the upper 16 bits of an f32.
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `input.len() != output.len()`.
///
/// # Example
/// ```
/// let bf16_one: u16 = 0x3F80; // 1.0 in bf16
/// let mut out = [0.0_f32; 1];
/// lanes::convert::bf16_to_f32(&[bf16_one], &mut out).unwrap();
/// assert_eq!(out[0], 1.0_f32);
/// ```
pub fn bf16_to_f32(input: &[u16], output: &mut [f32]) -> Result<(), Error> {
    if input.len() != output.len() {
        return Err(Error::LengthMismatch {
            expected: input.len(),
            actual: output.len(),
        });
    }
    let backend = Backend::detect();
    kernels::dispatch_bf16_to_f32(backend, input, output);
    Ok(())
}

/// Convert f32 values to bf16 with round-to-nearest-even.
///
/// Each element of `input` is narrowed to bf16 using the round-to-nearest-even
/// rule. NaN inputs produce a quiet NaN output (with the quiet bit forced).
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `input.len() != output.len()`.
///
/// # Example
/// ```
/// let mut out = [0u16; 1];
/// lanes::convert::f32_to_bf16(&[1.0_f32], &mut out).unwrap();
/// assert_eq!(out[0], 0x3F80); // 1.0 in bf16
/// ```
pub fn f32_to_bf16(input: &[f32], output: &mut [u16]) -> Result<(), Error> {
    if input.len() != output.len() {
        return Err(Error::LengthMismatch {
            expected: input.len(),
            actual: output.len(),
        });
    }
    let backend = Backend::detect();
    kernels::dispatch_f32_to_bf16(backend, input, output);
    Ok(())
}

/// Dot product of two f16 slices, computed in f32.
///
/// Each pair of elements is converted from f16 to f32, multiplied, and
/// accumulated in f32 precision. This avoids the precision loss of
/// accumulating in f16.
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
///
/// # Example
/// ```
/// let f16_one: u16 = 0x3C00; // 1.0 in f16
/// let f16_two: u16 = 0x4000; // 2.0 in f16
/// let result = lanes::convert::dot_f16(&[f16_one; 4], &[f16_two; 4]).unwrap();
/// assert_eq!(result, 8.0_f32); // 4 × (1.0 × 2.0)
/// ```
pub fn dot_f16(a: &[u16], b: &[u16]) -> Result<f32, Error> {
    if a.len() != b.len() {
        return Err(Error::LengthMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let backend = Backend::detect();
    Ok(kernels::dispatch_dot_f16(backend, a, b))
}

/// Dot product of two bf16 slices, computed in f32.
///
/// Each pair of elements is converted from bf16 to f32, multiplied, and
/// accumulated in f32 precision. This avoids the precision loss of
/// accumulating in bf16.
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `a.len() != b.len()`.
///
/// # Example
/// ```
/// let bf16_one: u16 = 0x3F80; // 1.0 in bf16
/// let bf16_two: u16 = 0x4000; // 2.0 in bf16
/// let result = lanes::convert::dot_bf16(&[bf16_one; 4], &[bf16_two; 4]).unwrap();
/// assert_eq!(result, 8.0_f32); // 4 × (1.0 × 2.0)
/// ```
pub fn dot_bf16(a: &[u16], b: &[u16]) -> Result<f32, Error> {
    if a.len() != b.len() {
        return Err(Error::LengthMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let backend = Backend::detect();
    Ok(kernels::dispatch_dot_bf16(backend, a, b))
}
