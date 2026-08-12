//! `ReLU` kernel implementations.
//!
//! `relu(x) = max(x, 0)` elementwise. The simplest activation — no exp, no
//! transcendentals; a pure clamp. SIMD is a single `max` against zero per
//! lane. Empty input → empty output; NaN propagates (max of NaN is NaN in
//! the hardware/SIMD semantics used here — scalar uses `f32::max` which is
//! NaN-agnostic, matching the min/max kernel contract).
//!
//! Safety: every SIMD kernel is an `unsafe fn` gated by `#[target_feature]`;
//! the dispatch layer verifies the CPU feature before calling.

/// Scalar `ReLU` reference. Writes into `out` (same length as `values`).
#[inline]
pub(crate) fn relu(values: &[f32], out: &mut [f32]) {
    for (i, &v) in values.iter().enumerate() {
        out[i] = v.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // exact: relu is a pure clamp
    fn relu_known_values() {
        let v = [-3.0_f32, -0.5, 0.0, 1.0, 5.0];
        let mut out = [0.0_f32; 5];
        relu(&v, &mut out);
        // Exact: relu is a pure clamp, so equality is exact.
        for (got, want) in out.iter().zip([0.0_f32, 0.0, 0.0, 1.0, 5.0]) {
            assert_eq!(got, &want);
        }
    }

    #[test]
    fn relu_empty() {
        let mut out = [1.0_f32; 2];
        relu(&[], &mut out[..0]);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact: relu maps -0.0 to +0.0
    fn relu_negative_zero() {
        let mut out = [0.0_f32; 1];
        relu(&[-0.0], &mut out);
        assert_eq!(out[0], 0.0);
    }
}
