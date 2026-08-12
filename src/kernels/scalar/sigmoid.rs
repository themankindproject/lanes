//! Sigmoid kernel implementations.
//!
//! Sigmoid is a one-pass elementwise map: `sigmoid(x) = 1 / (1 + exp(-x))`.
//! It is not a reduction, so it does not fit the `simd_reduce!` skeleton;
//! each backend implements the map over its vector width, reusing its
//! vector `exp` kernel (`vexp_*`).
//!
//! Range behavior (matches IEEE exp semantics): large positive x saturate to
//! 1.0 (exp(-x) → 0), large negative x saturate to 0.0 (exp(-x) → inf, and
//! inf/inf collapses via 1/(1+inf) = 0 in the scalar tail; the vector path
//! uses the same division so both agree). NaN propagates.
//!
//! Safety: like the other kernels, every SIMD kernel here is an `unsafe fn`
//! gated by `#[target_feature]`; the dispatch layer verifies the CPU feature
//! before calling. Uses the crate's `no_std` exp, so this module is fully
//! `no_std`-clean.

use crate::kernels::exp;

/// Scalar sigmoid reference. Writes into `out` (same length as `values`).
#[inline]
pub(crate) fn sigmoid(values: &[f32], out: &mut [f32]) {
    for (i, &v) in values.iter().enumerate() {
        out[i] = 1.0 / (1.0 + exp::exp(-v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_known_values() {
        let v = [0.0_f32, 1.0, -1.0];
        let mut out = [0.0_f32; 3];
        sigmoid(&v, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6, "sigmoid(0)={}", out[0]);
        assert!((out[1] - 0.731_058_6).abs() < 1e-6, "sigmoid(1)={}", out[1]);
        assert!(
            (out[2] - 0.268_941_4).abs() < 1e-6,
            "sigmoid(-1)={}",
            out[2]
        );
    }

    #[test]
    fn sigmoid_saturates() {
        // Large positive → 1.0, large negative → 0.0, monotone.
        let v = [100.0_f32, -100.0, 0.0, 10.0, -10.0];
        let mut out = [0.0_f32; 5];
        sigmoid(&v, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6, "sigmoid(100)={}", out[0]);
        assert!(out[1].abs() < 1e-6, "sigmoid(-100)={}", out[1]);
        assert!(out[3] > out[4], "sigmoid must be monotone");
        assert!((out[2] - 0.5).abs() < 1e-6, "sigmoid(0)={}", out[2]);
    }

    #[test]
    fn sigmoid_empty() {
        let mut out = [1.0_f32; 2];
        sigmoid(&[], &mut out[..0]);
    }

    #[test]
    fn sigmoid_identity() {
        // sigmoid(x) + sigmoid(-x) == 1 (symmetry).
        for x in [-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
            let mut o = [0.0_f32; 2];
            sigmoid(&[x, -x], &mut o);
            assert!(
                (o[0] + o[1] - 1.0).abs() < 1e-5,
                "x={x}: {} + {} != 1",
                o[0],
                o[1]
            );
        }
    }
}
