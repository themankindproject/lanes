//! GELU kernel implementations.
//!
//! `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` —
//! the standard tanh approximation of the Gaussian-error GELU used in
//! production LLMs (GPT-2, etc.). Elementwise, one-pass map.
//!
//! `tanh` is computed from the crate's `exp`: `tanh(z) = 1 - 2/(exp(2z)+1)`,
//! which saturates correctly (exp(2z) → inf/0 gives tanh → ±1) and reuses
//! the vector `vexp_*` kernels, so no new transcendental is needed. The
//! tanh-approximation form is chosen over the exact erf form deliberately
//! (erf would need a new polynomial; the tanh form is the production
//! standard and accurate to ~1e-3 vs exact GELU).
//!
//! Range behavior: `gelu(x) ≈ x` for large positive x (tanh → 1), `≈ 0` for
//! large negative x; smooth, differentiable, no saturation cliffs. NaN
//! propagates. Empty input → empty output.
//!
//! Safety: every SIMD kernel is an `unsafe fn` gated by `#[target_feature]`;
//! the dispatch layer verifies the CPU feature before calling.

use crate::kernels::exp;

/// Constants for the tanh approximation.
const GELU_A: f32 = 0.797_884_6; // sqrt(2/pi) in f32
const GELU_B: f32 = 0.044_715;

/// Scalar GELU reference (tanh approximation). Writes into `out`.
#[inline]
pub(crate) fn gelu(values: &[f32], out: &mut [f32]) {
    for (i, &x) in values.iter().enumerate() {
        let z = GELU_A * (x + GELU_B * x * x * x);
        // tanh(z) = 1 - 2/(exp(2z)+1); saturates to ±1 via exp under/overflow.
        let tanh_z = 1.0 - 2.0 / (exp::exp(2.0 * z) + 1.0);
        out[i] = 0.5 * x * (1.0 + tanh_z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_known_values() {
        // gelu(0) = 0, gelu(1) ≈ 0.8413, gelu(-1) ≈ -0.1587 (exact GELU);
        // tanh approx: gelu(1) ≈ 0.84119, gelu(-1) ≈ -0.15881.
        let v = [0.0_f32, 1.0, -1.0];
        let mut out = [0.0_f32; 3];
        gelu(&v, &mut out);
        assert!(out[0].abs() < 1e-6, "gelu(0)={}", out[0]);
        assert!((out[1] - 0.841_19).abs() < 2e-4, "gelu(1)={}", out[1]);
        assert!((out[2] + 0.158_81).abs() < 2e-4, "gelu(-1)={}", out[2]);
    }

    #[test]
    fn gelu_saturates() {
        // Large positive → x, large negative → ~0.
        let v = [100.0_f32, -100.0, 5.0, -5.0];
        let mut out = [0.0_f32; 4];
        gelu(&v, &mut out);
        assert!((out[0] - 100.0).abs() < 1e-3, "gelu(100)={}", out[0]);
        assert!(out[1].abs() < 1e-3, "gelu(-100)={}", out[1]);
        assert!(out[2] > 4.99, "gelu(5)={}", out[2]);
        assert!(out[3].abs() < 1e-3, "gelu(-5)={}", out[3]);
    }

    #[test]
    fn gelu_empty() {
        let mut out = [1.0_f32; 2];
        gelu(&[], &mut out[..0]);
    }

    #[test]
    fn gelu_matches_exact_at_known_points() {
        // Exact GELU = x·Φ(x) (erf-based), computed in f64. The tanh approx
        // is within ~1e-3 absolute in the bulk; at the far tails (|x| > 3)
        // its relative deviation grows but the values are tiny (< 0.005).
        // Tolerances reflect that.
        let points = [
            (-3.0_f32, -0.004_05),
            (-1.0, -0.158_66),
            (0.0, 0.0),
            (1.0, 0.841_34),
            (3.0, 2.995_95),
        ];
        for (x, exact) in points {
            let mut o = [0.0_f32; 1];
            gelu(&[x], &mut o);
            assert!(
                (o[0] - exact).abs() < 5e-3,
                "x={x}: gelu={} vs exact {exact}",
                o[0]
            );
        }
    }
}
