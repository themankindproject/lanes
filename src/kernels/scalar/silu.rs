//! `SiLU` (`Swish`) kernel implementations.
//!
//! `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`, elementwise. Like sigmoid
//! it is a one-pass map; it reuses the sigmoid path with one extra multiply
//! (or equivalently `x / (1 + exp(-x))` — one divide, no separate sigmoid).
//!
//! Range behavior: saturates to `x` for large positive x (sigmoid → 1), to 0
//! for large negative x (x·0 = 0), minimum ≈ −0.278 at x ≈ −1.28. NaN
//! propagates. Empty input → empty output.
//!
//! Safety: every SIMD kernel is an `unsafe fn` gated by `#[target_feature]`;
//! the dispatch layer verifies the CPU feature before calling. Uses the
//! crate's `no_std` exp, so this module is fully `no_std`-clean.

use crate::kernels::exp;

/// Scalar `SiLU` reference. Writes into `out` (same length as `values`).
#[inline]
pub(crate) fn silu(values: &[f32], out: &mut [f32]) {
    for (i, &v) in values.iter().enumerate() {
        out[i] = v / (1.0 + exp::exp(-v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_known_values() {
        let v = [0.0_f32, 1.0, -1.0];
        let mut out = [0.0_f32; 3];
        silu(&v, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6, "silu(0)={}", out[0]);
        assert!((out[1] - 0.731_058_6).abs() < 1e-6, "silu(1)={}", out[1]);
        assert!((out[2] + 0.268_941_4).abs() < 1e-6, "silu(-1)={}", out[2]);
    }

    #[test]
    fn silu_saturates() {
        // Large positive → x, large negative → 0.
        let v = [100.0_f32, -100.0, 5.0, -5.0];
        let mut out = [0.0_f32; 4];
        silu(&v, &mut out);
        assert!((out[0] - 100.0).abs() < 1e-4, "silu(100)={}", out[0]);
        assert!(out[1].abs() < 1e-4, "silu(-100)={}", out[1]);
        assert!(out[2] > 0.0, "silu(5)>0");
        assert!(out[3] < 0.0, "silu(-5)<0");
    }

    #[test]
    fn silu_empty() {
        let mut out = [1.0_f32; 2];
        silu(&[], &mut out[..0]);
    }

    #[test]
    fn silu_matches_x_times_sigmoid() {
        // silu(x) == x * sigmoid(x) for a few values.
        for x in [-3.0_f32, -1.5, 0.0, 0.5, 2.0] {
            let mut o = [0.0_f32; 1];
            silu(&[x], &mut o);
            let expected = x / (1.0 + exp::exp(-x));
            assert!(
                (o[0] - expected).abs() < 1e-6,
                "x={x}: {} vs {expected}",
                o[0]
            );
        }
    }
}
