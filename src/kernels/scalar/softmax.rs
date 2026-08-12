//! Softmax kernel implementations.
//!
//! Softmax is a three-pass map: find the max, compute `exp(x - max)` and its
//! sum, then scale each lane by `1/sum`. It is not a pure reduction, so it
//! does not fit the `simd_reduce!` skeleton; each backend implements the
//! three passes over its vector width.
//!
//! Numerically-stable form: `softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
//! The max subtraction prevents overflow for large inputs. Empty input
//! returns an empty output. NaN inputs propagate (exp of NaN is NaN).
//!
//! Safety: like the reduction kernels, every SIMD kernel here is an
//! `unsafe fn` gated by `#[target_feature]`; the dispatch layer verifies the
//! CPU feature before calling.
//!
//! Uses the crate's own `no_std` [`crate::kernels::exp`] instead of `f32::exp`,
//! so this module is fully ``no_std``-clean.

use crate::kernels::exp;

/// Scalar softmax reference. Writes into `out` (same length as `values`).
#[inline]
pub(crate) fn softmax(values: &[f32], out: &mut [f32]) {
    let Some(max) = values.iter().copied().reduce(f32::max) else {
        return;
    };
    let mut sum = 0.0;
    for (i, &v) in values.iter().enumerate() {
        let e = exp::exp(v - max);
        out[i] = e;
        sum += e;
    }
    if sum != 0.0 {
        let inv = 1.0 / sum;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_empty() {
        let mut out = [1.0_f32; 2];
        softmax(&[], &mut out[..0]);
        // out unchanged (empty output)
    }

    #[test]
    fn softmax_sums_to_one() {
        let v = [1.0_f32, 2.0, 3.0];
        let mut out = [0.0_f32; 3];
        softmax(&v, &mut out);
        let s: f32 = out.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum={s}");
    }

    #[test]
    fn softmax_single() {
        let mut out = [0.0_f32];
        softmax(&[7.0], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_shift_invariance() {
        // Adding a constant to all inputs must not change the output.
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [11.0_f32, 12.0, 13.0, 14.0];
        let mut oa = [0.0_f32; 4];
        let mut ob = [0.0_f32; 4];
        softmax(&a, &mut oa);
        softmax(&b, &mut ob);
        for i in 0..4 {
            assert!((oa[i] - ob[i]).abs() < 1e-6, "lane {i}");
        }
    }
}
