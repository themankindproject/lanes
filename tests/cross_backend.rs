//! Cross-backend validation tests.
//!
//! These tests verify that the dispatched (potentially SIMD-accelerated)
//! results match the scalar reference implementation for known test vectors
//! where the answers are exact (integers representable as f32).

use lanes::stats::f32::{dot, max, min, prod, sum};

/// Helper: naive scalar sum for reference.
fn naive_sum(values: &[f32]) -> f32 {
    values.iter().sum()
}

/// Helper: naive scalar product for reference.
fn naive_prod(values: &[f32]) -> f32 {
    values.iter().product()
}

/// Helper: naive scalar min for reference.
fn naive_min(values: &[f32]) -> Option<f32> {
    values
        .iter()
        .copied()
        .reduce(|a, b| if a <= b { a } else { b })
}

/// Helper: naive scalar max for reference.
fn naive_max(values: &[f32]) -> Option<f32> {
    values
        .iter()
        .copied()
        .reduce(|a, b| if a >= b { a } else { b })
}

/// Helper: naive scalar dot for reference.
fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Helper: naive bit-level hamming reference.
fn naive_hamming(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones() as usize)
        .sum()
}

/// Helper: naive jaccard similarity reference.
fn naive_jaccard(a: &[u8], b: &[u8]) -> Option<f32> {
    let mut inter = 0usize;
    let mut union = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        inter += (x & y).count_ones() as usize;
        union += (x | y).count_ones() as usize;
    }
    (union != 0).then(|| inter as f32 / union as f32)
}

/// Deterministic pseudo-random bytes (LCG) so the test is reproducible.
fn lcg_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect()
}

/// Ascending sequence: 1, 2, 3, ..., N
fn ascending(n: usize) -> Vec<f32> {
    (1..=n).map(|x| x as f32).collect()
}

/// Descending sequence: N, N-1, ..., 1
fn descending(n: usize) -> Vec<f32> {
    (1..=n).rev().map(|x| x as f32).collect()
}

#[test]
fn cross_sum_ascending_64() {
    let data = ascending(64);
    assert_eq!(sum(&data), naive_sum(&data));
}

#[test]
fn cross_sum_ascending_256() {
    let data = ascending(256);
    assert_eq!(sum(&data), naive_sum(&data));
}

#[test]
fn cross_sum_ascending_1024() {
    let data = ascending(1024);
    assert_eq!(sum(&data), naive_sum(&data));
}

#[test]
fn cross_sum_all_zeros() {
    let data = vec![0.0_f32; 512];
    assert_eq!(sum(&data), 0.0);
    assert_eq!(sum(&data), naive_sum(&data));
}

#[test]
fn cross_sum_all_ones() {
    let data = vec![1.0_f32; 1000];
    assert_eq!(sum(&data), 1000.0);
    assert_eq!(sum(&data), naive_sum(&data));
}

#[test]
fn cross_sum_alternating_positive_negative() {
    // +1, -1, +1, -1, ... (even length = 0)
    let data: Vec<f32> = (0..512)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    assert_eq!(sum(&data), 0.0);
    assert_eq!(sum(&data), naive_sum(&data));
}

#[test]
fn cross_prod_all_ones() {
    let data = vec![1.0_f32; 512];
    assert_eq!(prod(&data), 1.0);
    assert_eq!(prod(&data), naive_prod(&data));
}

#[test]
fn cross_prod_ascending_64() {
    let data = ascending(64);
    assert_eq!(prod(&data), naive_prod(&data));
}

#[test]
fn cross_min_ascending_128() {
    let data = ascending(128);
    assert_eq!(min(&data), naive_min(&data));
}

#[test]
fn cross_min_descending_128() {
    let data = descending(128);
    assert_eq!(min(&data), naive_min(&data));
}

#[test]
fn cross_min_all_zeros() {
    let data = vec![0.0_f32; 512];
    assert_eq!(min(&data), Some(0.0));
    assert_eq!(min(&data), naive_min(&data));
}

#[test]
fn cross_min_all_ones() {
    let data = vec![1.0_f32; 512];
    assert_eq!(min(&data), Some(1.0));
    assert_eq!(min(&data), naive_min(&data));
}

#[test]
fn cross_min_alternating() {
    let data: Vec<f32> = (0..256)
        .map(|i| if i % 2 == 0 { 10.0 } else { -10.0 })
        .collect();
    assert_eq!(min(&data), Some(-10.0));
    assert_eq!(min(&data), naive_min(&data));
}

#[test]
fn cross_max_ascending_128() {
    let data = ascending(128);
    assert_eq!(max(&data), naive_max(&data));
}

#[test]
fn cross_max_descending_128() {
    let data = descending(128);
    assert_eq!(max(&data), naive_max(&data));
}

#[test]
fn cross_max_all_zeros() {
    let data = vec![0.0_f32; 512];
    assert_eq!(max(&data), Some(0.0));
    assert_eq!(max(&data), naive_max(&data));
}

#[test]
fn cross_max_all_ones() {
    let data = vec![1.0_f32; 512];
    assert_eq!(max(&data), Some(1.0));
    assert_eq!(max(&data), naive_max(&data));
}

#[test]
fn cross_max_alternating() {
    let data: Vec<f32> = (0..256)
        .map(|i| if i % 2 == 0 { 10.0 } else { -10.0 })
        .collect();
    assert_eq!(max(&data), Some(10.0));
    assert_eq!(max(&data), naive_max(&data));
}

#[test]
fn cross_dot_ascending_with_ones() {
    let a = ascending(64);
    let b = vec![1.0_f32; 64];
    assert_eq!(dot(&a, &b).unwrap(), naive_dot(&a, &b));
}

#[test]
fn cross_dot_identity() {
    // dot([1,2,3,...N], [1,2,3,...N]) = sum of squares = N*(N+1)*(2N+1)/6
    let n = 100;
    let data = ascending(n);
    let expected = (n * (n + 1) * (2 * n + 1) / 6) as f32;
    assert_eq!(dot(&data, &data).unwrap(), expected);
}

#[test]
fn cross_dot_all_zeros() {
    let a = vec![0.0_f32; 256];
    let b = vec![5.0_f32; 256];
    assert_eq!(dot(&a, &b).unwrap(), 0.0);
    assert_eq!(dot(&a, &b).unwrap(), naive_dot(&a, &b));
}

#[test]
fn cross_dot_all_ones() {
    let n = 500;
    let a = vec![1.0_f32; n];
    let b = vec![1.0_f32; n];
    assert_eq!(dot(&a, &b).unwrap(), n as f32);
    assert_eq!(dot(&a, &b).unwrap(), naive_dot(&a, &b));
}

#[test]
fn cross_dot_alternating_signs() {
    // a = [1, -1, 1, -1, ...], b = [1, 1, 1, 1, ...]
    // dot = 1 - 1 + 1 - 1 + ... = 0 (even count)
    let n = 256;
    let a: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let b = vec![1.0_f32; n];
    assert_eq!(dot(&a, &b).unwrap(), 0.0);
    assert_eq!(dot(&a, &b).unwrap(), naive_dot(&a, &b));
}

#[test]
fn cross_backend_odd_sizes() {
    // Test sizes that are NOT multiples of common SIMD widths (8 for AVX2, 16 for AVX-512)
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
        let data = ascending(n);
        assert_eq!(sum(&data), naive_sum(&data), "sum mismatch for size {n}");
        assert_eq!(min(&data), naive_min(&data), "min mismatch for size {n}");
        assert_eq!(max(&data), naive_max(&data), "max mismatch for size {n}");

        let ones = vec![1.0_f32; n];
        assert_eq!(
            dot(&data, &ones).unwrap(),
            naive_dot(&data, &ones),
            "dot mismatch for size {n}"
        );
    }
}

// ===========================================================================
// f64 cross-backend coverage: dispatched SIMD vs the scalar reference on
// exact-value inputs, including the NaN and tie-breaking contracts that
// have historically been backend-specific bugs (argmax/argmin).
// ===========================================================================

/// Naive f64 scalar references (exact-value inputs only).
fn naive_sum_f64(values: &[f64]) -> f64 {
    values.iter().sum()
}
fn naive_min_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::min)
}
fn naive_max_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::max)
}
fn naive_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn naive_argmax_f64(values: &[f64]) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let all_nan = values.iter().all(|x| x.is_nan());
    if all_nan {
        return Some(0);
    }
    Some(
        values
            .iter()
            .enumerate()
            .filter(|(_, x)| !x.is_nan())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap(),
    )
}
fn naive_argmin_f64(values: &[f64]) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let all_nan = values.iter().all(|x| x.is_nan());
    if all_nan {
        return Some(0);
    }
    Some(
        values
            .iter()
            .enumerate()
            .filter(|(_, x)| !x.is_nan())
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap(),
    )
}

#[test]
fn cross_f64_reductions_exact_values() {
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        assert_eq!(
            lanes::stats::f64::sum(&data),
            naive_sum_f64(&data),
            "sum mismatch for size {n}"
        );
        assert_eq!(
            lanes::stats::f64::min(&data),
            naive_min_f64(&data),
            "min mismatch for size {n}"
        );
        assert_eq!(
            lanes::stats::f64::max(&data),
            naive_max_f64(&data),
            "max mismatch for size {n}"
        );
        assert_eq!(
            lanes::stats::f64::argmax(&data),
            naive_argmax_f64(&data),
            "argmax mismatch for size {n}"
        );
        assert_eq!(
            lanes::stats::f64::argmin(&data),
            naive_argmin_f64(&data),
            "argmin mismatch for size {n}"
        );
        let ones = vec![1.0_f64; n];
        assert_eq!(
            lanes::stats::f64::dot(&data, &ones).unwrap(),
            naive_dot_f64(&data, &ones),
            "dot mismatch for size {n}"
        );
    }
}

#[test]
fn cross_f64_argmax_argmin_nan_and_ties() {
    // NaN-dethrone: a NaN seed must never win (the historical NEON bug).
    assert_eq!(
        lanes::stats::f64::argmax(&[f64::NAN, 1.0, f64::NAN]),
        Some(1)
    );
    assert_eq!(
        lanes::stats::f64::argmax(&[
            5.0,
            f64::NAN,
            3.0,
            f64::NAN,
            8.0,
            f64::NAN,
            1.0,
            f64::NAN,
            4.0
        ]),
        Some(4)
    );
    assert_eq!(
        lanes::stats::f64::argmin(&[f64::NAN, 3.0, 2.0, f64::NAN, 9.0]),
        Some(2)
    );
    // All-NaN falls back to index 0.
    assert_eq!(lanes::stats::f64::argmax(&[f64::NAN, f64::NAN]), Some(0));
    assert_eq!(lanes::stats::f64::argmin(&[f64::NAN, f64::NAN]), Some(0));
    // Tie spanning chunk boundaries: first global occurrence wins.
    let tied = [
        -1.456_816_089_375_683e144_f64,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
    ];
    assert_eq!(lanes::stats::f64::argmax(&tied), Some(1));
}

#[test]
fn cross_f64_distance_exact_values() {
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let l1: f64 = data.iter().map(|x| x.abs()).sum();
        let l2: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_eq!(
            lanes::distance::f64::l1_norm(&data),
            l1,
            "l1 mismatch for size {n}"
        );
        assert!(
            (lanes::distance::f64::l2_norm(&data) - l2).abs() < 1e-9,
            "l2 mismatch for size {n}"
        );
        assert_eq!(
            lanes::distance::f64::max_norm(&data),
            Some(n as f64 - 1.0),
            "max_norm mismatch for size {n}"
        );
    }
}

#[test]
fn cross_logsumexp_matches_naive() {
    // Backends reduce in different orders; the results may differ in the
    // last ulp, so compare against a tight tolerance instead of bitwise.
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257] {
        let data: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.5 - 2.0).collect();
        let naive = {
            let m = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = data.iter().map(|x| (x - m).exp()).sum();
            m + s.ln()
        };
        let got = lanes::ml::f32::logsumexp(&data);
        assert!(
            (got - naive).abs() < 1e-4 * naive.abs().max(1.0),
            "logsumexp mismatch for size {n}: got {got}, naive {naive}"
        );
    }
    // f64: same shape, tighter tolerance.
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
        let data: Vec<f64> = (0..n).map(|i| (i % 13) as f64 * 0.5 - 2.0).collect();
        let naive = {
            let m = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let s: f64 = data.iter().map(|x| (x - m).exp()).sum();
            m + s.ln()
        };
        let got = lanes::ml::f64::logsumexp(&data);
        assert!(
            (got - naive).abs() < 1e-9 * naive.abs().max(1.0),
            "logsumexp f64 mismatch for size {n}: got {got}, naive {naive}"
        );
    }
}

#[test]
fn cross_log_softmax_into_agrees_with_vec() {
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin() * 4.0).collect();
        let mut into = vec![0.0_f32; n];
        lanes::ml::f32::log_softmax_into(&data, &mut into).unwrap();
        let alloced = lanes::ml::f32::log_softmax(&data);
        for (a, b) in into.iter().zip(alloced.iter()) {
            assert!(
                (a - b).abs() < 1e-5 * a.abs().max(1.0),
                "log_softmax mismatch for size {n}"
            );
        }
        // exp of log-softmax must sum to 1 (softmax property).
        let sum: f32 = into.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-3, "exp-sum {sum} for size {n}");
    }
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin() * 4.0).collect();
        let mut into = vec![0.0_f64; n];
        lanes::ml::f64::log_softmax_into(&data, &mut into).unwrap();
        let alloced = lanes::ml::f64::log_softmax(&data);
        for (a, b) in into.iter().zip(alloced.iter()) {
            assert!(
                (a - b).abs() < 1e-9 * a.abs().max(1.0),
                "log_softmax f64 mismatch for size {n}"
            );
        }
        let sum: f64 = into.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "exp-sum {sum} for size {n}");
    }
}

#[test]
fn cross_layer_norm_into_agrees_with_vec() {
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos() * 3.0).collect();
        let mut into = vec![0.0_f32; n];
        lanes::ml::f32::layer_norm_into(&data, 1e-5, &mut into).unwrap();
        let alloced = lanes::ml::f32::layer_norm(&data, 1e-5);
        for (a, b) in into.iter().zip(alloced.iter()) {
            assert!(
                (a - b).abs() < 1e-5 * a.abs().max(1.0),
                "layer_norm mismatch for size {n}"
            );
        }
        // Unit variance after normalization (population variance). Skip
        // n == 1: variance of a single element is 0 by definition.
        if n > 1 {
            let mean: f32 = into.iter().sum::<f32>() / n as f32;
            let var: f32 = into.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            assert!((var - 1.0).abs() < 1e-2, "var {var} for size {n}");
        }
    }
    for n in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos() * 3.0).collect();
        let mut into = vec![0.0_f64; n];
        lanes::ml::f64::layer_norm_into(&data, 1e-10, &mut into).unwrap();
        let alloced = lanes::ml::f64::layer_norm(&data, 1e-10);
        for (a, b) in into.iter().zip(alloced.iter()) {
            assert!(
                (a - b).abs() < 1e-9 * a.abs().max(1.0),
                "layer_norm f64 mismatch for size {n}"
            );
        }
        // Unit variance after normalization (population variance). Skip
        // n == 1: variance of a single element is 0 by definition.
        if n > 1 {
            let mean: f64 = into.iter().sum::<f64>() / n as f64;
            let var: f64 = into.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
            assert!((var - 1.0).abs() < 1e-6, "var {var} for size {n}");
        }
    }
}

// ===========================================================================
// NaN parity: min/max/max_norm must match the scalar reference semantics
// on every backend (minNum/maxNum for min/max, total_cmp for max_norm),
// regardless of where the NaN sits relative to the vector chunk boundaries.
// ===========================================================================

/// Bit-exact Option comparison (NaN == NaN by bits, ±0 distinguished).
fn opt_bits_eq(a: Option<f32>, b: Option<f32>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
        (None, None) => true,
        _ => false,
    }
}

fn scalar_min(v: &[f32]) -> Option<f32> {
    v.iter().copied().reduce(f32::min)
}
fn scalar_max(v: &[f32]) -> Option<f32> {
    v.iter().copied().reduce(f32::max)
}
fn scalar_max_norm(v: &[f32]) -> Option<f32> {
    v.iter().copied().map(f32::abs).max_by(f32::total_cmp)
}

/// NaN at every position for lengths around each chunk boundary: the SIMD
/// result must equal the scalar reference (NaN ignored unless all-NaN).
#[test]
fn min_max_nan_parity_all_positions() {
    let nan = f32::NAN;
    for len in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        for pos in 0..len {
            let mut data: Vec<f32> = (0..len).map(|i| (i as f32) + 1.0).collect();
            data[pos] = nan;
            assert!(
                opt_bits_eq(min(&data), scalar_min(&data)),
                "min NaN parity broken: len {len}, nan at {pos}"
            );
            assert!(
                opt_bits_eq(max(&data), scalar_max(&data)),
                "max NaN parity broken: len {len}, nan at {pos}"
            );
            assert!(
                opt_bits_eq(
                    lanes::distance::f32::max_norm(&data),
                    scalar_max_norm(&data)
                ),
                "max_norm NaN parity broken: len {len}, nan at {pos}"
            );
        }
    }
}

/// All-NaN inputs of every length: min/max must return NaN (not the ±inf
/// seed), max_norm must return NaN.
#[test]
fn min_max_all_nan_returns_nan() {
    let nan = f32::NAN;
    for len in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
        let data = vec![nan; len];
        assert!(
            min(&data).is_some_and(f32::is_nan),
            "min(all-NaN len {len}) must be NaN"
        );
        assert!(
            max(&data).is_some_and(f32::is_nan),
            "max(all-NaN len {len}) must be NaN"
        );
        assert!(
            lanes::distance::f32::max_norm(&data).is_some_and(f32::is_nan),
            "max_norm(all-NaN len {len}) must be NaN"
        );
    }
}

/// NaN in a full chunk followed by real values (and vice versa): the NaN
/// chunk must not poison the result for min/max, and must poison max_norm.
#[test]
fn min_max_nan_chunk_isolation() {
    let nan = f32::NAN;
    let mut data = vec![nan; 16];
    data.extend_from_slice(&[5.0, 3.0, 8.0]);
    assert_eq!(min(&data), Some(3.0));
    assert_eq!(max(&data), Some(8.0));
    assert!(lanes::distance::f32::max_norm(&data).is_some_and(f32::is_nan));

    let mut data = vec![5.0, 3.0, 8.0];
    data.extend_from_slice(&[nan; 16]);
    assert_eq!(min(&data), Some(3.0));
    assert_eq!(max(&data), Some(8.0));
    assert!(lanes::distance::f32::max_norm(&data).is_some_and(f32::is_nan));
}

/// f64 twins of the NaN-parity checks (2/4/8-lane chunks).
#[test]
fn min_max_nan_parity_f64() {
    let nan = f64::NAN;
    let s_min = |v: &[f64]| v.iter().copied().reduce(f64::min);
    let s_max = |v: &[f64]| v.iter().copied().reduce(f64::max);
    let bits_eq = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
        (None, None) => true,
        _ => false,
    };
    for len in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
        for pos in 0..len {
            let mut data: Vec<f64> = (0..len).map(|i| (i as f64) + 1.0).collect();
            data[pos] = nan;
            assert!(
                bits_eq(lanes::stats::f64::min(&data), s_min(&data)),
                "f64 min NaN parity broken: len {len}, nan at {pos}"
            );
            assert!(
                bits_eq(lanes::stats::f64::max(&data), s_max(&data)),
                "f64 max NaN parity broken: len {len}, nan at {pos}"
            );
        }
        let all_nan = vec![nan; len];
        assert!(
            lanes::stats::f64::min(&all_nan).is_some_and(f64::is_nan),
            "f64 min(all-NaN len {len}) must be NaN"
        );
        assert!(
            lanes::stats::f64::max(&all_nan).is_some_and(f64::is_nan),
            "f64 max(all-NaN len {len}) must be NaN"
        );
    }
}

// Binary family: popcount reductions are integer-exact, so dispatched
// results must equal the naive reference exactly on every backend. The
// size list hits every chunk/tail boundary for the 16- and 32-byte
// kernels (±1 around 16 and 32, plus larger sizes).

#[test]
fn cross_binary_hamming() {
    for &n in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 255, 1024] {
        let a = lcg_bytes(0xDEAD_BEEF, n);
        let b = lcg_bytes(0xC0FF_EE00, n);
        assert_eq!(
            lanes::binary::hamming(&a, &b),
            Ok(naive_hamming(&a, &b)),
            "hamming mismatch at n={n}"
        );
    }
}

#[test]
fn cross_binary_jaccard() {
    for &n in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 255, 1024] {
        let a = lcg_bytes(0xFEED_FACE, n);
        let b = lcg_bytes(0x0BAD_F00D, n);
        assert_eq!(
            lanes::binary::jaccard(&a, &b),
            Ok(naive_jaccard(&a, &b)),
            "jaccard mismatch at n={n}"
        );
    }
    // All-zero union at every size.
    for &n in &[0usize, 1, 16, 33, 64] {
        let z = vec![0u8; n];
        assert_eq!(lanes::binary::jaccard(&z, &z), Ok(None));
    }
}

// i8 family: widening reductions are integer-exact, so dispatched results
// must equal the naive reference exactly on every backend. The size list
// hits every chunk boundary (±1 around 16 and 32) and the widening-epoch
// boundaries (dot widens every 1024 chunks, sum every 64 — a broken
// epoch flush shows up exactly at multiples of those chunk counts).

/// Helper: naive i8 dot reference.
fn naive_dot_i8(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

/// Helper: naive i8 sum reference.
fn naive_sum_i8(v: &[i8]) -> i64 {
    v.iter().map(|&x| i64::from(x)).sum()
}

/// Sizes covering chunk boundaries and widening-epoch boundaries for both
/// the 16-element (SSE2/NEON) and 32-element (AVX2) kernels.
const I8_SIZES: &[usize] = &[
    0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 255,
    // sum epochs: 64 chunks = 1024 (16-elem) / 2048 (32-elem) elements.
    1023, 1024, 1025, 2047, 2048, 2049,
    // dot epochs: 1024 chunks = 16384 / 32768 elements.
    16383, 16384, 16385, 32767, 32768, 32769,
];

#[test]
fn cross_i8_dot() {
    for &n in I8_SIZES {
        let a: Vec<i8> = lcg_bytes(0x18D0_0007, n)
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let b: Vec<i8> = lcg_bytes(0x5EED_1800, n)
            .into_iter()
            .map(|b| b as i8)
            .collect();
        assert_eq!(
            lanes::stats::i8::dot(&a, &b),
            Ok(naive_dot_i8(&a, &b)),
            "i8 dot mismatch at n={n}"
        );
    }
}

#[test]
fn cross_i8_sum() {
    for &n in I8_SIZES {
        let v: Vec<i8> = lcg_bytes(0x511CE, n).into_iter().map(|b| b as i8).collect();
        assert_eq!(
            lanes::stats::i8::sum(&v),
            naive_sum_i8(&v),
            "i8 sum mismatch at n={n}"
        );
    }
}

#[test]
fn cross_i8_sum_sq() {
    for &n in I8_SIZES {
        let v: Vec<i8> = lcg_bytes(0x511CE, n).into_iter().map(|b| b as i8).collect();
        let naive: i64 = v.iter().map(|&x| i64::from(x) * i64::from(x)).sum();
        assert_eq!(
            lanes::stats::i8::sum_sq(&v),
            naive,
            "i8 sum_sq mismatch at n={n}"
        );
    }
}

#[test]
fn cross_i8_min_max() {
    for &n in I8_SIZES {
        let v: Vec<i8> = lcg_bytes(0x1111_1111, n)
            .into_iter()
            .map(|b| b as i8)
            .collect();
        assert_eq!(
            lanes::stats::i8::min(&v),
            v.iter().copied().min(),
            "i8 min mismatch at n={n}"
        );
        assert_eq!(
            lanes::stats::i8::max(&v),
            v.iter().copied().max(),
            "i8 max mismatch at n={n}"
        );
    }
}

#[test]
fn cross_i8_count_zero() {
    for &n in I8_SIZES {
        // Mix in guaranteed zeros: every 7th byte forced to 0.
        let v: Vec<i8> = lcg_bytes(0x00AB_CDEF, n)
            .into_iter()
            .enumerate()
            .map(|(i, b)| if i % 7 == 0 { 0 } else { b as i8 })
            .collect();
        let naive = v.iter().filter(|&&x| x == 0).count();
        assert_eq!(
            lanes::stats::i8::count_zero(&v),
            naive,
            "i8 count_zero mismatch at n={n}"
        );
    }
}
