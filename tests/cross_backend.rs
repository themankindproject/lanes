//! Cross-backend validation tests.
//!
//! These tests verify that the dispatched (potentially SIMD-accelerated)
//! results match the scalar reference implementation for known test vectors
//! where the answers are exact (integers representable as f32).

use lanes::{dot, max, min, prod, sum};

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
