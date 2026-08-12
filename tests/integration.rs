//! Integration tests for the public `lanes` API.
//!
//! These tests exercise the library as an external consumer would,
//! using only the public re-exports from the crate root.

use lanes::{dot, max, min, prod, sum, Backend, Error};

#[test]
fn sum_empty_returns_zero() {
    assert_eq!(sum(&[]), 0.0);
}

#[test]
fn sum_single_element() {
    assert_eq!(sum(&[42.0_f32]), 42.0);
}

#[test]
fn sum_multiple_elements() {
    let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    assert_eq!(sum(&data), 55.0);
}

#[test]
fn sum_large_array() {
    let data = vec![1.0_f32; 100_000];
    let result = sum(&data);
    // Exact for powers-of-two representable integers.
    assert_eq!(result, 100_000.0);
}

#[test]
fn sum_negative_values() {
    assert_eq!(sum(&[-1.0, -2.0, -3.0, -4.0]), -10.0);
}

#[test]
fn prod_empty_returns_one() {
    assert_eq!(prod(&[]), 1.0);
}

#[test]
fn prod_single_element() {
    assert_eq!(prod(&[7.0_f32]), 7.0);
}

#[test]
fn prod_multiple_elements() {
    let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    assert_eq!(prod(&data), 720.0);
}

#[test]
fn prod_with_zero() {
    assert_eq!(prod(&[3.0, 0.0, 5.0]), 0.0);
}

// softmax_sums_to_one: covered by kernels::exp::tests (exp contract) and the
// per-backend softmax unit tests; the remaining cases exercise the public
// `lanes::ml` API surface (empty/single/overflow), which the unit tests don't.

#[cfg(feature = "alloc")]
#[test]
fn softmax_empty_returns_empty() {
    assert!(lanes::ml::softmax(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn softmax_single_is_one() {
    let out = lanes::ml::softmax(&[7.0_f32]);
    assert!((out[0] - 1.0).abs() < 1e-6);
}

#[cfg(feature = "alloc")]
#[test]
fn softmax_large_inputs_no_overflow() {
    // exp(1000) overflows, but max-subtraction keeps it finite.
    let out = lanes::ml::softmax(&[1000.0_f32, 1000.0, 999.0]);
    let s: f32 = out.iter().sum();
    assert!((s - 1.0).abs() < 1e-5, "sum={s}");
    assert!(out.iter().all(|x| x.is_finite()));
}

#[cfg(feature = "alloc")]
#[test]
fn sigmoid_range_and_symmetry() {
    // Outputs in [0, 1] (endpoints reached only by saturation), sigmoid(0)=0.5,
    // sigmoid(x)+sigmoid(-x)=1.
    let v = [-5.0_f32, -1.0, 0.0, 1.0, 5.0, 100.0, -100.0];
    let out = lanes::ml::sigmoid(&v);
    assert_eq!(out.len(), v.len());
    assert!(out.iter().all(|&x| (0.0..=1.0).contains(&x)));
    assert!((out[2] - 0.5).abs() < 1e-6);
    assert!((out[0] + out[4] - 1.0).abs() < 1e-5, "symmetry x=5");
    assert!((out[1] + out[3] - 1.0).abs() < 1e-5, "symmetry x=1");
    assert!((out[5] - 1.0).abs() < 1e-6, "saturate +100");
    assert!(out[6] < 1e-6, "saturate -100");
}

#[cfg(feature = "alloc")]
#[test]
fn sigmoid_empty_returns_empty() {
    assert!(lanes::ml::sigmoid(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn silu_known_values_and_saturation() {
    let v = [-1.0_f32, 0.0, 1.0, 100.0, -100.0];
    let out = lanes::ml::silu(&v);
    assert_eq!(out.len(), v.len());
    assert!((out[0] + 0.268_941_4).abs() < 1e-6, "silu(-1)={}", out[0]);
    assert!(out[1].abs() < 1e-6, "silu(0)={}", out[1]);
    assert!((out[2] - 0.731_058_6).abs() < 1e-6, "silu(1)={}", out[2]);
    assert!((out[3] - 100.0).abs() < 1e-4, "silu(100)={}", out[3]);
    assert!(out[4].abs() < 1e-4, "silu(-100)={}", out[4]);
}

#[cfg(feature = "alloc")]
#[test]
fn silu_empty_returns_empty() {
    assert!(lanes::ml::silu(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn gelu_known_values_and_saturation() {
    let v = [0.0_f32, 1.0, -1.0, 100.0, -100.0];
    let out = lanes::ml::gelu(&v);
    assert_eq!(out.len(), v.len());
    assert!(out[0].abs() < 1e-6, "gelu(0)={}", out[0]);
    assert!((out[1] - 0.84119).abs() < 2e-4, "gelu(1)={}", out[1]);
    assert!((out[2] + 0.15881).abs() < 2e-4, "gelu(-1)={}", out[2]);
    assert!((out[3] - 100.0).abs() < 1e-3, "gelu(100)={}", out[3]);
    assert!(out[4].abs() < 1e-3, "gelu(-100)={}", out[4]);
}

#[cfg(feature = "alloc")]
#[test]
fn gelu_empty_returns_empty() {
    assert!(lanes::ml::gelu(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn relu_known_values_and_empty() {
    let v = [-3.0_f32, -0.5, 0.0, 1.0, 5.0];
    let out = lanes::ml::relu(&v);
    assert_eq!(out, [0.0, 0.0, 0.0, 1.0, 5.0]);
    assert!(lanes::ml::relu(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn silu_equals_x_times_sigmoid() {
    let v = [-3.0_f32, -1.5, 0.5, 2.0, 7.0];
    let silu_out = lanes::ml::silu(&v);
    let sig = lanes::ml::sigmoid(&v);
    for i in 0..v.len() {
        assert!(
            (silu_out[i] - v[i] * sig[i]).abs() < 1e-5,
            "lane {i}: silu={} x*sig={}",
            silu_out[i],
            v[i] * sig[i]
        );
    }
}

#[test]
fn sum_sq_known_and_empty() {
    assert_eq!(lanes::stats::sum_sq(&[1.0_f32, 2.0, 3.0]), 14.0);
    assert_eq!(lanes::stats::sum_sq(&[]), 0.0);
    assert_eq!(lanes::stats::sum_sq(&[-2.0, 3.0]), 13.0);
}

#[test]
fn mean_known_and_empty() {
    assert_eq!(lanes::stats::mean(&[1.0_f32, 2.0, 3.0]), Some(2.0));
    assert_eq!(lanes::stats::mean(&[] as &[f32]), None);
    assert_eq!(lanes::stats::mean(&[5.0]), Some(5.0));
}

#[cfg(feature = "alloc")]
#[test]
fn variance_known_and_empty() {
    // Population variance of [1, 2, 3] = ((1-2)² + 0 + (3-2)²)/3 = 2/3.
    let v = lanes::stats::variance(&[1.0_f32, 2.0, 3.0]).unwrap();
    assert!((v - 2.0 / 3.0).abs() < 1e-6, "variance={v}");
    assert_eq!(lanes::stats::variance(&[] as &[f32]), None);
    assert_eq!(lanes::stats::variance(&[4.0]), Some(0.0));
}

#[test]
fn norms_known_and_empty() {
    assert_eq!(lanes::distance::l1_norm(&[-3.0_f32, 4.0]), 7.0);
    assert_eq!(lanes::distance::l1_norm(&[]), 0.0);
    assert_eq!(lanes::distance::max_norm(&[-3.0_f32, 4.0, -9.0]), Some(9.0));
    assert_eq!(lanes::distance::max_norm(&[] as &[f32]), None);
    {
        let l2 = lanes::distance::l2_norm(&[3.0_f32, 4.0]);
        assert!((l2 - 5.0).abs() < 1e-6, "l2={l2}");
    }
}

#[test]
fn family_reexports_match_root() {
    // The family modules and the root re-exports must agree.
    let v = [1.0_f32, 2.0, 3.0, 4.0];
    assert_eq!(lanes::sum(&v), lanes::stats::sum(&v));
    assert_eq!(lanes::prod(&v), lanes::stats::prod(&v));
    assert_eq!(lanes::min(&v), lanes::stats::min(&v));
    assert_eq!(lanes::max(&v), lanes::stats::max(&v));
    assert_eq!(
        lanes::dot(&v, &v).unwrap(),
        lanes::stats::dot(&v, &v).unwrap()
    );
}

#[cfg(feature = "alloc")]
#[test]
fn math_sqrt_known_and_empty() {
    let v = lanes::math::sqrt(&[1.0_f32, 4.0, 9.0, 0.0]);
    assert_eq!(v.len(), 4);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - 2.0).abs() < 1e-6);
    assert!((v[2] - 3.0).abs() < 1e-6);
    assert_eq!(v[3], 0.0);
    assert!(lanes::math::sqrt(&[]).is_empty());
    // NaN and negative → NaN (IEEE).
    assert!(lanes::math::sqrt(&[-1.0_f32])[0].is_nan());
    assert!(lanes::math::sqrt(&[f32::NAN])[0].is_nan());
    // sqrt(inf) = inf
    assert_eq!(lanes::math::sqrt(&[f32::INFINITY])[0], f32::INFINITY);
}

#[cfg(feature = "alloc")]
#[test]
fn math_clip_known_and_empty() {
    let v = lanes::math::clip(&[-5.0_f32, 0.5, 3.0, 10.0], -1.0, 2.0);
    assert_eq!(v, [-1.0, 0.5, 2.0, 2.0]);
    assert!(lanes::math::clip(&[], -1.0, 1.0).is_empty());
    // NaN propagates.
    assert!(lanes::math::clip(&[f32::NAN], -1.0, 1.0)[0].is_nan());
}

#[cfg(feature = "alloc")]
#[test]
fn math_rsqrt_known_and_empty() {
    let v = lanes::math::rsqrt(&[1.0_f32, 4.0, 16.0]);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - 0.5).abs() < 1e-6);
    assert!((v[2] - 0.25).abs() < 1e-6);
    assert!(lanes::math::rsqrt(&[]).is_empty());
    // rsqrt(±0) = ±inf, rsqrt(neg) = NaN, rsqrt(inf) = 0.
    assert_eq!(lanes::math::rsqrt(&[0.0_f32])[0], f32::INFINITY);
    assert!(lanes::math::rsqrt(&[-1.0_f32])[0].is_nan());
    assert_eq!(lanes::math::rsqrt(&[f32::INFINITY])[0], 0.0);
}

#[cfg(feature = "alloc")]
#[test]
fn math_exp_known_and_empty() {
    let v = lanes::math::exp(&[0.0_f32, 1.0]);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - 2.718_281_7).abs() < 1e-5);
    assert!(lanes::math::exp(&[]).is_empty());
    // Saturation: exp(100) = inf, exp(-100) ~ 0.
    assert_eq!(lanes::math::exp(&[100.0_f32])[0], f32::INFINITY);
    assert!(lanes::math::exp(&[-100.0_f32])[0].abs() < 1e-38);
    assert!(lanes::math::exp(&[f32::NAN])[0].is_nan());
}

#[test]
fn min_empty_returns_none() {
    assert_eq!(min(&[] as &[f32]), None);
}

#[test]
fn min_single_element() {
    assert_eq!(min(&[7.0_f32]), Some(7.0));
}

#[test]
fn min_multiple_elements() {
    assert_eq!(min(&[5.0, 3.0, 8.0, 1.0, 4.0]), Some(1.0));
}

#[test]
fn min_all_same() {
    assert_eq!(min(&[3.0, 3.0, 3.0]), Some(3.0));
}

#[test]
fn min_negative_values() {
    assert_eq!(min(&[2.0, -5.0, 3.0, -1.0]), Some(-5.0));
}

#[test]
fn max_empty_returns_none() {
    assert_eq!(max(&[] as &[f32]), None);
}

#[test]
fn max_single_element() {
    assert_eq!(max(&[7.0_f32]), Some(7.0));
}

#[test]
fn max_multiple_elements() {
    assert_eq!(max(&[5.0, 3.0, 8.0, 1.0, 4.0]), Some(8.0));
}

#[test]
fn max_all_same() {
    assert_eq!(max(&[3.0, 3.0, 3.0]), Some(3.0));
}

#[test]
fn max_negative_values() {
    assert_eq!(max(&[-2.0, -5.0, -3.0, -1.0]), Some(-1.0));
}

#[test]
fn dot_empty_returns_zero() {
    assert_eq!(dot(&[], &[]).unwrap(), 0.0);
}

#[test]
fn dot_single_element() {
    assert_eq!(dot(&[3.0], &[4.0]).unwrap(), 12.0);
}

#[test]
fn dot_multiple_elements() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert_eq!(dot(&a, &b).unwrap(), 70.0);
}

#[test]
fn dot_length_mismatch_returns_error() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.0_f32, 2.0];
    let result = dot(&a, &b);
    assert_eq!(
        result,
        Err(Error::LengthMismatch {
            expected: 3,
            actual: 2,
        })
    );
}

#[test]
fn dot_orthogonal_vectors() {
    // (1, 0) · (0, 1) = 0
    assert_eq!(dot(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 0.0);
}

#[test]
fn dot_large_arrays() {
    let n = 10_000;
    let a = vec![2.0_f32; n];
    let b = vec![3.0_f32; n];
    let result = dot(&a, &b).unwrap();
    assert_eq!(result, 60_000.0);
}

#[test]
fn backend_detect_returns_valid_variant() {
    let backend = Backend::detect();
    // Verify we can match on it — compilation is the real test.
    match backend {
        Backend::Scalar => {}
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => {}
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {}
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => {}
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {}
    }
}

#[test]
fn backend_detect_is_deterministic() {
    let b1 = Backend::detect();
    let b2 = Backend::detect();
    assert_eq!(b1, b2);
}
