//! Integration tests for the public `lanes` API.
//!
//! These tests exercise the library as an external consumer would,
//! using only the public re-exports from the crate root.

use lanes::stats::f32::{dot, max, min, prod, sum};
use lanes::{Backend, Error};

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
    assert!(lanes::ml::f32::softmax(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn softmax_single_is_one() {
    let out = lanes::ml::f32::softmax(&[7.0_f32]);
    assert!((out[0] - 1.0).abs() < 1e-6);
}

#[cfg(feature = "alloc")]
#[test]
fn softmax_large_inputs_no_overflow() {
    // exp(1000) overflows, but max-subtraction keeps it finite.
    let out = lanes::ml::f32::softmax(&[1000.0_f32, 1000.0, 999.0]);
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
    let out = lanes::ml::f32::sigmoid(&v);
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
    assert!(lanes::ml::f32::sigmoid(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn silu_known_values_and_saturation() {
    let v = [-1.0_f32, 0.0, 1.0, 100.0, -100.0];
    let out = lanes::ml::f32::silu(&v);
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
    assert!(lanes::ml::f32::silu(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn gelu_known_values_and_saturation() {
    let v = [0.0_f32, 1.0, -1.0, 100.0, -100.0];
    let out = lanes::ml::f32::gelu(&v);
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
    assert!(lanes::ml::f32::gelu(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn relu_known_values_and_empty() {
    let v = [-3.0_f32, -0.5, 0.0, 1.0, 5.0];
    let out = lanes::ml::f32::relu(&v);
    assert_eq!(out, [0.0, 0.0, 0.0, 1.0, 5.0]);
    assert!(lanes::ml::f32::relu(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn silu_equals_x_times_sigmoid() {
    let v = [-3.0_f32, -1.5, 0.5, 2.0, 7.0];
    let silu_out = lanes::ml::f32::silu(&v);
    let sig = lanes::ml::f32::sigmoid(&v);
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
    assert_eq!(lanes::stats::f32::sum_sq(&[1.0_f32, 2.0, 3.0]), 14.0);
    assert_eq!(lanes::stats::f32::sum_sq(&[]), 0.0);
    assert_eq!(lanes::stats::f32::sum_sq(&[-2.0, 3.0]), 13.0);
}

#[test]
fn mean_known_and_empty() {
    assert_eq!(lanes::stats::f32::mean(&[1.0_f32, 2.0, 3.0]), Some(2.0));
    assert_eq!(lanes::stats::f32::mean(&[] as &[f32]), None);
    assert_eq!(lanes::stats::f32::mean(&[5.0]), Some(5.0));
}

#[cfg(feature = "alloc")]
#[test]
fn variance_known_and_empty() {
    // Population variance of [1, 2, 3] = ((1-2)² + 0 + (3-2)²)/3 = 2/3.
    let v = lanes::stats::f32::variance(&[1.0_f32, 2.0, 3.0]).unwrap();
    assert!((v - 2.0 / 3.0).abs() < 1e-6, "variance={v}");
    assert_eq!(lanes::stats::f32::variance(&[] as &[f32]), None);
    assert_eq!(lanes::stats::f32::variance(&[4.0]), Some(0.0));
}

#[test]
fn norms_known_and_empty() {
    assert_eq!(lanes::distance::f32::l1_norm(&[-3.0_f32, 4.0]), 7.0);
    assert_eq!(lanes::distance::f32::l1_norm(&[]), 0.0);
    assert_eq!(
        lanes::distance::f32::max_norm(&[-3.0_f32, 4.0, -9.0]),
        Some(9.0)
    );
    assert_eq!(lanes::distance::f32::max_norm(&[] as &[f32]), None);
    {
        let l2 = lanes::distance::f32::l2_norm(&[3.0_f32, 4.0]);
        assert!((l2 - 5.0).abs() < 1e-6, "l2={l2}");
    }
}

#[test]
fn stats_dot_errors_on_length_mismatch() {
    let short = [1.0_f32, 2.0];
    let long = [1.0_f32, 2.0, 3.0];
    assert!(lanes::stats::f32::dot(&short, &long).is_err());
}

#[cfg(feature = "alloc")]
#[test]
fn math_sqrt_known_and_empty() {
    let v = lanes::math::f32::sqrt(&[1.0_f32, 4.0, 9.0, 0.0]);
    assert_eq!(v.len(), 4);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - 2.0).abs() < 1e-6);
    assert!((v[2] - 3.0).abs() < 1e-6);
    assert_eq!(v[3], 0.0);
    assert!(lanes::math::f32::sqrt(&[]).is_empty());
    // NaN and negative → NaN (IEEE).
    assert!(lanes::math::f32::sqrt(&[-1.0_f32])[0].is_nan());
    assert!(lanes::math::f32::sqrt(&[f32::NAN])[0].is_nan());
    // sqrt(inf) = inf
    assert_eq!(lanes::math::f32::sqrt(&[f32::INFINITY])[0], f32::INFINITY);
}

#[cfg(feature = "alloc")]
#[test]
fn math_clip_known_and_empty() {
    let v = lanes::math::f32::clip(&[-5.0_f32, 0.5, 3.0, 10.0], -1.0, 2.0);
    assert_eq!(v, [-1.0, 0.5, 2.0, 2.0]);
    assert!(lanes::math::f32::clip(&[], -1.0, 1.0).is_empty());
    // NaN propagates.
    assert!(lanes::math::f32::clip(&[f32::NAN], -1.0, 1.0)[0].is_nan());
}

#[cfg(feature = "alloc")]
#[test]
fn math_rsqrt_known_and_empty() {
    let v = lanes::math::f32::rsqrt(&[1.0_f32, 4.0, 16.0]);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - 0.5).abs() < 1e-6);
    assert!((v[2] - 0.25).abs() < 1e-6);
    assert!(lanes::math::f32::rsqrt(&[]).is_empty());
    // rsqrt(±0) = ±inf, rsqrt(neg) = NaN, rsqrt(inf) = 0.
    assert_eq!(lanes::math::f32::rsqrt(&[0.0_f32])[0], f32::INFINITY);
    assert!(lanes::math::f32::rsqrt(&[-1.0_f32])[0].is_nan());
    assert_eq!(lanes::math::f32::rsqrt(&[f32::INFINITY])[0], 0.0);
}

#[cfg(feature = "alloc")]
#[test]
fn math_exp_known_and_empty() {
    let v = lanes::math::f32::exp(&[0.0_f32, 1.0]);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - 2.718_281_7).abs() < 1e-5);
    assert!(lanes::math::f32::exp(&[]).is_empty());
    // Saturation: exp(100) = inf, exp(-100) ~ 0.
    assert_eq!(lanes::math::f32::exp(&[100.0_f32])[0], f32::INFINITY);
    assert!(lanes::math::f32::exp(&[-100.0_f32])[0].abs() < 1e-38);
    assert!(lanes::math::f32::exp(&[f32::NAN])[0].is_nan());
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
fn argmax_empty_returns_none() {
    assert_eq!(lanes::stats::f32::argmax(&[] as &[f32]), None);
}

#[test]
fn argmax_single_element() {
    assert_eq!(lanes::stats::f32::argmax(&[7.0_f32]), Some(0));
}

#[test]
fn argmax_multiple_elements() {
    assert_eq!(
        lanes::stats::f32::argmax(&[5.0, 3.0, 8.0, 1.0, 4.0]),
        Some(2)
    );
}

#[test]
fn argmax_first_occurrence_wins() {
    // The max 8.0 appears at index 2 and 4; first must win.
    assert_eq!(
        lanes::stats::f32::argmax(&[5.0, 3.0, 8.0, 1.0, 8.0]),
        Some(2)
    );
}

#[test]
fn argmax_negative_values() {
    assert_eq!(
        lanes::stats::f32::argmax(&[-2.0, -5.0, -3.0, -1.0]),
        Some(3)
    );
}

#[test]
fn argmax_consistent_with_max() {
    let data = [1.0_f32, -2.0, 3.5, 0.5, 9.0, 2.0];
    let i = lanes::stats::f32::argmax(&data).unwrap();
    assert_eq!(data[i], lanes::stats::f32::max(&data).unwrap());
}

#[test]
fn argmin_empty_returns_none() {
    assert_eq!(lanes::stats::f32::argmin(&[] as &[f32]), None);
}

#[test]
fn argmin_single_element() {
    assert_eq!(lanes::stats::f32::argmin(&[7.0_f32]), Some(0));
}

#[test]
fn argmin_multiple_elements() {
    assert_eq!(
        lanes::stats::f32::argmin(&[5.0, 3.0, 8.0, 1.0, 4.0]),
        Some(3)
    );
}

#[test]
fn argmin_first_occurrence_wins() {
    // The min 1.0 appears at index 3 and 5; first must win.
    assert_eq!(
        lanes::stats::f32::argmin(&[5.0, 3.0, 8.0, 1.0, 4.0, 1.0]),
        Some(3)
    );
}

#[test]
fn argmin_negative_values() {
    assert_eq!(
        lanes::stats::f32::argmin(&[-2.0, -5.0, -3.0, -1.0]),
        Some(1)
    );
}

#[test]
fn argmin_consistent_with_min() {
    let data = [1.0_f32, -2.0, 3.5, -0.5, 9.0, 2.0];
    let i = lanes::stats::f32::argmin(&data).unwrap();
    assert_eq!(data[i], lanes::stats::f32::min(&data).unwrap());
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

// ===========================================================================
// f64 family tests
// ===========================================================================

#[test]
fn f64_sum_known_and_empty() {
    assert_eq!(lanes::stats::f64::sum(&[1.0_f64, 2.0, 3.0]), 6.0);
    assert_eq!(lanes::stats::f64::sum(&[]), 0.0);
    assert_eq!(lanes::stats::f64::sum(&[-1.0, -2.0, -3.0]), -6.0);
}

#[test]
fn f64_prod_known_and_empty() {
    assert_eq!(lanes::stats::f64::prod(&[2.0_f64, 3.0, 4.0]), 24.0);
    assert_eq!(lanes::stats::f64::prod(&[]), 1.0);
    assert_eq!(lanes::stats::f64::prod(&[3.0, 0.0, 5.0]), 0.0);
}

#[test]
fn f64_min_max_known_and_empty() {
    assert_eq!(lanes::stats::f64::min(&[3.0_f64, 1.0, 4.0]), Some(1.0));
    assert_eq!(lanes::stats::f64::min(&[] as &[f64]), None);
    assert_eq!(lanes::stats::f64::max(&[3.0_f64, 1.0, 4.0]), Some(4.0));
    assert_eq!(lanes::stats::f64::max(&[] as &[f64]), None);
}

#[test]
fn f64_argmax_argmin_known() {
    assert_eq!(
        lanes::stats::f64::argmax(&[5.0_f64, 3.0, 8.0, 1.0, 8.0]),
        Some(2)
    );
    assert_eq!(
        lanes::stats::f64::argmin(&[5.0_f64, 3.0, 8.0, 1.0, 4.0, 1.0]),
        Some(3)
    );
    assert_eq!(lanes::stats::f64::argmax(&[] as &[f64]), None);
    assert_eq!(lanes::stats::f64::argmin(&[] as &[f64]), None);
}

#[test]
fn f64_sum_sq_mean() {
    assert_eq!(lanes::stats::f64::sum_sq(&[1.0_f64, 2.0, 3.0]), 14.0);
    assert_eq!(lanes::stats::f64::mean(&[1.0_f64, 2.0, 3.0]), Some(2.0));
    assert_eq!(lanes::stats::f64::mean(&[] as &[f64]), None);
}

#[cfg(feature = "alloc")]
#[test]
fn f64_variance_known() {
    let v = lanes::stats::f64::variance(&[1.0_f64, 2.0, 3.0]).unwrap();
    assert!((v - 2.0 / 3.0).abs() < 1e-12, "variance={v}");
    assert_eq!(lanes::stats::f64::variance(&[4.0]), Some(0.0));
    assert_eq!(lanes::stats::f64::variance(&[] as &[f64]), None);
}

#[test]
fn f64_dot_known_and_mismatch() {
    assert_eq!(
        lanes::stats::f64::dot(&[1.0_f64, 2.0], &[3.0_f64, 4.0]).unwrap(),
        11.0
    );
    assert!(lanes::stats::f64::dot(&[1.0_f64], &[1.0_f64, 2.0]).is_err());
    assert_eq!(lanes::stats::f64::dot(&[], &[]).unwrap(), 0.0);
}

#[test]
fn f64_norms_known_and_empty() {
    assert_eq!(lanes::distance::f64::l1_norm(&[-3.0_f64, 4.0]), 7.0);
    assert_eq!(lanes::distance::f64::l1_norm(&[]), 0.0);
    assert_eq!(
        lanes::distance::f64::max_norm(&[-3.0_f64, 4.0, -9.0]),
        Some(9.0)
    );
    assert_eq!(lanes::distance::f64::max_norm(&[] as &[f64]), None);
    let l2 = lanes::distance::f64::l2_norm(&[3.0_f64, 4.0]);
    assert!((l2 - 5.0).abs() < 1e-12, "l2={l2}");
}

#[cfg(feature = "alloc")]
#[test]
fn f64_math_sqrt_clip_rsqrt_exp() {
    let v = lanes::math::f64::sqrt(&[1.0_f64, 4.0, 9.0, 0.0]);
    assert!((v[0] - 1.0).abs() < 1e-12);
    assert!((v[1] - 2.0).abs() < 1e-12);
    assert!((v[2] - 3.0).abs() < 1e-12);
    assert_eq!(v[3], 0.0);
    assert!(lanes::math::f64::sqrt(&[-1.0_f64])[0].is_nan());

    let c = lanes::math::f64::clip(&[-5.0_f64, 0.5, 3.0, 10.0], -1.0, 2.0);
    assert_eq!(c, [-1.0, 0.5, 2.0, 2.0]);

    let r = lanes::math::f64::rsqrt(&[1.0_f64, 4.0, 16.0]);
    assert!((r[0] - 1.0).abs() < 1e-12);
    assert!((r[1] - 0.5).abs() < 1e-12);
    assert!((r[2] - 0.25).abs() < 1e-12);

    let e = lanes::math::f64::exp(&[0.0_f64, 1.0]);
    assert!((e[0] - 1.0).abs() < 1e-12);
    assert!((e[1] - std::f64::consts::E).abs() < 1e-12);
    assert_eq!(lanes::math::f64::exp(&[1000.0_f64])[0], f64::INFINITY);
    assert!(lanes::math::f64::exp(&[-1000.0_f64])[0].abs() < 1e-300);
    assert!(lanes::math::f64::exp(&[f64::NAN])[0].is_nan());
}

#[cfg(feature = "alloc")]
#[test]
fn f64_softmax_sums_to_one() {
    let out = lanes::ml::f64::softmax(&[1.0_f64, 2.0, 3.0]);
    let s: f64 = out.iter().sum();
    assert!((s - 1.0).abs() < 1e-12, "sum={s}");
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(lanes::ml::f64::softmax(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn f64_sigmoid_silu_gelu_relu() {
    let s = lanes::ml::f64::sigmoid(&[0.0_f64, 1.0, -1.0]);
    assert!((s[0] - 0.5).abs() < 1e-12, "s[0]={}", s[0]);
    assert!(
        (s[1] - 0.731_058_578_630_092_5).abs() < 1e-12,
        "s[1]={}",
        s[1]
    );
    assert!(
        (s[2] - 0.268_941_421_369_907_5).abs() < 1e-12,
        "s[2]={}",
        s[2]
    );

    let si = lanes::ml::f64::silu(&[0.0_f64, 1.0, -1.0]);
    assert!(si[0].abs() < 1e-12);
    assert!((si[1] - 0.731_058_578_630_092_5).abs() < 1e-12);
    assert!((si[2] + 0.268_941_421_369_907_5).abs() < 1e-12);

    let g = lanes::ml::f64::gelu(&[0.0_f64, 1.0, -1.0]);
    assert!(g[0].abs() < 1e-12);
    assert!((g[1] - 0.841_192_029_433_373).abs() < 2e-4);
    assert!((g[2] + 0.158_807_970_566_627).abs() < 2e-4);

    let r = lanes::ml::f64::relu(&[-3.0_f64, -0.5, 0.0, 1.0, 5.0]);
    assert_eq!(r, [0.0, 0.0, 0.0, 1.0, 5.0]);
    assert!(lanes::ml::f64::relu(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn std_dev_known_and_empty() {
    let s = lanes::stats::f32::std_dev(&[1.0_f32, 2.0, 3.0]).unwrap();
    assert!((s - (2.0_f32 / 3.0).sqrt()).abs() < 1e-6, "s={s}");
    assert_eq!(lanes::stats::f32::std_dev(&[4.0]), Some(0.0));
    assert_eq!(lanes::stats::f32::std_dev(&[] as &[f32]), None);

    let s = lanes::stats::f64::std_dev(&[1.0_f64, 2.0, 3.0]).unwrap();
    assert!((s - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12, "s={s}");
    assert_eq!(lanes::stats::f64::std_dev(&[4.0]), Some(0.0));
    assert_eq!(lanes::stats::f64::std_dev(&[] as &[f64]), None);
}

#[cfg(feature = "alloc")]
#[test]
fn tanh_known_and_saturation() {
    let t = lanes::math::f32::tanh(&[0.0_f32, 1.0, -1.0, 50.0, -50.0]);
    assert!(t[0].abs() < 1e-6, "t[0]={}", t[0]);
    assert!((t[1] - 0.761_594_2).abs() < 1e-5, "t[1]={}", t[1]);
    assert!((t[2] + 0.761_594_2).abs() < 1e-5, "t[2]={}", t[2]);
    assert!((t[3] - 1.0).abs() < 1e-6, "t[3]={}", t[3]);
    assert!((t[4] + 1.0).abs() < 1e-6, "t[4]={}", t[4]);
    assert!(lanes::math::f32::tanh(&[]).is_empty());

    let t = lanes::math::f64::tanh(&[0.0_f64, 1.0, -1.0, 100.0, -100.0]);
    assert!(t[0].abs() < 1e-12);
    assert!(
        (t[1] - 0.761_594_155_955_764_9).abs() < 1e-12,
        "t[1]={}",
        t[1]
    );
    assert!(
        (t[2] + 0.761_594_155_955_764_9).abs() < 1e-12,
        "t[2]={}",
        t[2]
    );
    assert!((t[3] - 1.0).abs() < 1e-12);
    assert!((t[4] + 1.0).abs() < 1e-12);
    assert!(lanes::math::f64::tanh(&[]).is_empty());
}

#[cfg(feature = "alloc")]
#[test]
fn rms_norm_known_and_empty() {
    // mean(x²) for [3,4] = 12.5; rms = √12.5 ≈ 3.535534.
    let v = lanes::ml::f32::rms_norm(&[3.0_f32, 4.0], 0.0);
    assert!((v[0] - 3.0 / 12.5_f32.sqrt()).abs() < 1e-5, "v[0]={}", v[0]);
    assert!((v[1] - 4.0 / 12.5_f32.sqrt()).abs() < 1e-5, "v[1]={}", v[1]);
    assert!(lanes::ml::f32::rms_norm(&[], 1e-5).is_empty());
    // eps guards the all-zero case.
    let z = lanes::ml::f32::rms_norm(&[0.0_f32; 4], 1e-4);
    assert!(z.iter().all(|x| x.abs() < 1e-6));

    let v = lanes::ml::f64::rms_norm(&[3.0_f64, 4.0], 0.0);
    assert!(
        (v[0] - 3.0 / 12.5_f64.sqrt()).abs() < 1e-12,
        "v[0]={}",
        v[0]
    );
    assert!(
        (v[1] - 4.0 / 12.5_f64.sqrt()).abs() < 1e-12,
        "v[1]={}",
        v[1]
    );
    assert!(lanes::ml::f64::rms_norm(&[], 1e-5).is_empty());
    let z = lanes::ml::f64::rms_norm(&[0.0_f64; 4], 1e-8);
    assert!(z.iter().all(|x| x.abs() < 1e-12));
}

#[test]
fn cosine_similarity_known_cases() {
    // Identical vectors → 1.0.
    assert_eq!(
        lanes::ml::f32::cosine_similarity(&[1.0_f32, 2.0], &[1.0_f32, 2.0]),
        Ok(Some(1.0))
    );
    // Orthogonal → 0.0.
    assert!(
        lanes::ml::f32::cosine_similarity(&[1.0_f32, 0.0], &[0.0_f32, 1.0])
            .unwrap()
            .unwrap()
            .abs()
            < 1e-6
    );
    // Opposite → -1.0.
    assert_eq!(
        lanes::ml::f32::cosine_similarity(&[1.0_f32], &[-1.0_f32]),
        Ok(Some(-1.0))
    );
    // Length mismatch → error.
    assert!(lanes::ml::f32::cosine_similarity(&[1.0_f32], &[1.0_f32, 2.0]).is_err());
    // Empty → None.
    assert_eq!(lanes::ml::f32::cosine_similarity(&[], &[]), Ok(None));
    // Zero vector → None.
    assert_eq!(
        lanes::ml::f32::cosine_similarity(&[0.0_f32], &[1.0_f32]),
        Ok(None)
    );

    assert!(
        (lanes::ml::f64::cosine_similarity(&[1.0_f64, 2.0], &[1.0_f64, 2.0])
            .unwrap()
            .unwrap()
            - 1.0)
            .abs()
            < 1e-12
    );
    assert!(
        lanes::ml::f64::cosine_similarity(&[1.0_f64, 0.0], &[0.0_f64, 1.0])
            .unwrap()
            .unwrap()
            .abs()
            < 1e-12
    );
    assert_eq!(
        lanes::ml::f64::cosine_similarity(&[1.0_f64], &[-1.0_f64]),
        Ok(Some(-1.0))
    );
    assert!(lanes::ml::f64::cosine_similarity(&[1.0_f64], &[1.0_f64, 2.0]).is_err());
    assert_eq!(
        lanes::ml::f64::cosine_similarity(&[0.0_f64], &[1.0_f64]),
        Ok(None)
    );
}
