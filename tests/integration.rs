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
    let v = lanes::math::f32::clip(&[-5.0_f32, 0.5, 3.0, 10.0], -1.0, 2.0).unwrap();
    assert_eq!(v, [-1.0, 0.5, 2.0, 2.0]);
    assert!(lanes::math::f32::clip(&[], -1.0, 1.0).unwrap().is_empty());
    // NaN propagates.
    assert!(lanes::math::f32::clip(&[f32::NAN], -1.0, 1.0).unwrap()[0].is_nan());
    // Inverted bounds are rejected.
    assert_eq!(
        lanes::math::f32::clip(&[1.0_f32], 2.0, -1.0),
        Err(lanes::Error::InvalidBounds)
    );
    // NaN bounds are rejected.
    assert_eq!(
        lanes::math::f32::clip(&[1.0_f32], f32::NAN, 1.0),
        Err(lanes::Error::InvalidBounds)
    );
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
fn argmax_ignores_nan_unless_all_nan() {
    assert_eq!(
        lanes::stats::f32::argmax(&[f32::NAN, 1.0, f32::NAN]),
        Some(1)
    );
    assert_eq!(
        lanes::stats::f32::argmax(&[f32::NAN, 3.0, 2.0, f32::NAN, 9.0]),
        Some(4)
    );
    assert_eq!(
        lanes::stats::f32::argmax(&[
            5.0,
            f32::NAN,
            3.0,
            f32::NAN,
            8.0,
            f32::NAN,
            1.0,
            f32::NAN,
            4.0
        ]),
        Some(4)
    );
    assert_eq!(
        lanes::stats::f32::argmax(&[f32::NAN, f32::NAN, f32::NAN]),
        Some(0)
    );
}

#[test]
fn argmin_ignores_nan_unless_all_nan() {
    assert_eq!(
        lanes::stats::f32::argmin(&[f32::NAN, 1.0, f32::NAN]),
        Some(1)
    );
    assert_eq!(
        lanes::stats::f32::argmin(&[f32::NAN, 3.0, 2.0, f32::NAN, 9.0]),
        Some(2)
    );
    assert_eq!(
        lanes::stats::f32::argmin(&[
            5.0,
            f32::NAN,
            3.0,
            f32::NAN,
            8.0,
            f32::NAN,
            1.0,
            f32::NAN,
            4.0
        ]),
        Some(6)
    );
    assert_eq!(
        lanes::stats::f32::argmin(&[f32::NAN, f32::NAN, f32::NAN]),
        Some(0)
    );
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
    // Verify we can match on it — compilation is the real test. The
    // wildcard arm is required: `Backend` is `#[non_exhaustive]`.
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
        _ => {}
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
fn f64_argmax_argmin_ignore_nan_unless_all_nan() {
    assert_eq!(
        lanes::stats::f64::argmax(&[f64::NAN, 1.0, f64::NAN]),
        Some(1)
    );
    assert_eq!(
        lanes::stats::f64::argmax(&[f64::NAN, 3.0, 2.0, f64::NAN, 9.0]),
        Some(4)
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
    assert_eq!(
        lanes::stats::f64::argmin(&[
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
        Some(6)
    );
    assert_eq!(
        lanes::stats::f64::argmax(&[f64::NAN, f64::NAN, f64::NAN]),
        Some(0)
    );
    assert_eq!(
        lanes::stats::f64::argmin(&[f64::NAN, f64::NAN, f64::NAN]),
        Some(0)
    );
}

#[test]
fn f64_argmax_argmin_ties_across_chunks() {
    // Tie spanning chunk boundaries: the first global occurrence wins, even
    // though the tie sits in a lower SIMD lane after the chunk loop.
    let a = [
        -1.456_816_089_375_683e144_f64,
        2.904_355_210_078_954_5e-144,
        -1.261_706_705_752_713_4e144,
        -595_821_443.733_333_2,
        -595_821_443.513_725_4,
        -1.456_815_989_101_385_2e144,
        5.853_478_681_697_126e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        5.853_637_718_687_906e170,
        7.261_553_200_147_971e-95,
        -2.721_116_718_732_734_6e306,
        3.237_86e-319,
    ];
    assert_eq!(lanes::stats::f64::argmax(&a), Some(7));
    assert_eq!(lanes::stats::f64::argmin(&a), Some(14));
    // f32 tie inside a single chunk still resolves to the first occurrence.
    assert_eq!(lanes::stats::f32::argmax(&[1.0, 5.0, 5.0, 1.0]), Some(1));
    assert_eq!(lanes::stats::f32::argmin(&[1.0, 5.0, 5.0, 1.0]), Some(0));
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

    let c = lanes::math::f64::clip(&[-5.0_f64, 0.5, 3.0, 10.0], -1.0, 2.0).unwrap();
    assert_eq!(c, [-1.0, 0.5, 2.0, 2.0]);
    assert_eq!(
        lanes::math::f64::clip(&[1.0_f64], 2.0, -1.0),
        Err(lanes::Error::InvalidBounds)
    );

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
        Ok(1.0)
    );
    // Orthogonal → 0.0.
    assert!(
        lanes::ml::f32::cosine_similarity(&[1.0_f32, 0.0], &[0.0_f32, 1.0])
            .unwrap()
            .abs()
            < 1e-6
    );
    // Opposite → -1.0.
    assert_eq!(
        lanes::ml::f32::cosine_similarity(&[1.0_f32], &[-1.0_f32]),
        Ok(-1.0)
    );
    // Length mismatch → error.
    assert!(lanes::ml::f32::cosine_similarity(&[1.0_f32], &[1.0_f32, 2.0]).is_err());
    // Empty → EmptyInput error.
    assert_eq!(
        lanes::ml::f32::cosine_similarity(&[], &[]),
        Err(lanes::Error::EmptyInput)
    );
    // Zero vector → 0.0 (no direction to share).
    assert_eq!(
        lanes::ml::f32::cosine_similarity(&[0.0_f32], &[1.0_f32]),
        Ok(0.0)
    );

    assert!(
        (lanes::ml::f64::cosine_similarity(&[1.0_f64, 2.0], &[1.0_f64, 2.0]).unwrap() - 1.0).abs()
            < 1e-12
    );
    assert!(
        lanes::ml::f64::cosine_similarity(&[1.0_f64, 0.0], &[0.0_f64, 1.0])
            .unwrap()
            .abs()
            < 1e-12
    );
    assert_eq!(
        lanes::ml::f64::cosine_similarity(&[1.0_f64], &[-1.0_f64]),
        Ok(-1.0)
    );
    assert!(lanes::ml::f64::cosine_similarity(&[1.0_f64], &[1.0_f64, 2.0]).is_err());
    assert_eq!(
        lanes::ml::f64::cosine_similarity(&[0.0_f64], &[1.0_f64]),
        Ok(0.0)
    );
    assert_eq!(
        lanes::ml::f64::cosine_similarity(&[], &[]),
        Err(lanes::Error::EmptyInput)
    );
}

#[test]
fn ln_matches_std() {
    for &x in &[1.0_f32, std::f32::consts::E, 2.0, 0.5, 0.001, 1e30, 1e-30] {
        let got = lanes::math::f32::ln(&[x])[0];
        assert!(
            (got - x.ln()).abs() < 1e-5 * x.ln().abs().max(1.0),
            "ln({x})"
        );
    }
    for &x in &[1.0_f64, std::f64::consts::E, 2.0, 0.5, 0.001, 1e200, 1e-200] {
        let got = lanes::math::f64::ln(&[x])[0];
        assert!(
            (got - x.ln()).abs() < 1e-12 * x.ln().abs().max(1.0),
            "ln_f64({x})"
        );
    }
    assert!(lanes::math::f32::ln(&[0.0_f32])[0].is_infinite());
    assert!(lanes::math::f32::ln(&[-1.0_f32])[0].is_nan());
    assert_eq!(lanes::math::f32::ln(&[f32::INFINITY])[0], f32::INFINITY);
}

#[test]
fn ln_full_chunk_accuracy() {
    // Multi-chunk input forces the SIMD register kernel (not the scalar
    // tail), which the single-element tests above never exercise.
    let xs_f64: Vec<f64> = (1..=128).map(|i| 0.5 + i as f64 * 0.03125).collect();
    for (&x, &got) in xs_f64.iter().zip(&lanes::math::f64::ln(&xs_f64)) {
        let want = x.ln();
        let ulps = (got.to_bits() as i128 - want.to_bits() as i128).abs();
        assert!(ulps <= 2, "ln_f64({x}) = {got} want {want} ({ulps} ulps)");
    }
    let xs_f32: Vec<f32> = (1..=256).map(|i| 0.5 + i as f32 * 0.25).collect();
    for (&x, &got) in xs_f32.iter().zip(&lanes::math::f32::ln(&xs_f32)) {
        let want = x.ln();
        let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
        assert!(ulps <= 2, "ln({x}) = {got} want {want} ({ulps} ulps)");
    }
}

#[test]
fn logsumexp_stable_and_correct() {
    let s = lanes::ml::f32::logsumexp(&[1.0_f32, 2.0, 3.0]);
    assert!((s - 3.407_606).abs() < 1e-5);
    assert_eq!(lanes::ml::f32::logsumexp(&[]), f32::NEG_INFINITY);
    // Huge constants shift the result, not the shape.
    let a = lanes::ml::f32::logsumexp(&[1.0_f32, 2.0, 3.0]);
    let b = lanes::ml::f32::logsumexp(&[101.0_f32, 102.0, 103.0]);
    assert!((b - a - 100.0).abs() < 1e-4);
    let d = lanes::ml::f64::logsumexp(&[1.0_f64, 2.0, 3.0]);
    assert!((d - 3.407_605_964_444_385).abs() < 1e-12);
    assert_eq!(lanes::ml::f64::logsumexp(&[]), f64::NEG_INFINITY);
}

#[test]
fn layer_norm_zero_mean_unit_var() {
    for v in [vec![1.0_f32, 2.0, 3.0], vec![-2.0_f32, 5.0, -1.0, 4.0]] {
        let out = lanes::ml::f32::layer_norm(&v, 1e-5);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        let var: f32 = out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32;
        assert!((var - 1.0).abs() < 1e-4, "var {var}");
    }
    // Constant vector: zero variance → layer_norm is all zeros (the
    // standard definition; there is no unit variance to normalize to).
    let out = lanes::ml::f32::layer_norm(&[1000.0_f32, 1000.0, 1000.0], 1e-5);
    assert!(out.iter().all(|&x| x.abs() < 1e-6));
    let out = lanes::ml::f64::layer_norm(&[1.0_f64, 2.0, 3.0], 1e-10);
    let mean: f64 = out.iter().sum::<f64>() / 3.0;
    assert!(mean.abs() < 1e-12);
    assert_eq!(lanes::ml::f32::layer_norm(&[], 1e-5), Vec::<f32>::new());
}

#[test]
fn geometric_mean_matches_product() {
    let g = lanes::stats::f32::geometric_mean(&[1.0_f32, 4.0, 16.0]).unwrap();
    assert!((g - 4.0).abs() < 1e-5);
    let g = lanes::stats::f64::geometric_mean(&[1.0_f64, 4.0, 16.0]).unwrap();
    assert!((g - 4.0).abs() < 1e-12);
    // Distinct failure modes: empty input vs non-positive values.
    assert_eq!(
        lanes::stats::f32::geometric_mean(&[]),
        Err(lanes::Error::EmptyInput)
    );
    assert_eq!(
        lanes::stats::f32::geometric_mean(&[1.0_f32, -1.0]),
        Err(lanes::Error::NonPositiveInput { index: 1 })
    );
    assert_eq!(
        lanes::stats::f32::geometric_mean(&[1.0_f32, 0.0]),
        Err(lanes::Error::NonPositiveInput { index: 1 })
    );
    // NaN is not an error: it propagates to a NaN result.
    assert!(
        lanes::stats::f64::geometric_mean(&[1.0_f64, f64::NAN])
            .unwrap()
            .is_nan()
    );
}

#[test]
fn softplus_matches_canonical() {
    // Canonical: max(x,0) + ln_1p(e^-|x|) computed in f64.
    for &x in &[0.0_f32, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0, 1e-7, -1e-7] {
        let got = lanes::ml::f32::softplus(&[x])[0];
        let want = (x as f64).max(0.0) + (-(x as f64).abs()).exp().ln_1p();
        assert!(
            (got as f64 - want).abs() < 1e-6 * want.max(1.0),
            "softplus({x}) = {got} want {want}"
        );
    }
    for &x in &[0.0_f64, 1.0, -1.0, 1000.0, -1000.0, 1e-13, -1e-13] {
        let got = lanes::ml::f64::softplus(&[x])[0];
        let want = x.max(0.0) + (-x.abs()).exp().ln_1p();
        assert!(
            (got - want).abs() < 1e-12 * want.max(1.0),
            "softplus_f64({x}) = {got} want {want}"
        );
    }
    assert_eq!(lanes::ml::f32::softplus(&[f32::INFINITY])[0], f32::INFINITY);
    assert_eq!(lanes::ml::f32::softplus(&[f32::NEG_INFINITY])[0], 0.0);
    assert!(lanes::ml::f32::softplus(&[f32::NAN])[0].is_nan());
}

#[test]
fn log_softmax_exp_sums_to_one() {
    // exp(log_softmax(x)) IS softmax(x): sums to 1, monotone order.
    for v in [
        vec![1.0_f32, 2.0, 3.0],
        vec![-5.0_f32, 0.0, 5.0, 100.0],
        vec![3.478e9_f32; 8], // large-common-offset precision case
    ] {
        let ls = lanes::ml::f32::log_softmax(&v);
        let s: f64 = ls.iter().map(|&x| (x as f64).exp()).sum();
        assert!((s - 1.0).abs() < 1e-4, "exp-sum {s} for {v:?}");
        assert!(ls.iter().all(|x| x.is_finite()));
    }
    let ls = lanes::ml::f64::log_softmax(&[1.0_f64, 2.0, 3.0]);
    let s: f64 = ls.iter().map(|x| x.exp()).sum();
    assert!((s - 1.0).abs() < 1e-12);
    assert_eq!(lanes::ml::f32::log_softmax(&[]), Vec::<f32>::new());
}

#[test]
fn log_softmax_into_short_buffer_errors() {
    let mut out = [0.0_f32; 2];
    assert_eq!(
        lanes::ml::f32::log_softmax_into(&[1.0, 2.0, 3.0], &mut out),
        Err(lanes::Error::LengthMismatch {
            expected: 3,
            actual: 2
        })
    );
}

#[test]
fn layer_norm_into_short_buffer_errors() {
    let mut out = [0.0_f64; 2];
    assert_eq!(
        lanes::ml::f64::layer_norm_into(&[1.0, 2.0, 3.0], 1e-9, &mut out),
        Err(lanes::Error::LengthMismatch {
            expected: 3,
            actual: 2
        })
    );
}

#[test]
fn math_abs_sub_known_and_empty() {
    let a = [1.0_f32, -5.0, 3.0];
    let b = [4.0_f32, -2.0, 3.0];
    assert_eq!(lanes::math::f32::abs_sub(&a, &b).unwrap(), [3.0, 3.0, 0.0]);
    assert!(lanes::math::f32::abs_sub(&[], &[]).unwrap().is_empty());
    let a64 = [1.0_f64, -5.0];
    let b64 = [4.0_f64, -2.0];
    assert_eq!(lanes::math::f64::abs_sub(&a64, &b64).unwrap(), [3.0, 3.0]);
}

#[test]
fn math_abs_sub_errors_on_length_mismatch() {
    assert_eq!(
        lanes::math::f32::abs_sub(&[1.0, 2.0], &[1.0]),
        Err(lanes::Error::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn stats_counts_known_and_empty() {
    let v = [
        0.0_f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.5,
    ];
    assert_eq!(lanes::stats::f32::count_zero(&v), 2);
    assert_eq!(lanes::stats::f32::count_nan(&v), 1);
    assert_eq!(lanes::stats::f32::count_infinite(&v), 2);
    let empty: [f32; 0] = [];
    assert_eq!(lanes::stats::f32::count_zero(&empty), 0);
    assert_eq!(lanes::stats::f32::count_nan(&empty), 0);
    assert_eq!(lanes::stats::f32::count_infinite(&empty), 0);
    let v64 = [0.0_f64, -0.0, f64::NAN, f64::INFINITY, 1.5];
    assert_eq!(lanes::stats::f64::count_zero(&v64), 2);
    assert_eq!(lanes::stats::f64::count_nan(&v64), 1);
    assert_eq!(lanes::stats::f64::count_infinite(&v64), 1);
}

#[test]
fn math_hypot_known_overflow_and_specials() {
    // 3-4-5 triangle.
    let r = lanes::math::f32::hypot(&[3.0_f32], &[4.0_f32]).unwrap();
    assert!((r[0] - 5.0).abs() < 1e-6);
    // Overflow case: naive sqrt(x²+y²) overflows, hypot must not.
    let big = [2.0e19_f32];
    let r = lanes::math::f32::hypot(&big, &big).unwrap();
    assert!(r[0].is_finite(), "hypot overflowed: {}", r[0]);
    // Specials: inf wins over NaN.
    let r = lanes::math::f32::hypot(&[f32::INFINITY], &[f32::NAN]).unwrap();
    assert_eq!(r[0], f32::INFINITY);
    let r = lanes::math::f32::hypot(&[f32::NAN], &[1.0]).unwrap();
    assert!(r[0].is_nan());
    let r = lanes::math::f32::hypot(&[5.0], &[0.0]).unwrap();
    assert_eq!(r[0], 5.0);
    // f64 twins.
    let r64 = lanes::math::f64::hypot(&[3.0_f64], &[4.0_f64]).unwrap();
    assert!((r64[0] - 5.0).abs() < 1e-12);
    let r64 = lanes::math::f64::hypot(&[f64::INFINITY], &[f64::NAN]).unwrap();
    assert_eq!(r64[0], f64::INFINITY);
}

#[test]
fn math_hypot_errors_on_length_mismatch() {
    assert_eq!(
        lanes::math::f32::hypot(&[1.0, 2.0], &[1.0]),
        Err(lanes::Error::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn math_powi_known_and_specials() {
    assert_eq!(lanes::math::f32::powi(&[2.0_f32, 3.0], 3), [8.0, 27.0]);
    assert_eq!(lanes::math::f32::powi(&[2.0], -2), [0.25]);
    // powi(x, 0) == 1 for every x, incl. NaN/inf.
    let r = lanes::math::f32::powi(&[f32::NAN, f32::INFINITY, 0.0], 0);
    assert_eq!(r, [1.0, 1.0, 1.0]);
    // Empty.
    assert!(lanes::math::f32::powi(&[], 5).is_empty());
    // f64 twins.
    assert_eq!(lanes::math::f64::powi(&[2.0_f64, 3.0], 3), [8.0, 27.0]);
    assert_eq!(lanes::math::f64::powi(&[2.0], -2), [0.25]);
}

#[test]
fn math_powi_into_errors_on_length_mismatch() {
    let mut out = [0.0_f32; 1];
    assert_eq!(
        lanes::math::f32::powi_into(&[1.0, 2.0], 2, &mut out),
        Err(lanes::Error::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn distance_squared_distance_known_empty_and_mismatch() {
    assert_eq!(
        lanes::distance::f32::squared_distance(&[1.0, 2.0], &[4.0, 6.0]),
        Ok(25.0)
    );
    assert_eq!(lanes::distance::f32::squared_distance(&[], &[]), Ok(0.0));
    assert_eq!(
        lanes::distance::f32::squared_distance(&[1.0, 2.0], &[1.0]),
        Err(lanes::Error::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        lanes::distance::f64::squared_distance(&[1.0, 2.0], &[4.0, 6.0]),
        Ok(25.0)
    );
}

// ---------------------------------------------------------------------------
// kl_divergence / js_divergence (issue #5)
// ---------------------------------------------------------------------------

/// f64-precision reference for the known-vector tests (hand-checkable:
/// KL(p‖q) = Σ pᵢ ln(pᵢ/qᵢ); JS = (KL(p‖m) + KL(q‖m))/2, m = (p+q)/2).
fn ref_kl_f64(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&a, &b)| a * (a / b).ln()).sum()
}

fn ref_js_f64(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q)
        .map(|(&a, &b)| {
            let m = (a + b) * 0.5;
            a * (a / m).ln() + b * (b / m).ln()
        })
        .sum::<f64>()
        * 0.5
}

#[test]
fn kl_divergence_known_vector() {
    let p = [0.1_f32, 0.9];
    let q = [0.2_f32, 0.8];
    let got = lanes::distance::f32::kl_divergence(&p, &q).unwrap();
    let want = ref_kl_f64(&[0.1, 0.9], &[0.2, 0.8]);
    assert!(
        (f64::from(got) - want).abs() < 1e-6,
        "got {got}, want {want}"
    );

    let p64 = [0.1_f64, 0.9];
    let q64 = [0.2_f64, 0.8];
    let got64 = lanes::distance::f64::kl_divergence(&p64, &q64).unwrap();
    assert!((got64 - want).abs() < 1e-12, "got {got64}, want {want}");
}

#[test]
fn js_divergence_known_vector() {
    let p = [0.1_f32, 0.9];
    let q = [0.2_f32, 0.8];
    let got = lanes::distance::f32::js_divergence(&p, &q).unwrap();
    let want = ref_js_f64(&[0.1, 0.9], &[0.2, 0.8]);
    assert!(
        (f64::from(got) - want).abs() < 1e-6,
        "got {got}, want {want}"
    );

    let p64 = [0.1_f64, 0.9];
    let q64 = [0.2_f64, 0.8];
    let got64 = lanes::distance::f64::js_divergence(&p64, &q64).unwrap();
    assert!((got64 - want).abs() < 1e-12, "got {got64}, want {want}");
}

#[test]
fn divergence_self_is_zero() {
    let p = [0.25_f32, 0.25, 0.5];
    // Every term is p·ln(1) = 0 exactly (ln(1) = 0 in fdlibm).
    assert_eq!(lanes::distance::f32::kl_divergence(&p, &p), Ok(0.0));
    assert_eq!(lanes::distance::f32::js_divergence(&p, &p), Ok(0.0));
    let p64 = [0.25_f64, 0.25, 0.5];
    assert_eq!(lanes::distance::f64::kl_divergence(&p64, &p64), Ok(0.0));
    assert_eq!(lanes::distance::f64::js_divergence(&p64, &p64), Ok(0.0));
}

#[test]
fn js_is_symmetric_kl_is_not() {
    let p = [0.1_f32, 0.9];
    let q = [0.2_f32, 0.8];
    let js_pq = lanes::distance::f32::js_divergence(&p, &q).unwrap();
    let js_qp = lanes::distance::f32::js_divergence(&q, &p).unwrap();
    assert!(
        (js_pq - js_qp).abs() < 1e-6,
        "JS must be symmetric: {js_pq} vs {js_qp}"
    );
    let kl_pq = lanes::distance::f32::kl_divergence(&p, &q).unwrap();
    let kl_qp = lanes::distance::f32::kl_divergence(&q, &p).unwrap();
    assert!(
        (kl_pq - kl_qp).abs() > 1e-4,
        "KL must be asymmetric for this pair: {kl_pq} vs {kl_qp}"
    );
}

#[test]
fn divergence_empty_is_zero_and_mismatch_errors() {
    assert_eq!(lanes::distance::f32::kl_divergence(&[], &[]), Ok(0.0));
    assert_eq!(lanes::distance::f32::js_divergence(&[], &[]), Ok(0.0));
    assert_eq!(lanes::distance::f64::kl_divergence(&[], &[]), Ok(0.0));
    assert_eq!(lanes::distance::f64::js_divergence(&[], &[]), Ok(0.0));
    assert_eq!(
        lanes::distance::f32::kl_divergence(&[0.5, 0.5], &[1.0]),
        Err(lanes::Error::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        lanes::distance::f32::js_divergence(&[0.5, 0.5], &[1.0]),
        Err(lanes::Error::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        lanes::distance::f64::kl_divergence(&[0.5, 0.5], &[1.0]),
        Err(lanes::Error::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn kl_zero_and_nan_semantics_documented() {
    // p=0, q>0: naive IEEE term is 0 · ln(0) = 0 · -inf = NaN (documented;
    // differs from scipy rel_entr's 0 convention).
    let r = lanes::distance::f32::kl_divergence(&[0.0, 1.0], &[0.5, 0.5]).unwrap();
    assert!(r.is_nan(), "expected NaN, got {r}");
    // p>0, q=0: term is p · ln(+inf) = +inf (the divergence is unbounded).
    let r = lanes::distance::f32::kl_divergence(&[1.0], &[0.0]).unwrap();
    assert_eq!(r, f32::INFINITY);
    // NaN input propagates.
    let r = lanes::distance::f32::kl_divergence(&[f32::NAN, 0.5], &[0.5, 0.5]).unwrap();
    assert!(r.is_nan());
    let r = lanes::distance::f32::js_divergence(&[f32::NAN, 0.5], &[0.5, 0.5]).unwrap();
    assert!(r.is_nan());
    // f64 twins.
    let r = lanes::distance::f64::kl_divergence(&[1.0], &[0.0]).unwrap();
    assert_eq!(r, f64::INFINITY);
    let r = lanes::distance::f64::kl_divergence(&[0.0, 1.0], &[0.5, 0.5]).unwrap();
    assert!(r.is_nan());
}

#[test]
fn divergence_matches_scalar_on_chunked_lengths() {
    // Lengths crossing the 4/8/16-lane chunk boundaries plus a scalar tail,
    // compared against the f64 reference with a summation-order tolerance.
    for &n in &[7_usize, 8, 9, 15, 16, 17, 31, 32, 33, 100] {
        let p: Vec<f32> = (1..=n).map(|i| (i as f32) * 0.001 + 0.01).collect();
        let q: Vec<f32> = (1..=n)
            .map(|i| ((n - i + 1) as f32) * 0.001 + 0.01)
            .collect();
        let p64: Vec<f64> = p.iter().map(|&x| f64::from(x)).collect();
        let q64: Vec<f64> = q.iter().map(|&x| f64::from(x)).collect();

        let kl = lanes::distance::f32::kl_divergence(&p, &q).unwrap();
        let kl_ref = ref_kl_f64(&p64, &q64);
        assert!(
            (f64::from(kl) - kl_ref).abs() < 1e-4,
            "n={n}: kl {kl} vs ref {kl_ref}"
        );

        let js = lanes::distance::f32::js_divergence(&p, &q).unwrap();
        let js_ref = ref_js_f64(&p64, &q64);
        assert!(
            (f64::from(js) - js_ref).abs() < 1e-5,
            "n={n}: js {js} vs ref {js_ref}"
        );

        let kl64 = lanes::distance::f64::kl_divergence(&p64, &q64).unwrap();
        assert!(
            (kl64 - kl_ref).abs() < 1e-9,
            "n={n}: kl64 {kl64} vs ref {kl_ref}"
        );
        let js64 = lanes::distance::f64::js_divergence(&p64, &q64).unwrap();
        assert!(
            (js64 - js_ref).abs() < 1e-9,
            "n={n}: js64 {js64} vs ref {js_ref}"
        );
    }
}

// ---------------------------------------------------------------------------
// binary family: hamming / jaccard over packed bitmaps
// ---------------------------------------------------------------------------

#[test]
fn binary_hamming_known_values() {
    // Bit-level, not byte-level: 0b01 vs 0b11 differ in exactly 1 bit.
    assert_eq!(lanes::binary::hamming(&[0b01], &[0b11]), Ok(1));
    assert_eq!(
        lanes::binary::hamming(&[0b1010_1010], &[0b0110_0110]),
        Ok(4)
    );
    assert_eq!(lanes::binary::hamming(&[], &[]), Ok(0));
    assert_eq!(lanes::binary::hamming(&[0xFF; 4], &[0x00; 4]), Ok(32));
    assert_eq!(lanes::binary::hamming(&[0xAA, 0x55], &[0xAA, 0x55]), Ok(0));
}

#[test]
fn binary_hamming_length_mismatch() {
    match lanes::binary::hamming(&[1, 2, 3], &[1, 2]) {
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn binary_jaccard_known_values() {
    // a = 0b1010_1010, b = 0b0110_0110:
    // AND = 0b0010_0010 (2 bits), OR = 0b1110_1110 (6 bits) -> 2/6 = 1/3.
    let j = lanes::binary::jaccard(&[0b1010_1010], &[0b0110_0110])
        .unwrap()
        .unwrap();
    assert!((j - 1.0 / 3.0).abs() < 1e-6);

    // Identical non-zero bitmaps -> similarity 1.
    assert_eq!(lanes::binary::jaccard(&[0xFF], &[0xFF]), Ok(Some(1.0)));

    // Disjoint bitmaps -> similarity 0.
    assert_eq!(lanes::binary::jaccard(&[0xF0], &[0x0F]), Ok(Some(0.0)));

    // All-zero union -> None (including the empty case).
    assert_eq!(lanes::binary::jaccard(&[0x00], &[0x00]), Ok(None));
    assert_eq!(lanes::binary::jaccard(&[], &[]), Ok(None));
}

#[test]
fn binary_jaccard_length_mismatch() {
    match lanes::binary::jaccard(&[1], &[1, 2]) {
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, 1);
            assert_eq!(actual, 2);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// stats::i8 — first general integer family (i8 with i32 accumulation)
// ---------------------------------------------------------------------------

#[test]
fn i8_dot_known_values() {
    // Mixed signs: 1*5 + (-2)*3 + 3*(-1) + (-4)*(-2) = 5 - 6 - 3 + 8 = 4.
    assert_eq!(
        lanes::stats::i8::dot(&[1, -2, 3, -4], &[5, 3, -1, -2]),
        Ok(4)
    );
    // Extremes: (-128)*(-128) = 16384, needs the i32 accumulator.
    assert_eq!(
        lanes::stats::i8::dot(&[-128, 127], &[-128, 127]),
        Ok(16384 + 16129)
    );
    assert_eq!(lanes::stats::i8::dot(&[], &[]), Ok(0));
    assert_eq!(lanes::stats::i8::dot(&[7; 8], &[3; 8]), Ok(168));
}

#[test]
fn i8_dot_length_mismatch() {
    match lanes::stats::i8::dot(&[1, 2, 3], &[1, 2]) {
        Err(lanes::Error::LengthMismatch { expected, actual }) => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn i8_sum_known_values() {
    assert_eq!(lanes::stats::i8::sum(&[1, -2, 3, -4]), -2);
    // i32 accumulation: 127 * 100 = 12700 overflows i8 but not i32.
    assert_eq!(lanes::stats::i8::sum(&[127; 100]), 12700);
    assert_eq!(lanes::stats::i8::sum(&[-128; 3]), -384);
    assert_eq!(lanes::stats::i8::sum(&[]), 0);
}
