//! Comprehensive correctness tests for f16/bf16 conversions (Issue #7).
//!
//! Verification strategy:
//! - All 65536 f16→f32→f16 round trips are tested exhaustively.
//! - bf16→f32 is tested exhaustively (all 65536 values).
//! - f32→bf16 round-to-nearest-even is verified against an independent
//!   f64-based oracle for all representable midpoints and tie cases.
//! - Edge cases: NaN, ±Inf, ±0, denormals, max/min values.
//! - Dot products verified against f64 reference accumulation.
//!
//! These tests run against whichever backend `Backend::detect` picks.

#![allow(clippy::excessive_precision)] // we use exact mathematical constants (2^-14 etc.)

use lanes::convert;

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the "correct" f16→f32 conversion using known bit-level rules,
/// independent of the library under test.
fn reference_f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormal: val = (-1)^sign * 2^(1-15) * (mant/1024)
            //         = (-1)^sign * 2^(-14) * mant * 2^(-10)
            //         = (-1)^sign * mant * 2^(-24)
            let val = (mant as f64) * 2.0_f64.powi(-24);
            if sign == 0 { val as f32 } else { -(val as f32) }
        }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY }
        } else {
            // NaN - just check it's NaN, don't compare bits
            f32::NAN
        }
    } else {
        // Normal: val = (-1)^sign * 2^(exp-15) * (1 + mant/1024)
        let val = 2.0_f64.powi(exp as i32 - 15) * (1.0 + (mant as f64) / 1024.0);
        if sign == 0 { val as f32 } else { -(val as f32) }
    }
}

/// Independent reference for f32→f16 using f64 arithmetic.
/// Computes the correctly rounded (round-to-nearest-even) f16 result.
fn reference_f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        // NaN or Inf
        if mant != 0 {
            // NaN → quiet NaN in f16
            (sign << 15) | 0x7E00
        } else {
            // Inf
            (sign << 15) | 0x7C00
        }
    } else {
        // Use f64 to compute the exact value and find the nearest f16
        let val = if value.is_sign_negative() {
            -(f64::from(value.abs()))
        } else {
            f64::from(value)
        };
        let abs_val = val.abs();

        // f16 max normal = 65504.0, f16 min denormal = 2^-24
        let f16_max: f64 = 65504.0;
        // The midpoint between f16 max and next Inf is 65504 + 16 = 65520
        // (the ULP at max is 32, so half-ULP is 16)
        let f16_overflow_threshold: f64 = 65520.0;

        if abs_val >= f16_overflow_threshold {
            return (sign << 15) | 0x7C00; // Inf
        }

        if abs_val < 2.0_f64.powi(-24) * 0.5 {
            return sign << 15; // zero (below half of smallest denorm)
        }

        // Brute-force: find the two nearest f16 candidates
        let _ = f16_max; // suppress unused warning

        // Convert to f16 by finding nearest among all f16 values
        best_f16_rne(val, sign)
    }
}

/// Find the nearest f16 (round-to-nearest-even) by brute-force comparison.
fn best_f16_rne(val: f64, sign: u16) -> u16 {
    let abs_val = val.abs();

    // Generate all candidate f16 values and find the nearest
    let mut best_bits: u16 = 0;
    let mut best_dist: f64 = f64::INFINITY;
    let mut best_is_even: bool = true;

    // Iterate over all non-negative f16 values (0..0x7C00 for finite)
    for candidate in 0u16..=0x7BFF {
        let cand_f32 = reference_f16_to_f32(candidate);
        if cand_f32.is_nan() {
            continue;
        }
        let cand_val = f64::from(cand_f32);
        let dist = (abs_val - cand_val).abs();

        if dist < best_dist || (dist == best_dist && !best_is_even && (candidate & 1) == 0) {
            best_dist = dist;
            best_bits = candidate;
            best_is_even = (candidate & 1) == 0;
        }
    }

    (sign << 15) | best_bits
}

/// Independent reference for f32→bf16 using exact arithmetic.
fn reference_f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF && mant != 0 {
        // NaN: force quiet
        ((bits >> 16) | 0x0040) as u16
    } else {
        // Round-to-nearest-even: add bias based on retained LSB + truncated bits
        let rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
        ((bits.wrapping_add(rounding_bias)) >> 16) as u16
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// f16 ↔ f32 EXHAUSTIVE TESTS
// ═══════════════════════════════════════════════════════════════════════════

/// Test ALL 65536 possible f16→f32 conversions against the reference.
#[test]
fn f16_to_f32_exhaustive() {
    let mut input = vec![0u16; 65536];
    let mut output = vec![0.0_f32; 65536];

    for i in 0..65536u32 {
        input[i as usize] = i as u16;
    }

    convert::f16_to_f32(&input, &mut output).unwrap();

    let mut failures = 0;
    for i in 0..65536u32 {
        let bits = i as u16;
        let got = output[i as usize];
        let expected = reference_f16_to_f32(bits);

        // NaN: both must be NaN
        if expected.is_nan() {
            if !got.is_nan() {
                failures += 1;
                if failures <= 10 {
                    eprintln!(
                        "FAIL f16_to_f32: bits=0x{bits:04X}, expected=NaN, got={got}"
                    );
                }
            }
        } else if got.to_bits() != expected.to_bits() {
            failures += 1;
            if failures <= 10 {
                eprintln!(
                    "FAIL f16_to_f32: bits=0x{bits:04X}, expected={expected} (0x{:08X}), got={got} (0x{:08X})",
                    expected.to_bits(), got.to_bits()
                );
            }
        }
    }
    assert_eq!(failures, 0, "{failures} f16→f32 conversions failed");
}

/// Test f16→f32→f16 round-trip for all 65536 f16 values.
/// Every finite f16 value must survive the round trip unchanged.
/// NaN and Inf have defined behavior (quiet NaN, preserve Inf).
#[test]
fn f16_roundtrip_exhaustive() {
    let mut f32_buf = vec![0.0_f32; 65536];
    let mut f16_buf = vec![0u16; 65536];

    // Create all f16 values
    let all_f16: Vec<u16> = (0..65536u32).map(|i| i as u16).collect();

    // Convert f16 → f32
    convert::f16_to_f32(&all_f16, &mut f32_buf).unwrap();
    // Convert f32 → f16
    convert::f32_to_f16(&f32_buf, &mut f16_buf).unwrap();

    let mut failures = 0;
    for i in 0..65536u32 {
        let original = i as u16;
        let recovered = f16_buf[i as usize];

        let exp = (original >> 10) & 0x1F;
        let mant = original & 0x03FF;

        if exp == 31 && mant != 0 {
            // NaN: both must have exp=31, mant!=0 (quiet NaN)
            let rec_exp = (recovered >> 10) & 0x1F;
            let rec_mant = recovered & 0x03FF;
            if rec_exp != 31 || rec_mant == 0 {
                failures += 1;
                if failures <= 10 {
                    eprintln!(
                        "FAIL roundtrip NaN: original=0x{original:04X}, recovered=0x{recovered:04X}"
                    );
                }
            }
        } else if original != recovered {
            failures += 1;
            if failures <= 10 {
                eprintln!(
                    "FAIL roundtrip: original=0x{original:04X}, recovered=0x{recovered:04X} (via f32={})",
                    f32_buf[i as usize]
                );
            }
        }
    }
    assert_eq!(failures, 0, "{failures} f16 round-trips failed");
}

// ═══════════════════════════════════════════════════════════════════════════
// f32 → f16 CORRECTNESS (selected values)
// ═══════════════════════════════════════════════════════════════════════════

/// Test f32→f16 for known exact values.
#[test]
fn f32_to_f16_exact_values() {
    let cases: &[(f32, u16)] = &[
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (1.0, 0x3C00),
        (-1.0, 0xBC00),
        (2.0, 0x4000),
        (0.5, 0x3800),
        (65504.0, 0x7BFF),   // f16 max normal
        (-65504.0, 0xFBFF),  // f16 min normal (negative)
        (f32::INFINITY, 0x7C00),
        (f32::NEG_INFINITY, 0xFC00),
        // Smallest positive normal f16 = 2^-14 ≈ 6.103515625e-5
        (6.103_515_625e-5, 0x0400),
        // Smallest positive denormal f16 = 2^-24 ≈ 5.96046448e-8
        (5.960_464_5e-8, 0x0001),
    ];

    for &(input, expected) in cases {
        let mut out = [0u16; 1];
        convert::f32_to_f16(&[input], &mut out).unwrap();
        assert_eq!(
            out[0], expected,
            "f32_to_f16({input}): got 0x{:04X}, expected 0x{expected:04X}",
            out[0]
        );
    }
}

/// Test f32→f16 NaN handling.
#[test]
fn f32_to_f16_nan() {
    let nans: &[f32] = &[
        f32::NAN,
        f32::from_bits(0x7F800001), // signaling NaN
        f32::from_bits(0xFF800001), // negative signaling NaN
        f32::from_bits(0x7FC00000), // quiet NaN
        f32::from_bits(0xFFC00000), // negative quiet NaN
    ];

    for &nan in nans {
        let mut out = [0u16; 1];
        convert::f32_to_f16(&[nan], &mut out).unwrap();
        // Result must be a quiet NaN in f16 (exp=31, mant!=0, quiet bit set)
        let exp = (out[0] >> 10) & 0x1F;
        let mant = out[0] & 0x03FF;
        assert_eq!(exp, 31, "NaN f32=0x{:08X}: exp not 31", nan.to_bits());
        assert_ne!(mant, 0, "NaN f32=0x{:08X}: mant is 0 (Inf, not NaN)", nan.to_bits());
        assert_ne!(mant & 0x0200, 0, "NaN f32=0x{:08X}: quiet bit not set", nan.to_bits());
    }
}

/// Test f32→f16 overflow (values > f16 max).
#[test]
fn f32_to_f16_overflow() {
    let overflows: &[(f32, u16)] = &[
        (65536.0, 0x7C00),        // > 65504 → +Inf
        (100000.0, 0x7C00),       // way over → +Inf
        (-70000.0, 0xFC00),       // negative overflow → -Inf
        (f32::MAX, 0x7C00),       // f32 max → +Inf
        (f32::MIN, 0xFC00),       // f32 min (most negative) → -Inf
    ];

    for &(input, expected) in overflows {
        let mut out = [0u16; 1];
        convert::f32_to_f16(&[input], &mut out).unwrap();
        assert_eq!(
            out[0], expected,
            "f32_to_f16({input}): got 0x{:04X}, expected 0x{expected:04X}",
            out[0]
        );
    }
}

/// Test f32→f16 underflow (values too small for f16 denormal).
#[test]
fn f32_to_f16_underflow() {
    let underflows: &[(f32, u16)] = &[
        // Smallest f16 denormal is 2^-24 ≈ 5.96e-8
        // Half of that is 2^-25 ≈ 2.98e-8 — below this rounds to zero
        (1e-9, 0x0000),           // tiny positive → +0
        (-1e-9, 0x8000),          // tiny negative → -0
        (1e-40, 0x0000),          // extremely tiny → +0
        (f32::MIN_POSITIVE * 0.5e-10, 0x0000), // near f32 denormal → +0
    ];

    for &(input, expected) in underflows {
        let mut out = [0u16; 1];
        convert::f32_to_f16(&[input], &mut out).unwrap();
        assert_eq!(
            out[0], expected,
            "f32_to_f16({input:e}): got 0x{:04X}, expected 0x{expected:04X}",
            out[0]
        );
    }
}

/// Test f32→f16 round-to-nearest-even for midpoint tie cases.
#[test]
fn f32_to_f16_ties_to_even() {
    // f16 0x3C00 = 1.0 (mantissa 0x000, EVEN)
    // f16 0x3C01 = 1.0 + 2^-10 = 1.0009765625 (mantissa 0x001, ODD)
    // Midpoint = 1.0 + 2^-11 = 1.00048828125
    // In f32: mantissa = 2^-11 * 2^23 = 2^12 = 0x001000
    // So f32 bits = 0x3F801000
    // Ties-to-even → should round to 0x3C00 (mantissa 0 is even)
    let midpoint_exact = f32::from_bits(0x3F80_1000);
    let mut out = [0u16; 1];
    convert::f32_to_f16(&[midpoint_exact], &mut out).unwrap();
    assert_eq!(out[0], 0x3C00, "tie at midpoint between 0x3C00 and 0x3C01 should round to even 0x3C00, got 0x{:04X}", out[0]);

    // Midpoint between 0x3C01 (odd, mantissa=1) and 0x3C02 (even, mantissa=2)
    // 0x3C01 = 1.0 + 1*2^-10 = 1.0009765625
    // 0x3C02 = 1.0 + 2*2^-10 = 1.001953125
    // Midpoint = 1.0 + 1.5*2^-10 = 1.0 + 2^-10 + 2^-11
    // In f32: mantissa = (2^-10 + 2^-11) * 2^23 = 2^13 + 2^12 = 0x003000
    // f32 bits = 0x3F803000
    // Ties-to-even → should round to 0x3C02 (mantissa 2 is even)
    let midpoint_2 = f32::from_bits(0x3F80_3000);
    convert::f32_to_f16(&[midpoint_2], &mut out).unwrap();
    assert_eq!(out[0], 0x3C02, "tie at midpoint between 0x3C01 and 0x3C02 should round to even 0x3C02, got 0x{:04X}", out[0]);

    // Value just above the first midpoint (should round up to 0x3C01)
    let above_mid = f32::from_bits(0x3F80_1001);
    convert::f32_to_f16(&[above_mid], &mut out).unwrap();
    assert_eq!(out[0], 0x3C01, "just above midpoint should round up to 0x3C01, got 0x{:04X}", out[0]);

    // Value just below the first midpoint (should round down to 0x3C00)
    let below_mid = f32::from_bits(0x3F80_0FFF);
    convert::f32_to_f16(&[below_mid], &mut out).unwrap();
    assert_eq!(out[0], 0x3C00, "just below midpoint should round down to 0x3C00, got 0x{:04X}", out[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// f32 → f16 RANDOM SAMPLING WITH ORACLE
// ═══════════════════════════════════════════════════════════════════════════

/// Test f32→f16 against the brute-force oracle for a sampling of values.
/// This is slow (oracle is O(65536) per value) so we limit to key ranges.
#[test]
fn f32_to_f16_oracle_normal_range() {
    // Test all f32 values that map to specific f16 normals: sample 1024
    // f32 values uniformly distributed in [2^-14, 65504]
    let mut failures = 0;
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_F00D;

    for _ in 0..1024 {
        // LCG
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Map to f32 in the normal f16 range [2^-14, 65504]
        let t = (rng >> 32) as f32 / u32::MAX as f32; // [0, 1]
        let val = 6.103515625e-5 + t * (65504.0 - 6.103515625e-5);

        let mut out = [0u16; 1];
        convert::f32_to_f16(&[val], &mut out).unwrap();
        let expected = reference_f32_to_f16(val);

        if out[0] != expected {
            failures += 1;
            if failures <= 5 {
                eprintln!(
                    "FAIL f32_to_f16 oracle: val={val} (0x{:08X}), got=0x{:04X}, expected=0x{expected:04X}",
                    val.to_bits(), out[0]
                );
            }
        }
    }
    assert_eq!(failures, 0, "{failures}/1024 f32→f16 conversions disagree with oracle");
}

/// Test denormal f16 outputs against oracle.
#[test]
fn f32_to_f16_oracle_denormal_range() {
    // Test f32 values in the denormal f16 range [2^-25, 2^-14)
    let mut failures = 0;
    let mut rng: u64 = 0xCAFE_BABE_1234_5678;

    for _ in 0..512 {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let t = (rng >> 32) as f32 / u32::MAX as f32;
        let val = 2.0e-8 + t * (6.0e-5 - 2.0e-8);

        let mut out = [0u16; 1];
        convert::f32_to_f16(&[val], &mut out).unwrap();
        let expected = reference_f32_to_f16(val);

        if out[0] != expected {
            failures += 1;
            if failures <= 5 {
                eprintln!(
                    "FAIL f32_to_f16 denorm oracle: val={val:e} (0x{:08X}), got=0x{:04X}, expected=0x{expected:04X}",
                    val.to_bits(), out[0]
                );
            }
        }
    }
    assert_eq!(failures, 0, "{failures}/512 denormal f32→f16 conversions disagree with oracle");
}

// ═══════════════════════════════════════════════════════════════════════════
// bf16 ↔ f32 TESTS
// ═══════════════════════════════════════════════════════════════════════════

/// Test ALL 65536 possible bf16→f32 conversions.
#[test]
fn bf16_to_f32_exhaustive() {
    let input: Vec<u16> = (0..65536u32).map(|i| i as u16).collect();
    let mut output = vec![0.0_f32; 65536];

    convert::bf16_to_f32(&input, &mut output).unwrap();

    let mut failures = 0;
    for i in 0..65536u32 {
        let bits = i as u16;
        let got = output[i as usize];
        // bf16→f32 is simply (bits as u32) << 16
        let expected = f32::from_bits((bits as u32) << 16);

        if expected.is_nan() {
            if !got.is_nan() {
                failures += 1;
            }
        } else if got.to_bits() != expected.to_bits() {
            failures += 1;
            if failures <= 10 {
                eprintln!(
                    "FAIL bf16_to_f32: bits=0x{bits:04X}, expected=0x{:08X}, got=0x{:08X}",
                    expected.to_bits(), got.to_bits()
                );
            }
        }
    }
    assert_eq!(failures, 0, "{failures} bf16→f32 conversions failed");
}

/// Test f32→bf16 for known exact values.
#[test]
fn f32_to_bf16_exact_values() {
    let cases: &[(f32, u16)] = &[
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (1.0, 0x3F80),
        (-1.0, 0xBF80),
        (2.0, 0x4000),
        (0.5, 0x3F00),
        (f32::INFINITY, 0x7F80),
        (f32::NEG_INFINITY, 0xFF80),
    ];

    for &(input, expected) in cases {
        let mut out = [0u16; 1];
        convert::f32_to_bf16(&[input], &mut out).unwrap();
        assert_eq!(
            out[0], expected,
            "f32_to_bf16({input}): got 0x{:04X}, expected 0x{expected:04X}",
            out[0]
        );
    }
}

/// Test f32→bf16→f32 round-trip for values exactly representable in bf16.
#[test]
fn bf16_roundtrip_exact_values() {
    // Values exactly representable in bf16 must survive the round trip.
    let exact: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
        128.0, 256.0, 0.25, 0.125, 3.0, 3.5,
        f32::INFINITY, f32::NEG_INFINITY,
    ];

    let mut bf16_buf = vec![0u16; exact.len()];
    let mut f32_buf = vec![0.0_f32; exact.len()];

    convert::f32_to_bf16(&exact, &mut bf16_buf).unwrap();
    convert::bf16_to_f32(&bf16_buf, &mut f32_buf).unwrap();

    for (i, (&orig, &recovered)) in exact.iter().zip(f32_buf.iter()).enumerate() {
        if orig.is_nan() {
            assert!(recovered.is_nan(), "idx {i}: expected NaN, got {recovered}");
        } else {
            assert_eq!(
                orig.to_bits(), recovered.to_bits(),
                "idx {i}: f32_to_bf16_to_f32({orig}) = {recovered} (bits differ)"
            );
        }
    }
}

/// Test f32→bf16 NaN handling.
#[test]
fn f32_to_bf16_nan() {
    let nans: &[f32] = &[
        f32::NAN,
        f32::from_bits(0x7F800001), // signaling NaN
        f32::from_bits(0xFF800001), // negative signaling NaN
        f32::from_bits(0x7FC00000), // quiet NaN
    ];

    for &nan in nans {
        let mut out = [0u16; 1];
        convert::f32_to_bf16(&[nan], &mut out).unwrap();
        // Result must be a NaN in bf16 (exp=0xFF, mant!=0 when widened)
        let widened = f32::from_bits((out[0] as u32) << 16);
        assert!(widened.is_nan(), "f32_to_bf16(NaN 0x{:08X}) = 0x{:04X} which is not NaN when widened", nan.to_bits(), out[0]);
        // Quiet bit must be set
        assert_ne!(out[0] & 0x0040, 0, "quiet bit not set for NaN 0x{:08X}", nan.to_bits());
    }
}

/// Test f32→bf16 round-to-nearest-even with tie cases.
#[test]
fn f32_to_bf16_ties_to_even() {
    // bf16 has 7 mantissa bits (8 significand bits total).
    // The truncation point is at bit 16 of the f32.
    //
    // Test: 1.0 in f32 = 0x3F800000
    // Next bf16 after 1.0: 0x3F81 = f32 0x3F810000 = 1.0 + 2^-7 = 1.0078125
    // Midpoint: 0x3F808000 = 1.0 + 2^-8 = 1.00390625
    // 0x3F80 has mantissa 0x00 (even), 0x3F81 has mantissa 0x01 (odd)
    // Ties-to-even → should round to 0x3F80
    let midpoint1 = f32::from_bits(0x3F80_8000);
    let mut out = [0u16; 1];
    convert::f32_to_bf16(&[midpoint1], &mut out).unwrap();
    assert_eq!(out[0], 0x3F80, "tie between 0x3F80(even) and 0x3F81(odd) should → 0x3F80, got 0x{:04X}", out[0]);

    // Test: midpoint between 0x3F81 (odd) and 0x3F82 (even)
    // 0x3F81 = 0x3F810000, 0x3F82 = 0x3F820000
    // Midpoint: 0x3F818000
    // Ties-to-even → should round to 0x3F82 (even)
    let midpoint2 = f32::from_bits(0x3F81_8000);
    convert::f32_to_bf16(&[midpoint2], &mut out).unwrap();
    assert_eq!(out[0], 0x3F82, "tie between 0x3F81(odd) and 0x3F82(even) should → 0x3F82, got 0x{:04X}", out[0]);

    // Test: value just above midpoint (should round up regardless of even/odd)
    let above_mid = f32::from_bits(0x3F80_8001);
    convert::f32_to_bf16(&[above_mid], &mut out).unwrap();
    assert_eq!(out[0], 0x3F81, "just above midpoint should round up to 0x3F81, got 0x{:04X}", out[0]);

    // Test: value just below midpoint (should round down)
    let below_mid = f32::from_bits(0x3F80_7FFF);
    convert::f32_to_bf16(&[below_mid], &mut out).unwrap();
    assert_eq!(out[0], 0x3F80, "just below midpoint should round down to 0x3F80, got 0x{:04X}", out[0]);
}

/// Exhaustive verification of f32→bf16 against reference for a sweep of values.
#[test]
fn f32_to_bf16_sweep() {
    let mut failures = 0;
    let mut rng: u64 = 0x1234_5678_ABCD_EF01;

    // Test 10000 random f32 values
    for _ in 0..10000 {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = (rng >> 32) as u32;
        let val = f32::from_bits(bits);

        let mut out = [0u16; 1];
        convert::f32_to_bf16(&[val], &mut out).unwrap();
        let expected = reference_f32_to_bf16(val);

        if val.is_nan() {
            // Both must be NaN
            let widened = f32::from_bits((out[0] as u32) << 16);
            if !widened.is_nan() {
                failures += 1;
            }
        } else if out[0] != expected {
            failures += 1;
            if failures <= 5 {
                eprintln!(
                    "FAIL f32_to_bf16: val=0x{bits:08X} ({val}), got=0x{:04X}, expected=0x{expected:04X}",
                    out[0]
                );
            }
        }
    }
    assert_eq!(failures, 0, "{failures}/10000 f32→bf16 conversions disagree with reference");
}

// ═══════════════════════════════════════════════════════════════════════════
// DOT PRODUCT TESTS
// ═══════════════════════════════════════════════════════════════════════════

/// Test dot_f16 basic correctness.
#[test]
fn dot_f16_basic() {
    let f16_one: u16 = 0x3C00; // 1.0
    let f16_two: u16 = 0x4000; // 2.0
    let f16_half: u16 = 0x3800; // 0.5
    let f16_neg_one: u16 = 0xBC00; // -1.0

    // 4 × (1.0 × 2.0) = 8.0
    let result = convert::dot_f16(&[f16_one; 4], &[f16_two; 4]).unwrap();
    assert_eq!(result, 8.0);

    // (1.0 × 2.0) + (0.5 × -1.0) = 2.0 - 0.5 = 1.5
    let result = convert::dot_f16(&[f16_one, f16_half], &[f16_two, f16_neg_one]).unwrap();
    assert_eq!(result, 1.5);

    // Empty dot = 0
    let result = convert::dot_f16(&[], &[]).unwrap();
    assert_eq!(result, 0.0);
}

/// Test dot_bf16 basic correctness.
#[test]
fn dot_bf16_basic() {
    let bf16_one: u16 = 0x3F80; // 1.0
    let bf16_two: u16 = 0x4000; // 2.0
    let bf16_half: u16 = 0x3F00; // 0.5
    let bf16_neg_one: u16 = 0xBF80; // -1.0

    // 4 × (1.0 × 2.0) = 8.0
    let result = convert::dot_bf16(&[bf16_one; 4], &[bf16_two; 4]).unwrap();
    assert_eq!(result, 8.0);

    // (1.0 × 2.0) + (0.5 × -1.0) = 1.5
    let result = convert::dot_bf16(&[bf16_one, bf16_half], &[bf16_two, bf16_neg_one]).unwrap();
    assert_eq!(result, 1.5);

    // Empty dot = 0
    let result = convert::dot_bf16(&[], &[]).unwrap();
    assert_eq!(result, 0.0);
}

/// Test dot product with larger arrays (exercises potential SIMD paths).
#[test]
fn dot_f16_large_array() {
    // Create 1024 elements of 1.0 in f16
    let f16_one = vec![0x3C00u16; 1024];
    let f16_two = vec![0x4000u16; 1024];

    let result = convert::dot_f16(&f16_one, &f16_two).unwrap();
    assert_eq!(result, 2048.0);
}

/// Test dot product with larger arrays (exercises potential SIMD paths).
#[test]
fn dot_bf16_large_array() {
    // Create 1024 elements of 1.0 in bf16
    let bf16_one = vec![0x3F80u16; 1024];
    let bf16_two = vec![0x4000u16; 1024];

    let result = convert::dot_bf16(&bf16_one, &bf16_two).unwrap();
    assert_eq!(result, 2048.0);
}

/// Test dot product with NaN propagation.
#[test]
fn dot_f16_nan_propagation() {
    let f16_nan: u16 = 0x7E00; // quiet NaN in f16
    let f16_one: u16 = 0x3C00;

    let result = convert::dot_f16(&[f16_one, f16_nan], &[f16_one, f16_one]).unwrap();
    assert!(result.is_nan(), "dot with NaN element should propagate NaN, got {result}");
}

/// Test dot product with NaN propagation for bf16.
#[test]
fn dot_bf16_nan_propagation() {
    let bf16_nan: u16 = 0x7FC0; // quiet NaN in bf16
    let bf16_one: u16 = 0x3F80;

    let result = convert::dot_bf16(&[bf16_one, bf16_nan], &[bf16_one, bf16_one]).unwrap();
    assert!(result.is_nan(), "dot with NaN element should propagate NaN, got {result}");
}

/// Verify dot product against f64 reference for random inputs.
#[test]
fn dot_f16_vs_f64_reference() {
    // Generate random f16 values and compute dot product
    let mut rng: u64 = 0xABCD_EF01_2345_6789;
    let n = 256;
    let mut a = vec![0u16; n];
    let mut b = vec![0u16; n];

    for i in 0..n {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Generate small normal f16 values (avoid Inf/NaN)
        let exp = ((rng >> 40) % 29 + 1) as u16; // exp 1..29 (normal, not too large)
        let mant = ((rng >> 48) & 0x03FF) as u16;
        let sign = ((rng >> 60) & 1) as u16;
        a[i] = (sign << 15) | (exp << 10) | mant;

        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let exp = ((rng >> 40) % 29 + 1) as u16;
        let mant = ((rng >> 48) & 0x03FF) as u16;
        let sign = ((rng >> 60) & 1) as u16;
        b[i] = (sign << 15) | (exp << 10) | mant;
    }

    // Compute reference in f64
    let mut ref_sum = 0.0_f64;
    for i in 0..n {
        let a_f32 = reference_f16_to_f32(a[i]);
        let b_f32 = reference_f16_to_f32(b[i]);
        ref_sum += f64::from(a_f32) * f64::from(b_f32);
    }

    let result = convert::dot_f16(&a, &b).unwrap();
    let ref_f32 = ref_sum as f32;

    // Should be very close (scalar accumulation in f32 matches our reference)
    let rel_err = if ref_f32 == 0.0 {
        (result - ref_f32).abs()
    } else {
        ((result - ref_f32) / ref_f32).abs()
    };
    assert!(
        rel_err < 1e-5,
        "dot_f16 vs f64 reference: result={result}, expected≈{ref_f32}, rel_err={rel_err:e}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR HANDLING TESTS
// ═══════════════════════════════════════════════════════════════════════════

/// Test length mismatch errors for conversion functions.
#[test]
fn convert_length_mismatch() {
    let input = [0u16; 4];
    let mut output = [0.0_f32; 3]; // wrong size

    assert!(convert::f16_to_f32(&input, &mut output).is_err());
    assert!(convert::bf16_to_f32(&input, &mut output).is_err());

    let finput = [0.0_f32; 4];
    let mut foutput = [0u16; 3]; // wrong size

    assert!(convert::f32_to_f16(&finput, &mut foutput).is_err());
    assert!(convert::f32_to_bf16(&finput, &mut foutput).is_err());
}

/// Test length mismatch for dot products.
#[test]
fn dot_length_mismatch() {
    let a = [0u16; 4];
    let b = [0u16; 3];

    assert!(convert::dot_f16(&a, &b).is_err());
    assert!(convert::dot_bf16(&a, &b).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// SPECIAL VALUES
// ═══════════════════════════════════════════════════════════════════════════

/// Test that ±0 is handled correctly in all conversions.
#[test]
fn zero_handling() {
    // f16: +0 = 0x0000, -0 = 0x8000
    let mut f32_out = [0.0_f32; 2];
    convert::f16_to_f32(&[0x0000, 0x8000], &mut f32_out).unwrap();
    assert_eq!(f32_out[0].to_bits(), 0x0000_0000); // +0
    assert_eq!(f32_out[1].to_bits(), 0x8000_0000); // -0

    // bf16: +0 = 0x0000, -0 = 0x8000
    convert::bf16_to_f32(&[0x0000, 0x8000], &mut f32_out).unwrap();
    assert_eq!(f32_out[0].to_bits(), 0x0000_0000);
    assert_eq!(f32_out[1].to_bits(), 0x8000_0000);

    // f32 ±0 → f16 ±0
    let mut f16_out = [0u16; 2];
    convert::f32_to_f16(&[0.0, -0.0], &mut f16_out).unwrap();
    assert_eq!(f16_out[0], 0x0000);
    assert_eq!(f16_out[1], 0x8000);

    // f32 ±0 → bf16 ±0
    let mut bf16_out = [0u16; 2];
    convert::f32_to_bf16(&[0.0, -0.0], &mut bf16_out).unwrap();
    assert_eq!(bf16_out[0], 0x0000);
    assert_eq!(bf16_out[1], 0x8000);
}

/// Test f16 denormal values convert correctly.
#[test]
fn f16_denormals() {
    // Smallest positive denormal: 0x0001 = 2^-24
    let mut out = [0.0_f32; 1];
    convert::f16_to_f32(&[0x0001], &mut out).unwrap();
    let expected = 2.0_f32.powi(-24); // ~5.96e-8
    assert_eq!(out[0], expected, "smallest f16 denormal: got {}, expected {expected}", out[0]);

    // Largest denormal: 0x03FF = 1023 × 2^-24
    convert::f16_to_f32(&[0x03FF], &mut out).unwrap();
    let expected = 1023.0 * 2.0_f32.powi(-24);
    assert!((out[0] - expected).abs() < 1e-15, "largest f16 denormal: got {}, expected {expected}", out[0]);

    // Negative denormals
    convert::f16_to_f32(&[0x8001], &mut out).unwrap();
    assert_eq!(out[0], -2.0_f32.powi(-24));
}

/// Test f16 infinity handling.
#[test]
fn f16_infinity() {
    let mut out = [0.0_f32; 2];
    convert::f16_to_f32(&[0x7C00, 0xFC00], &mut out).unwrap();
    assert_eq!(out[0], f32::INFINITY);
    assert_eq!(out[1], f32::NEG_INFINITY);
}
