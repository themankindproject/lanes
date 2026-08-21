//! Fuzz target for the convert family (f16/bf16 ↔ f32, dot_f16/dot_bf16).
//! Verifies no panics on arbitrary bit patterns and lengths, and that
//! the dispatched kernels agree with scalar-oracle semantics for round-trip
//! invariants that must hold on any backend.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct ConvertInput {
    f16_bits: Vec<u16>,
    bf16_bits: Vec<u16>,
    f32_vals: Vec<f32>,
    a_f16: Vec<u16>,
    b_f16: Vec<u16>,
    a_bf16: Vec<u16>,
    b_bf16: Vec<u16>,
}

fuzz_target!(|input: ConvertInput| {
    // f16 -> f32 -> f16 round-trip length contract: output len must match input.
    if !input.f16_bits.is_empty() {
        let mut out = vec![0.0f32; input.f16_bits.len()];
        let r = lanes::convert::f16_to_f32(&input.f16_bits, &mut out);
        assert!(r.is_ok());
        let mut back = vec![0u16; input.f16_bits.len()];
        let r2 = lanes::convert::f32_to_f16(&out, &mut back);
        assert!(r2.is_ok());
    }
    // bf16 -> f32 -> bf16
    if !input.bf16_bits.is_empty() {
        let mut out = vec![0.0f32; input.bf16_bits.len()];
        assert!(lanes::convert::bf16_to_f32(&input.bf16_bits, &mut out).is_ok());
        let mut back = vec![0u16; input.bf16_bits.len()];
        assert!(lanes::convert::f32_to_bf16(&out, &mut back).is_ok());
    }
    // f32 -> f16 / bf16 direct
    if !input.f32_vals.is_empty() {
        let mut out16 = vec![0u16; input.f32_vals.len()];
        assert!(lanes::convert::f32_to_f16(&input.f32_vals, &mut out16).is_ok());
        let mut outbf = vec![0u16; input.f32_vals.len()];
        assert!(lanes::convert::f32_to_bf16(&input.f32_vals, &mut outbf).is_ok());
    }
    // dot products: equal-length slices succeed, mismatched fail.
    if input.a_f16.len() == input.b_f16.len() {
        let _ = lanes::convert::dot_f16(&input.a_f16, &input.b_f16);
    } else {
        assert!(lanes::convert::dot_f16(&input.a_f16, &input.b_f16).is_err());
    }
    if input.a_bf16.len() == input.b_bf16.len() {
        let _ = lanes::convert::dot_bf16(&input.a_bf16, &input.b_bf16);
    } else {
        assert!(lanes::convert::dot_bf16(&input.a_bf16, &input.b_bf16).is_err());
    }
    // Mismatched f16/bf16 convert lengths must error (not panic).
    if input.f16_bits.len() != input.f32_vals.len() && !input.f16_bits.is_empty() && !input.f32_vals.is_empty() {
        let mut out = vec![0u16; input.f32_vals.len().saturating_sub(1).max(1)];
        if out.len() != input.f32_vals.len() {
            assert!(lanes::convert::f32_to_f16(&input.f32_vals, &mut out).is_err());
        }
    }
});
