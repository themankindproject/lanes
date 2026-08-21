//! Convert example: f16/bf16 round-trips and mixed-precision dot.

fn main() {
    println!("=== lanes convert ===\n");
    let backend = lanes::Backend::detect();
    println!("backend: {:?} ({})\n", backend, backend.as_str());

    // Known f16 bit patterns (1.0 = 0x3C00, 0.5 = 0x3800)
    let f16_vals: Vec<u16> = vec![0x3C00, 0x3800, 0x4000, 0x4200]; // 1, 0.5, 2, 3
    let mut f32_buf = vec![0.0f32; f16_vals.len()];
    lanes::convert::f16_to_f32(&f16_vals, &mut f32_buf).unwrap();
    println!("f16 {f16_vals:04X?} -> f32 {f32_buf:?}");

    let mut back = vec![0u16; f32_buf.len()];
    lanes::convert::f32_to_f16(&f32_buf, &mut back).unwrap();
    println!("f32 {f32_buf:?} -> f16 {back:04X?} (round-trip)\n");

    // bf16 round-trip
    let f32_vals = vec![1.0f32, 0.5, 2.0, 3.0, f32::NAN, f32::INFINITY];
    let mut bf16_buf = vec![0u16; f32_vals.len()];
    lanes::convert::f32_to_bf16(&f32_vals, &mut bf16_buf).unwrap();
    let mut f32_back = vec![0.0f32; f32_vals.len()];
    lanes::convert::bf16_to_f32(&bf16_buf, &mut f32_back).unwrap();
    println!("f32 {f32_vals:?}");
    println!(" -> bf16 {bf16_buf:04X?}");
    println!(" -> f32 {f32_back:?}\n");

    // Mixed-precision dot (note bf16 encodings differ from f16: bf16 1.0 = 0x3F80, 2.0 = 0x4000, 3.0 = 0x4040)
    let a_f16: Vec<u16> = vec![0x3C00, 0x4000]; // f16 [1, 2]
    let b_f16: Vec<u16> = vec![0x4200, 0x3C00]; // f16 [3, 1]
    let d16 = lanes::convert::dot_f16(&a_f16, &b_f16).unwrap();
    println!("dot_f16([1,2],[3,1])  = {d16} (expect 5)");
    assert!((d16 - 5.0).abs() < 1e-5);
    let a_bf: Vec<u16> = vec![0x3F80, 0x4000]; // bf16 [1, 2]
    let b_bf: Vec<u16> = vec![0x4040, 0x3F80]; // bf16 [3, 1]
    let d_bf = lanes::convert::dot_bf16(&a_bf, &b_bf).unwrap();
    println!("dot_bf16([1,2],[3,1]) = {d_bf} (expect 5)");
    assert!((d_bf - 5.0).abs() < 1e-5);
    println!("\n✓ convert example passed");
}
