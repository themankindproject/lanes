//! Dispatch information example for the `lanes` crate.
//!
//! Shows which SIMD backend was selected and platform details.
//!
//! Run with: `cargo run --example dispatch_info`

fn main() {
    println!("=== lanes dispatch information ===\n");

    println!("Platform:");
    println!("  arch:   {}", std::env::consts::ARCH);
    println!("  os:     {}", std::env::consts::OS);
    println!("  family: {}", std::env::consts::FAMILY);
    println!();

    let backend = lanes::Backend::detect();
    println!("Detected SIMD backend: {:?}", backend);
    println!();

    println!("Selection logic:");
    println!("  lanes probes CPU features at startup and selects the fastest");
    println!("  available instruction set. The result is cached in a OnceLock");
    println!("  so subsequent calls have zero dispatch overhead.\n");

    #[cfg(target_arch = "x86_64")]
    {
        println!("  x86_64 hierarchy (best to worst):");
        println!("    1. AVX-512F — 512-bit vectors (Zen 4, Ice Lake+)");
        println!("    2. AVX2 + FMA — 256-bit vectors (Haswell+)");
        println!("    3. SSE2 — 128-bit vectors (mandatory on x86-64)");
        println!("    4. Scalar — portable fallback");
        println!();
        println!("  CPU feature detection via `is_x86_feature_detected!`:");
        println!("    avx512f: {}", is_x86_feature_detected!("avx512f"));
        println!("    avx2:    {}", is_x86_feature_detected!("avx2"));
        println!("    fma:     {}", is_x86_feature_detected!("fma"));
        println!("    sse2:    {}", is_x86_feature_detected!("sse2"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("  aarch64 hierarchy:");
        println!("    1. NEON — 128-bit vectors (mandatory on all ARMv8-A)");
        println!("    2. Scalar — portable fallback");
        println!();
        println!("  NEON is always available on aarch64, so the NEON backend");
        println!("  is unconditionally selected.");
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("  This architecture has no specialized SIMD backend.");
        println!("  The scalar (portable) backend is used.");
    }

    println!();

    let test = vec![1.0_f32; 16];
    let result = lanes::stats::f32::sum(&test);
    println!("Validation: sum([1.0; 16]) = {result} (expected 16.0)");

    if (result - 16.0).abs() < f32::EPSILON {
        println!("  ✓ Backend is working correctly.");
    } else {
        println!("  ✗ Unexpected result!");
    }
}
