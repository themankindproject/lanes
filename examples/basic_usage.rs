//! Basic usage example for the `lanes` crate.
//!
//! Demonstrates sum, min, max, and dot product operations
//! with timing information.
//!
//! Run with: `cargo run --example basic_usage`

use std::time::Instant;

fn main() {
    println!("=== lanes basic usage ===\n");

    // Print detected backend.
    let backend = lanes::Backend::detect();
    println!("Detected backend: {:?}\n", backend);

    let data: Vec<f32> = (1..=10_000).map(|x| x as f32).collect();

    let start = Instant::now();
    let total = lanes::stats::f32::sum(&data);
    let elapsed = start.elapsed();
    println!("sum of 1..=10000: {total}");
    println!("  elapsed: {elapsed:?}\n");

    // Deliberately approximate constants to exercise generic float data.
    #[allow(clippy::approx_constant)]
    let values = vec![3.14_f32, 2.71, 1.41, 1.73, 2.23, 0.577];

    let start = Instant::now();
    let minimum = lanes::stats::f32::min(&values);
    let maximum = lanes::stats::f32::max(&values);
    let elapsed = start.elapsed();
    println!("values: {values:?}");
    println!("  min: {minimum:?}");
    println!("  max: {maximum:?}");
    println!("  elapsed: {elapsed:?}\n");

    let n = 100_000;
    let a = vec![2.0_f32; n];
    let b = vec![3.0_f32; n];

    let start = Instant::now();
    let product = lanes::stats::f32::dot(&a, &b).expect("lengths match");
    let elapsed = start.elapsed();
    println!("dot([2.0; {n}], [3.0; {n}]): {product}");
    println!("  elapsed: {elapsed:?}\n");

    let short = [1.0_f32, 2.0];
    let long = [1.0_f32, 2.0, 3.0];
    match lanes::stats::f32::dot(&short, &long) {
        Ok(v) => println!("unexpected success: {v}"),
        Err(e) => println!("Expected error for mismatched lengths: {e}"),
    }
}
