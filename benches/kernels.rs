//! Criterion benchmarks for `lanes` kernel operations.
//!
//! Run with: `cargo bench --bench kernels`
//!
//! Every algorithm is benchmarked against a naive iterator baseline so the
//! scalar-vs-SIMD speedup is measurable, and against the dispatched `lanes`
//! entry point at sizes spanning cache-resident to memory-bandwidth-bound.
//!
//! To compare backends, set `LANES_BACKEND` (see docs/benchmarking.md):
//!
//! ```sh
//! LANES_BACKEND=scalar cargo bench --bench kernels
//! LANES_BACKEND=avx2   cargo bench --bench kernels
//! LANES_BACKEND=avx512 cargo bench --bench kernels
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Size ladder from cache-resident to memory-bandwidth-bound.
const SIZES: &[usize] = &[16, 32, 64, 128, 256, 1024, 4096, 16_384, 65_536, 1_000_000];

/// Deterministic random f32 data for reproducible benchmarks.
fn random_f32_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-1000.0..1000.0)).collect()
}

/// Naive iterator baselines (independent of `lanes`).
fn naive_sum(values: &[f32]) -> f32 {
    values.iter().sum()
}

fn naive_prod(values: &[f32]) -> f32 {
    values.iter().product()
}

fn naive_min(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::INFINITY, f32::min)
}

fn naive_max(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for &size in SIZES {
        let data = random_f32_vec(size, 42);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &data, |b, data| {
            b.iter(|| lanes::sum(black_box(data)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive_sum(black_box(data)));
        });
    }

    group.finish();
}

fn bench_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("prod");

    for &size in SIZES {
        let data = random_f32_vec(size, 43);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &data, |b, data| {
            b.iter(|| lanes::prod(black_box(data)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive_prod(black_box(data)));
        });
    }

    group.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot");

    for &size in SIZES {
        let a = random_f32_vec(size, 42);
        let b = random_f32_vec(size, 123);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::dot(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive_dot(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

fn bench_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("min");

    for &size in SIZES {
        let data = random_f32_vec(size, 42);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &data, |b, data| {
            b.iter(|| lanes::min(black_box(data)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive_min(black_box(data)));
        });
    }

    group.finish();
}

fn bench_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("max");

    for &size in SIZES {
        let data = random_f32_vec(size, 42);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &data, |b, data| {
            b.iter(|| lanes::max(black_box(data)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive_max(black_box(data)));
        });
    }

    group.finish();
}

criterion_group!(
    benches, bench_sum, bench_prod, bench_dot, bench_min, bench_max
);
criterion_main!(benches);
