//! Criterion benchmarks for `lanes` kernel operations.
//!
//! Run with: `cargo bench --bench kernels`
//!
//! Every algorithm is benchmarked against a naive iterator baseline so the
//! scalar-vs-SIMD speedup is measurable, and against the dispatched `lanes`
//! entry point at sizes spanning cache-resident to memory-bandwidth-bound.
//!
//! To compare backends, set `LANES_BACKEND` (e.g. `scalar`, `sse2`, `avx2`,
//! `avx512`, `neon` on the matching platform):
//!
//! ```sh
//! LANES_BACKEND=scalar cargo bench --bench kernels
//! LANES_BACKEND=avx2   cargo bench --bench kernels
//! LANES_BACKEND=avx512 cargo bench --bench kernels
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Size ladder from cache-resident to memory-bandwidth-bound.
const SIZES: &[usize] = &[16, 32, 64, 128, 256, 1024, 4096, 16_384, 65_536, 1_000_000];

/// xorshift64* RNG: 6 lines, zero deps, deterministic and fast enough for
/// benchmark data generation (uniformity is irrelevant — reproducibility is).
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Deterministic random f32 data for reproducible benchmarks.
fn random_f32_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    (0..n)
        .map(|_| {
            let hi = (rng.next_u64() >> 40) as u32;
            let frac = (hi as f32) * (1.0 / (1u32 << 24) as f32);
            (frac - 0.5) * 2000.0
        })
        .collect()
}

/// Deterministic random f64 data for reproducible benchmarks.
fn random_f64_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = XorShift64::new(seed);
    (0..n)
        .map(|_| {
            let u = rng.next_u64();
            let hi = (u >> 40) as u32;
            let frac = (hi as f64) * (1.0 / (1u32 << 24) as f64);
            (frac - 0.5) * 2000.0
        })
        .collect()
}

/// Naive iterator baselines (independent of `lanes`).
fn naive_sum(values: &[f32]) -> f32 {
    values.iter().sum()
}

fn naive_sum_f64(values: &[f64]) -> f64 {
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

fn naive_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Benchmark a sum-family reduction: `lanes` vs naive, across the size ladder.
fn bench_reduce<T, F, G>(
    c: &mut Criterion,
    name: &str,
    make_data: fn(usize, u64) -> Vec<T>,
    lanes: F,
    naive: G,
) where
    F: Fn(&[T]) -> T + Copy,
    G: Fn(&[T]) -> T + Copy,
{
    let mut group = c.benchmark_group(name);
    for &size in SIZES {
        let data = make_data(size, 42);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &data, |b, data| {
            b.iter(|| lanes(black_box(data)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive(black_box(data)));
        });
    }
    group.finish();
}

/// Benchmark a dot product: `lanes` vs naive, across the size ladder.
fn bench_dot_pair<T, F, G>(
    c: &mut Criterion,
    name: &str,
    make_data: fn(usize, u64) -> Vec<T>,
    lanes: F,
    naive: G,
) where
    F: Fn(&[T], &[T]) -> T + Copy,
    G: Fn(&[T], &[T]) -> T + Copy,
{
    let mut group = c.benchmark_group(name);
    for &size in SIZES {
        let a = make_data(size, 42);
        let b = make_data(size, 123);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    bench_reduce(c, "sum", random_f32_vec, lanes::stats::f32::sum, naive_sum);
}

fn bench_sum_f64(c: &mut Criterion) {
    bench_reduce(
        c,
        "sum_f64",
        random_f64_vec,
        lanes::stats::f64::sum,
        naive_sum_f64,
    );
}

fn bench_dot(c: &mut Criterion) {
    bench_dot_pair(
        c,
        "dot",
        random_f32_vec,
        |a, b| lanes::stats::f32::dot(a, b).unwrap(),
        naive_dot,
    );
}

fn bench_dot_f64(c: &mut Criterion) {
    bench_dot_pair(
        c,
        "dot_f64",
        random_f64_vec,
        |a, b| lanes::stats::f64::dot(a, b).unwrap(),
        naive_dot_f64,
    );
}

fn bench_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("prod");

    for &size in SIZES {
        let data = random_f32_vec(size, 43);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &data, |b, data| {
            b.iter(|| lanes::stats::f32::prod(black_box(data)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive_prod(black_box(data)));
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
            b.iter(|| lanes::stats::f32::min(black_box(data)));
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
            b.iter(|| lanes::stats::f32::max(black_box(data)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive_max(black_box(data)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sum,
    bench_prod,
    bench_dot,
    bench_min,
    bench_max,
    bench_sum_f64,
    bench_dot_f64
);
criterion_main!(benches);
