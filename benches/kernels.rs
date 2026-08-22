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

fn naive_abs_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).collect()
}

fn naive_hypot(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x.hypot(y)).collect()
}

fn naive_powi(values: &[f32], n: i32) -> Vec<f32> {
    values.iter().map(|&x| x.powi(n)).collect()
}

fn naive_squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn naive_count_zero(values: &[f32]) -> usize {
    values.iter().filter(|&&x| x == 0.0).count()
}

fn naive_kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter().zip(q).map(|(&a, &b)| a * (a / b).ln()).sum()
}

fn naive_js_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q)
        .map(|(&a, &b)| {
            let m = (a + b) * 0.5;
            a * (a / m).ln() + b * (b / m).ln()
        })
        .sum::<f32>()
        * 0.5
}

fn naive_kl_divergence_f64(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&a, &b)| a * (a / b).ln()).sum()
}

fn naive_js_divergence_f64(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q)
        .map(|(&a, &b)| {
            let m = (a + b) * 0.5;
            a * (a / m).ln() + b * (b / m).ln()
        })
        .sum::<f64>()
        * 0.5
}

/// Deterministic random probability-distribution data: strictly positive
/// values (divergences take `ln(p/q)`), reproducible per seed.
fn random_distribution_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    (0..n)
        .map(|_| {
            let hi = (rng.next_u64() >> 40) as u32;
            let frac = (hi as f32) * (1.0 / (1u32 << 24) as f32);
            frac + 1e-3
        })
        .collect()
}

/// `f64` twin of [`random_distribution_f32`].
fn random_distribution_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = XorShift64::new(seed);
    (0..n)
        .map(|_| {
            let u = rng.next_u64();
            let hi = (u >> 40) as u32;
            let frac = (hi as f64) * (1.0 / (1u32 << 24) as f64);
            frac + 1e-6
        })
        .collect()
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

/// Benchmark a two-input elementwise map: `lanes` vs naive, across the size
/// ladder.
fn bench_map2<T, F, G>(
    c: &mut Criterion,
    name: &str,
    make_data: fn(usize, u64) -> Vec<T>,
    lanes: F,
    naive: G,
) where
    F: Fn(&[T], &[T]) -> Vec<T> + Copy,
    G: Fn(&[T], &[T]) -> Vec<T> + Copy,
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

fn bench_abs_sub(c: &mut Criterion) {
    bench_map2(
        c,
        "abs_sub",
        random_f32_vec,
        |a, b| lanes::math::f32::abs_sub(a, b).unwrap(),
        naive_abs_sub,
    );
}

fn bench_hypot(c: &mut Criterion) {
    bench_map2(
        c,
        "hypot",
        random_f32_vec,
        |a, b| lanes::math::f32::hypot(a, b).unwrap(),
        naive_hypot,
    );
}

fn bench_powi(c: &mut Criterion) {
    let mut group = c.benchmark_group("powi");
    for &size in SIZES {
        let data = random_f32_vec(size, 44);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &data, |b, data| {
            b.iter(|| lanes::math::f32::powi(black_box(data), black_box(3)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive_powi(black_box(data), 3));
        });
    }
    group.finish();
}

fn bench_squared_distance(c: &mut Criterion) {
    bench_dot_pair(
        c,
        "squared_distance",
        random_f32_vec,
        |a, b| lanes::distance::f32::squared_distance(a, b).unwrap(),
        naive_squared_distance,
    );
}

fn bench_kl_divergence(c: &mut Criterion) {
    bench_dot_pair(
        c,
        "kl_divergence",
        random_distribution_f32,
        |a, b| lanes::distance::f32::kl_divergence(a, b).unwrap(),
        naive_kl_divergence,
    );
}

fn bench_js_divergence(c: &mut Criterion) {
    bench_dot_pair(
        c,
        "js_divergence",
        random_distribution_f32,
        |a, b| lanes::distance::f32::js_divergence(a, b).unwrap(),
        naive_js_divergence,
    );
}

fn bench_kl_divergence_f64(c: &mut Criterion) {
    bench_dot_pair(
        c,
        "kl_divergence_f64",
        random_distribution_f64,
        |a, b| lanes::distance::f64::kl_divergence(a, b).unwrap(),
        naive_kl_divergence_f64,
    );
}

fn bench_js_divergence_f64(c: &mut Criterion) {
    bench_dot_pair(
        c,
        "js_divergence_f64",
        random_distribution_f64,
        |a, b| lanes::distance::f64::js_divergence(a, b).unwrap(),
        naive_js_divergence_f64,
    );
}

fn bench_count_zero(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_zero");
    for &size in SIZES {
        let data = random_f32_vec(size, 45);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &data, |b, data| {
            b.iter(|| lanes::stats::f32::count_zero(black_box(data)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| naive_count_zero(black_box(data)));
        });
    }
    group.finish();
}

/// Deterministic random byte data (packed bitmaps) for the binary family.
fn random_u8_vec(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = XorShift64::new(seed);
    (0..n).map(|_| rng.next_u64() as u8).collect()
}

fn naive_hamming(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones() as usize)
        .sum()
}

fn naive_jaccard(a: &[u8], b: &[u8]) -> Option<f32> {
    let mut inter = 0usize;
    let mut union = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        inter += (x & y).count_ones() as usize;
        union += (x | y).count_ones() as usize;
    }
    (union != 0).then(|| inter as f32 / union as f32)
}

fn bench_binary_pair<R, F, G>(c: &mut Criterion, name: &str, lanes: F, naive: G)
where
    R: Copy,
    F: Fn(&[u8], &[u8]) -> R + Copy,
    G: Fn(&[u8], &[u8]) -> R + Copy,
{
    let mut group = c.benchmark_group(name);
    for &size in SIZES {
        let a = random_u8_vec(size, 46);
        let b = random_u8_vec(size, 124);
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

fn bench_hamming(c: &mut Criterion) {
    bench_binary_pair(
        c,
        "hamming",
        |a, b| lanes::binary::hamming(a, b).unwrap(),
        naive_hamming,
    );
}

fn bench_jaccard(c: &mut Criterion) {
    bench_binary_pair(
        c,
        "jaccard",
        |a, b| lanes::binary::jaccard(a, b).unwrap(),
        naive_jaccard,
    );
}

/// Deterministic random i8 data for the i8 family.
fn random_i8_vec(n: usize, seed: u64) -> Vec<i8> {
    let mut rng = XorShift64::new(seed);
    (0..n).map(|_| rng.next_u64() as i8).collect()
}

fn naive_dot_i8(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

fn naive_sum_i8(v: &[i8]) -> i64 {
    v.iter().map(|&x| i64::from(x)).sum()
}

fn bench_dot_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_i8");
    for &size in SIZES {
        let a = random_i8_vec(size, 47);
        let b = random_i8_vec(size, 125);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::stats::i8::dot(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive_dot_i8(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_sum_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_i8");
    for &size in SIZES {
        let v = random_i8_vec(size, 48);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::stats::i8::sum(black_box(&v)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive_sum_i8(black_box(&v)));
        });
    }
    group.finish();
}

fn bench_sum_sq_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_sq_i8");
    for &size in SIZES {
        let v = random_i8_vec(size, 49);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::stats::i8::sum_sq(black_box(&v)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(&v)
                    .iter()
                    .map(|&x| i64::from(x) * i64::from(x))
                    .sum::<i64>()
            });
        });
    }
    group.finish();
}

fn bench_min_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_i8");
    for &size in SIZES {
        let v = random_i8_vec(size, 50);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::stats::i8::min(black_box(&v)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| black_box(&v).iter().copied().min());
        });
    }
    group.finish();
}

fn bench_max_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_i8");
    for &size in SIZES {
        let v = random_i8_vec(size, 51);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::stats::i8::max(black_box(&v)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| black_box(&v).iter().copied().max());
        });
    }
    group.finish();
}

fn bench_count_zero_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_zero_i8");
    for &size in SIZES {
        let v = random_i8_vec(size, 52);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::stats::i8::count_zero(black_box(&v)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| black_box(&v).iter().filter(|&&x| x == 0).count());
        });
    }
    group.finish();
}

fn bench_l1_norm_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("l1_norm_i8");
    for &size in SIZES {
        let v = random_i8_vec(size, 53);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::distance::i8::l1_norm(black_box(&v)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(&v)
                    .iter()
                    .map(|&x| i64::from(x.unsigned_abs()))
                    .sum::<i64>()
            });
        });
    }
    group.finish();
}

fn bench_max_norm_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_norm_i8");
    for &size in SIZES {
        let v = random_i8_vec(size, 54);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::distance::i8::max_norm(black_box(&v)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| black_box(&v).iter().map(|&x| x.unsigned_abs()).max());
        });
    }
    group.finish();
}

fn bench_squared_distance_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("squared_distance_i8");
    for &size in SIZES {
        let a = random_i8_vec(size, 55);
        let b = random_i8_vec(size, 56);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::distance::i8::squared_distance(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(&a)
                    .iter()
                    .zip(black_box(&b).iter())
                    .map(|(&x, &y)| {
                        let d = i64::from(x) - i64::from(y);
                        d * d
                    })
                    .sum::<i64>()
            });
        });
    }
    group.finish();
}

fn random_f16_vec(n: usize, seed: u64) -> Vec<u16> {
    let mut rng = XorShift64::new(seed);
    (0..n).map(|_| (rng.next_u64() >> 48) as u16).collect()
}

fn random_bf16_vec(n: usize, seed: u64) -> Vec<u16> {
    let mut rng = XorShift64::new(seed);
    (0..n).map(|_| (rng.next_u64() >> 48) as u16).collect()
}

fn reference_f16_to_f32_scalar(bits: u16) -> f32 {
    // tiny scalar reference for naive bench
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 && mant == 0 {
        f32::from_bits(sign << 31)
    } else if exp == 31 && mant == 0 {
        if sign == 0 {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        }
    } else if exp == 31 {
        f32::NAN
    } else if exp == 0 {
        let v = (mant as f64) * 2.0_f64.powi(-24);
        if sign == 0 { v as f32 } else { -(v as f32) }
    } else {
        let v = 2.0_f64.powi(exp as i32 - 15) * (1.0 + (mant as f64) / 1024.0);
        if sign == 0 { v as f32 } else { -(v as f32) }
    }
}

fn bench_f16_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("f16_to_f32");
    for &size in SIZES {
        let v = random_f16_vec(size, 60);
        let out = vec![0.0f32; size];
        let out2 = vec![0.0f32; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |b, _| {
            b.iter(|| {
                lanes::convert::f16_to_f32(black_box(&v), black_box(&mut out.clone())).unwrap();
                black_box(&out);
            });
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| {
                let mut o = out2.clone();
                for (i, &bits) in v.iter().enumerate() {
                    o[i] = reference_f16_to_f32_scalar(bits);
                }
                black_box(o);
            });
        });
    }
    group.finish();
}

fn bench_f32_to_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_to_f16");
    for &size in SIZES {
        let v = random_f32_vec(size, 61);
        let out = vec![0u16; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |b, _| {
            b.iter(|| {
                lanes::convert::f32_to_f16(black_box(&v), black_box(&mut out.clone())).unwrap();
                black_box(&out);
            });
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| {
                let mut o = vec![0u16; size];
                for (i, &x) in v.iter().enumerate() {
                    let bits = x.to_bits();
                    let sign = (bits >> 31) as u16;
                    let exp = ((bits >> 23) & 0xFF) as i32;
                    let mant = bits & 0x007F_FFFF;
                    o[i] = if exp == 255 && mant != 0 {
                        (sign << 15) | 0x7E00
                    } else {
                        (x.to_bits() >> 16) as u16
                    };
                }
                black_box(o);
            });
        });
    }
    group.finish();
}

fn bench_bf16_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf16_to_f32");
    for &size in SIZES {
        let v = random_bf16_vec(size, 62);
        let out = vec![0.0f32; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |b, _| {
            b.iter(|| {
                lanes::convert::bf16_to_f32(black_box(&v), black_box(&mut out.clone())).unwrap();
                black_box(&out);
            });
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| {
                let mut o = vec![0.0f32; size];
                for (i, &bits) in v.iter().enumerate() {
                    o[i] = f32::from_bits((u32::from(bits)) << 16);
                }
                black_box(o);
            });
        });
    }
    group.finish();
}

fn bench_f32_to_bf16(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_to_bf16");
    for &size in SIZES {
        let v = random_f32_vec(size, 63);
        let out = vec![0u16; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |b, _| {
            b.iter(|| {
                lanes::convert::f32_to_bf16(black_box(&v), black_box(&mut out.clone())).unwrap();
                black_box(&out);
            });
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| {
                let mut o = vec![0u16; size];
                for (i, &x) in v.iter().enumerate() {
                    let bits = x.to_bits();
                    let bias = ((bits >> 16) & 1) + 0x7FFF;
                    o[i] = ((bits + bias) >> 16) as u16;
                }
                black_box(o);
            });
        });
    }
    group.finish();
}

fn bench_dot_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f16");
    for &size in SIZES {
        let a = random_f16_vec(size, 64);
        let b = random_f16_vec(size, 65);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::convert::dot_f16(black_box(&a), black_box(&b)).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| {
                let mut s = 0.0f32;
                for (&x, &y) in a.iter().zip(&b) {
                    s += reference_f16_to_f32_scalar(x) * reference_f16_to_f32_scalar(y);
                }
                black_box(s)
            });
        });
    }
    group.finish();
}

fn bench_dot_bf16(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_bf16");
    for &size in SIZES {
        let a = random_bf16_vec(size, 66);
        let b = random_bf16_vec(size, 67);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &size, |bench, _| {
            bench.iter(|| lanes::convert::dot_bf16(black_box(&a), black_box(&b)).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| {
                let mut s = 0.0f32;
                for (&x, &y) in a.iter().zip(&b) {
                    s +=
                        f32::from_bits((u32::from(x)) << 16) * f32::from_bits((u32::from(y)) << 16);
                }
                black_box(s)
            });
        });
    }
    group.finish();
}

fn bench_bitonic_sort_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitonic_sort_f32");
    for size in [8usize, 16, 32] {
        let base = random_f32_vec(size, 70);
        let mut buf = base.clone();
        group.throughput(criterion::Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &base, |b, base| {
            b.iter(|| {
                buf.copy_from_slice(black_box(base));
                lanes::sort::f32::bitonic_sort(black_box(&mut buf));
                black_box(&buf);
            });
        });
        let mut buf2 = base.clone();
        group.bench_with_input(BenchmarkId::new("naive", size), &base, |b, base| {
            b.iter(|| {
                buf2.copy_from_slice(black_box(base));
                buf2.sort_unstable_by(|a, b| a.total_cmp(b));
                black_box(&buf2);
            });
        });
    }
    group.finish();
}

fn bench_bitonic_sort_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitonic_sort_f64");
    for size in [8usize, 16, 32] {
        let base = random_f64_vec(size, 71);
        let mut buf = base.clone();
        group.throughput(criterion::Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("lanes", size), &base, |b, base| {
            b.iter(|| {
                buf.copy_from_slice(black_box(base));
                lanes::sort::f64::bitonic_sort(black_box(&mut buf));
                black_box(&buf);
            });
        });
        let mut buf2 = base.clone();
        group.bench_with_input(BenchmarkId::new("naive", size), &base, |b, base| {
            b.iter(|| {
                buf2.copy_from_slice(black_box(base));
                buf2.sort_unstable_by(|a, b| a.total_cmp(b));
                black_box(&buf2);
            });
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
    bench_dot_f64,
    bench_abs_sub,
    bench_hypot,
    bench_powi,
    bench_squared_distance,
    bench_kl_divergence,
    bench_js_divergence,
    bench_kl_divergence_f64,
    bench_js_divergence_f64,
    bench_count_zero,
    bench_hamming,
    bench_jaccard,
    bench_dot_i8,
    bench_sum_i8,
    bench_sum_sq_i8,
    bench_min_i8,
    bench_max_i8,
    bench_count_zero_i8,
    bench_l1_norm_i8,
    bench_max_norm_i8,
    bench_squared_distance_i8,
    bench_f16_to_f32,
    bench_f32_to_f16,
    bench_bf16_to_f32,
    bench_f32_to_bf16,
    bench_dot_f16,
    bench_dot_bf16,
    bench_bitonic_sort_f32,
    bench_bitonic_sort_f64
);
criterion_main!(benches);
