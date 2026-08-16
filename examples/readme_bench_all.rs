//! Throwaway full-coverage benchmark harness for the README table.
//! Times every public function: lanes vs a naive iterator baseline.
//! Run: cargo run --release --example readme_bench_all
use std::time::Instant;

const N: usize = 65_536;

fn time_us<T, F: FnMut() -> T>(mut f: F) -> f64 {
    f(); // warmup
    let t = Instant::now();
    let mut n = 0;
    while t.elapsed().as_millis() < 250 {
        std::hint::black_box(f());
        n += 1;
    }
    t.elapsed().as_secs_f64() * 1e6 / n as f64
}

fn row(family: &str, name: &str, lanes_us: f64, naive_us: f64) {
    println!(
        "{family}\t{name}\t{lanes_us:.2}\t{naive_us:.2}\t{:.1}",
        naive_us / lanes_us
    );
}

fn main() {
    println!("backend: {:?}", lanes::Backend::detect());
    println!("family\tfunction\tlanes_us\tnaive_us\tspeedup");

    // General data in [-2, 2].
    let a: Vec<f32> = (0..N).map(|i| (i as f32 * 0.001).sin() * 2.0).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32 * 0.0007).cos() * 2.0).collect();
    // Strictly positive data in [0.5, 3.5] for ln / geometric_mean.
    let pos: Vec<f32> = (0..N)
        .map(|i| (i as f32 * 0.0013).sin().abs() * 3.0 + 0.5)
        .collect();
    // Near-1 data in [0.999, 1.001] so `prod` neither over- nor underflows.
    let near1: Vec<f32> = (0..N)
        .map(|i| 1.0 + 0.001 * (i as f32 * 0.001).sin())
        .collect();
    // Packed bitmaps for the binary family.
    let ba: Vec<u8> = (0..N).map(|i| (i * 31) as u8).collect();
    let bb: Vec<u8> = (0..N).map(|i| (i * 17) as u8).collect();

    // ---------------- stats ----------------
    row(
        "stats",
        "sum",
        time_us(|| lanes::stats::f32::sum(&a)),
        time_us(|| a.iter().copied().sum::<f32>()),
    );
    row(
        "stats",
        "prod",
        time_us(|| lanes::stats::f32::prod(&near1)),
        time_us(|| near1.iter().copied().product::<f32>()),
    );
    row(
        "stats",
        "min",
        time_us(|| lanes::stats::f32::min(&a)),
        time_us(|| a.iter().copied().fold(f32::INFINITY, f32::min)),
    );
    row(
        "stats",
        "max",
        time_us(|| lanes::stats::f32::max(&a)),
        time_us(|| a.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
    );
    row(
        "stats",
        "argmax",
        time_us(|| lanes::stats::f32::argmax(&a)),
        time_us(|| {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &x) in a.iter().enumerate() {
                if x > bv {
                    bv = x;
                    best = i;
                }
            }
            best
        }),
    );
    row(
        "stats",
        "argmin",
        time_us(|| lanes::stats::f32::argmin(&a)),
        time_us(|| {
            let mut best = 0usize;
            let mut bv = f32::INFINITY;
            for (i, &x) in a.iter().enumerate() {
                if x < bv {
                    bv = x;
                    best = i;
                }
            }
            best
        }),
    );
    row(
        "stats",
        "sum_sq",
        time_us(|| lanes::stats::f32::sum_sq(&a)),
        time_us(|| a.iter().map(|&x| x * x).sum::<f32>()),
    );
    row(
        "stats",
        "mean",
        time_us(|| lanes::stats::f32::mean(&a)),
        time_us(|| a.iter().copied().sum::<f32>() / a.len() as f32),
    );
    row(
        "stats",
        "variance",
        time_us(|| lanes::stats::f32::variance(&a)),
        time_us(|| {
            let m = a.iter().copied().sum::<f32>() / a.len() as f32;
            a.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / a.len() as f32
        }),
    );
    row(
        "stats",
        "std_dev",
        time_us(|| lanes::stats::f32::std_dev(&a)),
        time_us(|| {
            let m = a.iter().copied().sum::<f32>() / a.len() as f32;
            (a.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / a.len() as f32).sqrt()
        }),
    );
    row(
        "stats",
        "geometric_mean",
        time_us(|| lanes::stats::f32::geometric_mean(&pos).unwrap()),
        time_us(|| (pos.iter().map(|&x| x.ln()).sum::<f32>() / pos.len() as f32).exp()),
    );
    row(
        "stats",
        "dot",
        time_us(|| lanes::stats::f32::dot(&a, &b).unwrap()),
        time_us(|| a.iter().zip(&b).map(|(&x, &y)| x * y).sum::<f32>()),
    );
    row(
        "stats",
        "count_zero",
        time_us(|| lanes::stats::f32::count_zero(&a)),
        time_us(|| a.iter().filter(|&&x| x == 0.0).count()),
    );
    row(
        "stats",
        "count_nan",
        time_us(|| lanes::stats::f32::count_nan(&a)),
        time_us(|| a.iter().filter(|x| x.is_nan()).count()),
    );
    row(
        "stats",
        "count_infinite",
        time_us(|| lanes::stats::f32::count_infinite(&a)),
        time_us(|| a.iter().filter(|x| x.is_infinite()).count()),
    );

    // ---------------- distance ----------------
    row(
        "distance",
        "l1_norm",
        time_us(|| lanes::distance::f32::l1_norm(&a)),
        time_us(|| a.iter().map(|&x| x.abs()).sum::<f32>()),
    );
    row(
        "distance",
        "l2_norm",
        time_us(|| lanes::distance::f32::l2_norm(&a)),
        time_us(|| a.iter().map(|&x| x * x).sum::<f32>().sqrt()),
    );
    row(
        "distance",
        "max_norm",
        time_us(|| lanes::distance::f32::max_norm(&a)),
        time_us(|| a.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max)),
    );
    row(
        "distance",
        "squared_distance",
        time_us(|| lanes::distance::f32::squared_distance(&a, &b).unwrap()),
        time_us(|| {
            a.iter()
                .zip(&b)
                .map(|(&x, &y)| {
                    let d = x - y;
                    d * d
                })
                .sum::<f32>()
        }),
    );
    row(
        "distance",
        "kl_divergence",
        time_us(|| lanes::distance::f32::kl_divergence(&pos, &near1).unwrap()),
        time_us(|| {
            pos.iter()
                .zip(&near1)
                .map(|(&p, &q)| p * (p / q).ln())
                .sum::<f32>()
        }),
    );
    row(
        "distance",
        "js_divergence",
        time_us(|| lanes::distance::f32::js_divergence(&pos, &near1).unwrap()),
        time_us(|| {
            pos.iter()
                .zip(&near1)
                .map(|(&p, &q)| {
                    let m = (p + q) * 0.5;
                    p * (p / m).ln() + q * (q / m).ln()
                })
                .sum::<f32>()
                * 0.5
        }),
    );

    // ---------------- binary ----------------
    row(
        "binary",
        "hamming",
        time_us(|| lanes::binary::hamming(&ba, &bb).unwrap()),
        time_us(|| {
            ba.iter()
                .zip(&bb)
                .map(|(&x, &y)| (x ^ y).count_ones() as usize)
                .sum::<usize>()
        }),
    );
    row(
        "binary",
        "jaccard",
        time_us(|| lanes::binary::jaccard(&ba, &bb).unwrap()),
        time_us(|| {
            let mut inter = 0usize;
            let mut union = 0usize;
            for (&x, &y) in ba.iter().zip(&bb) {
                inter += (x & y).count_ones() as usize;
                union += (x | y).count_ones() as usize;
            }
            (union != 0).then(|| inter as f32 / union as f32)
        }),
    );

    // ---------------- math ----------------
    row(
        "math",
        "sqrt",
        time_us(|| lanes::math::f32::sqrt(&pos)),
        time_us(|| pos.iter().map(|&x| x.sqrt()).collect::<Vec<_>>()),
    );
    row(
        "math",
        "clip",
        time_us(|| lanes::math::f32::clip(&a, -1.0, 1.0).unwrap()),
        time_us(|| a.iter().map(|&x| x.clamp(-1.0, 1.0)).collect::<Vec<_>>()),
    );
    row(
        "math",
        "rsqrt",
        time_us(|| lanes::math::f32::rsqrt(&pos)),
        time_us(|| pos.iter().map(|&x| 1.0 / x.sqrt()).collect::<Vec<_>>()),
    );
    row(
        "math",
        "exp",
        time_us(|| lanes::math::f32::exp(&a)),
        time_us(|| a.iter().map(|&x| x.exp()).collect::<Vec<_>>()),
    );
    row(
        "math",
        "ln",
        time_us(|| lanes::math::f32::ln(&pos)),
        time_us(|| pos.iter().map(|&x| x.ln()).collect::<Vec<_>>()),
    );
    row(
        "math",
        "tanh",
        time_us(|| lanes::math::f32::tanh(&a)),
        time_us(|| a.iter().map(|&x| x.tanh()).collect::<Vec<_>>()),
    );
    row(
        "math",
        "hypot",
        time_us(|| lanes::math::f32::hypot(&a, &b).unwrap()),
        time_us(|| {
            a.iter()
                .zip(&b)
                .map(|(&x, &y)| x.hypot(y))
                .collect::<Vec<_>>()
        }),
    );
    row(
        "math",
        "powi",
        time_us(|| lanes::math::f32::powi(&a, 3)),
        time_us(|| a.iter().map(|&x| x.powi(3)).collect::<Vec<_>>()),
    );
    row(
        "math",
        "abs_sub",
        time_us(|| lanes::math::f32::abs_sub(&a, &b).unwrap()),
        time_us(|| {
            a.iter()
                .zip(&b)
                .map(|(&x, &y)| (x - y).abs())
                .collect::<Vec<_>>()
        }),
    );

    // ---------------- ml ----------------
    row(
        "ml",
        "softmax",
        time_us(|| lanes::ml::f32::softmax(&a)),
        time_us(|| {
            let m = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut out: Vec<f32> = a.iter().map(|&x| (x - m).exp()).collect();
            let s: f32 = out.iter().sum();
            for o in out.iter_mut() {
                *o /= s;
            }
            out
        }),
    );
    row(
        "ml",
        "log_softmax",
        time_us(|| lanes::ml::f32::log_softmax(&a)),
        time_us(|| {
            let m = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = a.iter().map(|&x| (x - m).exp()).sum();
            let lns = s.ln();
            a.iter().map(|&x| x - m - lns).collect::<Vec<_>>()
        }),
    );
    row(
        "ml",
        "sigmoid",
        time_us(|| lanes::ml::f32::sigmoid(&a)),
        time_us(|| {
            a.iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect::<Vec<_>>()
        }),
    );
    row(
        "ml",
        "silu",
        time_us(|| lanes::ml::f32::silu(&a)),
        time_us(|| {
            a.iter()
                .map(|&x| x / (1.0 + (-x).exp()))
                .collect::<Vec<_>>()
        }),
    );
    row(
        "ml",
        "gelu",
        time_us(|| lanes::ml::f32::gelu(&a)),
        time_us(|| {
            let s2pi = (2.0 / std::f32::consts::PI).sqrt();
            a.iter()
                .map(|&x| 0.5 * x * (1.0 + (s2pi * (x + 0.044_715 * x * x * x)).tanh()))
                .collect::<Vec<_>>()
        }),
    );
    row(
        "ml",
        "relu",
        time_us(|| lanes::ml::f32::relu(&a)),
        time_us(|| a.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>()),
    );
    row(
        "ml",
        "softplus",
        time_us(|| lanes::ml::f32::softplus(&a)),
        time_us(|| a.iter().map(|&x| x.exp().ln_1p()).collect::<Vec<_>>()),
    );
    row(
        "ml",
        "rms_norm",
        time_us(|| lanes::ml::f32::rms_norm(&a, 1e-5)),
        time_us(|| {
            let ms = a.iter().map(|&x| x * x).sum::<f32>() / a.len() as f32;
            let r = 1.0 / (ms + 1e-5).sqrt();
            a.iter().map(|&x| x * r).collect::<Vec<_>>()
        }),
    );
    row(
        "ml",
        "layer_norm",
        time_us(|| lanes::ml::f32::layer_norm(&a, 1e-5)),
        time_us(|| {
            let m = a.iter().copied().sum::<f32>() / a.len() as f32;
            let v = a.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / a.len() as f32;
            let r = 1.0 / (v + 1e-5).sqrt();
            a.iter().map(|&x| (x - m) * r).collect::<Vec<_>>()
        }),
    );
    row(
        "ml",
        "cosine_similarity",
        time_us(|| lanes::ml::f32::cosine_similarity(&a, &b).unwrap()),
        time_us(|| {
            let d: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
            let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
            d / (na * nb)
        }),
    );
    row(
        "ml",
        "logsumexp",
        time_us(|| lanes::ml::f32::logsumexp(&a)),
        time_us(|| {
            let m = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            m + a.iter().map(|&x| (x - m).exp()).sum::<f32>().ln()
        }),
    );
}
