# lanes

[![Crates.io](https://img.shields.io/crates/v/lanes)](https://crates.io/crates/lanes)
[![Documentation](https://docs.rs/lanes/badge.svg)](https://docs.rs/lanes)
[![License](https://img.shields.io/crates/l/lanes)](LICENSE-MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/themankindproject/lanes/ci.yml?branch=main&label=CI)](https://github.com/themankindproject/lanes/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/themankindproject/lanes/branch/main/graph/badge.svg)](https://codecov.io/gh/themankindproject/lanes)
![Crates.io Downloads](https://img.shields.io/crates/d/lanes)
![Rust Version](https://img.shields.io/badge/rust-1.89%2B-blue)

High-performance numerical kernels with **runtime SIMD dispatch**: write
once, `lanes` picks the best backend for the CPU (scalar, SSE2, AVX2,
AVX-512F, NEON). Zero dependencies, `no_std`-friendly, and nothing
panics on bad input — fallible kernels return `Result`.

| Kernel | `lanes` | naive iterator | speedup |
| --- | ---: | ---: | ---: |
| `sum` | 6.2 µs | 147.2 µs | **23.9×** |
| `l2_norm` | 6.9 µs | 147.2 µs | **21.3×** |
| `dot` | 12.4 µs | 146.9 µs | **11.8×** |
| `tanh` | 120.5 µs | 771.5 µs | **6.4×** |
| `softmax` | 110.0 µs | 861.0 µs | **7.8×** |
| `exp` | 65.8 µs | 365.2 µs | **5.6×** |

<sub>f32, n = 65,536, AVX-512F backend on an i5-1135G7, release build.
"naive" is the plain iterator expression compiled with the same
settings; Rust/LLVM does not reassociate floating-point reductions
without fast-math, so the naive `sum`/`dot`/`l2_norm` stay scalar —
the idiomatic baseline `lanes` replaces. Full table for every public
function below. Reproduce with `cargo bench` (Criterion, sizes
16 … 1,000,000) or `cargo run --release --example readme_bench_all`.</sub>

## Install

```toml
[dependencies]
lanes = "0.1"
```

## Usage

Every function comes in `f32` and `f64` via a family submodule:

```rust
use lanes::stats::{f32, f64};

let a = vec![1.0_f32; 1024];
let b = vec![2.0_f32; 1024];

let total = f32::sum(&a);                    // 1024.0
let dot_product = f32::dot(&a, &b).unwrap(); // 2048.0
let s64 = f64::sum(&[1.0, 2.0, 3.0]);        // 6.0
```

The real value is chaining kernels in a hot loop with the
allocation-free `_into` forms — here a transformer-style activation
path (layer norm → GELU → softmax) over one buffer set, reused across
calls:

```rust
use lanes::ml::f32;

let hidden = vec![0.5_f32; 512]; // activations in
let mut normed = vec![0.0_f32; 512];
let mut activated = vec![0.0_f32; 512];
let mut probs = vec![0.0_f32; 512];

f32::layer_norm_into(&hidden, 1e-5, &mut normed).unwrap();
f32::gelu_into(&normed, &mut activated).unwrap();
f32::softmax_into(&activated, &mut probs).unwrap();
// `probs` is a valid distribution — three SIMD kernels, zero per-call
// allocations, one backend picked at startup.
```

## Functions

- **`stats`** — `sum`, `prod`, `min`, `max`, `argmax`, `argmin`, `sum_sq`,
  `mean`, `variance`, `std_dev`, `geometric_mean`, `dot`, `count_zero`,
  `count_nan`, `count_infinite` (the `i8` submodule adds exact integer
  `dot`/`sum`/`sum_sq`/`min`/`max`/`count_zero` with `i64` accumulation)
- **`distance`** — `l1_norm`, `l2_norm`, `max_norm`, `squared_distance`,
  `kl_divergence`, `js_divergence` (the `i8` submodule adds exact
  integer `l1_norm`, `max_norm`, `squared_distance`)
- **`binary`** — `hamming`, `jaccard` (bit-level distances over packed
  `&[u8]` bitmaps: a slice of `n` bytes is a binary vector of `8n`
  dimensions)
- **`math`** — `sqrt`, `clip`, `rsqrt`, `exp`, `ln`, `tanh`, `hypot`,
  `powi`, `abs_sub` (each also as `*_into`)
- **`special`** — `erf`, `erfc` (f64: ≤ 1 ulp / ≤ 3 ulp; f32: perfectly
  rounded via compute-in-f64-and-round-once)
- **`ml`** — `softmax`, `log_softmax`, `sigmoid`, `silu`, `gelu`, `relu`,
  `softplus`, `rms_norm`, `layer_norm`, `cosine_similarity`, `logsumexp`
  (every map-style op also as `*_into`)
- **`sort`** — `bitonic_sort` for small power-of-two slices (8/16/32): optimal sorting networks (19/60/185 compare-exchanges) dispatching per backend with deterministic `total_cmp` (NaN last, `-0 < +0`); other lengths fall back to `sort_unstable_by(total_cmp)` — `no_std`-clean, in-place, branch-free
- **`convert`** — `f16_to_f32`, `f32_to_f16`, `bf16_to_f32`, `f32_to_bf16`,
  `dot_f16`, `dot_bf16` (half-precision conversions with
  round-to-nearest-even; `no_std`-compatible via caller-provided buffers;
  f16 uses F16C hardware on `x86_64` when present, bf16 is vectorized
  integer shifts on every SIMD tier)

Reductions (`stats`, `distance`) return scalars and mostly work without
`alloc` (exceptions: `variance`, `std_dev`, `geometric_mean` need an
internal buffer); `math`/`special`/`ml` return `Vec`s and need the
`alloc` feature. The `_into` variants write into a caller-provided
buffer instead of allocating — reuse the buffer across calls in hot
loops:

```rust
let mut buf = vec![0.0_f32; 1024];
lanes::ml::f32::softmax_into(&a, &mut buf).unwrap();   // no allocation
lanes::math::f32::exp_into(&a, &mut buf).unwrap();     // no allocation
```

## Benchmarks (every public function)

Measured on the same machine as the summary table (f32, n = 65,536,
AVX-512F backend, release build). "naive" is the plain iterator
expression compiled with identical settings. Reproduce with
`cargo run --release --example readme_bench_all`.

Note: `erf`/`erfc` have no `std` baseline, so their naive column is the
Abramowitz–Stegun 7.1.26 polynomial — a *speed* reference only, not the
same accuracy class as the `lanes` kernels.

- **Speed.** They run below that baseline here because the shared bench
  distribution (arcsine on [−2, 2]) is tail-heavy for erf, and every
  tail element pays two correctly-rounded vector `exp`s to hold the
  accuracy contract. On small/mid-heavy inputs the kernels run at SIMD
  speed (measured 2–3× faster than the baseline there).
- **Accuracy.** The baseline is far less accurate: measured max absolute
  error 5.7e-7 over 2.6M points (worse than its advertised 1.5e-7 bound,
  because the bound assumes exact arithmetic), and up to ~1.3M ulp near
  zero where its ~1e-9 error floor dominates the shrinking true value.
  The `lanes` kernels are ≤ 1 ulp (f64 `erf`) / perfectly rounded (f32).

| Family | Function | `lanes` | naive | speedup |
| --- | --- | ---: | ---: | ---: |
| `stats` | `sum` | 6.2 µs | 147.3 µs | **23.9×** |
| `stats` | `prod` | 9.3 µs | 147.2 µs | **15.9×** |
| `stats` | `min` | 11.7 µs | 294.9 µs | **25.3×** |
| `stats` | `max` | 11.7 µs | 294.8 µs | **25.3×** |
| `stats` | `argmax` | 18.6 µs | 148.0 µs | **8.0×** |
| `stats` | `argmin` | 18.5 µs | 147.8 µs | **8.0×** |
| `stats` | `sum_sq` | 6.8 µs | 147.3 µs | **21.5×** |
| `stats` | `mean` | 6.2 µs | 147.3 µs | **23.8×** |
| `stats` | `variance` | 26.8 µs | 294.4 µs | **11.0×** |
| `stats` | `std_dev` | 27.4 µs | 295.1 µs | **10.8×** |
| `stats` | `geometric_mean` | 119.3 µs | 459.3 µs | **3.8×** |
| `stats` | `dot` | 12.7 µs | 147.8 µs | **11.7×** |
| `stats` | `count_zero` | 25.2 µs | 26.4 µs | 1.0× |
| `stats` | `count_nan` | 25.2 µs | 26.4 µs | 1.0× |
| `stats` | `count_infinite` | 26.0 µs | 33.7 µs | 1.3× |
| `distance` | `l1_norm` | 6.8 µs | 146.9 µs | **21.6×** |
| `distance` | `l2_norm` | 6.8 µs | 146.9 µs | **21.5×** |
| `distance` | `max_norm` | 11.6 µs | 290.0 µs | **25.1×** |
| `distance` | `squared_distance` | 12.5 µs | 147.1 µs | **11.8×** |
| `distance` | `kl_divergence` | 81.9 µs | 551.8 µs | **6.7×** |
| `distance` | `js_divergence` | 179.8 µs | 1334.3 µs | **7.4×** |
| `binary` | `hamming` | 4.7 µs | 122.2 µs | **25.8×** |
| `binary` | `jaccard` | 8.7 µs | 239.0 µs | **27.6×** |
| `stats::i8` | `dot` | 5.3 µs | 45.7 µs | **8.6×** |
| `stats::i8` | `sum` | 4.7 µs | 22.1 µs | **4.7×** |
| `stats::i8` | `sum_sq` | 4.9 µs | 41.4 µs | **8.5×** |
| `stats::i8` | `min` | 1.4 µs | 5.5 µs | **4.0×** |
| `stats::i8` | `max` | 1.4 µs | 5.3 µs | **3.8×** |
| `stats::i8` | `count_zero` | 8.8 µs | 48.5 µs | **5.5×** |
| `distance::i8` | `l1_norm` | 4.9 µs | 53.7 µs | **10.9×** |
| `distance::i8` | `max_norm` | 2.7 µs | 3.6 µs | 1.3× |
| `distance::i8` | `squared_distance` | 6.7 µs | 70.3 µs | **10.5×** |
| `math` | `sqrt` | 27.7 µs | 27.6 µs | 1.0× |
| `math` | `clip` | 11.7 µs | 14.0 µs | 1.2× |
| `math` | `rsqrt` | 30.0 µs | 55.3 µs | **1.8×** |
| `math` | `exp` | 65.8 µs | 365.8 µs | **5.6×** |
| `math` | `ln` | 64.2 µs | 476.4 µs | **7.4×** |
| `math` | `tanh` | 121.2 µs | 809.7 µs | **6.7×** |
| `special` | `erf` | 807.5 µs | 504.4 µs | 0.6× |
| `special` | `erfc` | 793.4 µs | 506.1 µs | 0.6× |
| `math` | `hypot` | 50.8 µs | 372.5 µs | **7.3×** |
| `math` | `powi` | 11.5 µs | 12.6 µs | 1.1× |
| `math` | `abs_sub` | 16.8 µs | 21.0 µs | 1.2× |
| `ml` | `softmax` | 111.3 µs | 870.3 µs | **7.8×** |
| `ml` | `log_softmax` | 117.1 µs | 766.1 µs | **6.5×** |
| `ml` | `sigmoid` | 81.6 µs | 395.2 µs | **4.8×** |
| `ml` | `silu` | 80.8 µs | 377.5 µs | **4.7×** |
| `ml` | `gelu` | 113.4 µs | 1199.2 µs | **10.6×** |
| `ml` | `relu` | 11.5 µs | 13.1 µs | 1.1× |
| `ml` | `softplus` | 250.1 µs | 1459.6 µs | **5.8×** |
| `ml` | `rms_norm` | 20.3 µs | 162.0 µs | **8.0×** |
| `ml` | `layer_norm` | 28.3 µs | 310.9 µs | **11.0×** |
| `ml` | `cosine_similarity` | 26.1 µs | 441.2 µs | **16.9×** |
| `ml` | `logsumexp` | 102.6 µs | 696.8 µs | **6.8×** |
| `sort` | `bitonic_sort` (n=32, scalar) | ~0.4 µs | ~0.4 µs | ~1.0× (parity; optimal networks 8:19/16:60/32:185 COEX; SIMD min/max+shuffle stages queued) |

**Reading the table honestly.** Two distinct regimes:

- **Reductions and transcendentals win big (3–25×).** Floating-point
  reductions (`sum`, `dot`, norms, `min`/`max`) can't be auto-vectorized
  by LLVM without fast-math (reassociation changes the result), so the
  naive baseline stays scalar. Compute-heavy maps (`exp`, `ln`, `tanh`,
  `gelu`, `softmax`, `hypot`) are dominated by the transcendental, which
  `lanes` evaluates with vectorized polynomial approximations.
- **Trivial elementwise ops are ~1× (`relu`, `clip`, `abs_sub`, `powi`,
  `sqrt`, `rsqrt`).** These need no reassociation, so the compiler
  auto-vectorizes the naive baseline too — both are memory-bandwidth
  bound and there's nothing left to win. `lanes` matches or slightly
  beats the naive baseline here (1.0–1.3×); the allocating wrappers
  build their output buffer without a zero-fill (the kernel writes every
  element), so they pay only one store pass. For hot loops, prefer the
  `_into` variants with a reused buffer to skip the allocation entirely.
  `lanes` still gives you the dispatch/`no_std`/`_into`/error-handling
  story for these, just not a large speedup over already-vectorized
  code.

## Error handling

Fallible kernels return `Result<_, lanes::Error>` instead of panicking:

- two-input ops (`dot`, `squared_distance`, `abs_sub`, `hypot`,
  `cosine_similarity`, `kl_divergence`, `js_divergence`, `hamming`,
  `jaccard`) →
  `Err(Error::LengthMismatch { expected, actual })`
  on unequal operand lengths
- every `*_into` variant → `Err(Error::LengthMismatch { .. })` when the
  output buffer has the wrong length
- `geometric_mean` → `Err(Error::EmptyInput)` on an empty slice,
  `Err(Error::NonPositiveInput { index })` when a value is ≤ 0 (NaN
  inputs propagate to a NaN result instead)
- `clip` → `Err(Error::InvalidBounds)` when `lo > hi` or a bound is NaN

Infallible kernels (reductions like `sum`, single-input maps like `exp`)
never fail.

## Backends

| Architecture | Backend | Selection |
| --- | --- | --- |
| x86-64 | AVX-512F (+ VPOPCNTDQ/VNNI when present) | runtime detection (`avx512f`); sub-features probed via `Avx512Caps` |
| x86-64 | AVX2 + FMA (+ F16C for half) | runtime detection (`avx2` + `fma`); half converts use F16C (`cvtph`) when `f16c` present |
| x86-64 | SSE2 | runtime detection; mandatory x86-64 baseline |
| aarch64 | NEON | mandatory on ARMv8-A |
| `wasm32` | WASM | scalar fallthrough now; `wasm32-unknown-unknown` wires the WASM backend (SIMD128 intrinsics are a drop-in next step) |
| anything else | Scalar | always available |

Detection runs once and is cached in a `OnceLock`, so dispatch overhead
after the first call is a single load. Every kernel has a portable
scalar fallback, and the unsafe SIMD code is isolated in the kernel
layer behind `platform::supports` gates (the algorithm layer is
`#![forbid(unsafe_code)]`).

`LANES_BACKEND=scalar|sse2|avx2|avx512|neon|wasm` forces a backend for
benchmarking or debugging; `cargo run --example dispatch_info` prints
what was detected and why.

## `no_std`

```toml
[dependencies]
lanes = { version = "0.1", default-features = false }
```

Without `std`, the architecture-guaranteed SIMD tier is selected
statically — SSE2 on x86-64, NEON on aarch64 (both mandatory baselines,
so no runtime probing), scalar elsewhere. The `stats` and `distance`
families work as-is (except `variance`/`std_dev`/`geometric_mean`,
which need `alloc`); enable `alloc` for the `Vec`-returning
`math`/`special`/`ml` families:

```rust
use lanes::stats::f32 as stats;

let total = stats::sum(&data);                  // no std needed
let norm = lanes::distance::f32::l2_norm(&data);
```

## Accuracy

- `exp`, `ln`, `sqrt`, `rsqrt` — ≤ 1 ulp vs `std`
  (fdlibm/SLEEF/musl-derived reductions), on every backend
- `tanh` — ≤ 2 ulp (derived from the `exp` kernel)
- `erf` — ≤ 1 ulp (f64), perfectly rounded (f32); `erfc` — ≤ 3 ulp
  (f64, the structural floor of the exp-product tail form), perfectly
  rounded (f32). Clean-room Remez coefficients fitted against an
  arbitrary-precision oracle
- `hypot` — overflow-safe (scales by `max(|a|, |b|)`), 1–2 ulp vs
  `std::hypot`, identical NaN/inf propagation
- `powi` — bit-exact with `std::powi`, including specials
- Reduction order is backend-dependent: results are deterministic
  *within* a backend but may differ in the last ulp *across* backends
- NaN rules are uniform across backends: `sum`/`dot` propagate NaN;
  `min`/`max` follow IEEE 754 `minNum`/`maxNum` (NaN ignored unless all
  inputs are NaN); `max_norm` returns NaN if any input is NaN

## Features

| Flag | Default | Effect |
| --- | --- | --- |
| `std` | on | Runtime CPU detection, `LANES_BACKEND` override. Off = `no_std`: the architecture baseline is picked statically (SSE2 on x86-64, NEON on aarch64, scalar elsewhere). |
| `alloc` | on (via `std`) | `Vec`-returning families (`math`, `ml`). |

## MSRV

**1.89** (stable AVX-512 `target_feature`).

## Development

```sh
cargo test --all-features                              # 380 tests
cargo clippy --all-features --all-targets -- -D warnings
cargo bench                                            # Criterion, 16 … 1M
cargo run --example basic_usage
cargo run --example dispatch_info
cargo run --release --example readme_bench_all         # README table numbers
```

Fuzz targets live in `fuzz/` (nightly + `cargo-fuzz`, not in CI). CI
runs fmt, clippy, tests, doctests, MSRV, Miri, a fuzz smoke run, native
aarch64, and llvm-cov coverage on every push and PR. See
[USAGE.md](USAGE.md) for the in-depth guide (per-kernel semantics,
specials, dispatch internals, release process).

## License

MIT OR Apache-2.0, at your option.
