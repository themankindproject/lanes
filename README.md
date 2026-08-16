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
| `sum` | 6.2 µs | 146.6 µs | **23.8×** |
| `l2_norm` | 6.9 µs | 147.3 µs | **21.4×** |
| `dot` | 12.3 µs | 146.9 µs | **11.9×** |
| `tanh` | 146.7 µs | 774.4 µs | **5.3×** |
| `softmax` | 140.6 µs | 845.6 µs | **6.0×** |
| `exp` | 95.1 µs | 355.9 µs | **3.7×** |

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

## Functions

- **`stats`** — `sum`, `prod`, `min`, `max`, `argmax`, `argmin`, `sum_sq`,
  `mean`, `variance`, `std_dev`, `geometric_mean`, `dot`, `count_zero`,
  `count_nan`, `count_infinite`
- **`distance`** — `l1_norm`, `l2_norm`, `max_norm`, `squared_distance`,
  `kl_divergence`, `js_divergence`
- **`math`** — `sqrt`, `clip`, `rsqrt`, `exp`, `ln`, `tanh`, `hypot`,
  `powi`, `abs_sub` (each also as `*_into`)
- **`ml`** — `softmax`, `log_softmax`, `sigmoid`, `silu`, `gelu`, `relu`,
  `softplus`, `rms_norm`, `layer_norm`, `cosine_similarity`, `logsumexp`
  (every map-style op also as `*_into`)

Reductions (`stats`, `distance`) return scalars and mostly work without
`alloc` (exceptions: `variance`, `std_dev`, `geometric_mean` need an
internal buffer); `math`/`ml` return `Vec`s and need the `alloc`
feature. The `_into` variants write into a caller-provided buffer
instead of allocating — reuse the buffer across calls in hot loops:

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

| Family | Function | `lanes` | naive | speedup |
| --- | --- | ---: | ---: | ---: |
| `stats` | `sum` | 6.2 µs | 146.6 µs | **23.8×** |
| `stats` | `prod` | 9.3 µs | 146.8 µs | **15.9×** |
| `stats` | `min` | 11.6 µs | 295.5 µs | **25.4×** |
| `stats` | `max` | 11.6 µs | 292.9 µs | **25.2×** |
| `stats` | `argmax` | 18.8 µs | 147.4 µs | **7.8×** |
| `stats` | `argmin` | 18.2 µs | 148.7 µs | **8.2×** |
| `stats` | `sum_sq` | 7.2 µs | 147.8 µs | **20.4×** |
| `stats` | `mean` | 6.4 µs | 147.7 µs | **23.0×** |
| `stats` | `variance` | 25.4 µs | 295.7 µs | **11.6×** |
| `stats` | `std_dev` | 24.0 µs | 293.8 µs | **12.3×** |
| `stats` | `geometric_mean` | 142.3 µs | 425.1 µs | **3.0×** |
| `stats` | `dot` | 12.3 µs | 146.9 µs | **11.9×** |
| `stats` | `count_zero` | 25.1 µs | 26.9 µs | 1.1× |
| `stats` | `count_nan` | 25.1 µs | 26.2 µs | 1.0× |
| `stats` | `count_infinite` | 26.1 µs | 33.5 µs | 1.3× |
| `distance` | `l1_norm` | 6.9 µs | 147.4 µs | **21.5×** |
| `distance` | `l2_norm` | 6.9 µs | 147.3 µs | **21.4×** |
| `distance` | `max_norm` | 11.6 µs | 289.5 µs | **25.0×** |
| `distance` | `squared_distance` | 12.5 µs | 147.0 µs | **11.8×** |
| `distance` | `kl_divergence` | 104.4 µs | 538.9 µs | **5.2×** |
| `distance` | `js_divergence` | 223.5 µs | 1307.1 µs | **5.8×** |
| `math` | `sqrt` | 27.6 µs | 27.6 µs | 1.0× |
| `math` | `clip` | 11.3 µs | 14.0 µs | 1.2× |
| `math` | `rsqrt` | 50.5 µs | 55.0 µs | 1.1× |
| `math` | `exp` | 95.1 µs | 355.9 µs | **3.7×** |
| `math` | `ln` | 87.7 µs | 452.4 µs | **5.2×** |
| `math` | `tanh` | 146.7 µs | 774.4 µs | **5.3×** |
| `math` | `hypot` | 50.5 µs | 367.2 µs | **7.3×** |
| `math` | `powi` | 11.3 µs | 12.7 µs | 1.1× |
| `math` | `abs_sub` | 16.6 µs | 20.4 µs | 1.2× |
| `ml` | `softmax` | 140.6 µs | 845.6 µs | **6.0×** |
| `ml` | `log_softmax` | 143.5 µs | 690.8 µs | **4.8×** |
| `ml` | `sigmoid` | 114.6 µs | 372.5 µs | **3.3×** |
| `ml` | `silu` | 113.2 µs | 372.4 µs | **3.3×** |
| `ml` | `gelu` | 155.0 µs | 1140.9 µs | **7.4×** |
| `ml` | `relu` | 10.9 µs | 12.5 µs | 1.2× |
| `ml` | `softplus` | 350.8 µs | 1492.2 µs | **4.3×** |
| `ml` | `rms_norm` | 20.1 µs | 159.4 µs | **7.9×** |
| `ml` | `layer_norm` | 28.1 µs | 307.0 µs | **10.9×** |
| `ml` | `cosine_similarity` | 25.9 µs | 439.4 µs | **17.0×** |
| `ml` | `logsumexp` | 134.8 µs | 707.1 µs | **5.2×** |

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
  `cosine_similarity`, `kl_divergence`, `js_divergence`) →
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
| x86-64 | AVX-512F | runtime detection (`avx512f`) |
| x86-64 | AVX2 + FMA | runtime detection (`avx2` + `fma`) |
| x86-64 | SSE2 | runtime detection; mandatory x86-64 baseline |
| aarch64 | NEON | mandatory on ARMv8-A |
| anything else | Scalar | always available |

Detection runs once and is cached in a `OnceLock`, so dispatch overhead
after the first call is a single load. Every kernel has a portable
scalar fallback, and the unsafe SIMD code is isolated in the kernel
layer behind `platform::supports` gates (the algorithm layer is
`#![forbid(unsafe_code)]`).

`LANES_BACKEND=scalar|sse2|avx2|avx512|neon` forces a backend for
benchmarking or debugging; `cargo run --example dispatch_info` prints
what was detected and why. WASM currently uses the scalar backend; the
code is kept OS-independent so a SIMD128 backend can be added later.

## `no_std`

```toml
[dependencies]
lanes = { version = "0.1", default-features = false }
```

Without `std`, the architecture-guaranteed SIMD tier is selected
statically — SSE2 on x86-64, NEON on aarch64 (both mandatory baselines,
so no runtime probing), scalar elsewhere. The `stats` and `distance`
families work as-is (except `variance`/`std_dev`/`geometric_mean`,
which need `alloc`); enable `alloc` for the `Vec`-returning `math`/`ml`
families:

```rust
use lanes::stats::f32 as stats;

let total = stats::sum(&data);                  // no std needed
let norm = lanes::distance::f32::l2_norm(&data);
```

## Accuracy

- `exp`, `ln`, `tanh`, `sqrt`, `rsqrt` — ≤ 1 ulp vs `std`
  (fdlibm/SLEEF/musl-derived reductions), on every backend
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
