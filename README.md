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
| `sum` | 6.1 µs | 146.7 µs | **23.9×** |
| `l2_norm` | 6.9 µs | 147.2 µs | **21.3×** |
| `dot` | 12.5 µs | 147.6 µs | **11.8×** |
| `tanh` | 157.4 µs | 791.9 µs | **5.0×** |
| `softmax` | 148.1 µs | 852.3 µs | **5.8×** |
| `exp` | 100.9 µs | 373.8 µs | **3.7×** |

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
- **`distance`** — `l1_norm`, `l2_norm`, `max_norm`, `squared_distance`
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
| `stats` | `sum` | 6.1 µs | 146.7 µs | **23.9×** |
| `stats` | `prod` | 9.2 µs | 146.9 µs | **15.9×** |
| `stats` | `min` | 11.6 µs | 293.0 µs | **25.4×** |
| `stats` | `max` | 11.6 µs | 293.0 µs | **25.2×** |
| `stats` | `argmax` | 18.4 µs | 147.3 µs | **8.0×** |
| `stats` | `argmin` | 18.6 µs | 147.7 µs | **7.9×** |
| `stats` | `sum_sq` | 6.8 µs | 146.7 µs | **21.6×** |
| `stats` | `mean` | 6.1 µs | 147.0 µs | **24.1×** |
| `stats` | `variance` | 23.9 µs | 293.5 µs | **12.3×** |
| `stats` | `std_dev` | 23.9 µs | 294.2 µs | **12.3×** |
| `stats` | `geometric_mean` | 146.6 µs | 414.8 µs | **2.8×** |
| `stats` | `dot` | 12.5 µs | 147.6 µs | **11.8×** |
| `stats` | `count_zero` | 25.1 µs | 26.0 µs | 1.0× |
| `stats` | `count_nan` | 25.0 µs | 26.1 µs | 1.0× |
| `stats` | `count_infinite` | 26.1 µs | 32.5 µs | 1.2× |
| `distance` | `l1_norm` | 6.9 µs | 148.6 µs | **21.6×** |
| `distance` | `l2_norm` | 6.9 µs | 147.2 µs | **21.3×** |
| `distance` | `max_norm` | 11.6 µs | 289.2 µs | **24.9×** |
| `distance` | `squared_distance` | 12.5 µs | 147.1 µs | **11.8×** |
| `math` | `sqrt` | 34.8 µs | 27.6 µs | 0.8× |
| `math` | `clip` | 19.0 µs | 14.2 µs | 0.7× |
| `math` | `rsqrt` | 58.0 µs | 55.5 µs | 1.0× |
| `math` | `exp` | 100.9 µs | 373.8 µs | **3.7×** |
| `math` | `ln` | 94.8 µs | 457.1 µs | **4.8×** |
| `math` | `tanh` | 157.4 µs | 791.9 µs | **5.0×** |
| `math` | `hypot` | 58.9 µs | 373.8 µs | **6.3×** |
| `math` | `powi` | 19.1 µs | 12.6 µs | 0.7× |
| `math` | `abs_sub` | 24.3 µs | 20.5 µs | 0.8× |
| `ml` | `softmax` | 148.1 µs | 852.3 µs | **5.8×** |
| `ml` | `log_softmax` | 164.5 µs | 697.1 µs | **4.2×** |
| `ml` | `sigmoid` | 121.8 µs | 372.4 µs | **3.1×** |
| `ml` | `silu` | 120.6 µs | 372.8 µs | **3.1×** |
| `ml` | `gelu` | 164.0 µs | 1226.1 µs | **7.5×** |
| `ml` | `relu` | 18.7 µs | 12.7 µs | 0.7× |
| `ml` | `softplus` | 360.6 µs | 1410.6 µs | **3.9×** |
| `ml` | `rms_norm` | 27.8 µs | 160.2 µs | **5.8×** |
| `ml` | `layer_norm` | 36.1 µs | 307.7 µs | **8.5×** |
| `ml` | `cosine_similarity` | 26.3 µs | 442.6 µs | **16.9×** |
| `ml` | `logsumexp` | 135.2 µs | 698.0 µs | **5.2×** |

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
  bound and there's nothing left to win. `lanes` still gives you the
  dispatch/`no_std`/`_into`/error-handling story for these, just not a
  speedup over already-vectorized code.

## Error handling

Fallible kernels return `Result<_, lanes::Error>` instead of panicking:

- two-input ops (`dot`, `squared_distance`, `abs_sub`, `hypot`,
  `cosine_similarity`) → `Err(Error::LengthMismatch { expected, actual })`
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
