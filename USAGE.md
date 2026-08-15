# lanes — Usage Guide

Complete usage documentation for the `lanes` crate: high-performance
numerical kernels with **runtime SIMD dispatch**. Write your code once;
`lanes` picks the best backend for the CPU at runtime (scalar, SSE2, AVX2,
AVX-512F, NEON).

- Crate: [`lanes` on crates.io](https://crates.io/crates/lanes)
- API docs: [docs.rs/lanes](https://docs.rs/lanes)
- Repository: <https://github.com/themankindproject/lanes>
- License: MIT OR Apache-2.0
- MSRV: **Rust 1.89** (stable AVX-512 `target_feature`)

---

## Table of contents

1. [Installation](#installation)
2. [Cargo features](#cargo-features)
3. [Quick start](#quick-start)
4. [API layout: precision-first design](#api-layout-precision-first-design)
5. [API reference](#api-reference)
   - [`stats` — statistical reductions](#stats--statistical-reductions)
   - [`distance` — norms and distances](#distance--norms-and-distances)
   - [`math` — elementwise math](#math--elementwise-math)
   - [`ml` — machine-learning kernels](#ml--machine-learning-kernels)
6. [Error handling](#error-handling)
7. [Backends and dispatch](#backends-and-dispatch)
8. [Floating-point semantics and accuracy](#floating-point-semantics-and-accuracy)
9. [Zero-allocation hot loops: the `_into` pattern](#zero-allocation-hot-loops-the_into-pattern)
10. [`no_std` usage](#no_std-usage)
11. [Runnable examples](#runnable-examples)
12. [Benchmarks](#benchmarks)
13. [Testing](#testing)
14. [Fuzzing](#fuzzing)
15. [Crate architecture](#crate-architecture)
16. [Contributing](#contributing)

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
lanes = "0.1"
```

The crate has **zero dependencies**. The default build enables `std`
(which implies `alloc`); see [Cargo features](#cargo-features) for
`no_std` configurations.

Build requirements:

| Requirement | Value |
|---|---|
| Minimum Rust version | 1.89 (pinned in `rust-toolchain.toml` for development) |
| Edition | 2024 |
| Supported architectures | x86-64 (SSE2/AVX2/AVX-512F), aarch64 (NEON), anything else (scalar) |
| WASM | Compiles and runs on the scalar backend; a SIMD128 backend is a future target |

---

## Cargo features

| Flag | Default | Effect |
|---|---|---|
| `std` | **on** | Runtime CPU detection (`is_x86_feature_detected!`), the `LANES_BACKEND` environment override, and `impl std::error::Error for lanes::Error`. Off = `no_std`. |
| `alloc` | on (via `std`) | Enables the `Vec`-returning families (`math`, `ml`), the `_into` variants, and the `alloc`-gated `stats` functions (`variance`, `std_dev`, `geometric_mean`). |

Feature combinations:

```toml
# Default: std + alloc, runtime CPU detection, full API.
lanes = "0.1"

# no_std with an allocator: static backend selection (SSE2 on x86-64,
# NEON on aarch64, scalar elsewhere), full API.
lanes = { version = "0.1", default-features = false, features = ["alloc"] }

# Bare no_std, no allocator: stats + distance only (minus the three
# alloc-gated stats functions).
lanes = { version = "0.1", default-features = false }
```

---

## Quick start

```rust
use lanes::stats::{f32, f64};

let a = vec![1.0_f32; 1024];
let b = vec![2.0_f32; 1024];

let total = f32::sum(&a);              // 1024.0
let dot_product = f32::dot(&a, &b)?;   // 2048.0
let s64 = f64::sum(&[1.0, 2.0, 3.0]);  // 6.0
# Ok::<(), lanes::Error>(())
```

Which backend am I running on?

```rust
let backend = lanes::Backend::detect();
println!("{backend:?}");   // e.g. Avx512, Avx2, Sse2, Neon, Scalar
```

---

## API layout: precision-first design

Every function comes in **two precisions**, selected by submodule:

```rust
lanes::stats::f32::sum(...)    // single precision
lanes::stats::f64::sum(...)    // double precision
```

The four public families, re-exported from the crate root:

| Module | Contents | Feature gate |
|---|---|---|
| `lanes::stats` | `sum`, `prod`, `min`, `max`, `argmax`, `argmin`, `sum_sq`, `mean`, `variance`, `std_dev`, `geometric_mean`, `dot`, `count_zero`, `count_nan`, `count_infinite` | always (`variance`/`std_dev`/`geometric_mean` need `alloc`) |
| `lanes::distance` | `l1_norm`, `l2_norm`, `max_norm`, `squared_distance` | always (fully `no_std`-clean) |
| `lanes::math` | `sqrt`, `clip`, `rsqrt`, `exp`, `ln`, `tanh`, `hypot`, `powi`, `abs_sub` — each also as `*_into` | `alloc` |
| `lanes::ml` | `softmax`, `log_softmax`, `sigmoid`, `silu`, `gelu`, `relu`, `softplus`, `rms_norm`, `layer_norm`, `cosine_similarity`, `logsumexp` — every map-style op also as `*_into` | `alloc` |

Plus two public items at the crate root:

- `lanes::Backend` — the SIMD backend enum, with `Backend::detect()`.
- `lanes::Error` — the error enum returned by the fallible functions.

The `f32` and `f64` submodules are exact mirrors of each other: same
names, same semantics, only the element type differs. All signatures
below are written with `T` = `f32` or `f64`.

---

## API reference

### `stats` — statistical reductions

Aggregates over a slice. All dispatch to the best available SIMD backend.

| Function | Signature | Empty input | Notes |
|---|---|---|---|
| `sum` | `fn sum(values: &[T]) -> T` | `0.0` | NaN-propagating |
| `prod` | `fn prod(values: &[T]) -> T` | `1.0` | NaN-propagating |
| `min` | `fn min(values: &[T]) -> Option<T>` | `None` | IEEE 754 `minNum`: NaNs ignored unless all inputs are NaN |
| `max` | `fn max(values: &[T]) -> Option<T>` | `None` | IEEE 754 `maxNum`: NaNs ignored unless all inputs are NaN |
| `argmax` | `fn argmax(values: &[T]) -> Option<usize>` | `None` | Index of max; ties → first occurrence; NaN handling follows `max` |
| `argmin` | `fn argmin(values: &[T]) -> Option<usize>` | `None` | Index of min; ties → first occurrence; NaN handling follows `min` |
| `sum_sq` | `fn sum_sq(values: &[T]) -> T` | `0.0` | Sum of squares |
| `mean` | `fn mean(values: &[T]) -> Option<T>` | `None` | Arithmetic mean |
| `variance` | `fn variance(values: &[T]) -> Option<T>` | `None` | **`alloc`-gated.** Population variance, numerically stable two-pass `sum((x−μ)²)/n` |
| `std_dev` | `fn std_dev(values: &[T]) -> Option<T>` | `None` | **`alloc`-gated.** `sqrt(variance(x))` (population) |
| `geometric_mean` | `fn geometric_mean(values: &[T]) -> Result<T, Error>` | `Err(Error::EmptyInput)` | **`alloc`-gated.** `exp(mean(ln(x)))`; `Err(Error::NonPositiveInput { index })` if any value ≤ 0; NaN inputs propagate to a NaN result |
| `dot` | `fn dot(a: &[T], b: &[T]) -> Result<T, Error>` | `Ok(0.0)` for two empty slices | `Err(LengthMismatch)` if lengths differ |
| `count_zero` | `fn count_zero(values: &[T]) -> usize` | `0` | Counts `+0.0` and `-0.0` |
| `count_nan` | `fn count_nan(values: &[T]) -> usize` | `0` | Counts NaN elements |
| `count_infinite` | `fn count_infinite(values: &[T]) -> usize` | `0` | Counts `+inf`/`-inf` elements |

Examples:

```rust
use lanes::stats::f32;

assert_eq!(f32::sum(&[1.0, 2.0, 3.0]), 6.0);
assert_eq!(f32::prod(&[2.0, 3.0, 4.0]), 24.0);
assert_eq!(f32::min(&[3.0, 1.0, 4.0]), Some(1.0));
assert_eq!(f32::max(&[3.0, 1.0, 4.0]), Some(4.0));
assert_eq!(f32::argmax(&[3.0, 1.0, 4.0]), Some(2));
assert_eq!(f32::argmin(&[3.0, 1.0, 4.0]), Some(1));
assert_eq!(f32::sum_sq(&[1.0, 2.0, 3.0]), 14.0);
assert_eq!(f32::mean(&[1.0, 2.0, 3.0]), Some(2.0));
assert_eq!(f32::dot(&[1.0, 2.0], &[3.0, 4.0]), Ok(11.0));
assert_eq!(f32::count_zero(&[0.0, -0.0, 1.0]), 2);
assert_eq!(f32::count_nan(&[f32::NAN, 1.0]), 1);
assert_eq!(f32::count_infinite(&[f32::INFINITY, f32::NEG_INFINITY, 1.0]), 2);

let v = f32::variance(&[1.0, 2.0, 3.0]).unwrap();
assert!((v - 2.0 / 3.0).abs() < 1e-6);

let g = f32::geometric_mean(&[1.0, 4.0, 16.0]).unwrap();
assert!((g - 4.0).abs() < 1e-5);
```

### `distance` — norms and distances

Fully `no_std`-clean (the `sqrt` used by `l2_norm` is the crate's own
std-free kernel).

| Function | Signature | Empty input | Notes |
|---|---|---|---|
| `l1_norm` | `fn l1_norm(values: &[T]) -> T` | `0.0` | Sum of absolute values |
| `l2_norm` | `fn l2_norm(values: &[T]) -> T` | `0.0` | Euclidean norm `sqrt(sum_sq)` |
| `max_norm` | `fn max_norm(values: &[T]) -> Option<T>` | `None` | Max absolute value; **NaN if any input is NaN** (all backends agree) |
| `squared_distance` | `fn squared_distance(a: &[T], b: &[T]) -> Result<T, Error>` | `Ok(0.0)` for two empty slices | `sum((a[i] − b[i])²)`; `Err(LengthMismatch)` if lengths differ |

Examples:

```rust
use lanes::distance::f32;

assert_eq!(f32::l1_norm(&[-3.0, 4.0]), 7.0);

let n = f32::l2_norm(&[3.0, 4.0]);
assert!((n - 5.0).abs() < 1e-6);

assert_eq!(f32::max_norm(&[-3.0, 4.0, -9.0]), Some(9.0));

let d = f32::squared_distance(&[1.0, 2.0], &[4.0, 6.0]);
assert_eq!(d, Ok(25.0));
```

### `math` — elementwise math

Per-element maps. Requires the `alloc` feature. Every function has two
forms:

- `f(values, ...) -> Vec<T>` (or `Result<Vec<T>, Error>` for the
  two-input maps and `clip`) — allocates a new output `Vec` (empty input
  → empty `Vec`).
- `f_into(values, ..., out: &mut [T]) -> Result<(), Error>` —
  allocation-free; writes into a caller-provided buffer. Returns
  `Err(Error::LengthMismatch { expected, actual })` if
  `out.len() != values.len()` (the backend kernels use unchecked writes,
  so the check keeps the safe API sound). An empty slice leaves `out`
  untouched.

Prefer the `_into` forms in hot loops; the allocating forms are thin
wrappers around them. See [the `_into` pattern](#zero-allocation-hot-loops-the_into-pattern).

| Function | Allocating signature | Semantics |
|---|---|---|
| `sqrt` | `fn sqrt(values: &[T]) -> Vec<T>` | IEEE 754: negative/NaN → NaN, `sqrt(±0) = ±0`, `sqrt(inf) = inf`. Correctly rounded (hardware) |
| `clip` | `fn clip(values: &[T], lo: T, hi: T) -> Result<Vec<T>, Error>` | `clamp(x, lo, hi)` per element. NaN values → NaN. `Err(Error::InvalidBounds)` if `lo > hi` or a bound is NaN (mirrors the `f32::clamp` precondition) |
| `rsqrt` | `fn rsqrt(values: &[T]) -> Vec<T>` | `1/sqrt(x)`. NaN/negative → NaN, `rsqrt(±0) = ±inf`, `rsqrt(inf) = 0` |
| `exp` | `fn exp(values: &[T]) -> Vec<T>` | `e^x`. f32 saturates to `0.0` below `x ≈ −104`, `inf` above `x ≈ 88.7`; f64 below `≈ −745.1`, above `≈ 709.8`. NaN propagates. Accuracy ≤ 2 ulp (f32) / ≤ 1 ulp (f64) vs `std` |
| `ln` | `fn ln(values: &[T]) -> Vec<T>` | IEEE 754: `ln(±0) = −inf`, `ln(x < 0) = NaN`, `ln(+inf) = +inf`, `ln(NaN) = NaN`. Accuracy ≤ 1 ulp vs `std` (fdlibm algorithm) |
| `tanh` | `fn tanh(values: &[T]) -> Vec<T>` | `tanh(x) = 1 − 2/(e^(2x) + 1)`. Saturates to ±1 via exp overflow/underflow; NaN propagates. Accuracy follows the `exp` kernel |
| `hypot` | `fn hypot(a: &[T], b: &[T]) -> Result<Vec<T>, Error>` | Overflow-safe `sqrt(a[i]² + b[i]²)` — scales by `max(|a[i]|, |b[i]|)` so large magnitudes don't spuriously overflow. Matches `std::hypot` within 1–2 ulp with identical special values: `hypot(inf, NaN) == inf`. `Err(Error::LengthMismatch)` if `a.len() != b.len()` |
| `powi` | `fn powi(values: &[T], n: i32) -> Vec<T>` | `values[i].powi(n)`, **bit-exact** with `std::powi` on every backend: `powi(x, 0) == 1` for every `x` (including NaN/inf), negative `n` takes the reciprocal, `powi(x, i32::MIN)` is `1 / x^(2^31)` |
| `abs_sub` | `fn abs_sub(a: &[T], b: &[T]) -> Result<Vec<T>, Error>` | `|a[i] − b[i]|`. NaN propagates (`abs` of NaN). `Err(Error::LengthMismatch)` if `a.len() != b.len()` |

`_into` signatures (same for f32/f64) — all return `Result<(), Error>`
(`Err(Error::LengthMismatch)` on a wrong-length `out`):

```rust
fn sqrt_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn clip_into(values: &[T], lo: T, hi: T, out: &mut [T]) -> Result<(), Error>;
fn rsqrt_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn exp_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn ln_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn tanh_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn hypot_into(a: &[T], b: &[T], out: &mut [T]) -> Result<(), Error>;
fn powi_into(values: &[T], n: i32, out: &mut [T]) -> Result<(), Error>;
fn abs_sub_into(a: &[T], b: &[T], out: &mut [T]) -> Result<(), Error>;
```

Examples:

```rust
use lanes::math::f32;

let v = f32::sqrt(&[1.0, 4.0, 9.0]);          // [1.0, 2.0, 3.0]
let c = f32::clip(&[-5.0, 0.5, 3.0, 10.0], -1.0, 2.0).unwrap();
assert_eq!(c, [-1.0, 0.5, 2.0, 2.0]);

let e = f32::exp(&[0.0, 1.0]);
assert!((e[1] - std::f32::consts::E).abs() < 1e-5);

let p = f32::powi(&[2.0, 3.0], 3);
assert_eq!(p, [8.0, 27.0]);

let d = f32::abs_sub(&[1.0, 5.0], &[4.0, 2.0]).unwrap();
assert_eq!(d, [3.0, 3.0]);

let h = f32::hypot(&[3.0], &[4.0]).unwrap();
assert!((h[0] - 5.0).abs() < 1e-6);

// Allocation-free form:
let input = [1.0_f32, 4.0, 16.0];
let mut out = vec![0.0_f32; input.len()];
f32::rsqrt_into(&input, &mut out).unwrap();    // [1.0, 0.5, 0.25]
```

### `ml` — machine-learning kernels

Higher-level ops composed from the core reductions. Requires the `alloc`
feature. Map-style ops follow the same allocating/`_into` split as
`math` (same error contract: `_into` returns `Err(Error::LengthMismatch)`
on a wrong-length output buffer).

| Function | Allocating signature | Semantics |
|---|---|---|
| `softmax` | `fn softmax(values: &[T]) -> Vec<T>` | Numerically stable: `exp(x_i − max(x)) / Σ_j exp(x_j − max(x))`. Max subtraction prevents overflow for large inputs. Outputs sum to 1 |
| `log_softmax` | `fn log_softmax(values: &[T]) -> Vec<T>` | `x_i − logsumexp(x)`, computed via the max-shift. What PyTorch's `nn.LogSoftmax` computes (paired with `nn.NLLLoss` it forms `nn.CrossEntropyLoss`). `exp(log_softmax(x))` sums to 1 |
| `sigmoid` | `fn sigmoid(values: &[T]) -> Vec<T>` | `1 / (1 + exp(−x))`. Outputs in `(0, 1)`, `sigmoid(0) = 0.5`, saturates to 1.0 / 0.0 without overflow |
| `silu` | `fn silu(values: &[T]) -> Vec<T>` | SiLU/Swish: `x / (1 + exp(−x))` — the smooth LLM activation (Llama, Qwen). Saturates to `x` for large positive, 0 for large negative; minimum ≈ −0.278 at x ≈ −1.28 |
| `gelu` | `fn gelu(values: &[T]) -> Vec<T>` | GELU **tanh approximation**: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` — the GPT-2-era production activation, accurate to ~1e-3 of exact GELU |
| `relu` | `fn relu(values: &[T]) -> Vec<T>` | `max(x, 0)` per element |
| `softplus` | `fn softplus(values: &[T]) -> Vec<T>` | `ln(1 + e^x)`, computed with the overflow-free form `max(x, 0) + ln1p(e^−|x|)` — exact to ~1 ulp across the full range, no exp overflow |
| `rms_norm` | `fn rms_norm(values: &[T], eps: T) -> Vec<T>` | `x_i / sqrt(mean(x²) + eps)` — the standard LLM normalization (Llama, Qwen). Typical `eps`: `1e-5` |
| `layer_norm` | `fn layer_norm(values: &[T], eps: T) -> Vec<T>` | `(x_i − mean(x)) / sqrt(variance(x) + eps)` — the standard pre-activation norm (complement to `rms_norm`, which drops the mean). NaNs propagate |
| `cosine_similarity` | `fn cosine_similarity(a: &[T], b: &[T]) -> Result<T, Error>` | `dot(a, b) / (|a|·|b|)`. See error contract below |
| `logsumexp` | `fn logsumexp(values: &[T]) -> T` | Numerically stable `ln(Σ_i exp(x_i))` = `max(x) + ln(Σ_i exp(x_i − max(x)))`. **Empty slice → `-infinity`**. The log-softmax denominator / cross-entropy log-normalizer |

`_into` signatures — all return `Result<(), Error>`
(`Err(Error::LengthMismatch)` on a wrong-length `out`):

```rust
fn softmax_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn log_softmax_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn sigmoid_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn silu_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn gelu_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn relu_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn softplus_into(values: &[T], out: &mut [T]) -> Result<(), Error>;
fn rms_norm_into(values: &[T], eps: T, out: &mut [T]) -> Result<(), Error>;
fn layer_norm_into(values: &[T], eps: T, out: &mut [T]) -> Result<(), Error>;
```

`cosine_similarity` contract:

- `Err(Error::LengthMismatch)` if `a.len() != b.len()`.
- `Err(Error::EmptyInput)` if both slices are empty (the angle between
  empty vectors is undefined).
- `Ok(0.0)` if either vector has zero norm — a zero vector has no
  direction, so it shares none with the other (the scikit-learn
  convention).
- Otherwise the result is in `[-1, 1]` up to rounding.

Examples:

```rust
use lanes::ml::f32;

let v = f32::softmax(&[1.0, 2.0, 3.0]);
let s: f32 = v.iter().sum();
assert!((s - 1.0).abs() < 1e-6);

// Stable even where naive exp overflows:
let big = f32::softmax(&[1000.0, 1000.0, 999.0]);
assert!(big.iter().all(|x| x.is_finite()));

assert!((f32::sigmoid(&[0.0])[0] - 0.5).abs() < 1e-6);
assert_eq!(f32::relu(&[-3.0, 0.0, 5.0]), [0.0, 0.0, 5.0]);

let lse = f32::logsumexp(&[1.0, 2.0, 3.0]);
assert!((lse - 3.407_606).abs() < 1e-5);

let sim = f32::cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
assert_eq!(sim, Ok(1.0));

// RMS norm with the conventional epsilon:
let normed = f32::rms_norm(&[3.0, 4.0], 1e-5);

// Allocation-free softmax in a hot loop:
let logits = [1.0_f32, 2.0, 3.0];
let mut buf = vec![0.0_f32; logits.len()];
f32::softmax_into(&logits, &mut buf).unwrap();
```

---

## Error handling

`lanes::Error` has four variants:

```rust
pub enum Error {
    /// Two input slices had different lengths when they must match.
    LengthMismatch { expected: usize, actual: usize },
    /// An input slice was empty but the operation requires at least one
    /// element (cosine_similarity, geometric_mean).
    EmptyInput,
    /// clip received bounds with lo > hi, or a NaN bound.
    InvalidBounds,
    /// geometric_mean saw a value <= 0 at `index`.
    NonPositiveInput { index: usize },
}
```

- `expected` is always the length of the **first** operand (or the input
  slice for an `_into` call), `actual` the length of the second (or the
  output buffer).
- Implements `Debug`, `Clone`, `PartialEq`, `Eq`, `Display`, and
  `core::error::Error` unconditionally (stable since Rust 1.81, below the
  MSRV) — so `no_std` users get the standard error trait too.

Which functions can fail:

| Function | `LengthMismatch` | `EmptyInput` | `InvalidBounds` | `NonPositiveInput` |
|---|---|---|---|---|
| `stats::dot` | yes | no (two empty slices → `Ok(0.0)`) | — | — |
| `stats::geometric_mean` | — | yes | — | yes (first `≤ 0` index) |
| `distance::squared_distance` | yes | no (two empty slices → `Ok(0.0)`) | — | — |
| `math::abs_sub` | yes | no (two empty slices → `Ok(vec![])`) | — | — |
| `math::hypot` | yes | no (two empty slices → `Ok(vec![])`) | — | — |
| `math::clip` | — | no (empty → `Ok(vec![])`) | yes | — |
| `ml::cosine_similarity` | yes | yes | — | — |
| every `*_into` variant | yes (wrong-length `out`) | no (empty input leaves `out` untouched) | `clip_into` only | — |

Everything else is infallible: functions with no defined value on empty
input return `Option::None` (`min`, `max`, `argmax`, `argmin`,
`max_norm`, `mean`, `variance`, `std_dev`). NaN inputs are never an
error — they propagate to NaN results, matching the crate's reduction
semantics.

```rust
match lanes::stats::f32::dot(&[1.0, 2.0], &[1.0, 2.0, 3.0]) {
    Ok(v) => println!("dot = {v}"),
    Err(lanes::Error::LengthMismatch { expected, actual }) => {
        println!("mismatch: expected {expected}, got {actual}")
    }
    Err(e) => println!("other error: {e}"),
}
```

---

## Backends and dispatch

### The backend enum

```rust
pub enum Backend {
    Scalar,                 // always available, any architecture
    #[cfg(target_arch = "x86_64")] Sse2,     // 128-bit, mandatory on x86-64
    #[cfg(target_arch = "x86_64")] Avx2,     // 256-bit, AVX2 + FMA
    #[cfg(target_arch = "x86_64")] Avx512,   // 512-bit, AVX-512F
    #[cfg(target_arch = "aarch64")] Neon,    // 128-bit, mandatory on ARMv8-A
}
```

Variants are target-dependent: only backends compiled for the current
architecture exist. `Backend` implements `Debug`, `Clone`, `Copy`,
`PartialEq`, `Eq`.

### Selection

`Backend::detect()` selects the best backend and **caches the result in
a `OnceLock`** for the process lifetime — subsequent calls are
essentially free, and every kernel call has zero meaningful dispatch
overhead.

With the `std` feature, detection probes CPU features at runtime:

| Architecture | Hierarchy (best first) | Detection |
|---|---|---|
| x86-64 | AVX-512F → AVX2+FMA → SSE2 → Scalar | `is_x86_feature_detected!("avx512f")`, `("avx2") && ("fma")`, `("sse2")` |
| aarch64 | NEON → Scalar | NEON is mandatory on ARMv8-A, selected unconditionally |
| anything else | Scalar | — |

Without `std` there is no runtime probing, but the architecture baseline
still guarantees a SIMD tier: **SSE2 on x86-64 and NEON on aarch64 are
selected statically** (both are mandatory baselines); all other targets
use scalar.

A backend is only ever invoked after passing a support gate
(`platform::supports`): it must be compiled in **and** actually supported
by the host CPU. A requested-but-unsupported backend is never invoked.

### Forcing a backend: `LANES_BACKEND`

With the `std` feature, set the environment variable `LANES_BACKEND` to
force a backend for benchmarking or debugging:

```sh
LANES_BACKEND=scalar  cargo test
LANES_BACKEND=sse2    cargo bench
LANES_BACKEND=avx2    ./my_binary
LANES_BACKEND=avx512  ./my_binary
LANES_BACKEND=neon    ./my_binary      # aarch64 only
```

Rules:

- Accepted values: `scalar`, `sse2`, `avx2`, `avx512`, `neon`
  (case-insensitive, whitespace-trimmed).
- The request is honoured **only if** the backend is compiled in and
  supported by the CPU; otherwise detection proceeds as usual (silent
  fallback — an unsupported backend is never invoked).
- Unknown values are ignored.
- Read once at first detection and cached; set it before the first
  `lanes` call in the process.
- Not available in `no_std` builds (no environment).

### Inspecting the backend

```rust
fn main() {
    println!("{:?}", lanes::Backend::detect());
}
```

The repo ships a fuller version of this as an example — see
[Runnable examples](#runnable-examples).

---

## Floating-point semantics and accuracy

Documented precisely so results are predictable across backends.

**Reduction order is backend-dependent.** Scalar kernels reduce strictly
left-to-right; SIMD kernels reduce in fixed-width chunks (with four
independent accumulator chains for the add-based reductions) and then
combine the chunk results. For inputs whose intermediate values exceed
exact representation, results may differ in the last ulp between
backends. Do not assume bit-identical results across backends for
arbitrary input; assume determinism *within* a backend for the same
input.

**NaN / special-value semantics** (identical on every backend):

| Function family | Semantics |
|---|---|
| `sum`, `dot` (and `sum_sq`, `l1_norm`, etc.) | Any NaN input → NaN result |
| `min`, `max`, `argmax`, `argmin` | IEEE 754 `minNum`/`maxNum` (matching `f32::min`/`f32::max`): NaNs ignored unless every input is NaN, then the result is NaN (arg* return the first index) |
| `max_norm` | NaN if **any** input is NaN (matches the scalar `total_cmp` reference, where NaN sorts above all) |
| Signed zero | For `min`/`max` inputs containing both `-0.0` and `+0.0` as the extremum, the sign of the result is backend-dependent (the values compare equal; the sign follows the backend's combine order) |

**Accuracy contracts** (enforced by the test suite — see [Testing](#testing)):

| Kernel | Guarantee |
|---|---|
| `exp` | ≤ 2 ulp vs `std::exp` (f32), ≤ 1 ulp (f64); exact IEEE saturation, never NaN for finite out-of-range input |
| `ln` | ≤ 1 ulp vs `std::ln` (fdlibm algorithm) |
| `tanh` | derived from the `exp` kernel; saturates to ±1 |
| `sqrt`, `rsqrt` | correctly rounded (hardware instruction on every SIMD backend; std-free fallback within ~1 ulp in `no_std`) |
| `hypot` | ≤ 1–2 ulp vs `std::hypot`, identical NaN/inf propagation (`hypot(inf, nan) == inf`) |
| `powi` | **bit-exact** vs `std::powi` on every backend |
| `abs_sub` | bit-exact vs `(x - y).abs()` |
| `softplus` | ~1 ulp across the full range (overflow-free formulation) |
| `gelu` | tanh approximation, ~1e-3 of exact GELU (by design) |
| Summation-style reductions | Error bounded by Higham's analysis, `|err| ≤ γ_n · Σ|x_i|` — verified by property tests with a tolerance derived from input magnitudes |

**`no_std` scalar helpers.** In `no_std` builds the crate uses its own
std-free implementations of the transcendental/elementary functions
(`exp` — degree-13 polynomial after f64 range reduction; `ln` — fdlibm;
`sqrt` — Newton iteration; `hypot` — SLEEF-style scale-by-max; `powi` —
compiler-builtins' exponentiation-by-squaring, bit-identical to std).
In `std` builds they delegate to the hardware/libc versions.

---

## Zero-allocation hot loops: the `_into` pattern

Every map-style op in `math` and `ml` has an `_into` variant that writes
into a caller-provided buffer instead of allocating. Allocate once,
reuse the buffer across calls:

```rust
use lanes::{math, ml};

let mut buf = vec![0.0_f32; 1024];

for batch in &batches {
    ml::f32::softmax_into(batch, &mut buf)?;   // no allocation
    math::f32::exp_into(batch, &mut buf)?;     // no allocation
    // ... consume buf ...
}
# Ok::<(), lanes::Error>(())
```

Contract of every `_into` function:

- `out.len()` must equal the input length — a mismatch returns
  `Err(Error::LengthMismatch { expected, actual })` (this check is what
  keeps the unchecked SIMD writes sound).
- An empty input leaves `out` untouched and returns `Ok(())` immediately.
- Two-input ops (`hypot_into`, `abs_sub_into`) require `a.len() ==
  b.len() == out.len()`.
- `clip_into` additionally returns `Err(Error::InvalidBounds)` when
  `lo > hi` or a bound is NaN.

The allocating forms (`softmax`, `exp`, ...) are thin wrappers: they
allocate a zeroed `Vec` of the right length and call the `_into` kernel.

---

## `no_std` usage

`lanes` is `#![no_std]`-clean when built without the `std` feature.
There is no OS-specific code anywhere; WASM currently runs on the
scalar backend and the code is kept free of OS dependencies so a SIMD128
backend can be added later.

| Configuration | Backend selection | Available API |
|---|---|---|
| `default-features = false` | Static: SSE2 (x86-64), NEON (aarch64), scalar (elsewhere) | `stats` + `distance`, minus `variance`/`std_dev`/`geometric_mean` |
| `default-features = false, features = ["alloc"]` | Same static selection | Full API (`math`, `ml`, and the three alloc-gated stats functions included) |
| default (`std`) | Runtime detection + `LANES_BACKEND` | Full API |

Verify a `no_std` build with:

```sh
cargo check --no-default-features
cargo check --no-default-features --features alloc
cargo check --no-default-features --target wasm32-unknown-unknown
```

Notes for `no_std` consumers:

- The crate declares `extern crate alloc` unconditionally; your target
  needs an allocator for the `alloc`-gated API, but the
  `stats`/`distance` core works without one.
- `lanes::Error` implements `Display` but only implements
  `std::error::Error` when `std` is enabled.
- All scalar transcendental fallbacks (exp/ln/sqrt/hypot/powi) are
  written from scratch, dependency-free, and verified against `std` by
  the test suite.

---

## Runnable examples

The repo ships two examples (excluded from the published crate tarball):

```sh
# Basic usage: sum/min/max/dot with timing, detected backend, and the
# LengthMismatch error path.
cargo run --example basic_usage

# Dispatch information: platform details, detected backend, the selection
# hierarchy, live CPU feature flags, and a validation sum.
cargo run --example dispatch_info
```

---

## Benchmarks

Criterion benchmarks live in `benches/kernels.rs` (excluded from the
published tarball). Every algorithm is benchmarked against a naive
iterator baseline at sizes spanning cache-resident to
memory-bandwidth-bound:

```
16, 32, 64, 128, 256, 1024, 4096, 16384, 65536, 1_000_000
```

Benchmarked groups: `sum`, `prod`, `dot`, `min`, `max`, `sum_f64`,
`dot_f64`, `abs_sub`, `hypot`, `powi`, `squared_distance`, `count_zero`.

Run:

```sh
cargo bench --bench kernels
```

Compare backends with the environment override:

```sh
LANES_BACKEND=scalar cargo bench --bench kernels
LANES_BACKEND=avx2   cargo bench --bench kernels
LANES_BACKEND=avx512 cargo bench --bench kernels
```

Benchmark data is generated with a deterministic dependency-free
xorshift64* generator (seeded), so runs are reproducible. HTML reports
land in `target/criterion/`.

---

## Testing

The test suite (excluded from the published tarball) has four layers:

| File | What it verifies |
|---|---|
| `tests/integration.rs` | The public API as an external consumer sees it: identities, empty-input contracts, error paths, activation ranges/symmetries, overflow stability |
| `tests/cross_backend.rs` | Dispatched (SIMD) results exactly match naive references on integer-exact vectors where answers are bit-exact |
| `tests/proptest_kernels.rs` | Property-based tests (`proptest`): random inputs vs naive iterators, with tolerances derived from Higham's summation error analysis on input magnitudes |
| `tests/numerical.rs` | Strict accuracy contracts: `hypot` max-ULP sweeps vs `std` (50k pairs across 2^±60 / 2^±510 magnitudes), `powi` bit-exactness for exponents −40..=40 plus specials, `squared_distance` error bounds vs an f64 reference, `abs_sub` bit-exactness — all from a seeded LCG so failures reproduce |

Plus per-backend unit tests inside `src/kernels/**` and doctests on
every public function.

Run everything:

```sh
cargo test --all-features
```

Pin a specific backend for any test run (the numerical tests are
designed for this):

```sh
LANES_BACKEND=scalar  cargo test --test numerical
LANES_BACKEND=sse2    cargo test --test numerical
LANES_BACKEND=avx2    cargo test --test numerical
LANES_BACKEND=avx512  cargo test --test numerical   # on AVX-512F hardware
```

CI additionally runs: `cargo fmt --check`, clippy with
`-D warnings` (pedantic), doctests as a separate job, the full suite on
the MSRV toolchain (1.89.0), native aarch64 tests on real ARM64
hardware, Miri (UB checking with `-Zmiri-strict-provenance` on the
scalar kernel tests), an llvm-cov coverage upload, and a fuzz smoke
test.

---

## Fuzzing

The `fuzz/` workspace (nightly-only, not part of the published crate)
contains five `cargo-fuzz` targets:

| Target | Coverage |
|---|---|
| `fuzz_reductions` | `sum`, `prod`, `min`, `max`, `sum_sq`, `mean`, `l1_norm`, `l2_norm`, `max_norm` — no panics on NaN/inf/denormals/any length; empty-input identities |
| `fuzz_contracts` | `dot`, `sqrt`, and the ML activations — error contracts, IEEE sqrt contract, relu bit-exactness, sigmoid range, softmax sum |
| `fuzz_exp` | `exp`, `tanh` — ulp bounds vs std on arbitrary input, exact saturation |
| `fuzz_f64` | The double-precision family — no panics, cheap exact properties vs naive references |
| `fuzz_norms` | `rms_norm`, `cosine_similarity` — scale invariance, error contracts, zero-norm convention, output range |

Run (requires nightly and `cargo install cargo-fuzz`):

```sh
cargo +nightly fuzz run --fuzz-dir fuzz fuzz_contracts
cargo +nightly fuzz run --fuzz-dir fuzz fuzz_exp -- -runs=1000000
cargo +nightly fuzz list --fuzz-dir fuzz
```

Corpora live under `fuzz/corpus/<target>/` (gitignored, regenerated by
cargo-fuzz). CI smoke-runs every target for 1000 executions to keep the
harnesses healthy; long fuzzing sessions are a development activity.

---

## Crate architecture

```
public API (lanes::stats / distance / math / ml, f32 + f64 submodules)
    │  safe code only — #![forbid(unsafe_code)] on the algorithms layer
    ▼
algorithm layer (src/algorithms/*)
    │  input validation, Backend::detect() (cached in a OnceLock)
    ▼
kernel dispatch (src/kernels/mod.rs)
    │  one `dispatch!` macro table wiring every op to its five backends
    ▼
backend kernels
    src/kernels/scalar/    portable reference + universal fallback
    src/kernels/x86/sse2.rs, avx2.rs, avx512.rs   (#[cfg(x86_64)])
    src/kernels/aarch64/neon.rs                    (#[cfg(aarch64)])
    src/kernels/{exp,ln,sqrt,hypot,powi}.rs — std-free scalar helpers
```

- All `unsafe` is confined to the kernel layer behind `pub(crate)`
  visibility; the crate forbids `unsafe_op_in_unsafe_fn`, so every
  unsafe operation is an explicit, reviewed block. Each unsafe fn
  documents the invariant that makes it safe (the enclosing
  `#[target_feature]` gate plus the caller's runtime feature check).
- Shared macro skeletons (`src/kernels/macros.rs`) generate the
  chunked-loop structure for every backend, so each backend kernel is a
  short macro invocation rather than hand-written unsafe.
- The dispatch layer is a single declarative table: 66 `dispatch!`
  entries (33 ops × f32/f64) wiring each operation to
  scalar/SSE2/AVX2/AVX-512/NEON.

---

## Contributing

See `CONTRIBUTING.md` for exact recipes (adding a reduction, adding a
SIMD backend, adding a benchmark/property test). Checks before pushing:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo check --no-default-features
cargo check --no-default-features --target wasm32-unknown-unknown
```

The development toolchain is pinned to 1.89.0 (`rust-toolchain.toml`);
formatting is `rustfmt` with `max_width = 100`, edition 2024 style.
Releases are cut by tagging `v*` (the release workflow runs a publish
dry-run; actual `cargo publish` is manual).

---

## License

MIT OR Apache-2.0, at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
