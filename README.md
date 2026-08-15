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
AVX-512F, NEON).

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

let total = f32::sum(&a);              // 1024.0
let dot_product = f32::dot(&a, &b)?;   // 2048.0
let s64 = f64::sum(&[1.0, 2.0, 3.0]);  // 6.0
# Ok::<(), lanes::Error>(())
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

All reduce to `f32`/`f64` via the `lanes::stats::f32::*`-style paths;
`math`/`ml` return `Vec`s and need the `alloc` feature. The `_into`
variants write into a caller-provided buffer instead of allocating —
reuse the buffer across calls in hot loops:

```rust
let mut buf = vec![0.0_f32; 1024];
lanes::ml::f32::softmax_into(&a, &mut buf)?;   // no allocation
lanes::math::f32::exp_into(&a, &mut buf)?;     // no allocation
# Ok::<(), lanes::Error>(())
```

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

## Features

| Flag | Default | Effect |
| --- | --- | --- |
| `std` | on | Runtime CPU detection, `LANES_BACKEND` override. Off = `no_std`: the architecture baseline is picked statically (SSE2 on x86-64, NEON on aarch64, scalar elsewhere). |
| `alloc` | on (via `std`) | `Vec`-returning families (`math`, `ml`). |

`LANES_BACKEND=scalar|sse2|avx2|avx512|neon` forces a backend for
benchmarking or debugging.

## MSRV

**1.89** (stable AVX-512 `target_feature`).

## License

MIT OR Apache-2.0, at your option.
