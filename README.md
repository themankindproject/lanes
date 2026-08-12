# lanes

[![Crates.io](https://img.shields.io/crates/v/lanes)](https://crates.io/crates/lanes)
[![Documentation](https://docs.rs/lanes/badge.svg)](https://docs.rs/lanes)
[![License](https://img.shields.io/crates/l/lanes)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/themankindproject/lanes/ci.yml?branch=main&label=CI)](https://github.com/themankindproject/lanes/actions/workflows/ci.yml)
![Crates.io Downloads](https://img.shields.io/crates/d/lanes)
![Rust Version](https://img.shields.io/badge/rust-1.89%2B-blue)

High-performance computational algorithm kernels with **runtime SIMD dispatch**.

`lanes` provides a small, deliberate set of optimized numerical kernels
that automatically select the best available SIMD instruction set at
runtime. Write your code once; `lanes` picks the backend.

## Kernels

| Function | Description | Backends |
| --- | --- | --- |
| `lanes::sum(&[f32]) -> f32` | Sum, `0.0` for empty input | scalar, SSE2, AVX2, AVX-512F, NEON |
| `lanes::prod(&[f32]) -> f32` | Product, `1.0` for empty input | scalar, SSE2, AVX2, AVX-512F, NEON |
| `lanes::min(&[f32]) -> Option<f32>` | Minimum (`None` for empty) | scalar, SSE2, AVX2, AVX-512F, NEON |
| `lanes::max(&[f32]) -> Option<f32>` | Maximum (`None` for empty) | scalar, SSE2, AVX2, AVX-512F, NEON |
| `lanes::dot(&[f32], &[f32]) -> Result<f32, Error>` | Dot product, length-checked | scalar, SSE2, AVX2+FMA, AVX-512F, NEON |

The scope is intentionally narrow — **few, fast, correct kernels**, not a
zoo of wrappers. The architecture (see
[docs/architecture.md](docs/architecture.md)) is built so new algorithm
families (stats, signal, compression, …) slot in without API churn.

## How dispatch works

```text
public fn sum(&[f32])
      │   input validation
      ▼
Backend::detect()      ── once per process (OnceLock), probes cpuid/auxv,
      │                    honors LANES_BACKEND diagnostic override
      ▼
dispatch_sum(backend)  ── one match on a Copy enum
      ▼
kernel                 ── scalar │ sse2 │ avx2 │ avx512 │ neon
```

- The decision is made **once** and cached; per-call overhead is one atomic
  load plus one branch.
- Every SIMD kernel is gated by a runtime feature check
  (`platform::supports`) before it is ever called — a kernel is never
  executed on hardware that lacks the instruction set.
- Scalar is the reference implementation and is always available.
- `LANES_BACKEND=scalar|sse2|avx2|avx512|neon` forces a backend for
  benchmarking or debugging (honored only when the CPU actually supports it).

## Supported platforms

| Architecture | Backend | Selection |
| --- | --- | --- |
| `x86_64` | AVX-512F (512-bit) | runtime detection (`avx512f`) |
| `x86_64` | AVX2 + FMA (256-bit) | runtime detection (`avx2` + `fma`) |
| `x86_64` | SSE2 (128-bit) | mandatory on x86-64; runtime detection (`sse2`) |
| `aarch64` | NEON (128-bit) | mandatory on ARMv8-A |
| any | Scalar | always available |
| `wasm32` | Scalar (future: SIMD128) | compiled in `no_std` mode; CI-checked |

## Floating-point semantics (read before trusting bit-level results)

- **Reduction order is backend-dependent.** Scalar reduces left-to-right;
  SIMD kernels reduce in fixed-width chunks then combine. Results may
  differ in the last ulp for inputs outside the exactly-representable range.
- **`sum`/`dot` propagate NaN.** Any NaN input yields a NaN result.
- **`min`/`max` differ on NaN input between backends.** Scalar uses IEEE 754
  `minNum`/`maxNum` (`f32::min`/`f32::max`: NaN inputs are ignored unless
  everything is NaN). SIMD kernels follow the hardware instruction, which
  propagates a NaN present in the data. NaN-free inputs agree exactly.
- No backend performs unsafe reordering optimizations; nothing is "fast math".

## Installation

```toml
[dependencies]
lanes = "0.1"
```

```rust
use lanes::{dot, sum};

let a = vec![1.0_f32; 1024];
let b = vec![2.0_f32; 1024];

let dot_product = dot(&a, &b)?;
let total = sum(&a);
# Ok::<(), lanes::Error>(())
```

## Feature flags

| Flag | Default | Effect |
| --- | --- | --- |
| `std` | on | Runtime CPU detection, `LANES_BACKEND`, `std::error::Error`. Off = `no_std` (scalar only). |
| `alloc` | on (via `std`) | `Vec`-based convenience paths. |

## Building & testing

```sh
cargo fmt --all -- --check   # format
cargo check --all-features  # compile (std)
cargo check --no-default-features  # compile (no_std)
cargo test --all-features   # unit + integration + proptest + doctests
cargo test --no-default-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps
cargo deny check            # dependency/license audit
cargo bench --bench kernels # criterion benchmarks
```

See [docs/benchmarking.md](docs/benchmarking.md) for the benchmarking
methodology and how to compare backends.

## Documentation

- [docs/architecture.md](docs/architecture.md) — dispatch design, layered
  architecture, and how to add kernels
- [docs/benchmarking.md](docs/benchmarking.md) — benchmarking methodology
  and backend comparison

## Minimum Supported Rust Version

**1.89** (required for stable AVX-512 `target_feature`). Enforced in CI;
bumped only in minor releases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — concrete recipes for adding
algorithms, backends, benchmarks, and property tests.

CI runs `fmt`, `clippy`, `test`, a `no_std` check, and doc build on every
push and PR.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.
