# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `lanes::prod` — product reduction across all backends (scalar, SSE2,
  AVX2, AVX-512F, NEON); `1.0` for empty input.
- SSE2 backend tier (mandatory 128-bit baseline on x86-64) for
  `sum`, `prod`, `min`, `max`, `dot`; added to detection ladder and
  `LANES_BACKEND=sse2`.
- Shared reduction-kernel macros (`src/kernels/macros.rs`) that generate
  the chunked-loop skeleton for every backend; new reductions are now a
  few lines per backend instead of a hand-written unsafe copy.

## [0.1.0] - Unreleased

### Added

- Core algorithms `sum`, `min`, `max`, `dot` for `f32` slices with a small,
  stable public API (`lanes::{sum,min,max,dot,Backend,Error}`).
- Layered architecture: public API → algorithm layer → kernel layer →
  backend layer, with runtime dispatch cached in a `OnceLock`.
- Scalar reference kernels (always available) plus real SIMD kernels:
  - AVX2 + FMA (`sum`, `min`, `max`, `dot`),
  - AVX-512F (`sum`, `min`, `max`, `dot`),
  - NEON (`sum`, `min`, `max`, `dot`).
- Runtime CPU detection (`is_x86_feature_detected!`, aarch64 auxiliary
  vector) with `platform::supports` gates before every unsafe kernel call.
- `LANES_BACKEND` environment override for benchmarking and debugging.
- `no_std` support behind the `std` feature (scalar backend only).
- Error model: `Error::LengthMismatch` for `dot`; `min`/`max` return
  `Option`; `sum`/`dot` return plain values (no forced `Result`s).
- Testing: unit tests per backend, cross-backend equality tests on
  integer-exact vectors, integration tests, `proptest` property tests,
  and a cargo-fuzz target (`fuzz/`, nightly-only, not in CI).
- Criterion benchmarks for all kernels vs naive baselines at sizes
  `16 … 1_000_000` ([docs/benchmarking.md](docs/benchmarking.md)).
- CI (`fmt`, `check`, `test`, `clippy`, `doc`) on stable + MSRV 1.89
  across Linux/macOS/Windows; wasm32 `no_std` check; benchmark workflow
  with backend matrix; release dry-run workflow.

[Unreleased]: https://github.com/themankindproject/lanes/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/themankindproject/lanes/releases/tag/v0.1.0
