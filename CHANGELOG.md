# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `softplus` and `log_softmax` (`ml`) in both `f32` and `f64`.
  `softplus` gets SIMD kernels on every backend using the overflow-free
  `max(x, 0) + ln1p(e^-|x|)` form (references: musl `s_log1pf.c` /
  fdlibm `s_log1p.c`); `log_softmax` composes the existing kernels with
  the PyTorch `nn.LogSoftmax` max-shift form.
- `ln` (`math`), `logsumexp`/`layer_norm` (`ml`), and
  `geometric_mean` (`stats`) in both `f32` and `f64`. `ln` gets a
  full SIMD kernel on every backend (scalar, SSE2, AVX2, AVX-512F,
  NEON) with fdlibm/SLEEF reduction, ≤ 1 ulp vs `std`; the composed
  functions reuse the existing `exp`/`rms_norm`/reduction kernels.
- `std_dev` (`stats`), `tanh` (`math`), `rms_norm` and
  `cosine_similarity` (`ml`) in both `f32` and `f64`. `tanh` and
  `rms_norm` get SIMD kernels on every backend (scalar, SSE2, AVX2,
  AVX-512F, NEON); `std_dev`/`cosine_similarity` compose the existing
  reduction kernels.
- **f64 (double-precision) support** across every family. Each of
  `stats`, `distance`, `math`, `ml` is split into an `f32` and an
  `f64` submodule, so the same function name serves both precisions:
  `lanes::stats::f32::sum` and `lanes::stats::f64::sum`.
- `f64` kernels on all backends: scalar reference, SSE2 (2-lane), AVX2
  (4-lane), AVX-512F (8-lane), NEON (2-lane). Includes f64 `exp`/`sqrt`/
  `rsqrt`/`clip` maps and ML activations (`softmax`, `sigmoid`, `silu`,
  `gelu`, `relu`).
- The public API is precision-first: `lanes::stats::f32::*` /
  `lanes::stats::f64::*` (and the same split for `distance`, `math`,
  `ml`). The old flat `lanes::stats::sum` path is replaced by
  `lanes::stats::f32::sum`.
- `lanes::prod` — product reduction across all backends (scalar, SSE2,
  AVX2, AVX-512F, NEON); `1.0` for empty input.
- SSE2 backend tier (mandatory 128-bit baseline on x86-64) for
  `sum`, `prod`, `min`, `max`, `dot`; added to detection ladder and
  `LANES_BACKEND=sse2`.
- Shared reduction-kernel macros (`src/kernels/macros.rs`) that generate
  the chunked-loop skeleton for every backend; new reductions are now a
  few lines per backend instead of a hand-written unsafe copy.
- Layered architecture: public API → algorithm layer → kernel layer →
  backend layer, with runtime dispatch cached in a `OnceLock`.
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
  `16 … 1_000_000`.
- CI: fmt + clippy + test, doctest, MSRV, Miri, fuzz smoke, native
  aarch64, and llvm-cov coverage on every push and PR.

### Changed

- **Breaking:** the public API now requires a precision submodule
  (`lanes::stats::f32::sum`, not `lanes::stats::sum`). Update imports to
  the `f32` family for the previous behavior.

### Fixed

- AVX-512 `l1_norm`/`max_norm` (f32 + f64) used a hand-rolled
  `andnot`-with-sign-mask abs; replaced with the native `_mm512_abs_ps`/
  `_mm512_abs_pd`, roughly 2× faster on the AVX-512 path.

[Unreleased]: https://github.com/themankindproject/lanes/commits/main
