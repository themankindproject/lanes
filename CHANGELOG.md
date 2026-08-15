# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Error::EmptyInput` variant, returned by `cosine_similarity` when both
  input slices are empty (the angle between empty vectors is undefined).
- Allocation-free `_into` variants for every map-style op, in both `f32`
  and `f64`: `softmax_into`, `sigmoid_into`, `softplus_into`, `silu_into`,
  `gelu_into`, `relu_into`, `rms_norm_into` (`ml`) and `sqrt_into`,
  `clip_into`, `rsqrt_into`, `tanh_into`, `exp_into`, `ln_into` (`math`).
  Each writes into a caller-provided buffer (length-checked with an
  always-on assertion) so hot loops can reuse one allocation; the
  allocating forms are now thin wrappers around them.
- `no_std` builds now select the architecture-guaranteed SIMD tier
  statically instead of always falling back to scalar: SSE2 on x86-64
  and NEON on aarch64 (both are mandatory baselines, so no runtime CPU
  probing is needed). Other targets still use scalar.

### Changed

- **Breaking:** `cosine_similarity` now returns `Result<f32, Error>` /
  `Result<f64, Error>` instead of `Result<Option<f32>, Error>`. Empty
  inputs return `Err(Error::EmptyInput)` (previously `Ok(None)`), and a
  zero-norm vector now yields `Ok(0.0)` (previously `Ok(None)`) — a zero
  vector has no direction, so it shares none with the other, matching the
  scikit-learn convention.
- **Performance:** the add-based reductions (`sum`, `sum_sq`, `l1_norm`,
  `dot` — f32 and f64, all backends) now use four independent accumulator
  chains in the chunked loop, hiding the vector-add/FMA latency and
  sustaining one combine per cycle instead of one per latency window.
  `prod`/`min`/`max` keep the single-chain form (prod is exact-equality
  tested; min/max have 1-cycle latency). Reduction order is
  backend-dependent and documented as such.
- The `algorithms` layer is now `#![forbid(unsafe_code)]`, making the
  "all unsafe lives in the kernel layer" boundary compiler-enforced.

### Fixed

- **Correctness:** `min`, `max`, and `max_norm` now have identical NaN
  semantics on every backend (previously the SIMD backends followed raw
  hardware `minps`/`vminq` semantics, which are position-dependent and
  differ from the scalar reference). `min`/`max` use IEEE 754
  `minNum`/`maxNum` (a NaN is ignored unless every input is NaN, in which
  case the result is NaN); `max_norm` returns NaN if any input is NaN
  (matching the scalar `total_cmp` reference). Previously a NaN in the
  last lane of a vector chunk could poison the SIMD result while the
  scalar path ignored it, and an all-NaN input shorter than one chunk
  returned `±inf` instead of NaN.
- **Soundness:** the SSE2 backend's horizontal sum used `_mm_movehdup_ps`,
  an SSE3 intrinsic, inside code gated only on SSE2 — undefined behavior
  (SIGILL) on CPUs without SSE3. Replaced with an SSE1/SSE2-only shuffle
  reduction.
- **Soundness:** the AVX-512 backend used AVX-512DQ-only intrinsics while
  dispatch gates it on AVX-512F alone — SIGILL on F-only parts (e.g.
  Knights Landing). The float bitwise ops (`_mm512_and/or/xor/andnot_ps`
  and `_pd` twins) now route through AVX-512F integer-domain `_si512`
  ops, and the f64 `exp` rounding detours through i32 conversions
  (AVX-512F) instead of `_mm512_cvttpd_epi64`/`_mm512_cvtepi64_pd`
  (AVX-512DQ), matching the existing SSE2/AVX2 pattern.

- `log_softmax_into` and `layer_norm_into` (`ml`) in both `f32` and
  `f64` — allocation-free variants that write into a caller-provided
  buffer. Dedicated SIMD kernels on every backend (scalar, SSE2, AVX2,
  AVX-512F, NEON); `log_softmax`/`layer_norm` are now thin wrappers
  around them. `logsumexp` also gained a dedicated scalar-returning
  SIMD kernel (no intermediate buffer or clones).
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
- `log_softmax`/`layer_norm` now delegate to the new `_into` kernels;
  the internal `sub_scalar` kernels were removed (dead after the
  restructure).
- **Breaking:** `Backend::name()` and the `Hash` impl on `Backend` are
  removed (both had zero callers; `{:?}` debug printing covers every
  existing use).
- `simd_map_param!` is renamed to `simd_clip!` (its only caller is
  `clip`) with named `lo`/`hi` parameters; all 14 internal kernel
  macros are now `#[doc(hidden)]` on docs.rs.
- Benchmarks use a dependency-free xorshift64 generator instead of
  `rand`; the `rand` dev-dependency and the redundant `[profile.bench]`
  block are dropped.

### Removed

- `deny.toml` (no `cargo-deny` runner in CI or tooling) and
  `site/index.html` (a gitignored single-file placeholder).

### Fixed

- AVX-512 `l1_norm`/`max_norm` (f32 + f64) used a hand-rolled
  `andnot`-with-sign-mask abs; replaced with the native `_mm512_abs_ps`/
  `_mm512_abs_pd`, roughly 2× faster on the AVX-512 path.

[Unreleased]: https://github.com/themankindproject/lanes/commits/main
