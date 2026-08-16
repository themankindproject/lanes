# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The first release: everything below shipped with `0.1.0`,
so it is all listed as additions.

## [Unreleased]

### Added

- New `binary` kernel family — the first integer kernels: bit-level
  `hamming` (popcount of XOR) and `jaccard` (intersection-over-union
  similarity, `Ok(None)` on empty union) over packed `&[u8]` bitmaps,
  with scalar/SSE2/AVX2/AVX-512/NEON backends.
- New `stats::i8` submodule — the first general integer reductions:
  `dot` and `sum` over `&[i8]` with exact `i64` accumulation (no
  overflow for any slice that fits in memory). Backends: scalar, SSE2
  (`pmaddwd`), AVX2 (`vpmovsxbw` + `vpmaddwd`), AVX-512 (AVX2 kernels),
  NEON (`vmull_s8`/`vpadalq`).

## [0.1.0] - 2026-08-16

### Added

**Architecture and platform**

- `Error` and `Backend` are `#[non_exhaustive]`: new error variants and
  new backends may be added in minor releases without a major version
  bump; downstream `match`es keep a wildcard arm.
- Layered architecture: public API → algorithm layer → kernel layer →
  backend layer, with runtime CPU dispatch cached in a `OnceLock`.
- Runtime CPU detection (`is_x86_feature_detected!`, aarch64 auxiliary
  vector) with `platform::supports` gates before every unsafe kernel
  call. Backend tiers: scalar, SSE2, AVX2, AVX-512F, NEON.
- `LANES_BACKEND=scalar|sse2|avx2|avx512|neon` environment override for
  benchmarking and debugging.
- `no_std` support behind the `std` feature. `no_std` builds select the
  architecture-guaranteed SIMD tier statically — SSE2 on x86-64, NEON
  on aarch64 (both mandatory baselines, no runtime probing), scalar
  elsewhere.
- Feature flags: `default = ["std"]`, `std = ["alloc"]`; `alloc` gates
  the `Vec`-returning forms (`math`, `ml`).
- The algorithm layer is `#![forbid(unsafe_code)]`, making the "all
  unsafe lives in the kernel layer" boundary compiler-enforced.
- Shared reduction-kernel macros (`src/kernels/macros.rs`) that generate
  the chunked-loop skeleton for every backend; new reductions are a few
  lines per backend.
- The SSE2 backend uses only SSE1/SSE2 intrinsics, and the AVX-512
  backend only AVX-512F: float bitwise ops route through AVX-512F
  integer-domain `_si512` ops and the f64 `exp` rounding detours through
  i32 conversions, so both tiers run on any CPU that satisfies their
  dispatch gate.

**Precision-first API**

- Every family is split into `f32` and `f64` submodules, so the same
  function name serves both precisions: `lanes::stats::f32::sum` and
  `lanes::stats::f64::sum` (same split for `distance`, `math`, `ml`).
- `f64` kernels on all backends: scalar reference, SSE2 (2-lane), AVX2
  (4-lane), AVX-512F (8-lane), NEON (2-lane).

**Kernels**

- `stats`: `sum`, `prod`, `min`, `max`, `argmax`, `argmin`, `sum_sq`,
  `mean`, `variance`, `std_dev`, `geometric_mean`, `dot`, `count_zero`,
  `count_nan`, `count_infinite`.
- `distance`: `l1_norm`, `l2_norm`, `max_norm`, `squared_distance`
  (fused `sub + mul + reduce_add` in one pass), `kl_divergence`,
  `js_divergence` (fused `div + ln + mul + reduce_add` in one pass over
  the register-only fdlibm `ln` kernels; raw IEEE zero/NaN semantics, no
  input normalization, `js_divergence` returns the divergence rather than
  the sqrt-distance).
- `math`: `sqrt`, `clip`, `rsqrt`, `exp`, `ln`, `tanh`, `hypot`, `powi`,
  `abs_sub` — each also as an allocation-free `*_into` variant.
- `ml`: `softmax`, `log_softmax`, `sigmoid`, `silu`, `gelu`, `relu`,
  `softplus`, `rms_norm`, `layer_norm`, `cosine_similarity`,
  `logsumexp` — maps and norms also as `*_into` variants.
- `_into` variants write into a caller-provided buffer so hot loops can
  reuse one allocation; the allocating forms are thin wrappers.
- The allocating wrappers build their output buffer without a zero-fill
  (`with_capacity` + `set_len`, confined to a single kernel-layer helper
  and Miri-verified): the map kernel writes every element, so the
  `vec![0.0; n]` pre-fill would be pure wasted store traffic on
  memory-bound maps.
- `exp`, `ln`, `tanh`, `sqrt`, `rsqrt` get full SIMD kernels on every
  backend with fdlibm/SLEEF/musl-derived reductions, ≤ 1 ulp vs `std`.
- `softplus` uses the overflow-free `max(x, 0) + ln1p(e^-|x|)` form
  (references: musl `s_log1pf.c` / fdlibm `s_log1p.c`); `log_softmax`
  uses the PyTorch `nn.LogSoftmax` max-shift form; `logsumexp` has a
  dedicated scalar-returning SIMD kernel (no intermediate buffer).
- `hypot` is overflow-safe (scales by `max(|a|, |b|)` instead of
  squaring directly), matching `f32::hypot`/`f64::hypot` within 1–2 ulp
  with identical NaN/inf propagation.
- `powi` is bit-exact with `std::powi` (`powi(x, 0) == 1` for every
  `x` including NaN/inf; `no_std` uses a portable squaring
  implementation matching `compiler-builtins`).
- Add-based reductions (`sum`, `sum_sq`, `l1_norm`, `dot`) use four
  independent accumulator chains in the chunked loop, hiding the
  vector-add/FMA latency; `prod`/`min`/`max` keep the single-chain
  form. Reduction order is backend-dependent and documented as such.
- AVX-512 `l1_norm`/`max_norm` use the native `_mm512_abs_ps`/`_pd`.

**Error model**

- `lanes::Error` with four variants: `LengthMismatch { expected,
  actual }`, `EmptyInput`, `InvalidBounds`, `NonPositiveInput { index }`.
- Two-input ops (`dot`, `squared_distance`, `abs_sub`, `hypot`,
  `cosine_similarity`) return `Err(Error::LengthMismatch)` on unequal
  lengths; every `_into` variant returns `Result<(), Error>` and
  `Err(Error::LengthMismatch)` when the output buffer has the wrong
  length. No caller-facing kernel panics on bad input.
- `geometric_mean` returns `Err(Error::EmptyInput)` for an empty slice
  and `Err(Error::NonPositiveInput { index })` when a value is ≤ 0;
  NaN inputs propagate to a NaN result.
- `clip` returns `Err(Error::InvalidBounds)` when `lo > hi` or a bound
  is NaN (mirroring the `f32::clamp`/`f64::clamp` precondition); NaN
  *values* still propagate.
- `cosine_similarity` returns `Err(Error::EmptyInput)` on empty inputs
  and `Ok(0.0)` for a zero-norm vector (a zero vector has no direction,
  matching the scikit-learn convention).
- `min`/`max`/`argmax`/`argmin` return `Option` (`None` on empty
  input); infallible reductions (`sum`, `prod`, ...) return plain
  values.
- `Error` implements `core::error::Error` unconditionally (stable since
  Rust 1.81, below the 1.89 MSRV), so `no_std` users get the standard
  error trait without the `std` feature.
- Uniform NaN semantics across all backends: `min`/`max` follow IEEE
  754 `minNum`/`maxNum` (a NaN is ignored unless every input is NaN, in
  which case the result is NaN); `max_norm` returns NaN if any input is
  NaN.

**Tooling**

- Unit tests per backend, cross-backend equality tests on integer-exact
  vectors, integration tests, `proptest` property tests, and strict
  numerical-correctness tests (bit-exact / ULP-bounded vs `std` for
  `hypot`, `powi`, `abs_sub`, `squared_distance`).
- cargo-fuzz targets (`fuzz/`, nightly-only, not in CI).
- Criterion benchmarks for all kernels vs naive baselines at sizes
  `16 … 1_000_000`.
- CI: fmt + clippy + test, doctest, MSRV, Miri, fuzz smoke, native
  aarch64, and llvm-cov coverage on every push and PR.

[Unreleased]: https://github.com/themankindproject/lanes/commits/main
