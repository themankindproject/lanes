# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The first release: everything below shipped with `0.1.0`,
so it is all listed as additions.

## [Unreleased]

### Added

- **`convert` family** — f16/bf16 ↔ f32 slice conversions and
  mixed-precision dot products (`dot_f16`, `dot_bf16`). All narrowing
  conversions use IEEE 754 round-to-nearest-even. The module is
  `no_std`-compatible (no `alloc` required — all functions write into
  caller-provided buffers). Exhaustively verified: all 65,536 f16 and
  bf16 bit patterns round-trip correctly; tie-to-even cases and denormal
  handling confirmed against brute-force oracles. SIMD: f16 paths use
  F16C (`_mm_cvtph_ps` / `_mm256_cvtph_ps`, runtime-probed via cached
  `platform::has_f16c()`) on x86_64 with scalar fallback; bf16 paths are
  vectorized integer shifts (4-wide SSE2, 8-wide AVX2, 16-wide
  AVX-512F, 4-wide NEON, scalar wasm) with RNE bias; dot products fuse
  widen+multiply via FMA. Benchmarks cover all six convert kernels vs
  naive (#7)
- `Backend::available()` / `available_backends()` and `Backend::as_str()`
  for runtime introspection; `Backend::Wasm` on `wasm32` (scalar
  fallthrough now, SIMD128 hooks in place).
- `distance::{cosine,euclidean}_distance` thin wrappers.

### Fixed

- `jaccard_similarity` now divides via `f64` then casts to `f32`
  (`(intersection as f64 / union as f64) as f32`), restoring precision
  for unions above 16 M bits — the `f32` path lost ~3 bits above `2^24`.
- `variance_into` / `std_dev_into` (f32/f64): the `alloc` + `not(alloc)`
  bodies were fused inside one function with nested `cfg`s and a dead
  `allow(unreachable_code)` block. They are now two clean
  `#[cfg(feature="alloc")]` / `#[cfg(not(feature="alloc"))]` function
  definitions with no dead code and bit-identical contracts.
- `geometric_mean` (f32/f64): previously scanned with
  `position(|&x| x <= 0.0)` then allocated and re-scanned with scalar
  `x.ln()`. Now validates in a single pre-alloc scan then uses the
  vectorized `dispatch_ln` / `dispatch_ln_f64` map, eliminating the double
  scan and giving SIMD `ln` on the hot path.
- `LANES_BACKEND` env override: unknown names (typos) and requesting an
  unavailable backend now emit a `debug_assertions`-gated `eprintln!`
  warning (`[lanes] LANES_BACKEND='…' ignored: …`) instead of silently
  falling back to auto-detection.
- Addressed `cargo clippy --all-targets --all-features` warnings in
  `src/platform/mod.rs` (uninlined format args).

### Changed

- F16 fast paths: SSE2 (4-wide) and AVX2 (8-wide) now use F16C `cvtph` intrinsics when `f16c` is present (cached `has_f16c` probe); scalar fallback otherwise. AVX-512 reuses the AVX2 F16C path.
- VPOPCNTDQ wiring: `hamming`/`jaccard` on AVX-512 dispatch to `_mm512_popcnt_epi64` (64 B/iter) when `avx512vpopcntdq` is present; AVX2 shuffle path otherwise. VNNI `dot_i8` probe scaffold added.
- Benchmarks: six new Criterion groups for `convert` (`f16_to_f32`, `f32_to_f16`, `bf16_to_f32`, `f32_to_bf16`, `dot_f16`, `dot_bf16`) vs naive baselines.
- Added `src/kernels/wasm` stub so `cargo check --target
  wasm32-unknown-unknown` succeeds (scalar fallthrough; future place for
  SIMD128 kernels).
- Addressed `cargo clippy --all-targets --all-features` warnings in
  `src/platform/mod.rs` (uninlined format args).

## [0.1.3] - 2026-08-20

### Added

- AVX-512 sub-feature detection (`Avx512Caps`): runtime detection of
  `avx512vpopcntdq` and `avx512vnni` extensions, cached in `OnceLock`.
  Infrastructure for per-kernel dispatch on Ice Lake+ / Zen4+ CPUs.

- Online-softmax (Milakov & Gimelshein, 2018) for small arrays: fuses
  max-finding + exp-sum into a single streaming pass for n ≤ 4096 (f32) /
  n ≤ 2048 (f64), reducing memory traffic by 33% for cache-resident
  arrays. SIMD backends delegate to this path for small inputs.

### Changed

- `count_zero`, `count_nan`, `count_infinite` (all backends): 4-way
  unrolled `simd_count!` macro with independent `popcnt` chains hides
  movemask → popcnt latency and prevents LLVM from collapsing the SIMD
  path into the same auto-vectorized pattern as the naive scalar code.
  Measured speedup vs naive (f32, n = 65,536, AVX-512F): `count_zero`
  1.0× → 2.9×, `count_nan` 1.0× → 2.5×, `count_infinite` 1.3× → 3.2×.

- Software prefetch hints added to all unrolled reduction loops
  (`simd_reduce!`, `simd_reduce2!`): prefetches 2 quad-iterations ahead
  into L1 for arrays exceeding ~32 KB. Measured on AVX-512F (n = 1 M):
  `sum` −4.6%, `dot` −7.3%; at n = 65,536: `variance` −41%.

- `prop_rsqrt_matches_naive` tolerance widened from 2 → 3 ulp for
  near-subnormal inputs (`< 2 * f32::MIN_POSITIVE`), matching the
  observed precision of the SSE2/AVX2 Newton refinement path on these
  borderline values.

- New allocation-free `variance_into` / `std_dev_into` (f32 and f64):
  caller-provided scratch buffer replaces the heap allocation of the
  two-pass `variance` / `std_dev`; results are bit-identical.
- New `max_abs_i8` kernels (scalar/SSE2/AVX2/AVX-512/NEON) backing
  `distance::i8::max_norm`: single-pass `max(|v|)` in `u8` (so
  `|i8::MIN| = 128` is exact via i16 widening; NEON via `vabdl_s8` to
  avoid saturating `vabs`). Dispatched as `dispatch_max_abs_i8`
  (`Option<u8>`, `None` on empty).

- The vector f32 `exp` / `ln` kernels now fuse their Horner polynomial
  steps with FMA (`vfmadd` on AVX2/AVX-512, `vfmaq` on NEON). The fused
  single rounding is *more* accurate than the split mul+add, so the
  documented ≤ 2 ulp (exp) / ≤ 1 ulp (ln) contracts hold. Measured on
  AVX-512F (i5-1135G7, n = 65,536): `exp` 95 → 66 µs, `ln` 89 → 64 µs,
  `tanh` 147 → 121 µs, `softmax` 141 → 111 µs, `gelu` 155 → 113 µs,
  `softplus` 351 → 250 µs, `sigmoid` 115 → 82 µs. (f64 transcendental
  kernels are unchanged: the erf chunk/tail bit-exactness invariant
  requires them to match the scalar exp/ln bit-for-bit.)
- `rsqrt` (f32, AVX2 + AVX-512) now uses the hardware `rsqrtps` /
  `rsqrt14ps` approximation + Newton refinement instead of the exact
  `div(sqrt)` pair, and special/subnormal lanes are handled exactly:
  - AVX2: two FMA-formulated Newton steps (`y · fma(−x/2, y², 1.5)` —
    one rounding per step instead of three, matching the accuracy of a
    three-step mul+add chain; the tier is only dispatched with FMA).
  - AVX-512: three Newton steps (removes the last 3-ulp cases of the
    two-step chain).
  - Subnormal lanes take the exact `div(sqrt)` pair, correctly rounded
    (measured 1 ulp). This also fixes a correctness bug: the raw
    hardware `rsqrtps` returns −inf for negative subnormals, but IEEE
    `1/sqrt` says NaN; ±0 keep their IEEE ±inf via the exact path.
  - Specials keep the raw hardware values: ±0 → ±inf, +inf → 0,
    negative normals → NaN.
  - Measured accuracy vs `1/sqrt` (f64-computed reference): 1 ulp on
    subnormals and the 2^-126 boundary, ≤ 2 ulp worst-case at the top
    of the finite range.
  - Measured on AVX-512F (i5-1135G7, n = 65,536, best-of-50):
    all-normal data 35 → 44 µs (third Newton step), subnormal-heavy
    432 → 1033 µs (exact div+sqrt path), specials-heavy 42 → 56 µs.
    The SSE2 tier keeps the exact div+sqrt (measured faster than
    approx+refine there).
- The vector `erf` / `erfc` kernels on SSE2, AVX2, and NEON now take
  pure-region fast paths (mirroring the existing AVX-512 structure): a
  chunk whose lanes all fall in one piecewise region evaluates only that
  region's form, skipping the other regions' work — in particular the
  tail's two vector `exp`s. Results are bit-identical (the fast paths
  compute exactly what the general blend selected anyway). Measured on
  the tail-heavy shared bench distribution: SSE2 erf/erfc 6.3 → 3.4 ms,
  AVX2 3.3 → 1.7 ms.
- `variance` / `std_dev` (f32 and f64) now use a fused scalar helper
  `variance_fused_* { let d = x - mean; s += d*d }` plus
  `dispatch_variance_fused_f32/f64` SIMD kernels (SSE2/AVX2/AVX-512/NEON;
  scalar single-pass fallback via `center_f32/f64`). One pass over input
  instead of the former center-into-scratch + sum_sq pass. The `_into`
  variants reuse `dispatch_center_*` to honor the scratch contract
  bit-identically. Measured (dev, n = 65,536 f32, 20k iters): 4.62 ms →
  2.44 ms (1.89×); release expected larger (alloc dominates).
- `distance::i8::max_norm` now does a single scan via `max_abs_i8`
  instead of the former `min_i8` + `max_i8` two-pass composition;
  bit-identical in `u8` (`|i8::MIN| = 128`). Measured (release, 5k iters,
  harness `/tmp/verify_i8_max_abs_bin`): n = 1,024 → 3.97×, n = 65,536 →
  3.84× (50.4 ms → 13.1 ms), n = 1,048,576 → 3.55× (820 ms → 231 ms).

### Fixed

- `sqrt` / `sqrt_f64` `no_std` subnormal guard: scale by 2^100 / 2^-50
  before/after the Newton iteration (aarch64/x86 denormal fast path).

## [0.1.2] - 2026-08-17

### Added

- New `special` kernel family — `erf` and `erfc` for f32 and f64 with
  scalar/SSE2/AVX2/AVX-512/NEON backends. Clean-room Remez coefficients
  fitted against an arbitrary-precision oracle; accuracy: f64 `erf`
  ≤ 1 ulp, f64 `erfc` ≤ 3 ulp (structural floor of the exp-product
  tail), f32 both perfectly rounded via compute-in-f64-and-round-once.
  Unblocks exact GELU (#9) and normal CDF (#11).
- New `binary` kernel family — the first integer kernels: bit-level
  `hamming` (popcount of XOR) and `jaccard` (intersection-over-union
  similarity, `Ok(None)` on empty union) over packed `&[u8]` bitmaps,
  with scalar/SSE2/AVX2/AVX-512/NEON backends.
- New `stats::i8` submodule — the first general integer reductions:
  `dot`, `sum`, `sum_sq`, `min`, `max`, `count_zero` over `&[i8]` with
  exact `i64` accumulation (no overflow for any slice that fits in
  memory). Backends: scalar, SSE2 (`pmaddwd`; min/max via sign-extend +
  `pminsw`/`pmaxsw`), AVX2 (`vpmovsxbw` + `vpmaddwd`; native `vpminsb`/
  `vpmaxsb`), AVX-512 (AVX2 kernels), NEON (`vmull_s8`/`vpadalq`;
  `vminq_s8`/`vmaxq_s8`).
- New `distance::i8` submodule — exact integer norms: `l1_norm`,
  `max_norm` (returns `Option<u8>` since `|i8::MIN| = 128` does not fit
  in `i8`), `squared_distance`. All widen to `i16` before any operation
 that could overflow; `max_norm` is composed from the `min`/`max`
 kernels (no dedicated kernel needed). Backends: scalar, SSE2
 (sign-extend + `pmaxsw`-abs idiom), AVX2 (`vpabsw`), AVX-512 (AVX2
 kernels), NEON (`vabdl_s8`).

### Fixed

- `simd_exp_f64!` (vector f64 exp) now returns denormals for results
  below 2^-1022, matching the scalar `exp_f64` contract (previously
  clamped to 0, diverging from scalar for inputs in (−745.13, −708.4)).

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

[Unreleased]: https://github.com/themankindproject/lanes/compare/v0.1.3...main
[0.1.3]: https://github.com/themankindproject/lanes/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/themankindproject/lanes/compare/v0.1.0...v0.1.2
