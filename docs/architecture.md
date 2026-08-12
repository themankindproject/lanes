# lanes architecture

This document describes how `lanes` is organized, how dispatch works, where
the unsafe code lives, and how to extend the crate without destabilizing the
public API.

## Layering

```
Public API        lanes::stats::f32/f64, distance::f32/f64, math::f32/f64,
     │            ml::f32/f64 — plus Backend, Error
     │
Algorithm layer   src/algorithms/           — input validation, backend lookup
     │
Kernel layer      src/kernels/            — dispatch fns (match on Backend)
     │                                       │                    │
     │            scalar/mod.rs            x86/sse2.rs, avx2.rs, avx512.rs
     │                                      aarch64/neon.rs
     │            macros.rs                — shared reduction skeletons
     ▼
Backend layer     src/platform/           — CPU feature detection + LANES_BACKEND
```

* **Public API** — the four families (`stats`, `distance`, `math`, `ml`),
  each split into `f32` and `f64` submodules, plus the `Backend`/`Error`
  types. This is the only stable surface.
* **Algorithm layer** — validates inputs (length checks, empty-slice rules),
  resolves the backend once, and calls the kernel dispatcher. No unsafe code.
* **Kernel layer** — `dispatch_*` functions map a `Backend` to the matching
  optimized kernel. Each kernel module is target-gated with `#[cfg]` and every
  SIMD kernel is an `unsafe fn` carrying `#[target_feature(...)]`.
* **Backend layer** — `platform::detect` probes CPU capabilities once and
  `platform::supports` gates every unsafe kernel invocation behind the
  corresponding `is_*_feature_detected!` check so a kernel is never executed
  on hardware that lacks the instruction set.

## Dispatch model

Two designs were considered:

1. **An `Engine` handle** the user constructs (`Engine::detect()`) and calls
   methods on. This adds API surface and a passing-around cost without any
   benefit: the backend decision is global (all operations benefit equally),
   and per-call overhead is one cached lookup.
2. **Free functions with a cached global decision** — `Backend::detect()`
   resolves once into a `OnceLock`, so every subsequent call is a single
   atomic load; the kernel dispatcher then does one `match` on a `Copy` enum.

The skeleton ships design 2: it is simpler, keeps the public API to free
functions plus two types, and makes "sum, then dot, then min" as cheap as if
the user inlined the backend themselves. If a future use case demonstrates a
need to pin different backends per call site, `LANES_BACKEND` (below) is the
bridge, and a scoped override API can be added without breaking anything.

### LANES_BACKEND diagnostic override

With the `std` feature, the environment variable `LANES_BACKEND` forces a
backend for the whole process. Accepted values: `scalar`, `sse2`, `avx2`,
`avx512`, `neon`. The requested backend is honored **only if it is compiled in
and actually detectable on the host CPU**; otherwise lanes falls back to
auto-detection. This is used by `docs/benchmarking.md`'s manual workflow to
produce comparable numbers per backend and by developers to reproduce bugs on
a specific path.

### Kernel code generation

The chunked SIMD reduction shape (vector loop → horizontal reduce → scalar
tail) is shared across every backend via the macros in `src/kernels/macros.rs`
(`simd_reduce!` for single-input reductions like `sum`/`min`/`max`,
`simd_reduce2!` for pairwise reductions like `dot`). Each backend module
supplies only the per-op identity, vector combine, horizontal reduce, and
scalar tail — so adding a new reduction to all three x86 tiers is one macro
invocation per backend, not a new copy of the unsafe skeleton. The generated
functions remain `#[target_feature(...)]` `unsafe fn`s gated by
`platform::supports`, so the safety model is unchanged.

## Scalar fallback

`kernels/scalar` is always compiled, always correct, and is the reference
implementation used by `proptest` and by every cross-backend unit test.
Dispatch can only reach SIMD kernels through `platform::supports`, and every
public function works with zero SIMD available.

## Unsafe boundaries

Policy (enforced by `#![forbid(unsafe_op_in_unsafe_fn)]`):

* All unsafe code lives in `src/kernels/{x86,aarch64}` (plus the generated
  bodies from `src/kernels/macros.rs`). Nothing else in the crate uses
  `unsafe`.
* Every `unsafe` kernel documents (a) the invariant that makes it safe — the
  `#[target_feature]` attribute plus the caller's runtime feature check —
  and (b) pointer-bounds reasoning for every `add`/`get_unchecked`.
* Calling a SIMD kernel is only possible via the dispatcher, which first
  passes through `platform::supports`.
* Intrinsics are wrapped in explicit `unsafe {}` blocks with `#![allow(
  unused_unsafe)]` per kernel module because current stdarch declares most
  intrinsics `safe`, while the MSRV toolchain still declares them `unsafe` —
  the explicit blocks compile on both.

## Floating-point semantics

Documented at crate level (see `src/lib.rs`): reduction order is
backend-dependent; NaN handling differs between scalar (`f32::min`/`max`,
IEEE `minNum`) and SIMD (hardware `vminps`/`vmaxps` which propagate NaN);
sums/dot products that overflow can surface as `inf` in one order and `NaN`
in another. Properties tested only hold where documented (bounded inputs,
NaN-free). See `docs/benchmarking.md` for why exact cross-backend equality
is asserted on integer-exact test vectors.

## Feature flags

| Feature | Default | Effect |
| --- | --- | --- |
| `std` | on | Runtime CPU detection, `LANES_BACKEND`, `std::error::Error`. Implies `alloc`. |
| `alloc` | on (via `std`) | `Vec`-returning families (`math`, `ml`, `stats::variance`). |
| (none) | — | `no_std` build; `Backend::detect()` returns `Scalar`; `stats`/`distance` only. |

There is intentionally no `nightly` or `wasm` feature yet: nothing in the
crate needs nightly, and WASM gets no special-cased code until a SIMD128
backend exists. Flags are added when they have real behavior.

## WASM strategy

The crate compiles for `wasm32-unknown-unknown` in `no_std` mode (checked in
CI) and uses the scalar backend there. Kernel modules are `#[cfg]`-gated per
architecture, so a future `wasm32` backend is additive: add
`kernels/wasm/simd128.rs`, a `Backend::WasmSimd128` variant, and a branch in
`platform`. No OS-specific API is used in any kernel. The macro skeletons
only use `core::arch` intrinsics, so a WASM backend can reuse the same
`simd_reduce!` / `simd_reduce2!` expansion with `simd128` intrinsics.

## MSRV policy

`rust-version = "1.89"` (AVX-512 `target_feature` stabilization). The MSRV is
enforced in CI (matrix row `1.89.0`) and only raised in a minor release.

## Adding algorithms / backends — see CONTRIBUTING.md

The contributing guide contains concrete, step-by-step recipes for adding a
new algorithm, a new SIMD backend, a new benchmark, and a new property test,
with exact file paths.