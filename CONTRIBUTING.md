# Contributing

Concrete recipes for the three common changes. Each recipe lists the exact
files; follow the existing style in those files.

## Adding a reduction (e.g. `stats`)

1. Add the scalar reference in `src/kernels/scalar/mod.rs` (and the `_f64`
   twin).
2. Add a `crate::simd_reduce!` (or `simd_reduce2!` for pairwise ops)
   invocation per backend in `src/kernels/x86/{sse2,avx2,avx512}.rs` and
   `src/kernels/aarch64/neon.rs`.
3. Add a `dispatch!` call in `src/kernels/mod.rs`.
4. Wrap it in `src/algorithms/stats.rs` (both `f32` and `f64` submodules,
   with a doctest example).
5. Add the public name to `README.md`, `src/lib.rs` crate docs, and
   `CHANGELOG.md`.

## Adding a SIMD backend

1. Create the module (`src/kernels/<arch>/<name>.rs`), reusing the macro
   skeletons; gate it with `#[cfg(target_arch = "...")]`.
2. Add the `Backend` variant in `src/dispatch.rs` (cfg-gated).
3. Add a `platform::supports` arm and the detection path in
   `src/platform/mod.rs`.
4. Add a matrix row / runner to `.github/workflows/ci.yml` if the hardware
   exists in CI.

## Adding a benchmark / property test

- Benchmarks: `benches/kernels.rs` (a `bench_*` fn + entry in
  `criterion_group!`).
- Property tests: `tests/proptest_kernels.rs`, following the
  `proptest!` style.

## Checks before pushing

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo check --no-default-features
cargo check --no-default-features --target wasm32-unknown-unknown
```
