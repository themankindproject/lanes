# lanes

High-performance computational algorithm kernels with runtime SIMD dispatch.

`lanes` provides a small, deliberate set of optimized numerical kernels
(`sum`, `prod`, `min`, `max`, `dot`) that automatically pick the best
available SIMD backend at runtime (SSE2, AVX2, AVX-512F, NEON), with a
portable scalar fallback. Write your code once; `lanes` picks the backend.

## Usage

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

## Building & testing

```sh
cargo test --all-features
cargo check --no-default-features   # no_std (scalar only)
cargo clippy --all-targets --all-features -- -D warnings
cargo bench --bench kernels         # criterion benchmarks
```

Set `LANES_BACKEND=scalar|sse2|avx2|avx512|neon` to force a backend for
benchmarking or debugging (honored only when the CPU supports it).

## Documentation

- [docs/architecture.md](docs/architecture.md) — dispatch design and how to add kernels
- [docs/benchmarking.md](docs/benchmarking.md) — benchmarking methodology

## License

MIT OR Apache-2.0.
