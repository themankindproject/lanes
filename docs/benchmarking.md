# Benchmarking methodology

How `lanes` should be measured, what the numbers mean, and how to reproduce
them.

## Scope

Every kernel ships a Criterion benchmark (`benches/kernels.rs`). Each
algorithm is measured:

* vs a **naive iterator baseline** written independently of `lanes`
  (so the speedup claim is real, not a comparison against the same code),
* at the size ladder `16, 32, 64, 128, 256, 1024, 4096, 16_384, 65_536,
  1_000_000`, which spans cache-resident to memory-bandwidth-bound
  workloads,
* with `Throughput::Elements` so numbers can be read as elements/second.

## Separating scalar, SIMD, and dispatch overhead

The public entry points dispatch automatically, so a single `cargo bench` run
only measures the best backend. To compare backends on the same machine, use
the diagnostic override:

```sh
LANES_BACKEND=scalar cargo bench --bench kernels
LANES_BACKEND=avx2   cargo bench --bench kernels
LANES_BACKEND=avx512 cargo bench --bench kernels   # needs an AVX-512 CPU
LANES_BACKEND=neon   cargo bench --bench kernels   # needs aarch64
```

Interpretation guide:

* `naive` at a given size = scalar reference on that machine.
* `lanes` under `LANES_BACKEND=scalar` = scalar + dispatch overhead
  (a single cached `OnceLock` read + one `match`; measurable at small sizes).
* `lanes` under a SIMD backend minus `lanes` under scalar = SIMD kernel gain
  beyond dispatch.
* Always include the backend in any reported figure:
  `"sum: 3.1× vs naive (AVX2, 64 KiB)"` — a raw number without a backend and
  size is not a claim.

## Correctness before performance

Exact cross-backend equality is asserted only on **integer-exact test
vectors** (bounded integer-valued `f32` inputs where every intermediate sum
is exactly representable). Property tests use bounded magnitudes so overflow
cannot introduce `inf`/`NaN` order effects. Floating-point reduction order is
backend-dependent by design; do not "fix" tests by loosening the naive
baseline, fix them by bounding the input domain.

## Environment

* Pin CPU frequency or at least use `taskset` for the benchmark process on
  noisy shared machines; report hardware in PRs.
* Data is generated with fixed seeds (`StdRng::seed_from_u64`) so runs are
  reproducible bit-for-bit.
* Do not run other load in parallel when measuring throughput at 1M
  elements — those numbers are bandwidth-bound.

## CI

`benches.yml` runs the ladder for `scalar` and `avx2` on push to `main`
(when `src/` or `benches/` change) and on demand via `workflow_dispatch`,
then uploads `target/criterion/` reports as artifacts. GH-hosted runners do
not expose AVX-512; add a self-hosted matrix row for it.

## Reporting

A good benchmark result in a PR includes:

```text
CPU: AMD EPYC ... / Apple M3 ...
backend: avx2
sum @ 1M: 1.8× vs naive;  sum @ 1024: 2.1× vs naive
```

`cargo bench --bench kernels` writes an HTML report to
`target/criterion/report/index.html`.