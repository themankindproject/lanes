//! Shared skeleton macros for SIMD reduction kernels.
//!
//! Every SIMD reduction (`sum`, `min`, `max`, `dot`) has the same shape:
//! a chunked vector loop with a fixed-width accumulator, a horizontal
//! reduction of the final accumulator, then a scalar tail. This module
//! provides macros that generate that skeleton; each backend module
//! supplies the per-op vector combine, horizontal-reduce, and scalar-tail
//! functions and expands the macros once per operation.
//!
//! Adding a new reduction (e.g. `prod`) therefore means adding a few small
//! vector functions per backend, not a new copy of the unsafe skeleton.
//!
//! Safety: every generated function carries `#[target_feature(...)]` and
//! remains `unsafe fn`; the caller (dispatch layer) must verify the
//! corresponding CPU feature before invoking it.

/// Generate a single-input reduction kernel (`sum`, `min`, `max`, `prod`, …).
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string (e.g. `"avx2"`).
/// * `$lanes` — vector width in `$t` elements (4, 8, 16 for f32; 2, 4, 8 for f64).
/// * `$load` — `fn(*const $t) -> V` loading `$lanes` elements from a
///   pointer (e.g. `|p| unsafe { _mm256_loadu_ps(p) }`).
/// * `$acc_ident` — expression yielding the identity accumulator for the
///   reduction (e.g. `_mm256_setzero_ps()` or `_mm256_set1_ps(f32::INFINITY)`).
/// * `$combine` — `fn(V, V) -> V` folding each loaded chunk into the
///   accumulator (e.g. `_mm256_add_ps` or `_mm256_min_ps`).
/// * `$reduce` — `fn(V) -> $t` horizontal reduction of the final
///   accumulator (e.g. `hsum_256` or `_mm512_reduce_add_ps`).
/// * `$tail` — a `fn($t, $t) -> $t` applied to `(result_so_far, element)`
///   for the scalar tail (e.g. `|r, v| r + v`, or `f32::min`).
#[macro_export]
#[doc(hidden)]
macro_rules! simd_reduce {
    // Unrolled variant: four independent accumulator chains hide the combine
    // latency (e.g. the 4-cycle `vaddps`) so the loop sustains one combine
    // per cycle — load bandwidth instead of latency-bound. `$merge` combines
    // two partial accumulators (plain vector add for the sum-family kernels).
    //
    // The final reduction order differs from the single-chain form, which is
    // fine for sum-family reductions (documented as backend-dependent) but
    // rules out `prod` (tested for exact equality) and gains nothing for
    // `min`/`max` (1-cycle latency); those keep the single-chain arm below.
    ($name:ident, $t:ty, $feat:literal, $lanes:expr, $load:expr, $acc_ident:expr, $combine:expr, $reduce:expr, $tail:expr, $merge:expr) => {
        /// SIMD reduction kernel (4-way unrolled). See the enclosing module
        /// for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available.
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t]) -> $t {
            let len = values.len();
            let ptr = values.as_ptr();
            let chunks = len / $lanes;
            let remainder = len % $lanes;

            // Four independent chains: each is latency-bound on `$combine`,
            // but with four in flight the scheduler issues one combine per
            // cycle. Inputs shorter than 4 chunks take the single-acc path.
            let mut acc0 = $acc_ident;
            let mut acc1 = $acc_ident;
            let mut acc2 = $acc_ident;
            let mut acc3 = $acc_ident;
            let quads = chunks / 4;
            for i in 0..quads {
                let base = i * 4 * $lanes;
                // SAFETY: base + 4*$lanes - 1 < quads*4*$lanes <= len.
                let v0 = $load(unsafe { ptr.add(base) });
                let v1 = $load(unsafe { ptr.add(base + $lanes) });
                let v2 = $load(unsafe { ptr.add(base + 2 * $lanes) });
                let v3 = $load(unsafe { ptr.add(base + 3 * $lanes) });
                acc0 = $combine(acc0, v0);
                acc1 = $combine(acc1, v1);
                acc2 = $combine(acc2, v2);
                acc3 = $combine(acc3, v3);
            }
            // Balanced merge tree: the two inner merges run in parallel.
            let mut acc = $merge($merge(acc0, acc1), $merge(acc2, acc3));
            for i in (quads * 4)..chunks {
                // SAFETY: i * $lanes + ($lanes - 1) < chunks * $lanes <= len.
                let v = $load(unsafe { ptr.add(i * $lanes) });
                acc = $combine(acc, v);
            }

            let mut result = $reduce(acc);

            let tail_start = chunks * $lanes;
            for i in 0..remainder {
                // SAFETY: tail_start + i < len, so the read is in bounds.
                let v = unsafe { *values.get_unchecked(tail_start + i) };
                result = $tail(result, v);
            }
            result
        }
    };

    ($name:ident, $t:ty, $feat:literal, $lanes:expr, $load:expr, $acc_ident:expr, $combine:expr, $reduce:expr, $tail:expr) => {
        /// SIMD reduction kernel. See the enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available.
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t]) -> $t {
            let len = values.len();
            let ptr = values.as_ptr();
            let chunks = len / $lanes;
            let remainder = len % $lanes;

            let mut acc = $acc_ident;
            for i in 0..chunks {
                // SAFETY: i * $lanes + ($lanes - 1) < chunks * $lanes <= len.
                let v = $load(unsafe { ptr.add(i * $lanes) });
                acc = $combine(acc, v);
            }

            let mut result = $reduce(acc);

            let tail_start = chunks * $lanes;
            for i in 0..remainder {
                // SAFETY: tail_start + i < len, so the read is in bounds.
                let v = unsafe { *values.get_unchecked(tail_start + i) };
                result = $tail(result, v);
            }

            result
        }
    };
}

/// Generate a softmax kernel (three-pass map: max → exp+sum → scale).
///
/// Softmax is not a reduction, so it needs its own skeleton. Each backend
/// supplies the vector ops (`$load`, `$max`, `$sub`, `$exp`, `$add`,
/// `$mul`, `$store`) and the horizontal reduction (`$reduce`); the macro
/// generates the 3-pass loop over chunks plus a scalar tail.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$store` — `fn(*mut $t, V)`.
/// * `$max` — `fn(V, V) -> V` elementwise max.
/// * `$sub` — `fn(V, V) -> V` elementwise sub.
/// * `$exp` — `fn(V) -> V` elementwise exp.
/// * `$add` — `fn(V, V) -> V` elementwise add.
/// * `$mul` — `fn(V, V) -> V` elementwise mul.
/// * `$reduce` — `fn(V) -> $t` horizontal sum.
/// * `$max_reduce` — `fn(V) -> $t` horizontal max (distinct from `$reduce`).
/// * `$set1` — `fn($t) -> V` broadcast a scalar to all lanes.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_softmax {
    ($name:ident, $t:ty, $feat:literal, $lanes:expr, $load:expr, $store:expr, $max:expr, $sub:expr, $exp:expr, $add:expr, $mul:expr, $reduce:expr, $max_reduce:expr, $set1:expr, $exp_scalar:expr) => {
        /// SIMD softmax kernel. See the enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available and that
        /// `values` and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t], out: &mut [$t]) {
            let len = values.len();
            if len == 0 {
                return;
            }
            let chunks = len / $lanes;
            let rem = len % $lanes;

            // Pass 1: max over all chunks (skip vector loads if none fit).
            let mut max = if chunks > 0 {
                let mut vmax = $load(unsafe { values.as_ptr() });
                for i in 1..chunks {
                    let v = $load(unsafe { values.as_ptr().add(i * $lanes) });
                    vmax = $max(vmax, v);
                }
                $max_reduce(vmax)
            } else {
                <$t>::NEG_INFINITY
            };
            for i in 0..rem {
                max = <$t>::max(max, unsafe { *values.get_unchecked(chunks * $lanes + i) });
            }

            // Pass 2: exp(x - max) and sum.
            let vmax_b = $set1(max);
            let mut sum = 0.0;
            for i in 0..chunks {
                let v = $load(unsafe { values.as_ptr().add(i * $lanes) });
                let e = $exp($sub(v, vmax_b));
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, e);
                sum += $reduce(e);
            }
            for i in 0..rem {
                let e = $exp_scalar(unsafe { *values.get_unchecked(chunks * $lanes + i) } - max);
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) = e };
                sum += e;
            }

            // Pass 3: scale by 1/sum.
            if sum != 0.0 {
                let inv = 1.0 / sum;
                let vinv = $set1(inv);
                for i in 0..chunks {
                    let v = $load(unsafe { out.as_ptr().add(i * $lanes) });
                    $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, $mul(v, vinv));
                }
                for i in 0..rem {
                    unsafe { *out.get_unchecked_mut(chunks * $lanes + i) *= inv };
                }
            }
        }
    };
}

/// Generate a two-input reduction kernel (`dot`, and future pairwise ops).
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `[$($feat:literal),+]` — one or more `target_feature` strings (the
///   FMA-using kernels pass `["avx2", "fma"]`).
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V` loading `$lanes` elements.
/// * `$acc_ident` — identity accumulator expression.
/// * `$combine` — `fn(V, V, V) -> V` folding the product chunk into the
///   accumulator (e.g. `_mm256_fmadd_ps`, or
///   `|acc, va, vb| _mm_add_ps(acc, _mm_mul_ps(va, vb))` without FMA).
/// * `$reduce` — horizontal reduction `fn(V) -> $t`.
/// * `$tail` — a `fn($t, $t, $t) -> $t` applied to
///   `(result_so_far, va, vb)` for the scalar tail (e.g. `|r, a, b| r + a * b`).
#[macro_export]
#[doc(hidden)]
macro_rules! simd_reduce2 {
    // Unrolled variant: four independent FMA/accumulate chains hide the
    // combine latency (4-cycle `vfmadd` on most cores). Same rationale and
    // ordering caveat as the unrolled `simd_reduce!` arm; dot is tested with
    // a Higham tolerance, so the changed reduction order is acceptable.
    ($name:ident, $t:ty, [$( $feat:literal ),+], $lanes:expr, $load:expr, $acc_ident:expr, $combine:expr, $reduce:expr, $tail:expr, $merge:expr) => {
        /// SIMD two-input reduction kernel (4-way unrolled). See the
        /// enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU features are available and that
        /// `a` and `b` have equal lengths.
        #[target_feature($(enable = $feat),*)]
        pub(crate) unsafe fn $name(a: &[$t], b: &[$t]) -> $t {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let chunks = len / $lanes;
            let remainder = len % $lanes;

            let mut acc0 = $acc_ident;
            let mut acc1 = $acc_ident;
            let mut acc2 = $acc_ident;
            let mut acc3 = $acc_ident;
            let quads = chunks / 4;
            for i in 0..quads {
                let base = i * 4 * $lanes;
                // SAFETY: base + 4*$lanes - 1 < quads*4*$lanes <= len, so
                // both pointers are in bounds.
                let a0 = $load(unsafe { a_ptr.add(base) });
                let a1 = $load(unsafe { a_ptr.add(base + $lanes) });
                let a2 = $load(unsafe { a_ptr.add(base + 2 * $lanes) });
                let a3 = $load(unsafe { a_ptr.add(base + 3 * $lanes) });
                let b0 = $load(unsafe { b_ptr.add(base) });
                let b1 = $load(unsafe { b_ptr.add(base + $lanes) });
                let b2 = $load(unsafe { b_ptr.add(base + 2 * $lanes) });
                let b3 = $load(unsafe { b_ptr.add(base + 3 * $lanes) });
                acc0 = $combine(acc0, a0, b0);
                acc1 = $combine(acc1, a1, b1);
                acc2 = $combine(acc2, a2, b2);
                acc3 = $combine(acc3, a3, b3);
            }
            // Balanced merge tree: the two inner merges run in parallel.
            let mut acc = $merge($merge(acc0, acc1), $merge(acc2, acc3));
            for i in (quads * 4)..chunks {
                // SAFETY: i * $lanes + ($lanes - 1) < chunks * $lanes <= len,
                // so both pointers are in bounds.
                let va = $load(unsafe { a_ptr.add(i * $lanes) });
                let vb = $load(unsafe { b_ptr.add(i * $lanes) });
                acc = $combine(acc, va, vb);
            }

            let mut result = $reduce(acc);

            let tail_start = chunks * $lanes;
            for i in 0..remainder {
                // SAFETY: tail_start + i < len, so both reads are in bounds.
                let va = unsafe { *a.get_unchecked(tail_start + i) };
                let vb = unsafe { *b.get_unchecked(tail_start + i) };
                result = $tail(result, va, vb);
            }
            result
        }
    };

    ($name:ident, $t:ty, [$( $feat:literal ),+], $lanes:expr, $load:expr, $acc_ident:expr, $combine:expr, $reduce:expr, $tail:expr) => {
        /// SIMD two-input reduction kernel. See the enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU features are available and that
        /// `a` and `b` have equal lengths.
        #[target_feature($(enable = $feat),*)]
        pub(crate) unsafe fn $name(a: &[$t], b: &[$t]) -> $t {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let chunks = len / $lanes;
            let remainder = len % $lanes;

            let mut acc = $acc_ident;
            for i in 0..chunks {
                // SAFETY: i * $lanes + ($lanes - 1) < chunks * $lanes <= len,
                // so both pointers are in bounds.
                let va = $load(unsafe { a_ptr.add(i * $lanes) });
                let vb = $load(unsafe { b_ptr.add(i * $lanes) });
                acc = $combine(acc, va, vb);
            }

            let mut result = $reduce(acc);

            let tail_start = chunks * $lanes;
            for i in 0..remainder {
                // SAFETY: tail_start + i < len, so both reads are in bounds.
                let va = unsafe { *a.get_unchecked(tail_start + i) };
                let vb = unsafe { *b.get_unchecked(tail_start + i) };
                result = $tail(result, va, vb);
            }

            result
        }
    };
}

/// Generate a NaN-parity reduction kernel for the min/max family (`min`,
/// `max`, `max_norm`).
///
/// Hardware `minps`/`vminq`-style ops have NaN semantics that differ from
/// the scalar reference (`f32::min`/`f32::max` minNum/maxNum semantics, and
/// `total_cmp` for `max_norm`) and are even position-dependent between
/// backends. This skeleton restores parity:
///
/// * every loaded chunk is passed through `$clean`, which replaces NaN
///   lanes with the reduction identity (neutral under `$combine`), so the
///   vector loop and horizontal reduce never see a NaN;
/// * a boolean flag is accumulated across chunks (`$detect`) and the scalar
///   tail (`$tail_flag`);
/// * `$finish` maps `(result, flag)` to the final value, encoding the
///   per-op NaN rule:
///   * `min`/`max` — flag = "a non-NaN element was seen"; if false (input
///     non-empty but all NaN) the result is NaN, matching
///     `reduce(f32::min)` / `reduce(f32::max)`;
///   * `max_norm` — flag = "a NaN element was seen"; if true the result is
///     NaN, matching `map(abs).max_by(total_cmp)` (NaN sorts above all).
///
/// For NaN-free inputs this is the same chunked loop as [`simd_reduce!`]
/// (the clean/detect ops are cheap masks), so performance on the common
/// path is essentially unchanged.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$acc_ident` — identity accumulator (e.g. `+inf` for `min`).
/// * `$combine` — `fn(V, V) -> V` elementwise combine.
/// * `$reduce` — `fn(V) -> $t` horizontal reduction.
/// * `$tail` — `fn($t, $t) -> $t` scalar tail combine (`f32::min`/`f32::max`
///   already ignore NaN, so the tail needs no cleaning).
/// * `$detect` — `fn(V) -> bool`, true if the chunk sets the flag.
/// * `$clean` — `fn(V) -> V`, NaN lanes replaced with the identity.
/// * `$tail_flag` — `fn($t) -> bool`, per-element flag for the tail.
/// * `$finish` — `fn($t, bool) -> $t`, final NaN-rule application.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and that `values` is non-empty.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_minmax {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr, $load:expr,
        $acc_ident:expr, $combine:expr, $reduce:expr, $tail:expr,
        $detect:expr, $clean:expr, $tail_flag:expr, $finish:expr
    ) => {
        /// SIMD min/max-family reduction with scalar NaN parity. See the
        /// enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available and that
        /// `values` is non-empty.
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t]) -> $t {
            let len = values.len();
            debug_assert!(len > 0);
            let ptr = values.as_ptr();
            let chunks = len / $lanes;
            let remainder = len % $lanes;

            let mut acc = $acc_ident;
            let mut flagged = false;
            for i in 0..chunks {
                // SAFETY: i * $lanes + ($lanes - 1) < chunks * $lanes <= len.
                let v = $load(unsafe { ptr.add(i * $lanes) });
                flagged |= $detect(v);
                acc = $combine(acc, $clean(v));
            }

            let mut result = $reduce(acc);

            let tail_start = chunks * $lanes;
            for i in 0..remainder {
                // SAFETY: tail_start + i < len, so the read is in bounds.
                let v = unsafe { *values.get_unchecked(tail_start + i) };
                flagged |= $tail_flag(v);
                result = $tail(result, v);
            }
            $finish(result, flagged)
        }
    };
}

/// Generate a register-only vector `exp` kernel (`vexp_128`, `vexp_256`, …).
///
/// One instance per backend; every backend implements the same algorithm
/// (range reduce `x = n·ln2 + r` with the 2-part `ln2` split, round-half
/// fixup, degree-9 Taylor poly in `r`, scale by `2^n` with f32-range
/// clamps). The macro exists so the ~60-line unsafe body lives once, not
/// once per backend.
///
/// `V` is the backend's float vector type (`$vt`), `IV` its integer vector
/// type (`$ivt`). All comparisons produce FULL-WIDTH mask vectors (each
/// backend's closure expands its native mask to all-ones/all-zeros lanes).
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature before invoking it.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_exp {
    (
        $name:ident, $t:ty, $feat:literal, $vt:ty, $ivt:ty,
        $set1:expr, $set1i:expr,
        $mul:expr, $add:expr, $sub:expr,
        $andf:expr, $andnotf:expr, $orf:expr,
        $cmpgt_f:expr,
        $cast_iv:expr, $cast_vi:expr,
        $cvtt_f2i:expr, $slli_i:expr, $add_i:expr,
        $cmpgt_i:expr, $cmplt_i:expr,
        $and_i:expr, $andnot_i:expr, $or_i:expr
    ) => {
        /// Vector `exp`, register-only. Matches the scalar `kernels::exp::exp`
        /// to ≤ 2 ulp on the normal range, saturates below/above (see the
        /// sse2 invocation's doc for the full contract).
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        unsafe fn $name(v: $vt) -> $vt {
            // n = round(x * log2(e)) via add-magic (2^23 + 2^22, not
            // plain 2^23: for negative t the sum lands in the ulp-0.5
            // bin below 2^23, giving half-integer n and a sqrt(2) error).
            let t = $mul(v, $set1(1.442_695_0));
            let c2_23 = $set1(12_582_912.0);
            let n = $sub($add(t, c2_23), c2_23);
            // r = x - n*ln2_hi - n*ln2_lo (2-part split: no cancellation).
            let r = $sub(
                $sub(v, $mul(n, $set1(0.693_145_75))),
                $mul(n, $set1(1.428_606_77e-6)),
            );
            // Round-half fixup: nearest-even can pick the wrong side at
            // exact half-integers; if |r| > ln2/2, n += sign(r), retry.
            let abs_r = $andnotf($cast_iv($set1i(i32::MIN)), r);
            let too_big = $cmpgt_f(abs_r, $set1(0.346_573_6));
            let sign_of_r = $andf(r, $cast_iv($set1i(i32::MIN)));
            let n_adj = $andf($orf(sign_of_r, $set1(1.0)), too_big);
            let n = $add(n, n_adj);
            let r = $sub(
                $sub(v, $mul(n, $set1(0.693_145_75))),
                $mul(n, $set1(1.428_606_77e-6)),
            );
            // Zero r on saturated lanes BEFORE the poly: for huge |x| the
            // float→int convert saturates and r = x - n·ln2 is garbage
            // (the add-magic n is unchanged for |t| > 2^23), making the
            // Taylor poly NaN/inf (NaN × 0 = NaN defeats the exponent
            // clamp below). r = 0 gives poly = 1, so the clamped exponent
            // scale produces the correct 0 or inf.
            let n_int_early = $cvtt_f2i(n);
            let sat = $or_i(
                $cmplt_i(n_int_early, $set1i(-126)),
                $cmpgt_i(n_int_early, $set1i(128)),
            );
            // NaN lanes must keep propagating NaN, not zero r: cvtt(NaN)
            // saturates to i32::MIN so they would be classified as
            // under-saturated. Detect NaN via bits (exponent all-ones +
            // nonzero mantissa) and exempt them from the mask.
            let nan = $cmpgt_i(
                $and_i($cast_vi(v), $set1i(0x7fff_ffff)),
                $set1i(0x7f80_0000),
            );
            let sat = $andnot_i(nan, sat);
            let r = $andnotf($cast_iv(sat), r);
            // exp(r) degree-9 Taylor, Horner in r (NOT r² — r² would drop
            // odd powers and compute a cosh-like fn). Error < 0.35^10/10!.
            let p1 = $add($set1(1.0 / 362_880.0), $mul(r, $set1(1.0 / 3_628_800.0)));
            let p2 = $add($set1(1.0 / 40_320.0), $mul(r, p1));
            let p3 = $add($set1(1.0 / 5_040.0), $mul(r, p2));
            let p4 = $add($set1(1.0 / 720.0), $mul(r, p3));
            let p5 = $add($set1(1.0 / 120.0), $mul(r, p4));
            let p6 = $add($set1(1.0 / 24.0), $mul(r, p5));
            let p7 = $add($set1(1.0 / 6.0), $mul(r, p6));
            let p8 = $add($set1(0.5), $mul(r, p7));
            let p9 = $add($set1(1.0), $mul(r, p8));
            let p = $add($set1(1.0), $mul(r, p9));
            // 2^n via exponent bits, clamped: n < -126 → 0 (denormal
            // range not matched — needs variable per-lane shifts; the
            // scalar exp returns f32 denormals there, which contribute
            // < 1e-38 to a softmax sum and are normalized away —
            // documented difference), n > 128 → inf (replace bits: the
            // raw shift overflows into the sign bit, giving -inf).
            let n_int = $cvtt_f2i(n);
            let under = $cmplt_i(n_int, $set1i(-126));
            let over = $cmpgt_i(n_int, $set1i(128));
            let n_bits = $slli_i($add_i(n_int, $set1i(127)));
            let n_bits = $andnot_i(under, n_bits);
            let n_bits = $andnot_i(over, n_bits);
            let n_bits = $or_i(n_bits, $and_i(over, $set1i(0x7F80_0000)));
            $mul(p, $cast_iv(n_bits))
        }
    };
}

/// Generate a register-only vector `exp` kernel for `f64` (`vexp_128d`,
/// `vexp_256d`, …).
///
/// Same algorithm as the scalar `crate::kernels::exp::exp_f64` (fdlibm
/// double-double ln2 reduction + degree-20 Taylor + 2^n scaling), but
/// vectorized. Rounding of `n = round(x·log2e)` uses the 2^52 add-magic
/// (valid for f64 mantissa rounding), and the exponent scale uses 52-bit
/// left shifts on `i64` lanes.
///
/// The f32 [`simd_exp!`] cannot be reused: its 2^23 magic, `i32` lanes, and
/// f32 exponent clamps are width-specific. This macro takes the same
/// backend-closure shape, but with `i64` integer vector ops.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature before invoking it.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_exp_f64 {
    (
        $name:ident, $feat:literal, $vt:ty, $ivt:ty,
        $set1:expr, $set1i:expr,
        $mul:expr, $add:expr, $sub:expr,
        $cast_iv:expr, $cast_vi:expr,
        $round_f2i:expr, $cvti2f:expr, $slli_i:expr, $add_i:expr,
        $cmpgt_i:expr, $cmplt_i:expr,
        $and_i:expr, $andnot_i:expr, $or_i:expr
    ) => {
        /// Vector `f64` exp, register-only. Matches the scalar
        /// `kernels::exp::exp_f64` to ≤ 2 ulp on the normal range.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        unsafe fn $name(v: $vt) -> $vt {
            // n = round(x * log2(e)), via a backend round-to-nearest
            // float→int conversion (the 2^52 add-magic is not portable:
            // on some backends it mis-rounds negative inputs).
            let t = $mul(v, $set1(1.442_695_040_888_963_4)); // log2(e)
            let n_int = $round_f2i(t);
            let n = $cvti2f(n_int);
            // r = x - n*ln2_hi - n*ln2_lo (fdlibm double-double reduction).
            let r = $sub(
                $sub(v, $mul(n, $set1(6.931_471_803_691_238e-1))),
                $mul(n, $set1(1.908_214_929_270_588e-10)),
            );
            // Zero r on saturated lanes BEFORE the poly: for huge |x| the
            // float→int convert saturates and r = x - n·ln2 is garbage,
            // making the Taylor poly NaN/inf (NaN × 0 = NaN defeats the
            // exponent clamp below). r = 0 gives poly = 1, so the clamped
            // exponent scale produces the correct 0 or inf.
            let sat = $or_i(
                $cmplt_i(n_int, $set1i(-1022)),
                $cmpgt_i(n_int, $set1i(1023)),
            );
            // NaN lanes must keep propagating NaN, not force p = 1:
            // round(NaN) saturates so they'd be classified as saturated.
            // Detect NaN via bits and exempt it from the mask.
            let nan = $cmpgt_i(
                $and_i($cast_vi(v), $set1i(0x7fff_ffff_ffff_ffff)),
                $set1i(0x7ff0_0000_0000_0000),
            );
            let sat = $andnot_i(nan, sat);
            // exp(r) degree-20 Taylor, Horner in r (descending coefficients).
            let p1 = $add(
                $set1(1.0 / 2_432_902_008_176_640_000.0),
                $mul(r, $set1(1.0 / 121_645_100_408_832_000.0)),
            );
            let p2 = $add($set1(1.0 / 6_402_373_705_728_000.0), $mul(r, p1));
            let p3 = $add($set1(1.0 / 355_687_428_096_000.0), $mul(r, p2));
            let p4 = $add($set1(1.0 / 20_922_789_888_000.0), $mul(r, p3));
            let p5 = $add($set1(1.0 / 1_307_674_368_000.0), $mul(r, p4));
            let p6 = $add($set1(1.0 / 87_178_291_200.0), $mul(r, p5));
            let p7 = $add($set1(1.0 / 6_227_020_800.0), $mul(r, p6));
            let p8 = $add($set1(1.0 / 479_001_600.0), $mul(r, p7));
            let p9 = $add($set1(1.0 / 39_916_800.0), $mul(r, p8));
            let p10 = $add($set1(1.0 / 3_628_800.0), $mul(r, p9));
            let p11 = $add($set1(1.0 / 362_880.0), $mul(r, p10));
            let p12 = $add($set1(1.0 / 40_320.0), $mul(r, p11));
            let p13 = $add($set1(1.0 / 5_040.0), $mul(r, p12));
            let p14 = $add($set1(1.0 / 720.0), $mul(r, p13));
            let p15 = $add($set1(1.0 / 120.0), $mul(r, p14));
            let p16 = $add($set1(1.0 / 24.0), $mul(r, p15));
            let p17 = $add($set1(1.0 / 6.0), $mul(r, p16));
            let p18 = $add($set1(0.5), $mul(r, p17));
            let p19 = $add($set1(1.0), $mul(r, p18));
            let p = $add($set1(1.0), $mul(r, p19));
            // 2^n via exponent bits (52-bit shift), clamped: n < -1022 → 0
            // (denormal range not matched by the shift path), n > 1023 →
            // inf (raw shift overflows into the sign bit, giving -inf).
            let under = $cmplt_i(n_int, $set1i(-1022));
            let over = $cmpgt_i(n_int, $set1i(1023));
            let n_bits = $slli_i($add_i(n_int, $set1i(1023)));
            let n_bits = $andnot_i(under, n_bits);
            let n_bits = $andnot_i(over, n_bits);
            let n_bits = $or_i(n_bits, $and_i(over, $set1i(0x7FF0_0000_0000_0000)));
            // On saturated lanes force p = 1.0 (r was garbage, so the poly
            // could be NaN/inf); the exponent clamp below then produces
            // the correct 0 or inf.
            let p_bits = $cast_vi(p);
            let p_bits = $andnot_i(sat, p_bits);
            let p_bits = $or_i(p_bits, $and_i(sat, $cast_vi($set1(1.0))));
            let p = $cast_iv(p_bits);
            $mul(p, $cast_iv(n_bits))
        }
    };
}

/// Generate a register-only vector `ln` kernel for `f32` (`vln_128`,
/// `vln_256`, `vln_512`, `vln_neon`).
///
/// Mirrors the scalar `kernels::ln::ln` (fdlibm `__ieee754_log`): extract
/// exponent `k` and mantissa `f` so `x = 2^k·(1+f)` with `√2/2 < 1+f < √2`,
/// then `ln(x) = k·ln2_hi + (f - s·(f - R))` with `s = f/(2+f)` and the
/// degree-14 Lg polynomial `R(s²)`. Accuracy ≤ 1 ulp vs `f32::ln` over the
/// full finite range (see the scalar's dense sweep).
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature before invoking it.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$feat` — `target_feature` string.
/// * `$vt` — float vector type.
/// * `$ivt` — same-width integer vector type.
/// * `$set1f` / `$set1i` — scalar broadcasts.
/// * `$add/$sub/$mul/$div` — float ops.
/// * `$cast_iv` — `int → float` bit-cast.
/// * `$cast_vi` — `float → int` bit-cast.
/// * `$and_i/$or_i/$slli_i/$srli_i/$cmpgt_i` — integer ops.
/// * `$cmpgt_f/$cmplt_f/$cmpeq_f` — float compares (float-mask vectors).
/// * `$andf/$andnotf/$orf` — float-domain bit ops (for masking).
#[macro_export]
#[doc(hidden)]
macro_rules! simd_ln {
    (
        $name:ident, $feat:literal, $vt:ty, $ivt:ty,
        $set1f:expr, $set1i:expr,
        $add:expr, $sub:expr, $mul:expr,
        $cast_iv:expr, $cast_ib:expr, $cast_vi:expr,
        $and_i:expr, $or_i:expr,
        $srli_i:expr,
        $cmpgt_f:expr, $cmplt_f:expr, $cmpeq_f:expr,
        $andf:expr, $andnotf:expr, $orf:expr
    ) => {
        /// Vector `f32` `ln`, register-only. Matches the scalar
        /// `kernels::ln::ln` to ≤ 1 ulp on the normal range.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available and `v` has no
        /// special-case lanes (x ≤ 0, subnormal, inf, NaN); the enclosing
        /// map kernel's scalar tail handles those. This register path
        /// assumes normal positive finite `x`.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        unsafe fn $name(v: $vt) -> $vt {
            // Subnormal scaling is branchless here but the contract
            // excludes subnormals from the vector path (see the scalar
            // `kernels::ln::ln` which handles them); the map macro's
            // scalar tail covers the remainder lanes where subnormals
            // typically appear. For safety this macro still scales
            // in-range subnormals correctly when they do occur.
            let min_norm = $set1f(1.175_494_35e-38); // 2^-126
            let is_sub = $cmplt_f(v, min_norm);
            let scale = $orf($andf(is_sub, $set1f(33_554_432.0)), $set1f(1.0)); // 2^25
            let x = $mul(v, scale);
            let k_adj = $andf(is_sub, $set1f(-25.0));
            // exponent/mantissa extraction.
            let bits = $cast_vi(x);
            // Exponent: shifted right 23 → numeric convert (cvtepi32_ps);
            // mantissa: bit-pattern float in [1, 2) → bitcast.
            let exp = $sub($cast_iv($srli_i(bits)), $set1f(127.0));
            let mant = $or_i($and_i(bits, $set1i(0x7f_ffff)), $set1i(0x3f80_0000));
            let m = $cast_ib(mant); // m ∈ [1, 2)
            // normalize: m ≥ √2 → m/2, k += 1.
            // m ≥ √2 (including the boundary float, matching fdlibm's
            // bit test): ge = NOT(m < √2).
            let ge_sqrt2 = $andnotf($cmplt_f(m, $set1f(1.414_213_5)), $cast_ib($set1i(-1)));
            let m = $add($mul($andf(ge_sqrt2, m), $set1f(0.5)), $andnotf(ge_sqrt2, m));
            let k = $add($add(exp, $andf(ge_sqrt2, $set1f(1.0))), k_adj);
            let f = $sub(m, $set1f(1.0));
            // SLEEF division-free core (same reduction as fdlibm, one
            // degree-8 minimax poly instead of the s-form's division):
            // ln(1+f) = f - f²/2 + f³·P(f) on f ∈ [-0.293, 0.414).
            let f2 = $mul(f, f);
            let f3 = $mul(f2, f);
            let mut p = $set1f(7.037_683_629_2e-02); // P8
            p = $add($mul(p, f), $set1f(-1.151_461_031_0e-01)); // P7
            p = $add($mul(p, f), $set1f(1.167_699_874_0e-01));
            p = $add($mul(p, f), $set1f(-1.242_014_084_6e-01));
            p = $add($mul(p, f), $set1f(1.424_932_278_7e-01));
            p = $add($mul(p, f), $set1f(-1.666_805_766_5e-01));
            p = $add($mul(p, f), $set1f(2.000_071_476_5e-01));
            p = $add($mul(p, f), $set1f(-2.499_999_399_3e-01));
            p = $add($mul(p, f), $set1f(3.333_333_117_4e-01));
            // ln(x) = k·ln2_hi + (f - f²/2 + f³·P) + k·ln2_lo (wide's order).
            let k2lo = $mul(k, $set1f(-2.121_944_40e-04));
            let mut normal = $mul(f3, p);
            normal = $add(k2lo, normal);
            normal = $add($sub(f, $mul($set1f(0.5), f2)), normal);
            let k2hi = $mul(k, $set1f(6.933_593_75e-01));
            normal = $add(k2hi, normal);
            // Special-case masks (branchless): x < 0 (excludes -0.0) → NaN,
            // x = ±0 → -inf, x = +inf → +inf, NaN → NaN (propagate).
            // Order: NaN and zero are disjoint (cmpeq with NaN is false).
            let zero_v = $set1f(0.0);
            let nan_v = $set1f(f32::NAN);
            let is_neg = $cmplt_f(v, zero_v);
            let is_zero = $cmpeq_f(v, zero_v);
            let is_inf = $cmpeq_f(v, $set1f(f32::INFINITY));
            let is_nan = $andnotf($cmpeq_f(v, v), $cast_ib($set1i(-1)));
            let mut result = normal;
            result = $orf($andf(is_neg, nan_v), $andnotf(is_neg, result));
            result = $orf(
                $andf(is_zero, $set1f(f32::NEG_INFINITY)),
                $andnotf(is_zero, result),
            );
            result = $orf(
                $andf(is_inf, $set1f(f32::INFINITY)),
                $andnotf(is_inf, result),
            );
            result = $orf($andf(is_nan, nan_v), $andnotf(is_nan, result));
            result
        }
    };
}

/// Generate a register-only vector `ln` kernel for `f64` (`vln_128d`,
/// `vln_256d`, `vln_512d`).
///
/// Same fdlibm `__ieee754_log` reduction as [`simd_ln!`] but with the f64
/// constants (bit-exact fdlibm values), 52-bit exponent shifts, and bias
/// 1023. Accuracy ≤ 1 ulp vs `f64::ln` over the full finite range.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature before invoking it.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_ln_f64 {
    (
        $name:ident, $feat:literal, $vt:ty, $ivt:ty,
        $set1f:expr, $set1i:expr,
        $add:expr, $sub:expr, $mul:expr, $div:expr,
        $cast_iv:expr, $cast_ib:expr, $cast_vi:expr,
        $and_i:expr, $or_i:expr,
        $srli_i:expr,
        $cmpgt_f:expr, $cmplt_f:expr, $cmpeq_f:expr,
        $andf:expr, $andnotf:expr, $orf:expr
    ) => {
        /// Vector `f64` `ln`, register-only. Matches the scalar
        /// `kernels::ln::ln_f64` to ≤ 1 ulp on the normal range.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available. Subnormal lanes
        /// are scaled by 2^54 branchlessly; special cases are masked.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        unsafe fn $name(v: $vt) -> $vt {
            let min_norm = $set1f(2.225_073_858_507_201_4e-308); // 2^-1022
            let is_sub = $cmplt_f(v, min_norm);
            let scale = $orf($andf(is_sub, $set1f(18_014_398_509_481_984.0)), $set1f(1.0)); // 2^54
            let x = $mul(v, scale);
            let k_adj = $andf(is_sub, $set1f(-54.0));
            let bits = $cast_vi(x);
            // Exponent: i64 → f64 numeric conversion does not exist below
            // AVX-512DQ, so use the 2^52 magic: (e | 2^52) - 2^52 is an
            // exact integer→float convert for e ∈ [0, 2046].
            let e_int = $srli_i(bits);
            let e_magic = $cast_ib($or_i(e_int, $set1i(0x4330_0000_0000_0000)));
            let exp = $sub(
                $sub(e_magic, $set1f(4_503_599_627_370_496.0)),
                $set1f(1023.0),
            );
            let mant = $or_i(
                $and_i(bits, $set1i(0x000f_ffff_ffff_ffff)),
                $set1i(0x3ff0_0000_0000_0000),
            );
            let m = $cast_ib(mant);
            // m ≥ √2 (including the boundary float, matching fdlibm's
            // bit test): ge = NOT(m < √2).
            let ge_sqrt2 = $andnotf(
                $cmplt_f(m, $set1f(1.414_213_562_373_095_1)),
                $cast_ib($set1i(-1)),
            );
            let m = $add($mul($andf(ge_sqrt2, m), $set1f(0.5)), $andnotf(ge_sqrt2, m));
            let k = $add($add(exp, $andf(ge_sqrt2, $set1f(1.0))), k_adj);
            let f = $sub(m, $set1f(1.0));
            let s = $div(f, $add($set1f(2.0), f));
            let z = $mul(s, s);
            let w = $mul(z, z);
            let w2 = $mul(w, w);
            let t1 = $mul(
                w,
                $add(
                    $set1f(3.999_999_999_940_941_908e-01), // Lg2
                    $add(
                        $mul(w, $set1f(2.222_219_843_214_978_396e-01)), // Lg4
                        $mul(w2, $set1f(1.531_383_769_920_937_332e-01)), // Lg6
                    ),
                ),
            );
            let t2 = $mul(
                z,
                $add(
                    $set1f(6.666_666_666_666_735_130e-01), // Lg1
                    $add(
                        $mul(w, $set1f(2.857_142_874_366_239_149e-01)), // Lg3
                        $add(
                            $mul(w2, $set1f(1.818_357_216_161_805_012e-01)), // Lg5
                            $mul($mul(w2, w), $set1f(1.479_819_860_511_658_591e-01)), // Lg7
                        ),
                    ),
                ),
            );
            let r = $add(t2, t1);
            let k2hi = $mul(k, $set1f(6.931_471_803_691_238_164_90e-01));
            let k2lo = $mul(k, $set1f(1.908_214_929_270_587_700_02e-10));
            let inner = $sub(f, $mul(s, $sub(f, r)));
            // fdlibm: dk·ln2_hi − ((s·(f−R) − dk·ln2_lo) − f)
            //      = dk·ln2_hi + f − s·(f−R) + dk·ln2_lo  (note: + k·ln2_lo)
            let normal = $add(k2hi, $add(inner, k2lo));
            // Special-case masks (same contract as the f32 kernel).
            let zero_v = $set1f(0.0);
            let nan_v = $set1f(f64::NAN);
            let is_neg = $cmplt_f(v, zero_v);
            let is_zero = $cmpeq_f(v, zero_v);
            let is_inf = $cmpeq_f(v, $set1f(f64::INFINITY));
            let is_nan = $andnotf($cmpeq_f(v, v), $cast_ib($set1i(-1)));
            let mut result = normal;
            result = $orf($andf(is_neg, nan_v), $andnotf(is_neg, result));
            result = $orf(
                $andf(is_zero, $set1f(f64::NEG_INFINITY)),
                $andnotf(is_zero, result),
            );
            result = $orf(
                $andf(is_inf, $set1f(f64::INFINITY)),
                $andnotf(is_inf, result),
            );
            result = $orf($andf(is_nan, nan_v), $andnotf(is_nan, result));
            result
        }
    };
}

/// Generate a one-pass vector map kernel (`sigmoid`, `silu`, `gelu`, `relu`, …).
///
/// Every elementwise activation has the same shape: a chunked vector loop
/// (load → `$op` → store) plus a scalar tail over the remainder. The macro
/// generates that skeleton once; each backend supplies the vector `$op`
/// (e.g. `|v| div(set1(1.0), add(set1(1.0), exp(neg(v))))` for sigmoid) and
/// the scalar `$scalar` for the tail.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$store` — `fn(*mut $t, V)`.
/// * `$op` — `fn(V) -> V` elementwise map.
/// * `$scalar` — `fn($t) -> $t` scalar tail map.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and equal-length slices.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_map {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $store:expr, $op:expr, $scalar:expr
    ) => {
        /// Vector elementwise map. See the scalar reference for semantics.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available and that `values`
        /// and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t], out: &mut [$t]) {
            let len = values.len();
            let chunks = len / $lanes;
            let rem = len % $lanes;
            for i in 0..chunks {
                let v = $load(unsafe { values.as_ptr().add(i * $lanes) });
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, $op(v));
            }
            for i in 0..rem {
                let x = unsafe { *values.get_unchecked(chunks * $lanes + i) };
                let mapped = $scalar(x);
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) = mapped };
            }
        }
    };
}

/// Generate a one-pass vector clip kernel (`clamp(x, lo, hi)` per lane).
///
/// Same skeleton as [`simd_map!`], but the generated function takes the
/// `lo`/`hi` bound parameters, passed to both the vector `$op` and the
/// scalar tail `$scalar`.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$store` — `fn(*mut $t, V)`.
/// * `$op` — `fn(V, $t, $t) -> V` elementwise clamp.
/// * `$scalar` — `fn($t, $t, $t) -> $t` scalar tail clamp.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and equal-length slices.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_clip {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $store:expr, $op:expr, $scalar:expr
    ) => {
        /// Vector elementwise clip. See the scalar reference for semantics.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available and that `values`
        /// and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t], lo: $t, hi: $t, out: &mut [$t]) {
            let len = values.len();
            let chunks = len / $lanes;
            let rem = len % $lanes;
            for i in 0..chunks {
                let v = $load(unsafe { values.as_ptr().add(i * $lanes) });
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, $op(v, lo, hi));
            }
            for i in 0..rem {
                let x = unsafe { *values.get_unchecked(chunks * $lanes + i) };
                let mapped = $scalar(x, lo, hi);
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) = mapped };
            }
        }
    };
}

/// Generate a two-input vector elementwise map kernel (`op(a[i], b[i])`).
///
/// Same skeleton as [`simd_map!`], but loads from two equal-length input
/// slices `a` and `b` and applies a binary `$op` per lane; the scalar tail
/// uses `$scalar`.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$store` — `fn(*mut $t, V)`.
/// * `$op` — `fn(V, V) -> V` binary elementwise op.
/// * `$scalar` — `fn($t, $t) -> $t` scalar tail op.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and that `a`, `b`, and `out` all have
/// equal lengths.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_map2 {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $store:expr, $op:expr, $scalar:expr
    ) => {
        /// Vector two-input elementwise map. See the scalar reference for
        /// semantics.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available and that `a`,
        /// `b`, and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(a: &[$t], b: &[$t], out: &mut [$t]) {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let chunks = len / $lanes;
            let rem = len % $lanes;
            for i in 0..chunks {
                let va = $load(unsafe { a.as_ptr().add(i * $lanes) });
                let vb = $load(unsafe { b.as_ptr().add(i * $lanes) });
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, $op(va, vb));
            }
            for i in 0..rem {
                let x = unsafe { *a.get_unchecked(chunks * $lanes + i) };
                let y = unsafe { *b.get_unchecked(chunks * $lanes + i) };
                let mapped = $scalar(x, y);
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) = mapped };
            }
        }
    };
}

/// Generate an elementwise integer-power kernel (`x.powi(n)` per lane).
///
/// The exponent `n` is a scalar shared by all lanes, so the
/// exponentiation-by-squaring loop has the identical multiply sequence in
/// every lane — the result is bit-exact with the scalar `compiler-builtins`
/// algorithm (and thus `std::powi`). `$mul`/`$div`/`$one` are the vector
/// multiply/divide/broadcast-one; `$scalar` handles the tail.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$store` — `fn(*mut $t, V)`.
/// * `$mul` — `fn(V, V) -> V` vector multiply.
/// * `$div` — `fn(V, V) -> V` vector divide.
/// * `$one` — `V` broadcast of `1.0` (expression, may be re-evaluated).
/// * `$scalar` — `fn($t, i32) -> $t` scalar tail power.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and equal-length `values`/`out`.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_powi {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $store:expr, $mul:expr, $div:expr, $one:expr, $scalar:expr
    ) => {
        /// Vector elementwise integer power. See the scalar reference for
        /// semantics.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available and that
        /// `values` and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t], n: i32, out: &mut [$t]) {
            let len = values.len();
            let chunks = len / $lanes;
            let rem = len % $lanes;
            let recip = n < 0;
            let exp = n.unsigned_abs();
            for i in 0..chunks {
                let v = $load(unsafe { values.as_ptr().add(i * $lanes) });
                // Exponentiation by squaring: identical multiply sequence to
                // the scalar compiler-builtins `pow`, so each lane is
                // bit-exact with `std::powi`.
                let mut base = v;
                let mut acc = $one;
                let mut e = exp;
                loop {
                    if (e & 1) != 0 {
                        acc = $mul(acc, base);
                    }
                    e >>= 1;
                    if e == 0 {
                        break;
                    }
                    base = $mul(base, base);
                }
                let r = if recip { $div($one, acc) } else { acc };
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, r);
            }
            for i in 0..rem {
                let x = unsafe { *values.get_unchecked(chunks * $lanes + i) };
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) = $scalar(x, n) };
            }
        }
    };
}

/// Generate a predicate-count reduction kernel (returns the number of lanes
/// satisfying a predicate, as `usize`).
///
/// Each vector chunk is reduced to a lane-mask via `$pred`, counted via
/// `$count_lanes`; the scalar tail uses the boolean `$tail` predicate. Not
/// alloc-gated (it is a reduction, not a map).
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$pred` — `fn(V) -> Mask` per-lane predicate mask.
/// * `$count_lanes` — `fn(Mask) -> usize` popcount of set lanes.
/// * `$tail` — `fn($t) -> bool` scalar predicate for the remainder.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_count {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $pred:expr, $count_lanes:expr, $tail:expr
    ) => {
        /// SIMD predicate-count reduction. See the scalar reference for
        /// semantics.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available.
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t]) -> usize {
            let len = values.len();
            let chunks = len / $lanes;
            let rem = len % $lanes;
            let mut count = 0usize;
            for i in 0..chunks {
                let v = $load(unsafe { values.as_ptr().add(i * $lanes) });
                count += $count_lanes($pred(v));
            }
            for i in 0..rem {
                let x = unsafe { *values.get_unchecked(chunks * $lanes + i) };
                if $tail(x) {
                    count += 1;
                }
            }
            count
        }
    };
}

/// Generate an RMS-norm kernel (two-pass: sum of squares → rsqrt → scale).
///
/// Pass 1 reduces `sum(x²)` over vector chunks plus a scalar tail; pass 2
/// scales every element by `rsqrt(sum/n + eps)`. The rsqrt is the scalar
/// `kernels::sqrt::sqrt` (IEEE-correct), so only the add/mul are vectorized;
/// the reduction dominates and this matches the other reduction kernels.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$store` — `fn(*mut $t, V)`.
/// * `$acc_ident` — zero identity vector.
/// * `$combine` — `fn(V, V) -> V` `acc + v*v` per lane.
/// * `$reduce` — `fn(V) -> $t` horizontal sum.
/// * `$scale` — `fn(V, $t) -> V` multiply every lane by the scalar `1/√(ms+eps)`.
/// * `$sqrt` — scalar sqrt (`kernels::sqrt::sqrt` / `sqrt_f64`).
#[macro_export]
#[doc(hidden)]
macro_rules! simd_rms_norm {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $store:expr, $acc_ident:expr, $combine:expr, $reduce:expr,
        $scale:expr, $sqrt:expr
    ) => {
        /// Vector RMS norm. See the scalar reference for semantics.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available and that `values`
        /// and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[inline]
        #[allow(clippy::cast_precision_loss)] // `len as $t` is inherent to the mean
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t], eps: $t, out: &mut [$t]) {
            let len = values.len();
            let chunks = len / $lanes;
            let rem = len % $lanes;
            let ptr = values.as_ptr();

            let mut acc = $acc_ident;
            for i in 0..chunks {
                let v = $load(unsafe { ptr.add(i * $lanes) });
                acc = $combine(acc, v);
            }
            let mut sum_sq = $reduce(acc);
            for i in 0..rem {
                let x = unsafe { *values.get_unchecked(chunks * $lanes + i) };
                sum_sq += x * x;
            }
            // Empty input: out untouched (caller may pass an empty out).
            if len == 0 {
                return;
            }
            let inv = 1.0 / $sqrt(sum_sq / len as $t + eps);
            for i in 0..chunks {
                let v = $load(unsafe { ptr.add(i * $lanes) });
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, $scale(v, inv));
            }
            for i in 0..rem {
                let x = unsafe { *values.get_unchecked(chunks * $lanes + i) };
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) = x * inv };
            }
        }
    };
}

/// Generate an index-tracking reduction kernel (`argmax`, `argmin`).
///
/// The chunked vector loop keeps a *pair* accumulator — the extremum value
/// vector and the matching index vector — and per chunk updates both via a
/// lane-wise compare + blend. A horizontal reduction extracts the first
/// extremum lane, then a scalar tail resolves the remainder.
///
/// Tie-breaking: strict `$cmp` (`>` for argmax, `<` for argmin) means the
/// first occurrence of the extremum wins, both across chunks and lanes.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V` loading `$lanes` elements.
/// * `$vidx` — constant integer vector `0..$lanes` (lane indices).
/// * `$set1i` — `fn(i32) -> IV` broadcast an integer scalar.
/// * `$addi` — `fn(IV, IV) -> IV` integer vector add.
/// * `$cmp` — `fn(V, V) -> Mask` lane-wise strict compare
///   (`>` for argmax, `<` for argmin).
/// * `$blend` — `fn(Mask, V, V) -> V` lane-wise select on the value vector
///   (`a` where mask, `b` elsewhere); the mask type is backend-specific.
/// * `$blend_idx` — `fn(Mask, IV, IV) -> IV` lane-wise select on the index
///   vector, using the same mask.
/// * `$cmp_scalar` — `fn($t, $t) -> bool` scalar strict compare, used by
///   the tail (`>` for argmax, `<` for argmin).
/// * `$reduce_pair` — `fn(V, IV) -> ($t, usize)` horizontal reduction
///   yielding the first extremum lane's value and index.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and that `values` is non-empty.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_argminmax {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $vidx:expr, $set1i:expr, $addi:expr,
        $cmp:expr, $blend:expr, $blend_idx:expr, $cmp_scalar:expr, $reduce_pair:expr
    ) => {
        /// SIMD index-tracking reduction. See the scalar reference for
        /// semantics (first occurrence of the extremum).
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available and that
        /// `values` is non-empty.
        ///
        /// Index arithmetic is i32 (SIMD lane width); slices beyond
        /// 2^31 elements would wrap — not a practical limit for this kernel.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t]) -> ($t, usize) {
            let len = values.len();
            debug_assert!(len > 0);
            let ptr = values.as_ptr();
            let chunks = len / $lanes;
            let remainder = len % $lanes;

            let mut result = if chunks == 0 {
                // No full chunk: seed from element 0, tail covers the rest.
                (unsafe { *ptr }, 0)
            } else {
                let mut vmax = $load(unsafe { ptr });
                let mut imax = $vidx;
                for i in 1..chunks {
                    let v = $load(unsafe { ptr.add(i * $lanes) });
                    let off = $addi($set1i((i * $lanes) as _), $vidx);
                    let mask = $cmp(v, vmax);
                    vmax = $blend(mask, v, vmax);
                    imax = $blend_idx(mask, off, imax);
                }
                $reduce_pair(vmax, imax)
            };

            let tail_start = chunks * $lanes;
            for i in 0..remainder {
                // SAFETY: tail_start + i < len.
                let v = unsafe { *values.get_unchecked(tail_start + i) };
                let j = tail_start + i;
                let (mv, mi) = result;
                // NaN-aware: a non-NaN candidate dethrones a NaN seed.
                result = if !v.is_nan() && (mv.is_nan() || $cmp_scalar(v, mv)) {
                    (v, j)
                } else {
                    (mv, mi)
                };
            }

            result
        }
    };
}

/// Generate a two-pass `logsumexp` reduction kernel: `max + ln(Σ exp(x − max))`.
///
/// Returns the log-sum-exp as a scalar — no output buffer. Pass 1 reduces
/// the max (vector chunks + scalar tail); pass 2 reduces `Σ exp(x − max)`
/// without storing anything; the result is `max + ln(sum)`.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type (`f32` or `f64`).
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$max` — `fn(V, V) -> V` elementwise max.
/// * `$sub` — `fn(V, V) -> V` elementwise sub.
/// * `$exp` — `fn(V) -> V` elementwise exp.
/// * `$reduce` — `fn(V) -> $t` horizontal sum.
/// * `$max_reduce` — `fn(V) -> $t` horizontal max.
/// * `$set1` — `fn($t) -> V` broadcast.
/// * `$exp_scalar` — `fn($t) -> $t` scalar exp for the tail.
/// * `$ln` — `fn($t) -> $t` scalar ln (the std-free kernel).
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_logsumexp {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $max:expr, $sub:expr, $exp:expr,
        $reduce:expr, $max_reduce:expr, $set1:expr,
        $exp_scalar:expr, $ln:expr
    ) => {
        /// SIMD log-sum-exp reduction. See the scalar reference for
        /// semantics; empty input yields `NEG_INFINITY`.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available.
        #[cfg(feature = "alloc")]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t]) -> $t {
            let len = values.len();
            if len == 0 {
                return <$t>::NEG_INFINITY;
            }
            let ptr = values.as_ptr();
            let chunks = len / $lanes;
            let rem = len % $lanes;

            // Pass 1: max over all chunks (skip vector loads if none fit).
            let mut m = if chunks > 0 {
                let mut vmax = $load(unsafe { ptr });
                for i in 1..chunks {
                    let v = $load(unsafe { ptr.add(i * $lanes) });
                    vmax = $max(vmax, v);
                }
                $max_reduce(vmax)
            } else {
                <$t>::NEG_INFINITY
            };
            for i in 0..rem {
                m = <$t>::max(m, unsafe { *values.get_unchecked(chunks * $lanes + i) });
            }

            // Pass 2: Σ exp(x − m), no store.
            let vm = $set1(m);
            let mut sum = 0.0;
            for i in 0..chunks {
                let v = $load(unsafe { ptr.add(i * $lanes) });
                sum += $reduce($exp($sub(v, vm)));
            }
            for i in 0..rem {
                sum += $exp_scalar(unsafe { *values.get_unchecked(chunks * $lanes + i) } - m);
            }

            m + $ln(sum)
        }
    };
}

/// Generate a three-pass `log_softmax` kernel writing into `out`:
/// `x_i − logsumexp(x)` elementwise, 0-alloc.
///
/// Pass 1 reduces the max; pass 2 reduces `Σ exp(x − m)` (no store); pass 3
/// writes `(x_i − m) − ln(sum)` directly. Reading `values` in the final pass
/// (instead of a stored intermediate) is what makes the kernel allocation-free.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type.
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$store` — `fn(*mut $t, V)`.
/// * `$max` — `fn(V, V) -> V` elementwise max.
/// * `$sub` — `fn(V, V) -> V` elementwise sub.
/// * `$exp` — `fn(V) -> V` elementwise exp.
/// * `$reduce` — `fn(V) -> $t` horizontal sum.
/// * `$max_reduce` — `fn(V) -> $t` horizontal max.
/// * `$set1` — `fn($t) -> V` broadcast.
/// * `$exp_scalar` — `fn($t) -> $t` scalar exp for the tail.
/// * `$ln` — `fn($t) -> $t` scalar ln.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and that `values` and `out` have
/// equal lengths.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_log_softmax {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $store:expr, $max:expr, $sub:expr, $exp:expr,
        $reduce:expr, $max_reduce:expr, $set1:expr,
        $exp_scalar:expr, $ln:expr
    ) => {
        /// SIMD log-softmax kernel. See the scalar reference for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available and that
        /// `values` and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t], out: &mut [$t]) {
            let len = values.len();
            if len == 0 {
                return;
            }
            let ptr = values.as_ptr();
            let chunks = len / $lanes;
            let rem = len % $lanes;

            // Pass 1: max (same shape as simd_softmax).
            let mut m = if chunks > 0 {
                let mut vmax = $load(unsafe { ptr });
                for i in 1..chunks {
                    let v = $load(unsafe { ptr.add(i * $lanes) });
                    vmax = $max(vmax, v);
                }
                $max_reduce(vmax)
            } else {
                <$t>::NEG_INFINITY
            };
            for i in 0..rem {
                m = <$t>::max(m, unsafe { *values.get_unchecked(chunks * $lanes + i) });
            }

            // Pass 2: Σ exp(x − m), no store.
            let vm = $set1(m);
            let mut sum = 0.0;
            for i in 0..chunks {
                let v = $load(unsafe { ptr.add(i * $lanes) });
                sum += $reduce($exp($sub(v, vm)));
            }
            for i in 0..rem {
                sum += $exp_scalar(unsafe { *values.get_unchecked(chunks * $lanes + i) } - m);
            }

            // Pass 3: out[i] = (x_i − m) − ln(sum). Subtracting ln(sum)
            // separately from (x_i − m) — never folding it into m — keeps
            // ln(sum) from vanishing in the ulp of a large m.
            let log_sum = $ln(sum);
            let vm = $set1(m);
            let vshift = $set1(log_sum);
            for i in 0..chunks {
                let v = $load(unsafe { ptr.add(i * $lanes) });
                let d = $sub(v, vm);
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, $sub(d, vshift));
            }
            for i in 0..rem {
                unsafe {
                    *out.get_unchecked_mut(chunks * $lanes + i) =
                        (*values.get_unchecked(chunks * $lanes + i) - m) - log_sum;
                }
            }
        }
    };
}

/// Generate a three-pass `layer_norm` kernel writing into `out`:
/// `(x_i − mean) / sqrt(var + eps)`, 0-alloc.
///
/// Pass 1 reduces the mean; pass 2 stores the centered value `x − mean` into
/// `out` while accumulating `Σ (x − mean)²`; pass 3 scales `out` by
/// `1/sqrt(var + eps)`. Reusing the stored centered values in pass 3 (instead
/// of re-reading `values`) is what makes the kernel allocation-free.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$t` — scalar element type.
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `$t` elements.
/// * `$load` — `fn(*const $t) -> V`.
/// * `$store` — `fn(*mut $t, V)`.
/// * `$add` — `fn(V, V) -> V` elementwise add.
/// * `$sub` — `fn(V, V) -> V` elementwise sub.
/// * `$acc_ident` — zero identity vector.
/// * `$combine` — `fn(V, V) -> V` folding `acc + v*v`.
/// * `$reduce` — `fn(V) -> $t` horizontal sum.
/// * `$set1` — `fn($t) -> V` broadcast.
/// * `$scale` — `fn(V, $t) -> V` multiply every lane by the scalar `1/√(var+eps)`.
/// * `$sqrt` — scalar `fn($t) -> $t` sqrt (the std-free kernel).
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and that `values` and `out` have
/// equal lengths.
#[macro_export]
#[doc(hidden)]
macro_rules! simd_layer_norm {
    (
        $name:ident, $t:ty, $feat:literal, $lanes:expr,
        $load:expr, $store:expr, $add:expr, $sub:expr,
        $acc_ident:expr, $combine:expr, $reduce:expr, $set1:expr,
        $scale:expr, $sqrt:expr
    ) => {
        /// SIMD layer norm kernel. See the scalar reference for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available and that
        /// `values` and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[allow(clippy::cast_precision_loss)] // `len as $t` is inherent to the mean
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[$t], eps: $t, out: &mut [$t]) {
            let len = values.len();
            if len == 0 {
                return;
            }
            let ptr = values.as_ptr();
            let chunks = len / $lanes;
            let rem = len % $lanes;

            // Pass 1: mean.
            let mut acc = $acc_ident;
            for i in 0..chunks {
                acc = $add(acc, $load(unsafe { ptr.add(i * $lanes) }));
            }
            let mut sum = $reduce(acc);
            for i in 0..rem {
                sum += unsafe { *values.get_unchecked(chunks * $lanes + i) };
            }
            let mean = sum / len as $t;

            // Pass 2: store centered values, accumulate Σ (x − mean)².
            let vmean = $set1(mean);
            let mut sq = $acc_ident;
            for i in 0..chunks {
                let c = $sub($load(unsafe { ptr.add(i * $lanes) }), vmean);
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, c);
                sq = $combine(sq, c);
            }
            let mut sum_sq = $reduce(sq);
            for i in 0..rem {
                let c = unsafe { *values.get_unchecked(chunks * $lanes + i) } - mean;
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) = c };
                sum_sq += c * c;
            }

            // Pass 3: scale by 1/sqrt(var + eps) (population variance).
            let inv = 1.0 / $sqrt(sum_sq / len as $t + eps);
            for i in 0..chunks {
                let v = $load(unsafe { out.as_ptr().add(i * $lanes) });
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, $scale(v, inv));
            }
            for i in 0..rem {
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) *= inv };
            }
        }
    };
}
