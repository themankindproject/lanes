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
/// * `$feat` — `target_feature` string (e.g. `"avx2"`).
/// * `$lanes` — vector width in `f32` elements (4, 8, 16).
/// * `$load` — `fn(*const f32) -> V` loading `$lanes` elements from a
///   pointer (e.g. `|p| unsafe { _mm256_loadu_ps(p) }`).
/// * `$acc_ident` — expression yielding the identity accumulator for the
///   reduction (e.g. `_mm256_setzero_ps()` or `_mm256_set1_ps(f32::INFINITY)`).
/// * `$combine` — `fn(V, V) -> V` folding each loaded chunk into the
///   accumulator (e.g. `_mm256_add_ps` or `_mm256_min_ps`).
/// * `$reduce` — `fn(V) -> f32` horizontal reduction of the final
///   accumulator (e.g. `hsum_256` or `_mm512_reduce_add_ps`).
/// * `$tail` — a `fn(f32, f32) -> f32` applied to `(result_so_far, element)`
///   for the scalar tail (e.g. `|r, v| r + v`, or `f32::min`).
#[macro_export]
macro_rules! simd_reduce {
    ($name:ident, $feat:literal, $lanes:expr, $load:expr, $acc_ident:expr, $combine:expr, $reduce:expr, $tail:expr) => {
        /// SIMD reduction kernel. See the enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available.
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[f32]) -> f32 {
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
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `f32` elements.
/// * `$load` — `fn(*const f32) -> V`.
/// * `$store` — `fn(*mut f32, V)`.
/// * `$max` — `fn(V, V) -> V` elementwise max.
/// * `$sub` — `fn(V, V) -> V` elementwise sub.
/// * `$exp` — `fn(V) -> V` elementwise exp.
/// * `$add` — `fn(V, V) -> V` elementwise add.
/// * `$mul` — `fn(V, V) -> V` elementwise mul.
/// * `$reduce` — `fn(V) -> f32` horizontal sum.
/// * `$max_reduce` — `fn(V) -> f32` horizontal max (distinct from `$reduce`).
/// * `$set1` — `fn(f32) -> V` broadcast a scalar to all lanes.
#[macro_export]
macro_rules! simd_softmax {
    ($name:ident, $feat:literal, $lanes:expr, $load:expr, $store:expr, $max:expr, $sub:expr, $exp:expr, $add:expr, $mul:expr, $reduce:expr, $max_reduce:expr, $set1:expr) => {
        /// SIMD softmax kernel. See the enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available and that
        /// `values` and `out` have equal lengths.
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[f32], out: &mut [f32]) {
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
                f32::NEG_INFINITY
            };
            for i in 0..rem {
                max = f32::max(max, unsafe { *values.get_unchecked(chunks * $lanes + i) });
            }

            // Pass 2: exp(x - max) and sum.
            let vmax_b = $set1(max);
            let mut sum = 0.0_f32;
            for i in 0..chunks {
                let v = $load(unsafe { values.as_ptr().add(i * $lanes) });
                let e = $exp($sub(v, vmax_b));
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, e);
                sum += $reduce(e);
            }
            for i in 0..rem {
                let e = $crate::kernels::exp::exp(
                    unsafe { *values.get_unchecked(chunks * $lanes + i) } - max,
                );
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

///
/// # Parameters
///
/// * `$name` — function name.
/// * `$feat1`, `$feat2` — the two `target_feature` strings.
/// * `$lanes` — vector width in `f32` elements.
/// * `$load` — `fn(*const f32) -> V` loading `$lanes` elements.
/// * `$acc_ident` — identity accumulator expression.
/// * `$combine` — `fn(V, V, V) -> V` folding the product chunk into the
///   accumulator.
/// * `$reduce` — horizontal reduction `fn(V) -> f32`.
/// * `$tail` — a `fn(f32, f32, f32) -> f32` applied to
///   `(result_so_far, va, vb)` for the scalar tail.
#[macro_export]
macro_rules! simd_reduce2_feat {
    ($name:ident, $feat1:literal, $feat2:literal, $lanes:expr, $load:expr, $acc_ident:expr, $combine:expr, $reduce:expr, $tail:expr) => {
        /// SIMD two-input reduction kernel. See the enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU features are available and that
        /// `a` and `b` have equal lengths.
        #[target_feature(enable = $feat1, enable = $feat2)]
        pub(crate) unsafe fn $name(a: &[f32], b: &[f32]) -> f32 {
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

/// Generate a two-input reduction kernel (`dot`, and future pairwise ops).
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `f32` elements.
/// * `$load` — `fn(*const f32) -> V` loading `$lanes` elements.
/// * `$acc_ident` — identity accumulator expression.
/// * `$combine` — `fn(V, V, V) -> V` folding the product chunk into the
///   accumulator (e.g. `_mm256_fmadd_ps`, or
///   `|acc, va, vb| _mm_add_ps(acc, _mm_mul_ps(va, vb))` without FMA).
/// * `$reduce` — horizontal reduction `fn(V) -> f32`.
/// * `$tail` — a `fn(f32, f32, f32) -> f32` applied to
///   `(result_so_far, va, vb)` for the scalar tail (e.g. `|r, a, b| r + a * b`).
#[macro_export]
macro_rules! simd_reduce2 {
    ($name:ident, $feat:literal, $lanes:expr, $load:expr, $acc_ident:expr, $combine:expr, $reduce:expr, $tail:expr) => {
        /// SIMD two-input reduction kernel. See the enclosing module for semantics.
        ///
        /// # Safety
        /// Caller must guarantee the CPU feature is available and that
        /// `a` and `b` have equal lengths.
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(a: &[f32], b: &[f32]) -> f32 {
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
macro_rules! simd_exp {
    (
        $name:ident, $feat:literal, $vt:ty, $ivt:ty,
        $set1:expr, $set1i:expr,
        $mul:expr, $add:expr, $sub:expr,
        $andf:expr, $andnotf:expr, $orf:expr,
        $cmpgt_f:expr,
        $cast_vi:expr, $cast_iv:expr,
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
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `f32` elements.
/// * `$load` — `fn(*const f32) -> V`.
/// * `$store` — `fn(*mut f32, V)`.
/// * `$op` — `fn(V) -> V` elementwise map.
/// * `$scalar` — `fn(f32) -> f32` scalar tail map.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and equal-length slices.
#[macro_export]
macro_rules! simd_map {
    (
        $name:ident, $feat:literal, $lanes:expr,
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
        pub(crate) unsafe fn $name(values: &[f32], out: &mut [f32]) {
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

/// Generate a one-pass vector map kernel with extra scalar parameters
/// (`clip` and future parameterized maps).
///
/// Same skeleton as [`simd_map!`], but the generated function takes two
/// extra `f32` parameters, passed to both the vector `$op` and the scalar
/// `$scalar`.
///
/// # Parameters
///
/// * `$name` — function name.
/// * `$feat` — `target_feature` string.
/// * `$lanes` — vector width in `f32` elements.
/// * `$load` — `fn(*const f32) -> V`.
/// * `$store` — `fn(*mut f32, V)`.
/// * `$op` — `fn(V, f32, f32) -> V` elementwise map (params after the vector).
/// * `$scalar` — `fn(f32, f32, f32) -> f32` scalar tail map.
///
/// # Safety
/// The generated function is `unsafe fn` with `#[target_feature]`; the
/// caller must verify the CPU feature and equal-length slices.
#[macro_export]
macro_rules! simd_map_param {
    (
        $name:ident, $feat:literal, $lanes:expr,
        $load:expr, $store:expr, $op:expr, $scalar:expr
    ) => {
        /// Vector elementwise map with parameters. See the scalar reference.
        ///
        /// # Safety
        /// Caller must ensure the CPU feature is available and that `values`
        /// and `out` have equal lengths.
        #[cfg(feature = "alloc")]
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(values: &[f32], p1: f32, p2: f32, out: &mut [f32]) {
            let len = values.len();
            let chunks = len / $lanes;
            let rem = len % $lanes;
            for i in 0..chunks {
                let v = $load(unsafe { values.as_ptr().add(i * $lanes) });
                $store(unsafe { out.as_mut_ptr().add(i * $lanes) }, $op(v, p1, p2));
            }
            for i in 0..rem {
                let x = unsafe { *values.get_unchecked(chunks * $lanes + i) };
                let mapped = $scalar(x, p1, p2);
                unsafe { *out.get_unchecked_mut(chunks * $lanes + i) = mapped };
            }
        }
    };
}
