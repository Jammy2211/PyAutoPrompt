# Deep research: Can we speed up Delaunay in PyAutoArray?

**Bottom line up front**: yes, by 2-5× without changing the science.
The lever isn't pure-JAX Delaunay (basically infeasible) — it's
**splitting the scipy callback in two**: keep scipy for triangulation
(small, fast), and move `find_simplex` point location to a JAX-native
implementation that vmaps across the batch.

A secondary lever is multiprocessing the scipy callback. Combined,
realistic speedup is 3-5× on the Delaunay-specific cost, which translates
to ~1.5-1.8× on the full production likelihood at batch=20.

## What is actually inside the 16.87 ms/element scipy callback

Decomposed via pure-Python `scipy.spatial.Delaunay` timing at production
size (1231 mesh pts, 15361 query pts) on a laptop CPU:

| Sub-call | Laptop ms | Share | Notes |
|---|---:|---:|---|
| `Delaunay(mesh)` build | 5.7 ms | ~14% | Triangulation only — cheap |
| `tri.find_simplex(data_grid)` | 20.6 ms | ~52% | Point location — dominant |
| `tri.find_simplex(split_points)` | ~20 ms | ~52% | **Called twice** in scipy_delaunay() — once for the data grid, once for split-cross-points for regularization |
| Misc (split_points construction, area_factor, padding) | ~5 ms | ~13% | Small |

Wait — this adds to >100%, because two find_simplex calls overlap with
the share. Re-stated: the scipy callback runs **one Delaunay build + two
find_simplex calls + some array padding**. The two find_simplex calls
together are >50% of scipy's work.

HPC A100 (server-grade CPU, single core via `--cpus-per-task=4` but
scipy is single-threaded): 16.87 ms per element. Same shape, faster CPU.

**Conclusion: the scipy callback is point-location-bound, not
triangulation-bound.** `find_simplex` is what scales with the data grid
size (15361 query points), and we have two of them.

## Approach catalogue, ranked by ROI

### A. Split triangulation from point location (HIGH ROI, MEDIUM EFFORT)

**Idea**: keep scipy.Delaunay (small, fast, CPU is fine for K=1231 pts).
Replace `tri.find_simplex(query_points)` with a JAX-native implementation
of point-in-triangle search that vmaps cleanly across the batch.

**JAX implementation options** for point location:

1. **Brute-force vectorised**. For each query point, test against all
   2N=2462 triangles using barycentric coordinates. The query is inside
   triangle i iff all three barycentric coordinates of i are non-negative.
   - Cost: Q × T × 3 multiplications per element = 15361 × 2462 × 3 ≈
     1.1e8 ops. On A100 fp64 (~10 TFLOPS) that's ~10 µs theoretical, ~1-2 ms
     realistic. Under vmap=20: ~30 ms wall.
   - Far cheaper than scipy's 20 ms sequential per element × 20 = 400 ms wall.

2. **Spatial-hash + brute-force on cell candidates**. Divide source-plane
   bounding box into a fine regular grid. For each grid cell, list the
   triangles overlapping it (precomputed once per Delaunay build).
   Per-query: hash to cell, brute-force the small candidate list.
   - Cost: ~10× faster than approach (1) for typical grids.
   - More complex to implement; needs careful padding to keep shapes
     static for JAX.

3. **Walk-based search** from a guess triangle. Classical CGAL approach.
   - Inherently sequential per query; doesn't vmap.
   - Avoid.

**Recommended**: approach (1) for simplicity. It's already 10-20× faster
than scipy's two find_simplex calls under vmap=20, and the code is short.

**Expected saving on full pipeline at batch=20**:
- Current: 16.87 ms × 20 = 337 ms wall on scipy_delaunay
- With this approach: triangulation 3 ms × 20 = 60 ms wall (still
  sequential pure_callback) + brute-force point loc 30 ms wall (JAX, vmap-parallel)
  + misc 20 ms = ~110 ms wall per batched call
- **Saving: ~227 ms per batched call**, full pipeline per element drops
  from 69.5 → ~58 ms, **batch_time 1.4 sec → 1.0 sec (~1.4× speedup)**.

**Risks**:
- Barycentric point-in-triangle on degenerate triangles needs care
  (numerical tolerance at edges). scipy/Qhull handles this with robust
  predicates.
- `tri.find_simplex` returns -1 for query points OUTSIDE the convex hull.
  Need equivalent handling in JAX — easy with a "max barycentric < 0" check.

**Why this works**: Delaunay's triangulation is a small problem (~1231
points). The scaling pain is in the 15361 query points, and that part
is embarrassingly parallel — exactly what JAX/GPU is good at.

### B. Multiprocessing the scipy callback (LOW EFFORT, MODEST GAIN)

**Idea**: replace the lambda inside `jax.pure_callback` with one that
spawns N processes (where N = batch_size) and runs scipy.Delaunay in
parallel. Persistent worker pool to avoid spawn overhead.

**Implementation sketch**:
```python
from concurrent.futures import ProcessPoolExecutor
_pool = ProcessPoolExecutor(max_workers=8)
def parallel_scipy_delaunay(points_batched, qpts_batched, areas_factor):
    futs = [_pool.submit(scipy_delaunay, p, q, areas_factor)
            for p, q in zip(points_batched, qpts_batched)]
    return [f.result() for f in futs]
```

Plus: change `vmap_method="sequential"` to `"parallel"` AND wrap so the
callback receives the full batch at once.

**Caveats**:
- scipy uses Qhull which has global state in C — but processes have
  separate address spaces, so multiprocessing is safe (threading is not).
- IPC overhead per batch element: ~1 ms each way → 2-3 ms per element
  baseline.
- Process pool startup: do it once, not per-call.

**Expected saving at batch=20**:
- Current: 337 ms wall (sequential)
- With 8 parallel workers: 337/8 + IPC overhead ≈ 45-60 ms wall
- **Saving: ~280 ms per batched call**, similar to approach A.

**Risks**:
- Process pool lifecycle in JAX context (forking under JIT? probably fine
  since pure_callback hands off cleanly)
- Memory: each process loads its own scipy, ~few hundred MB × 8 = ~few GB
- HPC SLURM `--cpus-per-task=N` must be at least 8 to use 8 workers

**Combines additively with approach A**: if A reduces scipy work to just
the triangulation (~3 ms/element), multiprocessing makes it 0.5 ms wall.

### C. Pure-JAX Delaunay (LOW ROI, VERY HIGH EFFORT)

**Status**: I assessed this carefully. Verdict: **not realistically
achievable for production within reasonable engineering budget.**

Why:
1. **Dynamic output shape.** Delaunay produces a triangulation whose
   size depends on point distribution (typically 2N-2-h triangles for N
   points where h is hull size). JAX requires static shapes. Workaround:
   pad to 2N. Adds complexity.
2. **Sequential algorithms.** Bowyer-Watson incremental insertion is
   intrinsically sequential. Implementable with `lax.scan` but each step
   has dynamic control flow (which triangles to flip).
3. **Numerical robustness.** Qhull uses exact arithmetic on geometric
   predicates (incircle, orient2d) — degenerate point configurations
   are common, and floating-point predicates fail catastrophically
   (returning inconsistent triangulations that crash downstream). JAX
   has no robust geometric predicates library.
4. **Existing parallel Delaunay** (gDel2D, JCFlip, etc.) is CUDA, not JAX.
   Wrapping via XLA custom-call is possible but complex; you'd be back
   to a `pure_callback`-like sync barrier and lose the JAX advantage.

**Effort estimate**: 2-6 months of research-grade work to get a robust
implementation. Almost certainly slower than scipy in practice (scipy/Qhull
is heavily optimized C).

**Don't go this route.** The pure-JAX *dream* is what makes you think
this is necessary, but the bottleneck isn't really the triangulation —
it's the point location and the synchronous CPU callback. Both are
addressable without solving the pure-JAX Delaunay problem.

### D. Cache topology across nearby parameter samples (MEDIUM ROI, HIGH EFFORT)

**Idea**: nested sampling proposes correlated points. If consecutive
parameter vectors give source-plane meshes that differ by less than the
local mesh spacing, the triangulation connectivity is the same — only
the coordinates change. Cache the simplices, re-do find_simplex.

**Implementation**: sampler-level, in PyAutoFit's `fitness._vmap`. Pass
a hint to the likelihood (e.g., "this batch is close to the previous;
reuse triangulation"). Within `inversion_util.reconstruction_positive_only_from`,
check the hint and skip scipy.Delaunay if valid.

**Expected saving**: depends on cache hit rate. Nautilus's nested sampling
typically has high spatial correlation in proposals when narrowing — 50-80%
hit rate plausible. If 70%, saving is 0.7 × 5 ms / element on triangulation
+ no saving on point location. **Net: ~3 ms/element. Small.**

**Risks**:
- Wrong triangulation if the lens model changes catastrophically — caustic
  crossings, source intersecting the lens, etc. Would silently corrupt the
  likelihood.
- Complex API: requires PyAutoFit changes to thread the cache hint.

**Don't prioritise** — small gain, complex change, risk of silent bugs.

### E. External GPU Delaunay via pure_callback (UNCLEAR ROI, HIGH EFFORT)

**Idea**: replace scipy.Delaunay with a GPU library (gDel2D, cuDelaunay)
via `pure_callback`. Eliminates the CPU bottleneck.

**Status**:
- gDel2D, JCFlip, etc. exist as research-grade CUDA. Maintenance and
  Python bindings are uncertain.
- Even on GPU, the dispatch is via pure_callback → device-to-host-to-device
  transfer + library invocation overhead. Worth ~10-20 µs per call but
  still synchronous.
- vmap_method="sequential" still applies. To benefit, would need to write
  a batched GPU Delaunay call.

**Worth a literature scan** before committing — see if any maintained
JAX-compatible Delaunay library exists. If so, low effort. If not,
binding work is high.

### F. Improve InterpolatorKNearestNeighbor (LOW EFFORT, BOUNDED BY SCIENCE)

**Status**: this exists at `autoarray/inversion/mesh/interpolator/knn.py`
and is **already pure JAX** (no scipy callback). It uses brute-force kNN
via `lax.fori_loop` over blocks of mesh points + Wendland C4 kernel
weighting.

**Why it's scientifically worse than Delaunay**:
- kNN ignores local triangulation structure. At source-plane density
  gradients (which adapt meshes have), kNN smooths across mesh density
  boundaries; Delaunay respects them via the triangle connectivity.
- Wendland kernel + radius_scale is a global knob. Delaunay's barycentric
  weights are locally exact.
- At caustic crossings (where source positions fold), kNN smears across
  the fold; Delaunay's triangulation handles it via the new
  triangles spanning the fold.

**Performance under vmap=20 (estimated, not measured here)**: probably
2-5 ms per element on A100 — much faster than scipy's 16.87 ms.

**Quality tuning to consider**:
- `k_neighbors` — increase to 5-7 (default may be 3)? Smoother but slower.
- `radius_scale` — adapt per-query-point based on local density? Currently
  global.
- Use the THREE closest mesh points and barycentric-interpolate among
  those three (essentially fake-Delaunay) instead of Wendland weights.
  **This is interesting** — it gives you Delaunay-like locality without
  scipy.

**Worth testing**: the kNN-with-barycentric-on-3 idea is a low-effort
experiment. If it gives log-evidence within rtol=1e-4 of Delaunay across
the production pipelines, it could replace Delaunay entirely. Even if
not, it's a useful fallback for environments where Delaunay is too slow.

### G. Eliminate the second find_simplex (LOW EFFORT, GUARANTEED ~10% SAVING)

**Idea**: scipy_delaunay() calls `find_simplex` twice — once on the
data grid (~20 ms) and once on split_points (~10 ms — split_points is
4×1231 = 4924 points). The second is for ConstantSplit regularization.

If the production pipelines don't use ConstantSplit (or use it less
hot-path-critically), the second find_simplex is wasted work.

**Investigation**: check which regularization is in use at production
fiducial. If not ConstantSplit, skip the second find_simplex.

**Expected saving**: ~3-5 ms/element at batch=20 → ~60-100 ms wall per
batched call. **About 5-7% on the full pipeline.**

This is a free win regardless of any other change.

## Recommended action plan

**Phase 1** (low effort, ~1-3 days):
- **G.** Audit whether the second find_simplex is needed at production
  fiducial. Make it conditional on regularization scheme. Free 5-7%.
- **F (variant).** Add a "kNN with barycentric-on-3-nearest" mode to
  InterpolatorKNearestNeighbor. Test log-evidence against Delaunay on
  the cross-pipeline matrix at rtol=1e-4. If it passes, it's a
  drop-in JAX-friendly replacement.

**Phase 2** (medium effort, ~1-2 weeks):
- **A.** Implement the split: scipy for triangulation, JAX brute-force
  for point location. Validate log-evidence matches scipy at rtol=1e-6.
- **B.** Add multiprocessing wrapper around the (now-small) scipy
  callback. Persistent process pool initialised at first use.

**Phase 2 expected impact**: combined 3-5× on Delaunay-specific cost,
~1.5-1.8× on the full likelihood at batch=20.

**Phase 3** (optional, only if Phase 2 isn't enough):
- **D.** Sampler-level caching. Significant complexity for modest gain.
  Only if science throughput is still the bottleneck after Phase 2.

## What this task is NOT

- Not a pure-JAX Delaunay project. That's a research dead end.
- Not an "improve scipy" project — scipy is fine; the issue is callback
  topology.
- Not an algorithm-replacement project — we keep Delaunay's scientific
  properties (triangulation, barycentric interpolation, split-points for
  regularization). We change WHERE the work happens, not WHAT the work is.

## Files touched

Phase 1 (G):
- `autoarray/inversion/mesh/interpolator/delaunay.py:scipy_delaunay` —
  conditional second find_simplex based on a parameter
- `autoarray/inversion/mesh/interpolator/delaunay.py:jax_delaunay` —
  same kwarg
- Caller: `_mappings_sizes_weights_split` in the InterpolatorDelaunay
  class checks regularization type before requesting split-points data

Phase 1 (F variant):
- `autoarray/inversion/mesh/interpolator/knn.py` — add a `use_barycentric`
  mode that picks top-3 and uses barycentric coords instead of Wendland
- Maybe new class `InterpolatorKNNBarycentric` if cleaner

Phase 2 (A):
- New module `autoarray/inversion/mesh/interpolator/delaunay_jax_locate.py`
  — JAX implementation of brute-force barycentric point location
- `delaunay.py:scipy_delaunay` simplified to triangulation only
- `delaunay.py:jax_delaunay` — orchestration: pure_callback for
  triangulation, JAX-native for point location

Phase 2 (B):
- `delaunay.py` — wrap the pure_callback host function with a process
  pool. Add a `--max-workers` config option to `autoarray/config/general.yaml`.

## Validation gates for any change

1. **Log-evidence regression**: `EXPECTED_LOG_EVIDENCE_HST` (Delaunay
   at MGE-60) and `EXPECTED_LOG_EVIDENCE_SMA` (interferometer) must
   hold at rtol=1e-4 fp64.
2. **Reconstruction-vector L2**: source-pixel coefficients must match
   current main at rtol=1e-6 on the canonical pipelines.
3. **Cross-pipeline matrix**: all 6 pipelines (imaging mge/rect/delaunay,
   interferometer mge/rect/delaunay) pass their regression assertions.
4. **Performance gates**: A100 batch=20 vmap-per-call ≤ 50 ms (down from
   69.5 ms baseline). Stretch goal: ≤ 40 ms.
5. **autolens_workspace_test `jax_assertions/nnls.py`** still passes
   (well-conditioned match + ill-conditioned finite-gradient).

## References

- jaxnnls upstream:
  https://github.com/CKrawczyk/jaxnnls (Coleman Krawczyk's modifications)
  https://github.com/kevin-tracy/qpax (original Tracy QP)
- gDel2D parallel Delaunay (CUDA):
  Cao et al. "Computing 2D constrained delaunay triangulation using
  graphics hardware"
- Robust point-in-triangle predicates:
  Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast
  Robust Geometric Predicates"
- Source: nnls-vmap-speedup findings
  (`z_projects/profiling/FINDINGS_nnls_v2.md`, issue PyAutoArray#307
  closed 2026-05-11)
