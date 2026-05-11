# InterpolatorKNearestNeighbor variant: barycentric weights on top-3 nearest

A pure-JAX wildcard for replacing scipy.spatial.Delaunay in PyAutoArray's
source-plane interpolation. The idea is small and testable, the
potential payoff is large — if it holds the log-evidence to rtol=1e-4
against true Delaunay, it's a drop-in replacement that eliminates the
scipy callback bottleneck entirely.

## Background from the nnls-vmap-speedup investigation

(Full findings: `z_projects/profiling/FINDINGS_nnls_v2.md`. Companion
research doc: `PyAutoPrompt/autoarray/delaunay_research.md`. Closed
issue: https://github.com/PyAutoLabs/PyAutoArray/issues/307.)

At production batch=20 on A100, the Delaunay imaging likelihood costs
69.5 ms per element. Decomposition:

```
scipy.spatial.Delaunay via pure_callback = 16.87 ms (24%)  <-- this prompt
other JAX-traced inversion setup         = ~25 ms (36%)
PSF FFT convolution                      = ~9 ms (13%)
log_ev (slogdet + matmul)                = 12 ms (17%)
NNLS reconstruction                      = 6.2 ms (9%)
misc                                     = ~0.5 ms (1%)
```

`pure_callback` with `vmap_method="sequential"` invokes scipy serially
per batch element → 16.87 × 20 = **337 ms wall per batched likelihood
call** is held on a single CPU running scipy/Qhull. The barycentric
weight computation (0.01 ms) and the sparse mapping matrix scatter
(0.15 ms) are essentially free — the 16.87 ms IS the scipy work.

Of that 16.87 ms, ~52% is `find_simplex` (point location for 15361
query points), called twice, and ~14% is the actual Delaunay
triangulation on 1231 mesh points. The bottleneck is point location
on CPU, not the triangulation per se.

## The science context the user supplied

PyAutoArray's premier interpolator is `InterpolatorDelaunay`. It uses
scipy.spatial.Delaunay to triangulate the source-plane mesh vertices
and find which triangle each over-sampled image-plane data pixel falls
in. The barycentric coordinates of the data pixel inside its triangle
give the three interpolation weights to the three triangle vertices.

`InterpolatorKNearestNeighbor` was a previous JAX-friendly attempt at
the same problem (`autoarray/inversion/mesh/interpolator/knn.py`). It
uses brute-force k-nearest-neighbor search in JAX (no scipy callback)
and weighs the neighbors with a **Wendland C4 kernel** (smoothed
inverse-distance, compact support).

Per the user, kNN performed scientifically much worse than Delaunay
because:
- The Wendland kernel has knobs (`k_neighbors`, `radius_scale`) that
  are hard to set to work for all lenses.
- Delaunay adapts cleanly to local mesh density; kNN smears across
  density gradients.
- At caustic crossings (folds in the source-plane mapping), Delaunay's
  triangulation correctly spans the fold; kNN smears across it.

So Delaunay is scientifically the right method but the least JAX-friendly.

## The wildcard idea

**Use kNN to pick the top-3 nearest neighbors in source plane, then
compute exact barycentric coordinates on the triangle those 3 form.**

This replaces Wendland kernel weights with **locally-exact barycentric
weights** — the same weights Delaunay uses, just on a triangle chosen
by Euclidean nearest-neighbor instead of by Delaunay triangulation.

When the 3 nearest mesh vertices happen to be the 3 vertices of the
containing Delaunay triangle (the common case for "interior" query
points), the result is bit-identical to Delaunay. When they aren't
(boundary points, density-gradient regions, caustic-crossing points),
the result is an approximation — but one that still respects local
mesh topology rather than smearing with a global kernel.

The potentially-killer subtlety: when the query point is OUTSIDE the
triangle formed by its 3 nearest neighbors, the barycentric coords
have a negative component. Two options:

1. **Clip and renormalize**: max(bary, 0), then normalize so they sum
   to 1. Gives a valid convex combination but slightly distorts.
2. **Fall back to kNN-Wendland** for those query points. Hybrid.
3. **Take more neighbors (k=4, 5, 6) and pick the 3 with non-negative
   barycentric coords**. Most expressive but more complex.

Start with option 1. If log-evidence rtol fails, escalate to option 3.

## Why this might work where pure kNN-Wendland didn't

- Barycentric weights on 3 nearest are **locally exact** when the 3 are
  the correct triangle vertices. The Wendland kernel never is.
- The "knobs" go away: no `radius_scale`, no kernel shape. Just k=3.
- At caustic crossings, the 3 nearest still bracket the local source-plane
  structure better than a smoothly-decaying kernel.

Where it might still fail:
- Boundary points (mesh edge) where the 3 nearest don't form a
  containing triangle, AND clip-renormalize doesn't approximate well.
- Highly anisotropic mesh density where the 3 nearest are nearly
  collinear (degenerate triangle → numerical instability in barycentric).

Both failure modes are testable.

## Performance expectations

`InterpolatorKNearestNeighbor` is already pure JAX with `lax.fori_loop`
over blocks of mesh points (brute-force kNN, no scipy callback). Adding
barycentric on top-3 is a small post-processing step on the kNN output.

Estimated runtime under vmap=20 on A100:
- kNN search (current code, k=3 instead of k=10): ~1-3 ms per element
  (brute-force, 1231 mesh × 15361 queries × 3 = 5.7e7 ops, easy GPU work)
- Barycentric weight computation on the 3 picked vertices: ~0.1 ms
  per element
- Total: ~2-4 ms per element

Replacing `InterpolatorDelaunay`'s 16.87 ms per element callback with
~3 ms pure JAX gives **~13 ms saving per element**, full pipeline drops
from 69.5 → 56 ms per element, batch_time 1.4 → 1.12 sec.
**~1.25× speedup overall, just from this swap.**

If it works, it's also vmap-parallel (no sequential callback), so
larger batch sizes will continue to amortize well — unlike the scipy
callback which is sequential-per-element forever.

## Implementation plan

### Step 1 — Add the variant

In `autoarray/inversion/mesh/interpolator/knn.py`:

- Add a new function `barycentric_weights_from_3_nearest(query, mesh,
  nearest_3_indices, xp)`:
  - Picks the 3 mesh vertices for each query
  - Computes barycentric coordinates of the query inside that triangle
  - Clips to 0 and renormalizes (option 1)
  - Returns weights shape (Q, 3) matching the existing format
- Add a new class `InterpolatorKNNBarycentric` extending
  `InterpolatorKNearestNeighbor`:
  - Overrides `_mappings_sizes_weights` to call kNN with k=3 then
    compute barycentric weights instead of Wendland

(Alternative: add a `use_barycentric: bool = False` flag on the
existing class. Cleaner separation but the class hierarchy is less
clear. Use whichever fits the existing pattern.)

### Step 2 — Register a new Mesh class

In `autoarray/inversion/mesh/mesh/`:

- Add `KNNBarycentric(Delaunay)` subclass that returns the new
  interpolator class. Mirrors the existing `KNearestNeighbor(Delaunay)`
  pattern.

### Step 3 — Validate against Delaunay

The hard gate. Take the existing canonical regression reference at
`autolens_workspace_developer/jax_profiling/jit/imaging/delaunay.py`
and add a parallel `delaunay_knn_barycentric.py` that uses the new
mesh. Same setup, same fiducial, same `EXPECTED_LOG_EVIDENCE_HST` for
reference — but the assertion at the bottom checks the NEW
interpolator's log-evidence against the same constant at **rtol=1e-3**
(looser tolerance, since we're approximating Delaunay).

The production-fiducial constants from the closed issue:
- Rectangular: `EXPECTED_LOG_EVIDENCE_HST = 24746.105672366088` (not
  applicable here; rectangular uses a different interpolator)
- **Delaunay: `EXPECTED_LOG_EVIDENCE_HST = 26288.321397232066`**
- Interferometer Delaunay: read the current value from
  `autolens_workspace_developer/jax_profiling/jit/interferometer/delaunay.py`

Validation gates (in priority order):
1. **`log_evidence` drift vs Delaunay at the production fiducial**:
   target rtol=1e-3 (3 dex tighter than rtol=1e-2 for mp, but looser
   than rtol=1e-4 for the same algorithm). If it passes rtol=1e-4,
   we can call it "Delaunay-equivalent". If it passes rtol=1e-3,
   we can call it "Delaunay-class approximation, science still good".
   If it fails rtol=1e-2, the wildcard didn't work — go back to
   pure-Delaunay path and consider the split-the-callback approach
   (`delaunay_research.md`).
2. **Source-pixel reconstruction L2** vs Delaunay: rtol=1e-2 acceptable
   (this is the science output, not a derived figure of merit).
3. **Cross-pipeline matrix**: imaging Delaunay (HST), interferometer
   Delaunay (SMA), both at MGE-60 fiducial.
4. **Strong-lensing edge cases**: try a known caustic-crossing lens
   model (the `autolens_workspace_test` script suite has these).
   Manually inspect whether the source reconstruction visually
   matches Delaunay's. If caustic regions show ringing or smearing
   that Delaunay doesn't show, the wildcard has a science problem.

### Step 4 — Measure the speedup

Re-run `z_projects/profiling/scripts/delaunay_vmap_probe.py` (set
`PROBE_BATCH_SIZE=20`) with the mesh swapped from Delaunay to the new
KNNBarycentric. Compare per-call timings to the existing
`output/imaging/delaunay/hpc_a100_fp64_vmap_probe.json`.

### Step 5 — Decision

- **If validation passes at rtol=1e-4**: ship as a drop-in replacement,
  make it the default Delaunay-like mesh. Don't even need to keep the
  scipy path.
- **If validation passes at rtol=1e-3 but not 1e-4**: ship as an
  alternative (`use_knn_barycentric` config flag), default off,
  available for users who care about wall-clock more than 4-digit
  log-evidence reproducibility.
- **If validation fails**: fall back to the split-callback approach
  in `delaunay_research.md`. The wildcard was worth trying because
  the implementation is small.

## Files to touch

**New code (PyAutoArray):**
- `autoarray/inversion/mesh/interpolator/knn.py` — add
  `barycentric_weights_from_3_nearest` + new interpolator class
- `autoarray/inversion/mesh/mesh/knn.py` — add `KNNBarycentric` mesh
  class
- `autoarray/inversion/mesh/__init__.py` — export it
- `test_autoarray/inversion/mesh/interpolator/test_knn_barycentric.py`
  — unit tests: known triangle interior cases (exact match to
  Delaunay), boundary cases (clipped weights still sum to 1),
  degenerate triangle (no NaN)

**Test / regression (autolens_workspace_developer):**
- `jax_profiling/jit/imaging/delaunay_knn_barycentric.py` — parallel
  regression reference using the new mesh, assertion at rtol=1e-3
  against Delaunay's value
- (Optional) interferometer equivalent

**Smoke / cross-conditioning (autolens_workspace_test):**
- `scripts/jax_assertions/knn_barycentric.py` — well-conditioned and
  ill-conditioned cases mirroring the existing `nnls.py` pattern,
  verifying the new interpolator is finite + matches Delaunay
  to its expected tolerance

## Read-only reference files

- `autoarray/inversion/mesh/interpolator/delaunay.py` —
  `scipy_delaunay`, `jax_delaunay`, `InterpolatorDelaunay`,
  `pixel_weights_delaunay_from` (the barycentric helper)
- `autoarray/inversion/mesh/interpolator/knn.py` —
  current `get_interpolation_weights`, `InterpolatorKNearestNeighbor`
- `autoarray/inversion/mesh/mesh/knn.py` — existing
  `KNearestNeighbor(Delaunay)` class
- `autoarray/inversion/mappers/abstract.py:255` —
  `Mapper.mapping_matrix` (the call site that uses the interpolator)
- `z_projects/profiling/scripts/delaunay_vmap_probe.py` — speed
  measurement harness
- `z_projects/profiling/FINDINGS_nnls_v2.md` — full investigation
  background (the v3/v4 sub-decomposition data is here)

## Out of scope

- Pure-JAX Delaunay (separate research project — see `delaunay_research.md`,
  assessed as ~3-6 months of work with uncertain payoff).
- Multiprocessing the scipy callback (separate optimization, also in
  `delaunay_research.md`).
- Changes to NNLS, log_ev, PSF FFT — none are the bottleneck at
  production batch=20 (NNLS = 9%, log_ev = 17%, PSF = 13%).
- Sampler-level Delaunay caching (PyAutoFit concern).

## Decision criteria up front

This is a **science-validated** speedup wildcard. Code is small (~50-100
lines new). The hard part is the validation: does kNN-barycentric give
the same scientific answer as Delaunay across the production lens
modeling matrix?

If yes — best possible outcome: drop scipy.spatial.Delaunay entirely,
get ~1.25× speedup on Delaunay production, zero external dep on scipy
in the inversion path.

If no — small sunk cost, fall back to the more conservative
split-the-callback approach in `delaunay_research.md`.

Either way the data tells us something useful.
