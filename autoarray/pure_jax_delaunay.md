## Pure-JAX Delaunay triangulation

Replace `scipy.spatial.Delaunay` (currently invoked via
`jax.pure_callback` in `autoarray/inversion/mesh/interpolator/delaunay.py:jax_delaunay`)
with a GPU-native pure-JAX implementation.

### Motivation

From the nnls-vmap-speedup investigation
(`z_projects/profiling/FINDINGS_nnls_v2.md`, issue
https://github.com/PyAutoLabs/PyAutoArray/issues/307, closed 2026-05-11):

At production batch=20 on A100, the Delaunay full-pipeline likelihood
costs 69.5 ms per element. Decomposition:

```
scipy.spatial.Delaunay via pure_callback = 16.87 ms (24%)  <-- this prompt
other JAX-traced inversion setup         = ~25 ms (36%)
PSF FFT convolution                      = ~9 ms (13%)
log_ev (slogdet + matmul)                = 12 ms (17%)
NNLS reconstruction                      = 6.2 ms (9%)
misc                                     = ~0.5 ms (1%)
```

`pure_callback` with `vmap_method="sequential"` invokes scipy
**sequentially per batch element** — at batch=20 that's
16.87 × 20 = 337 ms of wall-clock CPU work per batched likelihood
call. A pure-JAX implementation can run on GPU and parallelise across
the batch dimension, dropping this 337 ms to a few ms total wall.

Expected speedup: ~1.3x on the full pipeline from this alone
(69.5 → 53 ms per element → batch_time 1.4 → 1.06 sec). Up to ~2x
combined with further optimisation of the ~25 ms "other JAX-traced
inversion setup" — but that's a separate sub-decomposition study.

### What needs to be replaced

`autoarray/inversion/mesh/interpolator/delaunay.py:jax_delaunay()`
returns five arrays:

```python
return jax.pure_callback(
    lambda points, qpts: scipy_delaunay(np.asarray(points), np.asarray(qpts), areas_factor),
    (points_shape, simplices_padded_shape, mappings_shape,
     split_points_shape, splitted_mappings_shape),
    points, query_points, vmap_method="sequential",
)
```

The five outputs are:
- `points` — the input mesh (pass-through, free)
- `simplices_padded` — Delaunay triangulation, padded to 2N (the
  expensive bit)
- `mappings` — which triangle each query point falls in
- `split_points` — split mesh points (for area-weighted regularisation)
- `splitted_mappings` — triangle assignments for split points

### Approach options to consider

1. **Bowyer-Watson on GPU.** Classic incremental Delaunay algorithm,
   parallelisable. Research-grade implementation.
2. **Grid-based / hashmap acceleration.** Spatial hashing for the
   point-in-triangle queries.
3. **Sampler-side warm-starting / caching.** If consecutive nested-sampling
   proposals are correlated, reuse the previous triangulation and only
   locally update. Requires PyAutoFit changes.
4. **Alternative interpolation scheme** (inverse-distance-weighted,
   natural neighbour, etc.) — sidesteps Delaunay entirely. Changes the
   science slightly so needs validation.

### Validation

- Cross-check that the JAX implementation produces the same simplices /
  mappings as scipy (up to ordering — Delaunay isn't unique on degenerate
  inputs).
- Verify EXPECTED_LOG_EVIDENCE_HST constants in
  `autolens_workspace_developer/jax_profiling/jit/imaging/delaunay.py`
  (currently `26288.321397232066`) and the interferometer equivalent.
- Re-run `autolens_workspace_test/scripts/jax_assertions/nnls.py` and
  the existing canonical regression scripts.

### Files

- Read: `autoarray/inversion/mesh/interpolator/delaunay.py` (the
  callback wrapper)
- Read: `autoarray/inversion/mesh/mesh_geometry/delaunay.py` (the
  scipy.spatial.Voronoi side, separate concern but related)
- Add: `autoarray/inversion/mesh/interpolator/delaunay_jax.py` (the
  pure-JAX implementation)
- Modify: `autoarray/inversion/mesh/interpolator/delaunay.py:jax_delaunay`
  to dispatch to the new pure-JAX path

### Out of scope

- The 25 ms "other JAX-traced inversion setup". That's a separate
  decomposition study. Sub-steps to identify: image-plane mesh grid
  construction (with circle-edge points), AdaptImages weighting,
  over-sampler setup, source-plane data grid ray-tracing.
- The log_ev slogdet (12 ms). Could plausibly be reformulated using
  the Cholesky factor from NNLS instead of a fresh slogdet, but that's
  also a separate task.
