# Fix `RectangularSplineAdaptImage` JIT crash in `AdaptImages` plumbing

## Context

`autolens_workspace_developer/jax_profiling/imaging/pixelization_spline_vs_linear.py`
benchmarks four rectangular meshes side-by-side:

- `RectangularUniform`
- `RectangularAdaptDensity` (linear CDF)
- `RectangularSplineAdaptDensity` (spline CDF)
- `RectangularSplineAdaptImage` (spline CDF + adapt image)

After PR #31 re-centred every jax_profiling script on simulator-truth
priors, the first three meshes now run end-to-end at truth and produce
sensible `log_L` values (14310, 26701, 26543). The fourth — the only
`AdaptImage` variant — crashes on its first JIT call of
`analysis.log_likelihood_function(instance)`:

```
Mesh: RectangularSplineAdaptImage
  eager figure_of_merit = 29065.9805 (2.7s)   <-- OK
  !!! RectangularSplineAdaptImage FAILED: AttributeError:
      'NoneType' object has no attribute 'array'
Traceback (most recent call last):
  ...
  File ".../autolens/imaging/model/analysis.py", line 79,
      in log_likelihood_function
    return self.fit_from(instance=instance).figure_of_merit
           - log_likelihood_penalty
  File ".../autoarray/fit/fit_dataset.py", line 361,
      in figure_of_merit
    if self.inversion is not None:
AttributeError: 'NoneType' object has no attribute 'array'
```

Eager `FitImaging(...).figure_of_merit` returns 29065.98 cleanly on the
same `tracer` / `dataset` / `adapt_images`, so the adapt-image plumbing
is consistent; something diverges only once JAX tracing starts.

This blocks the adapt-image row of the spline-vs-linear benchmark and,
more broadly, any gradient-sampler use of `RectangularSplineAdaptImage`
or `RectangularAdaptImage` under `jax.jit`. The other three meshes show
the spline gives a modest (~22%) smoothness improvement over linear but
is not dominant — so confirming that the adapt-image variant behaves
the same way is the missing piece for a complete picture.

## Hypothesis

`AdaptImages.galaxy_image_dict` is a dict keyed on a `Galaxy` instance:

```python
adapt_images = AdaptImages(
    galaxy_image_dict={instance.galaxies.source: adapt_arr}
)
```

The key is the pre-trace `Galaxy` object. Under
`jax.jit(log_L)(params_tree)`, the analysis rebuilds galaxies from the
traced pytree, producing a *new* `Galaxy` instance whose identity (and
`__eq__` / `__hash__`) does not match the dict key. The lookup returns
`None` and a downstream `.array` access blows up inside
`fit_dataset.inversion` — the traceback's line 361 is an outer call; the
real failure is in the inversion cache/property or the linear-obj-func
that asks for the adapt image of the current source galaxy.

JAX filters its internal frames from the traceback
(`JAX_TRACEBACK_FILTERING=off` to see them), so the first diagnostic
step is to re-run the benchmark with that env var set and capture the
real `.array` call site.

## Task

1. Reproduce the crash from the benchmark:

   ```bash
   cd autolens_workspace_developer
   JAX_TRACEBACK_FILTERING=off \
     NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
     python jax_profiling/imaging/pixelization_spline_vs_linear.py \
     2>&1 | tee /tmp/spline_vs_linear_full_tb.log
   ```

   Confirm the real call site of the `.array` access inside
   `AdaptImages` / `Inversion` / `LightProfileLinearObjFuncList`.

2. Fix the key-mismatch so `AdaptImages` resolves under JIT. Options:

   - Key `galaxy_image_dict` by a stable path (e.g. the
     `"galaxies.source"` dotted path from the model collection) rather
     than by galaxy-instance identity.
   - Key by a galaxy-level hash that's preserved across
     tree\_map (class + redshift + index in the collection).
   - Pass the adapt image through the analysis via a side channel that
     doesn't go through a dict keyed on a traced object.

   Whichever shape lands, the public API — `AdaptImages(galaxy_image_dict={galaxy: arr})` — should stay ergonomic. The JIT-safe lookup can be an internal implementation detail.

3. Re-run the benchmark and confirm:

   - `RectangularSplineAdaptImage` produces an eager and JIT `log_L`
     that agree to within the NNLS-vs-solve gap seen for the other
     meshes.
   - `fd1_roughness` / `fd2_sup` / `fd2_std` land in the same order as
     `RectangularSplineAdaptDensity` and `RectangularAdaptDensity`.

4. Verify no regression in `jax_profiling/imaging/pixelization.py`,
   which also uses an adapt image path.

## Scope

Library-side fix in PyAutoArray or PyAutoGalaxy (wherever `AdaptImages`
is defined), plus a re-run of the workspace benchmark to confirm all
four meshes populate the comparison table. No changes needed to the
spline modules themselves — the crash is upstream of the mesh.

## Success criteria

- `pixelization_spline_vs_linear.py` completes without the
  `AdaptImage` error and writes all four rows to the results JSON
  and sweep PNG.
- Spline-vs-linear smoothness ratios reported for both density and
  adapt-image variants.

## Related

- Spline mesh shipped via PyAutoArray PR #30 and workspace PR
  `feature/rectangular-spline-mesh`.
- Truth-params conversion via `feature/jax-profiling-truth-params`
  (PR #31).
- First spline-vs-linear comparison at truth:
  `jax_profiling/imaging/results/spline_vs_linear_hst_v2026.4.13.6.json`
  and `..._sweep.png` — adapt-image row currently marked ERROR.
