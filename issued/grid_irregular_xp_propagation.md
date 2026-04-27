# Grid2DIrregular: propagate `xp` through derived constructors

## Context

This issue surfaced while writing the source-plane point-source JAX
profiling script (`autolens_workspace_developer/jax_profiling/point_source/source_plane.py`,
shipped in PyAutoLabs/autolens_workspace_developer#22).

Image-plane fitting (`al.FitPositionsImagePairAll`) JITs end-to-end. The
source-plane variant (`al.FitPositionsSource`) blocks at compile time
because `Grid2DIrregular` derived constructors do not propagate the `xp`
backend, so model_data ends up with `_xp=np` while holding JAX tracers.

## Reproduction

```python
import jax.numpy as jnp
import autolens as al

# ... build tracer + dataset, then:
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,
    use_jax=True,
)
jax.jit(lambda inst: analysis.log_likelihood_function(instance=inst))(params_tree)
# -> jax.errors.TracerArrayConversionError inside
#    Grid2DIrregular.squared_distances_to_coordinate_from
```

## Root cause

`@PyAutoArray/autoarray/structures/grids/irregular_2d.py` —
`Grid2DIrregular.grid_2d_via_deflection_grid_from` constructs the new
grid without propagating `xp`:

```python
def grid_2d_via_deflection_grid_from(self, deflection_grid):
    return Grid2DIrregular(values=self - deflection_grid)
```

When the receiver (`self`) is a numpy-backed `Grid2DIrregular` (the
observed dataset positions) but the deflection_grid carries JAX tracers,
the subtraction returns JAX tracers but the new wrapper's `_xp` defaults
to `np`. The next call into `squared_distances_to_coordinate_from` then
runs `self._xp.square(self.array - coordinate)` → `np.square(tracer)` →
`TracerArrayConversionError`.

## Proposed fix

Two complementary one-liners:

1. `grid_2d_via_deflection_grid_from` should pass `xp=self._xp`:

   ```python
   def grid_2d_via_deflection_grid_from(self, deflection_grid):
       return Grid2DIrregular(values=self - deflection_grid, xp=self._xp)
   ```

2. `AbstractFitPositions.__init__` should rewrap `data` with `xp=xp`
   so the observed positions match the analysis backend regardless of
   how the dataset was originally constructed (the dataset comes off
   disk via `al.from_json` which always builds a numpy-backed grid).

Either fix individually unblocks the source-plane JIT path. Both
together provide defence-in-depth.

## Audit

While there, sweep `Grid2DIrregular` (and `Grid2D` for parity) for any
other derived constructor that calls `Grid2DIrregular(values=...)`
without `xp=self._xp`. Likely candidates: any method returning a new
grid from arithmetic on `self`.

## Validation

After the fix, the gated assertion in
`autolens_workspace_developer/jax_profiling/point_source/source_plane.py`
will start firing automatically:

```python
EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE = -4496.798984131583

if full_pipeline_jits:
    np.testing.assert_allclose(
        float(full_result),
        EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE,
        rtol=1e-4,
    )
```

Re-run that script post-fix; it should print:

```
Full pipeline (JIT):        <time> s/call
JIT regression assertion PASSED: log_likelihood matches -4496.798984
```

instead of the current `BLOCKED (see module docstring)` line.

## Related

- Workspace PR documenting the blocker: PyAutoLabs/autolens_workspace_developer#22
- Workspace issue: PyAutoLabs/autolens_workspace_developer#21
- The `xp` threading conventions (relevant to the audit) are documented
  in `@PyAutoArray/CLAUDE.md` ("JAX Support" section).
