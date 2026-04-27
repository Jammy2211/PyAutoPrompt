# LensCalc.magnification_2d_via_hessian_from diverges between np and jnp

## Context

Follow-up from the `grid-irregular-xp-propagation` task. After fixing
`Grid2DIrregular.grid_2d_via_deflection_grid_from` to propagate `xp` and
rewrapping observed positions + magnifications inside
`AbstractFitPoint`, the full-pipeline source-plane JIT now runs end-to-end —
no more `TracerArrayConversionError`.

However the JIT-path log-likelihood disagrees with the eager-path result by
~0.1%, well outside float64 precision:

- eager (xp=np):  `-4496.798984`
- jax   (xp=jnp): `-4491.832094`  (diff ≈ 4.97, rtol ≈ 1.1e-3)

The `ray_trace_to_source_plane` prefix already asserts equal-to-1e-4 against
the numpy path, and standalone diagnosis confirmed:

- `residual_map` np vs jnp agrees to ~1e-8 (the ray-trace is identical).
- `magnifications_at_positions` is where the numbers diverge.

Concretely, evaluated at the same prior-median parameters on the same two
observed positions:

| Position | np magnification | jnp magnification | rel. diff |
|----------|------------------|-------------------|-----------|
| 0        | 10.60783132      | 10.6070619        | 7e-5      |
| 1        | 152.66911098     | 152.5846727       | 5e-4      |

Because chi-squared divides by `mag**-2 * noise**2`, even a 5e-4 relative
error on one point gets amplified into the ~0.1% log-likelihood drift seen
above.

## Root cause

Traced to `ag.LensCalc.hessian_from` in
`PyAutoGalaxy/autogalaxy/operate/lens_calc.py:350`. It dispatches on `xp`:

- `xp is np` → `_hessian_via_finite_difference` (lens_calc.py:413) — uses
  2-point central difference with `buffer=0.01`. Truncation error is
  **O(h²) ≈ 1e-4**.
- `xp is jnp` → `_hessian_via_jax` (lens_calc.py:383) — uses `jax.jacfwd`
  on `deflections_yx_scalar` vectorised with `jnp.vectorize`. This is
  **exact** forward-mode autodiff — accurate to float64.

So the two paths aren't "both mathematically correct and should match" —
the JAX path is more accurate, and the numpy path is truncation-limited.
At near-critical points (position 1 in the repro above, magnification = 152)
the O(1e-4) truncation error in the Hessian is amplified by the near-zero
Jacobian determinant, producing the observed ~5e-4 relative error in the
magnification.

## Task

Close the accuracy gap on the **numpy** path so it agrees with the JAX
path to float64 precision (~1e-8 rtol). Do this generically, without
touching any mass-profile API.

### Strategy — Richardson extrapolation of the existing FD stencil

Evaluate the existing central-difference Hessian at two step sizes (`h` and
`h/2`) and combine:

```
H_extrapolated = (4 * H(h/2) - H(h)) / 3
```

This cancels the leading O(h²) error term, leaving O(h⁴) ≈ 1e-8 — which
should match the JAX path. It is a pure wrapper around the existing
`_hessian_via_finite_difference(grid, buffer=h)`; no change to stencils,
no change to mass profiles, no new public API.

### Why not the alternatives

- **Analytic Hessians** via profile-specific `convergence_2d_from` +
  `shear_yx_2d_from`. Exact, but requires every mass profile to expose a
  matching pair of methods — new API surface and per-profile work. Noted
  as future work only; **out of scope for this task**.
- **5-point central stencil** — same O(h⁴) accuracy as Richardson at the
  same cost (4 evaluations per axis), but requires rewriting the stencil
  arithmetic in-place. Richardson wins on minimal-churn: the existing
  2-point code stays untouched.
- **Shrinking `buffer`** (e.g. 1e-5) — hits floating-point round-off and
  gets worse, not better.

### Implementation steps

1. In `PyAutoGalaxy/autogalaxy/operate/lens_calc.py`, leave
   `_hessian_via_finite_difference(self, grid, buffer=0.01)` unchanged —
   it remains the primitive stencil.
2. Rewrite the `xp is np` branch of `hessian_from` (lens_calc.py:379) to
   compute Richardson-extrapolated Hessians:

   ```python
   if xp is np:
       h = 0.01
       yy_h,  xy_h,  yx_h,  xx_h  = self._hessian_via_finite_difference(grid, buffer=h)
       yy_h2, xy_h2, yx_h2, xx_h2 = self._hessian_via_finite_difference(grid, buffer=h / 2)
       # Richardson: cancels leading O(h^2) error -> O(h^4).
       yy = (4.0 * yy_h2 - yy_h) / 3.0
       xy = (4.0 * xy_h2 - xy_h) / 3.0
       yx = (4.0 * yx_h2 - yx_h) / 3.0
       xx = (4.0 * xx_h2 - xx_h) / 3.0
       return yy, xy, yx, xx
   return self._hessian_via_jax(grid=grid, xp=xp)
   ```

   (Exact structure is at the implementer's discretion — a private helper
   is fine too.)
3. Verify convergence locally:
   - Pick a mass profile (e.g. `mp.Isothermal`) and a small `Grid2DIrregular`.
   - Compute `np` Hessian with the new Richardson path and `jnp` Hessian with
     the JAX path.
   - Assert `np.allclose(np_hessian, jnp_hessian, rtol=1e-8)`.

### Tests to add

Add a unit test in `test_autogalaxy/operate/test_lens_calc.py` (or the
closest existing file):

- `test__hessian_from__np_richardson_matches_jax_jacfwd_to_float64` — builds
  a small tracer and asserts `np.allclose(np_hess, jnp_hess, rtol=1e-8)` for
  all four Hessian components on the same grid.
- `test__magnification_2d_via_hessian_from__np_jnp_agree_to_float64` — same
  tracer, asserts the same rtol on the derived magnification.

### Workspace follow-up

Once the library fix lands:

- Re-run
  `autolens_workspace_developer/jax_profiling/point_source/source_plane.py`.
- Tighten the full-pipeline JIT regression assertion (currently `rtol=2e-3`
  pending this work — see the `if full_pipeline_jits:` block in
  `source_plane.py` around line 522) back to `rtol=1e-4` against
  `EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE = -4496.798984131583`.
- Remove the "pending the np/jnp parity fix" comment pointing at this
  prompt.

## Affected repos

- `PyAutoGalaxy` (primary — `LensCalc.hessian_from` machinery, unit tests)
- `autolens_workspace_developer` (tighten the regression assertion in
  `point_source/source_plane.py` once the library fix lands)

No changes needed in `PyAutoLens` — the tracer layer calls down into the
same `LensCalc.hessian_from` method and inherits the fix automatically.

## Suggested branch

`feature/lens-calc-magnification-xp-parity`

## How to reproduce

```bash
source ~/Code/PyAutoLabs-wt/<task>/activate.sh   # or main checkout
cd /home/jammy/Code/PyAutoLabs/autolens_workspace_developer
python jax_profiling/point_source/source_plane.py
```

Look for the final assertion:

```
point_source/source_plane: regression — JIT log_likelihood drifted
 ACTUAL: array(-4491.832094)
 DESIRED: array(-4496.798984)
```

The drift will disappear once the magnification calc is aligned across
backends.

## Original analysis

Comparison script (standalone) that pins the divergence to magnifications:

```python
fit_np  = analysis_eager.fit_from(instance=instance)
fit_jax = analysis_jax.fit_from(instance=params_tree)

# residuals — identical
np.testing.assert_allclose(
    fit_np.positions.residual_map.array,
    np.asarray(fit_jax.positions.residual_map.array),
    rtol=1e-8,
)

# magnifications — diverge at ~5e-4 rel
print(fit_np.positions.magnifications_at_positions.array)
print(fit_jax.positions.magnifications_at_positions.array)
```
