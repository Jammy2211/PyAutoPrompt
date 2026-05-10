Step 4 of the ellipse-JAX series. `DatasetInterp` in `@PyAutoGalaxy/autogalaxy/ellipse/dataset_interp.py` uses `scipy.interpolate.RegularGridInterpolator` for the data, noise-map, and mask. scipy is numpy-only, so this is the first hard JAX blocker for `AnalysisEllipse.log_likelihood_function`. There is no JAX-compatible 2D interpolator anywhere in the codebase — the only precedent is the 1D `_interp1d_jax` in `@PyAutoArray/autoarray/inversion/mesh/interpolator/rectangular_spline.py:86-106`. We need a 2D analogue.

Please:

1. Add a 2D regular-grid bilinear interpolator helper to PyAutoArray. Suggested location: `@PyAutoArray/autoarray/numerics/interp_2d.py` (create the `numerics/` subpackage if it doesn't already exist; check `@PyAutoArray/autoarray/__init__.py` for the right import surface). Two paths:

   - `_interp_2d_numpy(points, x_axis, y_axis, values, fill_value=0.0)` — matches the current `RegularGridInterpolator(bounds_error=False, fill_value=0.0)` semantics in `dataset_interp.py`. A direct call to `scipy.interpolate.RegularGridInterpolator` is fine here.
   - `_interp_2d_jax(points, x_axis, y_axis, values, fill_value=0.0)` — uses `jax.scipy.ndimage.map_coordinates(values, coords, order=1, cval=fill_value)`. Translate `(y, x)` world coordinates to pixel-fractional coordinates using `x_axis`, `y_axis` (assume regularly spaced — the existing scipy `points_interp` is built from `mask.derive_grid.all_false`, which is regular).
   - Public dispatcher `interp_2d(points, x_axis, y_axis, values, fill_value=0.0, xp=np)` that picks the path. Mirror the dispatch style in `rectangular_spline.py`.

2. Unit tests in `@PyAutoArray/test_autoarray/numerics/test_interp_2d.py`:
   - Random `(N, 2)` query points inside the grid: assert numpy and JAX paths agree to `rtol=1e-6`.
   - Out-of-bounds query points: assert both paths return `fill_value` for those rows.
   - Single-point query: assert shape is `(1,)` not `()`.
   - `xp=np` is the default — JAX-only tests gated by `pytest.importorskip("jax")` per `@PyAutoArray/CLAUDE.md` testing conventions.

3. Wire `DatasetInterp` to the new helper. In `@PyAutoGalaxy/autogalaxy/ellipse/dataset_interp.py`:
   - Drop the cached `data_interp`, `noise_map_interp`, `mask_interp` properties that return `RegularGridInterpolator` instances.
   - Replace with methods `data_interp(points, xp=np)`, `noise_map_interp(points, xp=np)`, `mask_interp(points, xp=np)` that call `aa.numerics.interp_2d(...)` directly. The interp axes (`points_interp`) can stay cached.
   - Keep the existing call sites in `fit_ellipse.py` working: `self.interp.data_interp(self._points_from_major_axis)` continues to take a `(N, 2)` array.

4. Do **not** touch `FitEllipse.points_from_major_axis_from`'s 300-iteration loop in this prompt. That's prompt 6. The mask interpolation calls inside the loop continue to use the numpy path because the surrounding code is still numpy-only — pass `xp=np` explicitly at those call sites.

5. Test bar:
   - `python -m pytest test_autoarray/numerics/test_interp_2d.py -v` passes.
   - `python -m pytest test_autogalaxy/ellipse/ -v` still passes (no behavioural change on the numpy path — same `fill_value=0.0`, same regular-grid semantics).
   - The reference numbers from prompt 2's workspace_test scripts are unchanged to `rtol=1e-10`.
