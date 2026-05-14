Step 6 of the ellipse-JAX series. Prompts 4 and 5 cleared the way: the data/noise/mask interpolator now has a JAX path, and `Ellipse` / `EllipseMultipole` accept `xp=jnp`. The remaining JAX blocker on `FitEllipse` is the 300-iteration mask-rejection loop in `points_from_major_axis_from` (`@PyAutoGalaxy/autogalaxy/ellipse/fit_ellipse.py:81-134`). The loop does dynamic-shape slicing (`points = points[unmasked_indices]`) and a Python `for` with an early `continue` based on traced values — none of that traces under `jax.jit`. The unit tests added in prompt 3 are the regression target for this rewrite.

The core idea for the JAX path: **oversample with a fixed shape, mask invalid points with NaN, and let downstream `nansum`/`nanmean` reductions in `chi_squared` / `residual_map` do the right thing**. Downstream is already NaN-aware (see `@PyAutoGalaxy/autogalaxy/ellipse/fit_ellipse.py:240, 296` — `np.nanmean`, `np.nansum`), so this strategy doesn't change the reduction layer.

Please:

1. Add `xp=np` to `FitEllipse.points_from_major_axis_from`. Add `use_jax: bool = False` (or read it off the dataset / a class attribute — match whatever pattern emerges from prompt 7's `AnalysisEllipse` wiring; pick whichever is least intrusive).

2. Keep the existing numpy loop **unchanged** under `if xp is np:`. The unit tests from prompt 3 must still pass byte-for-byte. Do not delete or reformat the loop body.

3. Add a JAX path under the `else:` branch:
   - Compute `total_points_required = ellipse.total_points_from(pixel_scale)` (still a Python `int`, fine to pass as a static argument to JIT).
   - Choose a fixed oversample factor `K` (suggest `K = 4`; tune with the workspace_test multipoles script if needed). Compute `oversample_total = total_points_required * K`.
   - Call `ellipse.points_from_major_axis_from(pixel_scale=..., n_i=oversample_total - total_points_required, xp=xp)` to get an `(oversample_total, 2)` array.
   - Apply multipole perturbations the same way (already JAX-safe after prompt 5).
   - Evaluate the mask interpolator: `mask_values = self.interp.mask_interp(points, xp=xp)`. Build `keep = mask_values == 0` as a boolean array.
   - Replace the `points = points[unmasked_indices]` dynamic-shape slice with `xp.where(keep[:, None], points, xp.nan)`. Output shape stays `(oversample_total, 2)`.

4. Update downstream call sites in `FitEllipse` so the `(oversample_total, 2)` JAX array flows through `data_interp` / `noise_map_interp` correctly. The `data_interp` JAX path (added in prompt 4) propagates NaNs through `map_coordinates`, but the masked output value is `fill_value=0.0` for OOB — that's fine for the masked rows because we already overwrote those positions with `nan` before the interp call. Sanity-check by setting up a tiny example and verifying `chi_squared` matches the numpy path to `rtol=1e-4`.

5. The 300-iteration safety raise in the numpy path stays under `if xp is np:` — JAX has no analogue of "the loop ran out". Document the JAX path's failure mode in a single docstring line: *"With xp=jnp, masked points are dropped via NaN propagation; if `K=4` is insufficient, the chi-squared will be biased by missing perimeter samples — increase K and re-pin the workspace_test reference numbers."*

6. Test bar:
   - `python -m pytest test_autogalaxy/ellipse/test_fit_ellipse.py -v` passes (prompt 3's tests, all numpy-path).
   - Add one new test in the same file that compares the numpy-path output against the JAX-path output for a non-trivial mask, asserting `nansum(chi_squared)` agrees to `rtol=1e-4`. Skip with `pytest.importorskip("jax")`.
   - The reference numbers from prompt 2's workspace_test scripts are unchanged on the numpy path.
