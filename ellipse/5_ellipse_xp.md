Step 5 of the ellipse-JAX series. With the 2D interpolator in place from prompt 4, the next blocker is the geometry math in `@PyAutoGalaxy/autogalaxy/ellipse/ellipse/ellipse.py` and `@PyAutoGalaxy/autogalaxy/ellipse/ellipse/ellipse_multipole.py`. Every routine on `Ellipse` uses bare `np.*`, and `EllipseMultipole.get_shape_angle` uses Python `while` loops to wrap an angle into `[-180/m, 180/m]` — both incompatible with `jax.jit` tracing. Convert these to the `xp=np` pattern documented in `@PyAutoGalaxy/CLAUDE.md` "JAX Support" section.

Please:

1. Add `xp=np` as a keyword argument to every method in `@PyAutoGalaxy/autogalaxy/ellipse/ellipse/ellipse.py` that returns a numerical array:
   - `Ellipse.angles_from_x0_from`
   - `Ellipse.ellipse_radii_from_major_axis_from`
   - `Ellipse.x_from_major_axis_from`
   - `Ellipse.y_from_major_axis_from`
   - `Ellipse.points_from_major_axis_from`

   Replace bare `np.*` with `xp.*` inside the function bodies (`xp.linspace`, `xp.sin`, `xp.cos`, `xp.divide`, `xp.add`, `xp.sqrt`, `xp.stack`). The `total_points_from` method stays numpy — its return type is a Python `int` and it's used to set static shapes outside the JIT trace.

   Special case in `points_from_major_axis_from`: the `idx = np.logical_or(np.isnan(x), np.isnan(y)); if np.sum(idx) > 0: raise NotImplementedError()` guard is JAX-incompatible (Python `if` on a traced value). Replace with `if xp is np:` around the guard — under JAX, NaNs propagate through downstream `nansum`/`nanmean` and we'd rather see them than crash inside a JIT trace.

2. Same treatment for `EllipseMultipole.points_perturbed_from` and `EllipseMultipoleScaled.points_perturbed_from` in `@PyAutoGalaxy/autogalaxy/ellipse/ellipse/ellipse_multipole.py`. Add `xp=np`, swap `np.*` for `xp.*`. The `multipole_comps_from` and `multipole_k_m_and_phi_m_from` helpers from `@PyAutoGalaxy/autogalaxy/convert.py` are called outside the math loop on Python tuples — leave those as-is unless they trip JIT (verify by tracing).

3. **Replace the `while` loops** in `EllipseMultipole.get_shape_angle` (`@PyAutoGalaxy/autogalaxy/ellipse/ellipse/ellipse_multipole.py:66-69`) with arithmetic that JAX can trace. The intent is "wrap `angle` into the open interval `(-180/m, 180/m]`". A direct replacement using `xp.mod` works:

   ```python
   period = 360.0 / self.m
   angle = xp.mod(angle + period / 2.0, period) - period / 2.0
   ```

   This produces values in `[-period/2, period/2)` rather than `(-period/2, period/2]`, which is a tiny boundary-case difference. Verify against the existing tests in `@PyAutoGalaxy/test_autogalaxy/ellipse/` and add a test pinning the new behaviour at the boundary (`angle = period/2.0`) so future changes don't drift unnoticed.

4. The existing call sites in `FitEllipse` and elsewhere don't pass `xp` — they get the numpy default and behaviour is unchanged. Don't thread `xp` through the call sites in this prompt; that happens in prompt 6 and 7 where it actually matters.

5. Add unit tests in `@PyAutoGalaxy/test_autogalaxy/ellipse/test_ellipse.py` that for one fixed `Ellipse` and one fixed `EllipseMultipole`, the `xp=np` and `xp=jnp` paths produce numerically identical points to `rtol=1e-6`. Gate the JAX side with `pytest.importorskip("jax")`.

6. Test bar:
   - `python -m pytest test_autogalaxy/ellipse/ -v` passes.
   - The reference numbers from prompt 2's workspace_test scripts are unchanged on the numpy path.
