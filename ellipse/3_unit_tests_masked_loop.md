Step 3 of the ellipse-JAX series. The 300-iteration mask-rejection loop in `FitEllipse.points_from_major_axis_from` (`@PyAutoGalaxy/autogalaxy/ellipse/fit_ellipse.py:81-134`) is the single nastiest piece of imperative code in the ellipse module — Python `for` loop, dynamic shape changes via `points = points[unmasked_indices]`, scipy interpolator calls, and a `raise ValueError` after 300 iterations. Prompt 6 will rewrite it for JAX. Before that we need unit tests in `@PyAutoGalaxy/test_autogalaxy/ellipse/test_fit_ellipse.py` that pin the current behaviour, so when the rewrite goes in we know nothing has shifted.

Please:

1. Add (or extend) tests in `@PyAutoGalaxy/test_autogalaxy/ellipse/test_fit_ellipse.py` covering the four exit paths of the loop:

   - **Zero-masked**: a mask that doesn't overlap the ellipse points. Assert `points.shape[0] == ellipse.total_points_from(pixel_scale)` and that the loop exits on the first iteration via the `total_points_required == total_points - total_points_masked` branch.

   - **Under-masked (trim path)**: a mask that drops a small fraction of points (e.g. ~10%). Assert the returned `points` has exactly `total_points_required` rows and that the trimmed indices are the latest ones in `unmasked_indices` (the slicing is `unmasked_indices[number_of_extra_points:]`).

   - **Over-masked (extra-points path)**: a mask that drops enough points to force a re-call of `ellipse.points_from_major_axis_from(..., n_i=i)` with a higher angular resolution. Assert the loop runs for at least one iteration and that the final `points.shape[0] == total_points_required`.

   - **Unreachable (`ValueError`)**: a degenerate mask where no `i ≤ 300` satisfies the constraint. Use `pytest.raises(ValueError)` and assert the error message matches the existing wording in `fit_ellipse.py`.

2. Use small `Mask2D` shapes (e.g. 30×30) so the tests run fast. Build the mask explicitly via `aa.Mask2D(...)` rather than `Mask2D.circular`, so the masked region is deterministic and the tests are robust to changes in the circular-mask helper.

3. Cover both with and without `multipole_list`. The multipole branch goes through the inner `for multipole in self.multipole_list:` block at lines 113-120 — add at least one test that hits this with `EllipseMultipole(m=4, multipole_comps=(0.05, 0.0))`.

4. Pin numerical reference values for at least one mask configuration: capture the returned `points` array via `np.testing.assert_allclose(points, expected, rtol=1e-12)` against a hard-coded reference. This is the strongest regression test — when prompt 6 swaps the loop for a JAX-friendly oversample-then-mask approach, the new path should reproduce these numbers (or the test must be re-pinned with a written justification).

5. Follow `@PyAutoGalaxy/CLAUDE.md` "Never use JAX in unit tests" — these tests stay numpy-only. The cross-numpy/JAX parity check happens in the workspace_test scripts from prompt 2.

6. Test bar: `python -m pytest test_autogalaxy/ellipse/test_fit_ellipse.py -v` passes, including the new tests.
