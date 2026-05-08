## Re-baseline the MGE imaging JIT profiling regression value

`@autolens_workspace_developer/jax_profiling/jit/imaging/mge.py` hardcodes a
regression assertion at the bottom:

```python
EXPECTED_LOG_LIKELIHOOD_HST = 27379.38890685539

np.testing.assert_allclose(log_likelihood_ref, EXPECTED_LOG_LIKELIHOOD_HST, rtol=1e-4, ...)
np.testing.assert_allclose(float(full_result), EXPECTED_LOG_LIKELIHOOD_HST, rtol=1e-4, ...)
np.testing.assert_allclose(np.array(result_vmap), EXPECTED_LOG_LIKELIHOOD_HST, rtol=1e-4, ...)
```

On canonical `main` the eager `log_likelihood_ref` is now `27542.080621803576`
— a 162.7-unit drift (~0.6% relative), well past `rtol=1e-4`. Confirmed during
the `fft-mixed-precision-fix` audit (`PyAutoArray#302`): the drift is *not*
caused by the FFT precision fix, since `log_likelihood_ref` comes from the
eager NumPy `FitImaging` path which the fix doesn't touch. Both pre-fix and
post-fix builds produce 27542.08 on the same dataset.

So the hardcoded value is just stale — some upstream change between when it
was last set and now (probably a simulator tweak, an over-sample default, or
a numerical-stability change in `Isothermal.deflections_yx_2d_from`) shifted
the truth-point likelihood. The script crashes at the regression assertion
on every run currently.

### What to do

1. Run the script once on a clean checkout to capture the new value. Keep
   the same dataset path / pixel scale / mask radius the script declares so
   the regression remains comparable across machines.

2. Confirm eager / full / vmap all agree to ~1e-11 (per the existing
   docstring at line 822 of `mge.py`). If they don't, that's a separate
   correctness bug, *not* a re-baseline.

3. Update `EXPECTED_LOG_LIKELIHOOD_HST` to the new measurement. Add a short
   inline comment noting which upstream commit the value was re-pinned
   against (use `git log -1 PyAutoArray/autoarray` or similar to find the
   most recent library SHA at the time of re-baselining).

4. Optionally tighten or loosen `rtol` based on the observed agreement
   between eager / JIT / vmap. The existing `rtol=1e-4` is conservative.

### Out of scope

- Investigating *why* the value drifted. The regression assertion is a smoke
  trip-wire, not a debugging tool. Whoever cared about reproducibility at the
  ~1e-4 level should have noticed the drift in CI, but `mge.py` doesn't run
  in CI today (it's a developer profiling script). If you want it in CI,
  that's a separate task.

- The other instrument variants (`euclid`, `jwst`, `ao`) — the script
  currently only sets the HST regression value. If those should also have
  hardcoded targets, that's a feature request, not a re-baseline.

### Why this didn't get bundled into PyAutoArray#302

The fft-mixed-precision-fix work scoped its plan to PyAutoArray +
autogalaxy_workspace_test + autolens_workspace_test. `autolens_workspace_developer`
was never in scope, and the mge.py drift is independent of the fix. Bundling
it in would have inflated the PR's blast radius without clear benefit.
