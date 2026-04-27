The new workspace smoke-test GitHub Actions (added via feature/smoke-test-ci) surfaced
pre-existing failures that were commented out to get the green baseline. Re-enable each
entry once its underlying bug is fixed.

## autofit_workspace/smoke_tests.txt

- `cookbooks/samples.py` — `IndexError: list index out of range` in bypass mode
  (`PYAUTO_TEST_MODE=2`). Was already in the original smoke list but apparently
  never exercised in bypass mode before. Investigate which `samples.*` list access
  runs past its length when the sampler is skipped.

## autogalaxy_workspace/smoke_tests.txt

- `group/modeling.py` — committed dataset (150x150) fights with
  `PYAUTO_SMALL_DATASETS=1` cap (15x15). Either add an `unset:
  [PYAUTO_SMALL_DATASETS]` override in `config/build/env_vars.yaml` for
  `group/modeling`, or delete the committed 150x150 dataset so the simulator
  regenerates it at 15x15.
- `multi/modeling.py` — same pattern with a committed 100x100 dataset.
- `ellipse/fit.py` — same pattern with the ellipse dataset.

Pick one approach (override vs strip-committed-data) and apply consistently.

## autolens_workspace/smoke_tests.txt

- `imaging/likelihood_function.py` — committed dataset (100x100) clashes with
  `PYAUTO_SMALL_DATASETS=1`. Same fix as the autogalaxy entries above.

## autofit_workspace_test/smoke_tests.txt

- `searches/Emcee.py` — `TypeError: only 0-dimensional arrays can be converted
  to Python scalars` on Python 3.13 + numpy 2.4. Failure is in
  `autofit/non_linear/samples/mcmc.py:147` (`median_pdf`) when converting an
  ndarray result to a scalar. Needs a numpy-2.x-safe conversion
  (`.item()` on a single element, or explicit indexing).

## autolens_workspace_test/smoke_tests.txt

- `jax_likelihood_functions/imaging/delaunay_mge.py` — `AttributeError:
  jax.interpreters.xla.pytype_aval_mappings was deprecated in JAX v0.5.0 and
  removed in JAX v0.7.0`. The internal call site needs migrating to
  `jax.core.pytype_aval_mappings`. Likely somewhere in PyAutoArray or
  PyAutoGalaxy JAX-interop code.

## Cross-cutting fixes already landed in this task

- PyAutoFit PR #1219: made `pandas.errors.SettingWithCopyWarning` import
  tolerate pandas >= 2.2. This unblocks many `autofit_workspace_test` scripts
  (DynestyStatic, Emcee, minimal_output, latent) and
  `autolens_workspace_test/database/scrape/general.py`.
- `autolens_workspace_test` CI installs `tensorflow-probability==0.25.0` so
  Matérn-kernel JAX likelihood functions can import.
- `autofit_workspace_test` CI installs `nautilus-sampler` so
  `scripts/searches/Nautilus.py` runs.
