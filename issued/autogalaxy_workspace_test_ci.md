Set up a GitHub Actions smoke-test workflow on @autogalaxy_workspace_test, mirroring the existing
setup in @autolens_workspace_test. This is the **blocker** for all sibling tasks in this folder —
once CI is green, the eight script-expansion tasks can land and be verified on every PR.

__Reference layout__

@autolens_workspace_test/.github/workflows/smoke_tests.yml and
@autolens_workspace_test/.github/scripts/run_smoke.py are the templates to copy. Supporting files
already present in that repo:

- `smoke_tests.txt` — list of scripts to run
- `config/build/env_vars.yaml` — per-script env overrides (test mode, small-dataset flag, JAX on/off)

@autogalaxy_workspace_test already has `smoke_tests.txt` at the repo root (currently empty of all
but aggregator entries). It does **not** yet have `.github/`, `.github/scripts/run_smoke.py`, or
`config/build/env_vars.yaml`.

__Deliverables__

1. `autogalaxy_workspace_test/.github/workflows/smoke_tests.yml` — fork the autolens version but
   **drop the PyAutoLens checkout + install step**. The matrix should check out
   PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy only. Keep the branch-matching logic, the
   Python 3.12 / 3.13 split, the numba install on 3.13, the tensorflow-probability pin, and the
   Slack-on-failure webhook (same channel).
2. `autogalaxy_workspace_test/.github/scripts/run_smoke.py` — copy verbatim from autolens
   (it is workspace-agnostic; it reads `smoke_tests.txt` and `config/build/env_vars.yaml` relative
   to itself).
3. `autogalaxy_workspace_test/config/build/env_vars.yaml` — start with the autolens defaults
   (`PYAUTOFIT_TEST_MODE=2`, `PYAUTO_WORKSPACE_SMALL_DATASETS=1`, `PYAUTO_DISABLE_JAX=1`,
   `PYAUTO_FAST_PLOTS=1`, `JAX_ENABLE_X64=True`, `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`) and the
   `jax_likelihood_functions/` override that unsets the small-dataset / disable-JAX flags.
   Drop any autolens-specific overrides (e.g. `database/scrape/`) unless they already apply.
4. Seed `smoke_tests.txt` with only scripts that exist **today** in autogalaxy_workspace_test
   (aggregator entries). The eight sibling tasks will append their new scripts as they ship.
5. PR includes a CI run showing the aggregator suite passing on both Python 3.12 and 3.13.

__Notes__

- Use LF line endings. All files in this repo are Unix LF — see repo CLAUDE.md.
- The smoke-test env var `PYAUTO_DISABLE_JAX=1` is honoured by `Analysis.__init__` in PyAutoFit
  (see the `smoke-test-optimization` work in active.md for context). JAX-likelihood scripts opt
  back in via the `jax_likelihood_functions/` override.

__Umbrella issue__

This is task 1/9 in the `expand autogalaxy_workspace_test coverage` epic — track progress under
the umbrella issue on `PyAutoLabs/autogalaxy_workspace_test`.
