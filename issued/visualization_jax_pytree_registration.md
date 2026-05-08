Both `autolens_workspace_test/scripts/imaging/visualization_jax.py` and
`autogalaxy_workspace_test/scripts/imaging/visualization_jax.py` are
**silently broken under JAX**: when run with `PYAUTO_DISABLE_JAX` unset
(i.e. JAX actually enabled), they fail with:

```
TypeError: Error interpreting argument to <bound method AnalysisImaging.fit_from>
as an abstract array. The problematic value is of type
<class 'autofit.mapper.model.ModelInstance'> and was passed to the function
at path instance.
```

Both scripts catch their own exception, print "PILOT FAILED", and then exit
with code 0 — so test runners don't notice. Discovered 2026-05-08 while
smoke-verifying the autogalaxy dispatch swap (PyAutoGalaxy #390). The
autolens version has been broken since the equivalent dispatch swap in
PR #443 (2026-04-19).

__Why this matters__

These scripts are meant to be the dedicated smoke verifier for the
`use_jax_for_visualization=True` JIT path on imaging — but in their
current state they silently no-op:

- They are **not in `smoke_tests.txt`** in either workspace_test repo.
- The workspaces' `env_vars.yaml` matches `imaging/visualization` and
  applies `PYAUTO_DISABLE_JAX=1` (default), which silently flips both
  `use_jax` flags off in `Analysis.__init__`. The script falls through to
  NumPy and "passes" without exercising the JIT path it claims to test.

Net effect: the JIT-cached visualization path has zero smoke coverage on
the standalone (non-Nautilus) call site. Coverage of the JIT path in a
live search exists via `imaging/modeling_visualization_jit.py` (which
DOES register the model — see lines 43-45) but the standalone
`fit_for_visualization` path is untested.

__Root cause__

`fit_for_visualization` lazily wraps `fit_from` in `jax.jit` when
`use_jax_for_visualization=True` (see
`@PyAutoFit/autofit/non_linear/analysis/analysis.py:114-122`). For
`jax.jit` to trace the call, the `instance: ModelInstance` argument must
be pytree-registered. `autofit.jax.pytrees` provides `enable_pytrees()`
+ `register_model(model)` for that — but neither `visualization_jax.py`
script calls them.

The working sibling `modeling_visualization_jit.py` does:

```python
from autofit.jax.pytrees import enable_pytrees, register_model
enable_pytrees()
...
register_model(model_mge)
```

Both `visualization_jax.py` scripts need the same.

__What to change__

Apply the same fix to both files:

1. `@autolens_workspace_test/scripts/imaging/visualization_jax.py`
2. `@autogalaxy_workspace_test/scripts/imaging/visualization_jax.py`

For each:

- Add at the top, after `import autolens as al` (or `autogalaxy as ag`):
  ```python
  from autofit.jax.pytrees import enable_pytrees, register_model

  enable_pytrees()
  ```
- After the `model = af.Collection(...)` line, before `analysis =
  ag.AnalysisImaging(...)`:
  ```python
  register_model(model)
  ```
- Replace the script-level try/except that swallows the JIT error with a
  hard re-raise so future regressions are caught instead of being printed
  and silently exit 0:
  ```python
  VisualizerImaging.visualize(
      analysis=analysis,
      paths=paths,
      instance=instance,
      during_analysis=False,
  )
  assert (image_path / "parametric" / "fit.png").exists() or (
      image_path / "fit.png"
  ).exists(), "fit.png was not produced"
  print("PILOT SUCCEEDED — JAX-backed visualization produced fit.png/tracer.png.")
  ```
  Drop the `try: ... except Exception: traceback.print_exc()` block. If
  the JIT path breaks again, the script should fail loudly.

__env_vars.yaml — second-order question__

Both workspaces' `config/build/env_vars.yaml` have an `imaging/visualization`
override that applies to both `visualization.py` and `visualization_jax.py`
(substring match) and currently sets `PYAUTO_DISABLE_JAX=1` for both.

After this fix lands, `visualization_jax.py` will actually need JAX to
run (it was always supposed to, but silently fell through). Two options:

- **Option A — narrow the existing override.** Change the pattern to
  `imaging/visualization.py` so it only matches the NumPy script; add a
  new override for `imaging/visualization_jax` that unsets `PYAUTO_DISABLE_JAX`
  alongside the existing unsets. Mirrors the pattern used for
  `imaging/modeling_visualization_jit`.
- **Option B — leave env unchanged, mark the script auto-skipping in CI.**
  Worse — keeps the silent-skip behaviour we are trying to remove.

Take Option A.

__Verification__

After both files are fixed, run each WITH `PYAUTO_DISABLE_JAX` unset:

```bash
# autolens
cd autolens_workspace_test
JAX_ENABLE_X64=True NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
  python scripts/imaging/visualization_jax.py

# autogalaxy
cd autogalaxy_workspace_test
JAX_ENABLE_X64=True NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
  python scripts/imaging/visualization_jax.py
```

Both should print `PILOT SUCCEEDED ...` and produce `fit.png` (or
`parametric/fit.png`). Re-run with the env_vars.yaml-resolved env (which
will now include `unset: [PYAUTO_DISABLE_JAX, ...]` for the
`imaging/visualization_jax` pattern) to confirm the build path also runs
JAX.

The pre-existing `imaging/visualization.py` (NumPy) and
`imaging/modeling_visualization_jit.py` must continue to pass.

__Out of scope__

- No library changes. The failure is purely test-script + env config.
- No additions to `smoke_tests.txt`. Per the user's smoke-test policy
  (`MEMORY.md`: "Smoke tests are a small curated subset"), promoting these
  scripts into smoke is a separate decision.
- No `register_model` audit elsewhere. If other workspace_test scripts
  also try `use_jax_for_visualization=True` without registering, they
  hit the same issue — but that's a follow-up prompt to author after
  this one lands and the pattern is established.

__Reference__

- `@PyAutoFit/autofit/jax/pytrees.py` — `enable_pytrees`, `register_model`
- `@autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py:43-45` — working sibling pattern
- `@autogalaxy_workspace_test/scripts/imaging/modeling_visualization_jit.py` — working sibling
- `@PyAutoFit/autofit/non_linear/analysis/analysis.py:82-122` — `fit_for_visualization` JIT dispatch
- PyAutoGalaxy #390 — dispatch swap that surfaced this issue (2026-05-08)
- PyAutoLens #443 — earlier autolens dispatch swap (2026-04-19) that left autolens script in same broken state
- `PyAutoPrompt/z_features/jax_visualization.md` — sequenced roadmap (informal Phase 1 follow-up)
