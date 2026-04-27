# Eager-numpy regression assertions for imaging / interferometer profiling scripts

## Context

`jax_profiling/point_source/source_plane.py` anchors its **eager-numpy**
baseline log-likelihood against a hardcoded constant
`EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE = -4496.798984131583` (see
`source_plane.py:506`). That pattern catches silent forward-pass regressions
in the numpy stack — changes in a profile/blurring/chi-squared step that would
drift the likelihood without tripping any existing np↔jnp cross-check.

The imaging and interferometer scripts have the pieces but not the assertion:

- they compute `log_likelihood_ref = fit.log_likelihood` (and often
  `log_evidence_ref = fit.figure_of_merit`) from the eager path at the top, and
- they already assert `float(full_result)` (JIT) and `np.array(result_vmap)`
  (vmap) against `EXPECTED_LOG_LIKELIHOOD_*` / `EXPECTED_LOG_EVIDENCE_*`,
- **but** the eager `_ref` baseline is never itself asserted against the
  hardcoded constant.

That's a gap: a regression in the eager numpy stack would change
`log_likelihood_ref` without the JIT/vmap assertions necessarily catching it
(the JIT and eager paths can drift in lockstep if the shared upstream code
changes).

## Task

Add an **eager-numpy regression assertion** to every
`jax_profiling/imaging/*.py` and `jax_profiling/interferometer/*.py` script
that currently uses the `EXPECTED_LOG_*` hardcoded-constant pattern. Model
the assertion on the point-source template:

```python
# From point_source/source_plane.py ~line 508
np.testing.assert_allclose(
    log_likelihood_ref,
    EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE,
    rtol=1e-4,
    err_msg=(
        f"point_source/source_plane: regression — eager log_likelihood drifted "
        f"(got {log_likelihood_ref}, expected {EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE:.6f}"
)
```

The assertion must run on the eager baseline **before** the existing JIT /
vmap assertions, so a failure points straight at the numpy stack rather than
at compilation.

Use whichever `_ref` variable each script already has (`log_likelihood_ref`
or `log_evidence_ref`) and match it to the existing `EXPECTED_LOG_LIKELIHOOD_*`
or `EXPECTED_LOG_EVIDENCE_*` constant. If a script happens to assert against
`log_evidence` (pixelization / delaunay) rather than `log_likelihood`, mirror
that here too.

## Affected scripts

### Imaging — scalar forward log-likelihood (full pattern)

- `jax_profiling/imaging/mge.py`
  (ref: `log_likelihood_ref`, constant: `EXPECTED_LOG_LIKELIHOOD_HST`)
- `jax_profiling/imaging/pixelization.py`
  (ref: `log_evidence_ref`, constant: `EXPECTED_LOG_EVIDENCE_HST`)
- `jax_profiling/imaging/delaunay.py`
  (ref: `log_evidence_ref`, constant: `EXPECTED_LOG_EVIDENCE_HST`)

### Interferometer — scalar forward log-likelihood (full pattern)

- `jax_profiling/interferometer/mge.py`
  (ref: `log_likelihood_ref`, constant: `EXPECTED_LOG_LIKELIHOOD_SMA`)
- `jax_profiling/interferometer/pixelization.py`
  (ref: `log_likelihood_ref`, constant follows the file's existing
  `EXPECTED_LOG_*` if present; otherwise introduce one and pin it to the
  observed eager value on a clean run)

### Gradient scripts — scope limited to scalar forward log-likelihood

Gradient scripts (`*_gradients.py`) use `jax.value_and_grad`, and numpy has
no autograd equivalent, so the gradient **vector** cannot be cross-checked
against a numpy reference. **Do not** add any assertion on the gradient
itself.

However, each gradient script still performs an eager forward-pass
log-likelihood computation before running autodiff (e.g.
`imaging/mge_gradients.py:247` prints `fit.log_likelihood` from the eager
`FitImaging`). That scalar forward pass is worth anchoring for exactly the
same reason it is worth anchoring in the non-gradient scripts: it catches
forward-stack regressions that would silently perturb every gradient
downstream.

Apply the assertion pattern to the **scalar** eager log-likelihood (or
log-evidence) only — and if the current script only prints the value without
capturing it, lift it to a `log_likelihood_ref = fit.log_likelihood` variable
first. Scripts to cover:

- `jax_profiling/imaging/mge_gradients.py`
- `jax_profiling/imaging/pixelization_gradients.py`
- `jax_profiling/interferometer/mge_gradients.py`

Skip any gradient script that does **not** compute an eager forward scalar
log-likelihood (unlikely — they all currently do), since in that case there
is nothing scalar to anchor. Do NOT introduce a new eager forward call just
for the assertion; only wire the assertion into values the script already
computes.

### Out of scope

- The gradient vector from `jax.value_and_grad` — no numpy reference exists.
- `imaging/mapper_grad_isolate.py`, `imaging/mapper_grad_probe.py`,
  `imaging/nnls_precondition_bench.py` — these are intermediate-step
  diagnostic scripts that do not compute a full-pipeline log-likelihood or
  use an `EXPECTED_LOG_*` constant. Leave them alone.
- `jax_profiling/point_source/source_plane.py` — already has the pattern.
  `point_source/image_plane.py` — add the pattern if it already captures a
  `log_likelihood_ref` and defines an `EXPECTED_LOG_*` constant; otherwise
  out of scope.

## Verification

After wiring each assertion:

```bash
source ~/Code/PyAutoLabs-wt/<task-name>/activate.sh
cd autolens_workspace_developer
python jax_profiling/imaging/mge.py
python jax_profiling/imaging/pixelization.py
python jax_profiling/imaging/delaunay.py
python jax_profiling/imaging/mge_gradients.py
python jax_profiling/imaging/pixelization_gradients.py
python jax_profiling/interferometer/mge.py
python jax_profiling/interferometer/pixelization.py
python jax_profiling/interferometer/mge_gradients.py
```

Every script should print an "Eager regression assertion PASSED" line
immediately after the eager fit and before the JIT / vmap profiling runs.

## Affected repos

- `autolens_workspace_developer` (only — workspace-only task, no library
  changes required)

## Suggested branch

`feature/eager-numpy-regression-assertions`

## Notes

- Use `rtol=1e-4` for consistency with the existing JIT / vmap assertions.
- If any eager `_ref` value drifts vs. the existing `EXPECTED_*` constant,
  that is the signal this prompt is designed to surface — stop and
  investigate rather than bumping the constant.
- The assertion must use `np.testing.assert_allclose`, not `assert abs(...)` —
  it produces better diagnostic output on failure.
