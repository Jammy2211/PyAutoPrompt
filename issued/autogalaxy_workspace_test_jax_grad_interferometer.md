Create `scripts/jax_grad/interferometer/` in @autogalaxy_workspace_test exercising `jax.grad` on
the autogalaxy interferometer likelihood path.

__Layout note__

Same subfolder-vs-flat divergence from autolens as task 6 — autogalaxy uses
`jax_grad/interferometer/`. Flag in PR body; don't retrofit autolens without user go-ahead.

__Scripts__

autolens has **no** interferometer `jax_grad` scripts today. This task creates them from scratch
for autogalaxy, using the corresponding `jax_likelihood_functions/interferometer/` scripts as
templates for model + dataset setup, then wrapping the likelihood in `jax.grad`.

Minimum coverage:

- `jax_grad/interferometer/lp.py`
- `jax_grad/interferometer/mge.py`

Additional variants only if the `jax_likelihood_functions/interferometer/` task surfaced a
ready-to-grad path.

**Skip**: `*_dspl.py`.

__Pytree prerequisite__

Task 4 must have landed (`AnalysisInterferometer` pytree registration on PyAutoGalaxy).

__`jax.grad` contract__

Same as task 6 — finite gradient, correct free-parameter shape.

__Deliverables__

1. `autogalaxy_workspace_test/scripts/jax_grad/interferometer/__init__.py`
2. Scripts above.
3. Appended to `smoke_tests.txt`.

__Depends on__

Task 4 (pytree registration on PyAutoGalaxy interferometer analysis).

__Umbrella issue__

Task 7/9. Track under the epic issue on `PyAutoLabs/autogalaxy_workspace_test`.
