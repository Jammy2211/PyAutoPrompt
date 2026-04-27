Create `scripts/jax_grad/multi/` in @autogalaxy_workspace_test exercising `jax.grad` on the
autogalaxy multi-dataset likelihood path.

__Layout note__

Same subfolder-vs-flat divergence as tasks 6 and 7. Flag in PR body.

__Scripts__

autolens has no multi `jax_grad` scripts today. Create from scratch using the
`jax_likelihood_functions/multi/` templates (task 5).

Minimum coverage:

- `jax_grad/multi/lp.py`
- `jax_grad/multi/mge.py`

Additional variants only if feasible.

**Skip**: `*_dspl.py`.

__Pytree prerequisite__

Task 5 scaffolding (multi-dataset factor-graph pytree registration) must be complete.

__`jax.grad` contract__

Gradient is taken over the full parameter vector spanning all datasets. Assert finite and the
shape matches the combined free-parameter count.

__Deliverables__

1. `autogalaxy_workspace_test/scripts/jax_grad/multi/__init__.py`
2. Scripts above.
3. Appended to `smoke_tests.txt`.

__Depends on__

Task 5 (multi-dataset pytree registration).

__Umbrella issue__

Task 8/9. Track under the epic issue on `PyAutoLabs/autogalaxy_workspace_test`.
