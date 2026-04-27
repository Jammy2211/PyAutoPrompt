Create `scripts/jax_grad/imaging/` in @autogalaxy_workspace_test exercising `jax.grad` on the
autogalaxy imaging likelihood path.

__Important — layout divergence from autolens__

@autolens_workspace_test/scripts/jax_grad/ currently has files **at the top level**
(`imaging_lp.py`, `imaging_mge.py`) — there is no `jax_grad/imaging/` subfolder. The user's
prompt for this epic explicitly asks for `jax_grad/imaging/`, `jax_grad/interferometer/`,
`jax_grad/multi/` subfolders on autogalaxy. Options:

1. Follow the user's directive — use subfolders on autogalaxy even though autolens is flat.
   Flag this drift in the PR description and ask the user whether to retrofit autolens.
2. Ask the user before creating the subfolder.

Recommended: go with (1) — the subfolder layout matches `jax_likelihood_functions/` and is more
extensible. Surface the autolens-retrofit question in the PR body.

__Scripts to port__

From autolens top-level `jax_grad/`:

- `imaging_lp.py` → `jax_grad/imaging/lp.py`
- `imaging_mge.py` → `jax_grad/imaging/mge.py`

Add any further variants (rectangular, delaunay, …) only if the corresponding
`jax_likelihood_functions/imaging/` port from task 3 uncovered a usable `grad`-ready path.
Default: match autolens's coverage one-for-one and leave extras to follow-ups.

**Skip**: anything `*_dspl.py`.

__Pytree prerequisite__

Shares the same `_register_fit_imaging_pytrees` requirement as task 3. By this point the
registration should already be landed on PyAutoGalaxy — if not, block on that first.

__`jax.grad` contract__

Each script prints the gradient vector of a scalar log-likelihood w.r.t. the parameter vector and
asserts it's finite and the shape matches the model's free parameter count. Follow the autolens
reference exactly for the assertions.

__Deliverables__

1. `autogalaxy_workspace_test/scripts/jax_grad/__init__.py`
2. `autogalaxy_workspace_test/scripts/jax_grad/imaging/__init__.py`
3. Ported scripts.
4. Appended to `smoke_tests.txt`.
5. PR body includes the autolens-retrofit question in `## Notes`.

__Depends on__

Task 3 (pytree registration on PyAutoGalaxy imaging analysis).

__Umbrella issue__

Task 6/9. Track under the epic issue on `PyAutoLabs/autogalaxy_workspace_test`.
