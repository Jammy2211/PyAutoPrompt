Today `Analysis.__init__` in PyAutoFit takes two independent flags:

```python
# @PyAutoFit/autofit/non_linear/analysis/analysis.py:36
def __init__(
    self,
    use_jax: bool = False,
    use_jax_for_visualization: bool = False,
    **kwargs,
):
```

A user who wants the JAX-accelerated visualization path has to remember to
set **both** flags. Across all production workspaces (`autolens_workspace`,
`autogalaxy_workspace`, `autofit_workspace`) zero scripts set
`use_jax_for_visualization=True` today (audit, 2026-05-08), even where
`use_jax=True` would benefit. The flag is effectively dead weight — it
exists to gate the still-evolving JIT visualization path, but once the
underlying coverage is green there is no reason it should not follow
`use_jax`.

This task changes the default so that whenever `use_jax=True`, the
jit-cached visualization path is on by default. Users keep the explicit
opt-out via `use_jax_for_visualization=False`.

__Why this matters__

This is **Phase 2** of `z_features/jax_visualization.md`. The user-facing
goal stated in the z_feature is:

> "I want us to be at a point where all default runs do JAX visualization
> and the notion of it being a separate thing is no longer relevant
> (unless the user doesn't have JAX installed or has `use_jax=False`)."

This prompt delivers that.

__Blockers — must land first__

All Phase 1 coverage prompts must be merged before flipping the default,
otherwise dataset types with no JAX viz smoke coverage will silently
regress for any user who runs them with `use_jax=True`:

- `autolens_workspace_test/jax_viz_interferometer_coverage.md`
- `autolens_workspace_test/jax_viz_point_source_coverage.md`
- `autogalaxy_workspace_test/jax_viz_dataset_coverage.md`

Phase 0 prerequisites (`fit_imaging_pytree.md`, the autogalaxy dispatch
swap, the autogalaxy other-datasets pytree registration) must also be
done. Verify all of these in `complete.md` before starting.

__What to change__

`@PyAutoFit/autofit/non_linear/analysis/analysis.py:36-79` — change the
default of `use_jax_for_visualization` from `False` to a sentinel that
follows `use_jax`. Recommended signature:

```python
def __init__(
    self,
    use_jax: bool = False,
    use_jax_for_visualization: Optional[bool] = None,
    **kwargs,
):
    ...
    if use_jax_for_visualization is None:
        use_jax_for_visualization = use_jax
    ...
```

`None` means "follow `use_jax`"; `True`/`False` are explicit opt-in/opt-out.
Resolution must happen **before** the existing
`if use_jax_for_visualization and not use_jax` guard at line 71 so the
guard's wording (`"requires use_jax=True; disabling..."`) still applies
correctly when a user explicitly passes `True` without `use_jax`.

The `PYAUTO_DISABLE_JAX=1` short-circuit at lines 42-45 already forces
both flags to `False`, so the env-var override path is unaffected — but
double-check that branch still resolves cleanly with the new sentinel.

The "JAX not installed" warning branch at lines 50-69 already sets
`use_jax_for_visualization = False`. That still works because the new
sentinel resolution happens after this branch — but be careful about
ordering: the warning branch sets `use_jax = False`, so the sentinel
resolution downstream picks up `False` correctly.

Update the docstring at lines 82-122 to reflect the new default behaviour.
The warning at line 71-76 must remain — passing
`use_jax_for_visualization=True, use_jax=False` explicitly is still a user
error (deserves a warning, then disabled).

__What to verify__

1. **Unit tests for the resolution logic.** Add cases to
   `@PyAutoFit/test_autofit/non_linear/analysis/test_analysis.py` (or
   create the file if missing) covering:
   - `Analysis()` → both off
   - `Analysis(use_jax=True)` → `_use_jax_for_visualization=True`
   - `Analysis(use_jax=True, use_jax_for_visualization=False)` → off (explicit opt-out works)
   - `Analysis(use_jax=False, use_jax_for_visualization=True)` → off + warning logged
   - `Analysis(use_jax=True, use_jax_for_visualization=None)` → on
   - `PYAUTO_DISABLE_JAX=1` env-var override forces both off regardless
   - JAX-not-installed branch still sets both off

2. **Workspace smoke.** Run `/smoke_test` on each workspace_test repo and
   on the production workspaces. The Phase 1 coverage protects against
   regressions; this is the moment of truth.

3. **No silent-warning regressions.** Grep production workspaces and tests
   for `use_jax_for_visualization=` and confirm no script's behaviour
   changes meaningfully. Any script that previously ran with `use_jax=True`
   and no explicit `use_jax_for_visualization` was implicitly opting out;
   after this change it will opt in. That is the intended behaviour but
   should be verified case-by-case if any script's runtime changes
   significantly.

__Out of scope__

- Phase 3 (production workspace tutorial adoption — opting tutorials into
  `use_jax=True`) is a separate prompt.
- Phase 4 (subprocess visualization).
- Phase 5 (live Jupyter / Colab cell).
- Removing the `use_jax_for_visualization` parameter entirely. The
  parameter stays — only its default changes.

__Reference__

- `@PyAutoFit/autofit/non_linear/analysis/analysis.py:36-79` — site of the change
- `@PyAutoFit/autofit/non_linear/analysis/analysis.py:82-122` — `fit_for_visualization` docstring to update
- `complete.md` entries `jax-visualization`, `mge-jit-visualization` (2026-04-19) — original Phase 0 ship notes
- `PyAutoPrompt/z_features/jax_visualization.md` — Phase 2 in the sequenced roadmap
