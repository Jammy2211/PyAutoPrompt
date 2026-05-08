# JAX Visualization — Sequenced Roadmap

This z_feature decomposes the multi-phase initiative to make JAX visualization
the default path whenever `use_jax=True` is set, and eventually run that
visualization in a separate process with live Jupyter / Colab cell updates.

Each phase below points to a concrete sub-prompt file. Phases that require
predecessor work to land first are marked with their dependency. Far-out
phases are intentionally left as stubs — their concrete prompts will be
authored once their predecessors merge.

The original motivating prose is preserved at the bottom under
**Background — original framing**.

## Phase 0 — In-flight prerequisites

These are already issued or pending; they unblock the rest of the roadmap.

| # | Title | Prompt file | Status |
|---|-------|-------------|--------|
| 0a | Path A feasibility study (FitImaging pytree) | `issued/fit_imaging_pytree.md` | issued |
| 0b | PyAutoGalaxy imaging visualizer dispatch swap | `autogalaxy/visualizer_fit_for_visualization_dispatch.md` | pending |
| 0c | PyAutoGalaxy interferometer / ellipse / quantity pytree registration | `autogalaxy/fit_pytree_registration_other_datasets.md` | pending |

## Phase 1 — Workspace_test JAX visualization coverage

Audit on 2026-05-08 confirmed coverage gaps in both workspace_test repos: only
`imaging/` has the full NumPy + JAX + jit visualization triplet. Filling the
gaps is mechanical (mirror the existing imaging scripts) but has to land
**before** Phase 2 flips the default — otherwise turning `use_jax_for_visualization`
on by default risks silently breaking dataset types that have no smoke
coverage.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 1A | autolens_workspace_test interferometer JAX viz coverage | `autolens_workspace_test/jax_viz_interferometer_coverage.md` | none — PyAutoLens dispatch is already wired |
| 1B | autolens_workspace_test point_source viz coverage | `autolens_workspace_test/jax_viz_point_source_coverage.md` | source_plane JIT feasibility check (existing CLAUDE.md note) |
| 1C | autogalaxy_workspace_test interferometer / ellipse / quantity coverage | `autogalaxy_workspace_test/jax_viz_dataset_coverage.md` | Phases 0b + 0c |

## Phase 2 — Default `use_jax_for_visualization` to follow `use_jax`

Once Phases 0 and 1 are green for every dataset type, flip the default in
`autofit.non_linear.analysis.analysis.Analysis.__init__` so that any analysis
constructed with `use_jax=True` automatically uses the jit-cached visualization
path — making the "two-flag" model invisible to users. Users who want the
slower NumPy plotter can still opt out with `use_jax_for_visualization=False`.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 2 | Default `use_jax_for_visualization` to follow `use_jax` in `Analysis.__init__` | `autofit/use_jax_for_visualization_default_on.md` | Phase 1 (all dataset types) |

## Phase 3 — Production workspace tutorial adoption (stub)

Today zero scripts in `autolens_workspace`, `autogalaxy_workspace`, or
`autofit_workspace` set `use_jax=True` in tutorials by default. Once Phase 2
defaults are flipped, opt-in tutorials that benefit (large pixelization fits,
multi-band, group lensing) should set `use_jax=True` so they pick up the JAX
visualization path automatically.

**Prompt to author when Phase 2 ships.** Likely splits per workspace.

## Phase 4 — Visualization in a separate process (stub)

Today `iterations_per_quick_update` triggers visualization synchronously in
the search process — the search pauses while plots are written. The next
unlock is dispatching visualization to a dedicated worker process so the
search never blocks on rendering.

Open architectural questions (will need their own feasibility prompt):

- `multiprocessing.Process` with a queue (FitImaging pickled / serialised across
  the boundary) vs `concurrent.futures.ProcessPoolExecutor` vs `subprocess.Popen`
  with disk-staged intermediate state.
- Whether `FitImaging` is picklable post-pytree-registration or whether we
  serialise raw arrays + reconstruct in the worker.
- Backpressure: what happens if the worker is still rendering the previous
  iteration when the next quick-update fires (drop, queue, coalesce).
- Failure semantics: a render crash must not take the search down.

**Prompt to author after Phase 3 lands** with one feasibility prototype first.

## Phase 5 — Live Jupyter / Colab quick-update cell + Colab default (stub)

When a search runs inside a Jupyter or Colab notebook, `subplot_fit.png` should
update **in place in a single cell** every `iterations_per_quick_update`
iterations rather than writing to disk only. On Colab GPU this should run fast
enough that watching the lens-modeling fit converge becomes the default
experience.

Components:

- `IPython.display` with a stable `display_id` so the image cell is overwritten,
  not appended (`PyAutoFit/autofit/non_linear/fitness.py` already uses
  `clear_output` for terminal text — this extends the same idea to the image).
- Detection: `get_ipython()` + Colab environment variable to switch default
  visualization channel.
- Interaction with Phase 4: subprocess viz must hand the rendered image back
  to the main kernel for `display_id` update; or alternatively the worker
  publishes images via a shared display-id channel.

**Prompt to author after Phase 4 lands.**

## Background — original framing

> Alongside these issues, its basically time to make it so all workspace examples when use_jax=True go through
> the full JAX visualization path, which will ensure really fast visualization and thus really fast updates.
>
> Remember we have examples already showing this works:
>
> - `autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py` and similar scripts there in.
> - `autolens_workspace_test/scripts/imaging/visualization_jax.py`
>
> I want us to be at a point where all default runs do JAX visualization and the notion of it being a separate
> thing is no longer relevant (unless the user doesn't have JAX installed or has `use_jax=False`).
>
> Remember also that ideally visualization (including via quick updates) would happen on a separate process
> to the search so we don't "pause" to visualize. We also want this to be something where Jupyter Notebooks
> update the quick visuals on the fly during modeling, albeit these could be follow up tasks to the first
> which is just getting it running.
>
> However, before doing lots of work, maybe we should do an assessment of `autolens_workspace_test` and
> `autogalaxy_workspace_test` and assess if we are still missing JAX Visualizer coverage in order to implement
> this seamlessly — my gut feeling is we actually want to build out
> `PyAutoPrompt/z_features/jax_visualization.md` into a logical sequence of steps that fully covers this
> step by step.
>
> A follow up issue will then be once all these runs work, to make it so a user can "watch" lens modeling happen,
> with the display updating every `iterations_per_quick_update`. Ideally, for a Python script this would be a matplotlib
> window which just updates as it does, not disrupting the user from doing other stuff, and in a Jupyter Notebook it'd
> appear as a cell which updates. We may want to add some other images and info to this, but for now just having
> `subplot_fit` display would probably be a good starting point.
>
> I also want this to be the default behaviour when a user is running a notebook via Google Colab, as on GPU it should
> run fast enough that they can thus easily watch the lens modeling run. This should ideally be how it works
> for all data types and Analysis class types.
