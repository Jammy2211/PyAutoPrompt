# autolens_profiling — New Repository Roadmap

This z_feature tracks the creation of a new dedicated GitHub repository,
`PyAutoLabs/autolens_profiling`, that consolidates all profiling work for
PyAutoLens currently scattered across `autolens_workspace_developer`.

**Scope of the new repo:**

- Likelihood function profiling for all supported science cases, dataset types,
  and model compositions (MGE, pixelization, Delaunay), on CPU, laptop GPU, and
  HPC GPU (e.g. A100).
- Simulator profiling and run-time tracking (imaging, interferometer,
  point_source, cluster, group, multi).
- Sampler / search profiling (Nautilus first; other samplers later).
- Easy-to-read, instrument-framed READMEs (HST, Euclid, etc.) with versioned
  result tables and graphs that survive across PyAutoLens releases.
- CI / GitHub Actions to keep the dashboards fresh.

**Explicitly out of scope (for now):**

- JAX gradient profiling — stays in `autolens_workspace_developer/jax_profiling/gradient/`.
  A note about this lives on the new repo's front page so readers know where
  to look. Likely folds in later once the gradient story stabilises.
- New profiling examples (e.g. group lensing) — this initiative migrates
  *existing* content only; new examples follow under their own prompts.

**Migration strategy:** mirror, don't move. `autolens_workspace_developer`
remains the source of truth during the migration. Each sub-prompt copies the
relevant scripts into the new repo and adds the README narrative; later sweeps
can decide whether to retire the `_developer` originals.

**Relationship to `Jammy2211/autolens_colab_profiling`:** treated as a
separate sibling repo for now (different scope — Colab-specific). Listed in
the new repo's README under "Related repos" but not folded in.

The original motivating prose is preserved at the bottom under
**Background — original framing**.

## Phase 0 — Repo bootstrap

Create the empty `PyAutoLabs/autolens_profiling` repo on GitHub, scaffold the
top-level files (README with vision/JAX-gradient note, LICENSE, .gitignore,
folder layout for `likelihood/` / `simulators/` / `searches/` / `results/`),
and check out a local clone next to the other PyAutoLabs repos.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 0 | Create repo + scaffolding | `autolens_profiling/bootstrap.md` | — |

## Phase 1 — Likelihood JIT profiling mirror

Mirror `autolens_workspace_developer/jax_profiling/jit/{imaging,interferometer,point_source,datacube}/*`
into `autolens_profiling/likelihood/`, preserving the step-by-step profile
narrative each script already prints. Each subfolder gets a README that
explains what is being profiled, surfaces the latest results table, and links
to the underlying JSON / PNG summaries under `results/`.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 1 | Mirror JIT likelihood profiling scripts + per-section READMEs | `autolens_profiling/likelihood_jit_mirror.md` | Phase 0 |

## Phase 2 — Simulators profiling mirror

Mirror `autolens_workspace_developer/jax_profiling/simulators/*` into
`autolens_profiling/simulators/`, with first-class run-time tracking for each
simulator. Where JIT for the simulator is not implemented yet, document the
placeholder and link to the upstream issue.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 2 | Mirror simulator profiling scripts + run-time tracking | `autolens_profiling/simulators_mirror.md` | Phase 0 |

## Phase 3 — Searches profiling (Nautilus first pass)

Stand up `autolens_profiling/searches/` with profiling for the Nautilus
sampler only, mirrored from `autolens_workspace_developer/searches_minimal/`.
Design the folder layout so that other samplers (Dynesty, Emcee, BlackJAX,
NumPyro, NSS, LBFGS, PocoMC) can be slotted in later under their own prompts.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 3 | Mirror Nautilus profiling, design for sampler expansion | `autolens_profiling/searches_nautilus_mirror.md` | Phase 0 |

## Phase 4 — Instrument-framed README dashboard

Build the public-facing READMEs (top-level + per-section) so that anyone
landing on GitHub immediately sees the latest run-times framed by astronomy
instrument (HST, Euclid, JWST, …) rather than just pixel counts. Auto-generate
the tables from the JSON summaries under `results/` so the numbers stay in
sync with whatever Phase 1–3 produced.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 4 | Top-level + per-section README dashboard with instrument framing | `autolens_profiling/instrument_readme_dashboard.md` | Phases 1, 2 |

## Phase 5 — CI / GitHub Actions

Wire up lint / format checks on PR, plus a `workflow_dispatch` workflow that
re-runs the profiling scripts (CPU first; GPU / HPC runners as a follow-up
decision) and commits updated JSON / PNG results + README tables. Frequency,
cost, and whether to use self-hosted runners are decisions for the sub-prompt.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 5 | GitHub Actions for lint + profile re-runs + README refresh | `autolens_profiling/ci_actions.md` | Phase 0 (refinement after Phase 4) |

## Background — original framing

> More and more work on autolens_workspace_developer is going towards profiling, including:
>
> - Setting up realistic science cases for profiling on CPU, different GPUS via JAX (e.g. `jax_profiling`).
> - Doing this across different datasets (e.g. `imaging`, `datacube`).
> - Make graphs of these which are retained over version to some extent to track run time.
>
> We are also beginning to add more and more of a searches aspect (e.g. `searches_minimal`) which track and profile
> run time for different samplers.
>
> I think we should begin to coordinate all this profiling effort in a single github repositroy
> `autolens_profiling` whose scope is:
>
> - To provide likelihood function profiling information for all supported science cases, dataset types and model composition (e.g. mge.py, pixelized, delaunay)
> - To provide this information on CPU, laptop GPU and HPC GPU (e.g. A100).
> - For the information to be easily visible navigating the GitHub pages (e.g. make the most up to date numbers tables and graphs in github README.md.
> - For there to be the step-by-step guides of each likelihood function with all the profiling step times, which autolens_workspace_Developeer already does just making sure you keep that.
> - For sampler profiling tests and their run times and stats to also be contained on this repo, likely in a separate high-level fodler to jax stuff. For the first pass, set up this up but only include Nautilus.
> - Lets keep JAX gradient stuff in autolens_workspace_developer for now, and view this as out of scope but likely to come into the repo in the future. Put a note about JAX gradients on the front page.
> - Lets include the autolens_workspace_developer/jax_profiling/simulators package in all this, and have first-class support and tracking for time simulators take to run, but remember that JAX jitting for these are not fully implemented everywhere so we may have placeholers and then follow that up.
> - GitHub actions and CI in whatever way you think is useful.
>
> Dont add any profiling scripts or whatnot that are missing from _develoeper (e.g. I will probably put a group one in sooner or later but the scope of this task is not to make any test examples).
>
> And thus for this to have good easy to read README.md's through for people to inspect and reprodocue. There should be a focus on
> the profiling examples using simulated datasets with values representative of real science cases (e.g. the imaging pckage already has options for "hst", "euclid"
> so we should make sure this instrument-level profiling is expressed in the information people can rad on the GitHub. Framing it in terms of Astronomy
> instruments and teelscopes, instead of just "number of pixels" or other such metrics is way more intuitive.
>
> Furthermore, I am soon going to write the PyAutoLens-JAX JOSS paper and this will be a great way to substantiate
> my claims about run times.
>
> Do a bit of deep research and thinking about other cool ways we can enhance a repo thats all about profiling and run times!
>
> Obviously this is likely to be a multi stage issue which we can put the high level prompt in z_features.
