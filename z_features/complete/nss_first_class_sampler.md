# `nss_jit` as a first-class PyAutoFit sampler — Sequenced Roadmap

This z_feature decomposes the multi-phase initiative to make `nss_jit`
(Nested Slice Sampling, JAX-native, yallup/nss) into a production
PyAutoFit `NonLinearSearch` — usable as `af.NSS(...)` alongside
`af.Nautilus(...)`, `af.DynestyStatic(...)`, etc. — so we can run
proper science fits with the per-eval speed advantage we've measured
in `z_projects/profiling/`.

Each phase below points to a concrete sub-prompt file. Phases that
require predecessor work to land first are marked with their
dependency. Far-out phases are intentionally left as stubs.

## Why this matters

`z_projects/profiling/FINDINGS_v3.md` measured the per-eval cost of
five samplers on the HST MGE imaging likelihood (15 free parameters,
RAL HPC A100 80GB):

| Sampler | ms/eval (best config) | Wall to converge |
|---|---|---|
| **`nss_jit`** | **0.52** | **7 min** |
| pocomc | 16.3 | 88 min |
| nautilus | 17.9 | 25 min |
| bjsmc | 32.6 | none converged |
| npess | 412 | 154 min |

`nss_jit` is ~30–800× faster per eval than the others. The 7-min wall
is the lowest in the comparison. Headline of the profiling project:
**`nss_jit` wins on wall time and per-eval cost** because it runs
entirely inside one JAX JIT compile — no Python↔JAX boundary per
likelihood call. That advantage only survives the move into PyAutoFit
if **the prior also runs inside JIT** — see Phase 0 below.

The user-facing goal is:

> "Be able to put `search = af.NSS(name=..., n_live=200, num_mcmc_steps=5,
> num_delete=50)` into any production `autolens_workspace` /
> `autogalaxy_workspace` script and have it work end-to-end —
> checkpointing, on-the-fly visualization, samples.csv / model.results
> / aggregator round-trip — just like `af.Nautilus`."

## Phase 0 — JAX-native priors (prerequisite for everything else)

This is the critical-path dependency. PyAutoFit's `Prior.value_for(unit)`
and `Prior.log_prior_from_value(value)` are currently scipy/NumPy-based
and return Python `float`s. Running the cube → physical mapping inside
`nss_jit`'s JIT'd likelihood today requires `jax.pure_callback` (an 18+
ms/eval host-callback) — that single design choice would erase ~30× of
`nss_jit`'s per-eval advantage and put it back in Nautilus's wall-time
league.

Without this phase the headline numbers disappear; with it,
`nss_jit`'s 0.5 ms/eval survives the integration.

| # | Title | Prompt file | Status |
|---|-------|-------------|--------|
| 0 | JAX-native priors: `value_for` + `log_prior_from_value` with `xp` dispatch on every Prior subclass | `autofit/priors_jax_native.md` | pending |

## Phase 1 — `af.NSS` `NonLinearSearch` wrapper

Once priors are JAX-native, the `af.NSS` class itself is mostly
mechanical — model it on `af.Nautilus` (`@PyAutoFit/autofit/non_linear/search/nest/nautilus/search.py`).
Standard `_fit(model, analysis)`, `samples_from`, output writing
(`samples.csv`, `samples_summary.json`, `samples_info.json`,
`search.json`, `settings.json`, `covariance.csv`), `Result` / `Aggregator`
round-trip.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 1 | `af.NSS` `NonLinearSearch` wrapper around `nss.ns.run_nested_sampling` | `autofit/nss_search_wrapper.md` | Phase 0 |

## Phase 2 — Checkpointing + resumption

Currently `nss.ns.run_nested_sampling` is a one-shot JIT'd
`jax.lax.while_loop` — no way to checkpoint mid-run. Production runs
need to survive SLURM timeouts (the profiling project hit the 4 h
SLURM cap on `pocomc c6_huge` and `npess c1/c2/c6`). PyAutoFit's
`Paths.search_internal/` is the standard checkpoint location.

Approach options (all need an upstream PR to `yallup/nss`):

- **(a)** Add a `max_steps` parameter so we can call `run_nested_sampling`
  in chunks, write `search_internal/state.json` between chunks.
- **(b)** Add a checkpoint callback fired every N inner iterations.
- **(c)** Expose the inner `(particles, weights, log_likelihood,
  log_priors)` state explicitly so we can serialise/deserialise.

**Prompt to author when Phase 1 is in flight.** Likely splits into an
upstream-nss PR prompt + a PyAutoFit wrapper prompt.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 2 | Checkpoint + resume from `search_internal/` between SLURM jobs | _stub — author after Phase 1_ | Phase 1 + upstream `yallup/nss` PR |

## Phase 3 — On-the-fly visualization (`iterations_per_quick_update`)

Nautilus and Dynesty both call `analysis.fit_for_visualization(instance)`
on the current best live point every `iterations_per_quick_update`. For
`nss_jit` this requires:

- The sampler to expose its current best live point periodically.
- A way to interrupt the JIT'd `while_loop` to Python so PyAutoFit can
  pick the best point and call `fit_for_visualization`.

This needs the same upstream nss callback hook as Phase 2 — they can
be authored together.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 3 | On-the-fly visualization hook driven by `iterations_per_quick_update` | _stub — author with Phase 2_ | Phase 2 |

## Phase 4 — Installation simplification

The profiling project's setup spent ~5 hours fighting the nss install
(see `z_projects/profiling/FINDINGS_v3.md` "Methodology saga"). The
problem: nss requires `handley-lab/blackjax` (not mainline PyPI
`blackjax`), and the `--no-deps` dance to avoid pulling
JAX-newer-than-0.4.38 along with optax/chex/distrax/lineax — none of
that is something we can ask a science user to do.

This phase makes `pip install autofit[nss]` or similar a single safe
command for a user installing PyAutoFit fresh.

Approach options (Phase 4 prompt evaluates all four):
- Vendor a minimal `nss.ns` into `autofit.non_linear.search.nss`,
  ditching the `yallup/nss` PyPI dep entirely.
- Coordinate with yallup to publish nss + the blackjax fork as a
  combined PyPI package.
- A `pyautofit[nss]` pip extra with the specific git+ URLs + no-deps
  flags baked in.
- A standalone `python -m autofit.install_nss` helper.

**Vendoring (option 1) is the pragmatic short-term path** — it
eliminates the install pain immediately for science users without
needing yallup's involvement, at the cost of a small maintenance
burden in PyAutoFit. Phase 4 ships option 1 + sketches option 2 as a
long-term direction.

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 4 | Install simplification — vendor nss into autofit, or `pip install autofit[nss]` extra | `autofit/nss_install_simplification.md` | Phase 1 (so we know what API surface to vendor) |

## Phase 5 — Workspace tutorial + dispatch into existing scripts

`autolens_workspace/scripts/imaging/modeling.py`, the SLaM pipelines,
the autogalaxy/autofit equivalents all hard-code `af.Nautilus(...)`.
Phase 5 adds an opt-in `af.NSS(...)` example in
`searches/` and updates the search-cookbook docstrings so production
users can adopt `nss_jit` from day one.

**Prompt to author when Phases 0–4 are merged.**

| # | Title | Prompt file | Depends on |
|---|-------|-------------|------------|
| 5 | `autolens_workspace/searches/nss.py` + autogalaxy + autofit cookbook entries | _stub — author after Phase 4_ | Phases 0–4 |

## Open architectural questions (need their own feasibility prompts)

These came up during the profiling experiments and would benefit from
isolated decisions before being baked into Phase 1's wrapper:

- **Prior mismatch with `nss` sampling space.** nss samples in physical
  parameter space, not unit-cube. Today our profiling scripts use a
  hard-bound box (uniform inside, −∞ outside). Once Phase 0 ships
  JAX-native priors, we could either (a) keep sampling in physical
  with the proper prior log-density, or (b) sample in unit-cube and
  call `prior.value_for(unit, xp=jnp)` inside the likelihood (matches
  the Nautilus and PocoMC pattern). (b) is closer to how the other
  PyAutoFit samplers think; (a) is what nss currently does. The
  Phase 1 prompt has to pick one and justify.

- **NaN gradient in the MGE+inversion likelihood** (from session-1's
  gradient probe `FAIL_NAN_OR_INF`). Phase 1 needs to ship the
  `jnp.where(jnp.isfinite, raw, -1e30)` guard the profiling project
  already uses, or this single pathology will kill HMC-driven follow-
  ups (and silently corrupt `nss_grad`).

- **JIT cache invalidation across runs.** Every fit recompiles
  `nss.ns.run_nested_sampling`'s `while_loop`. ~25–30 s per cold start.
  Production users running batched fits should not pay this each time.
  Investigate `jax.persistent_cache` or a process-level cache directory
  baked into PyAutoFit's `Paths`.

## Background — original framing

From the profiling project's Phase-3 findings (`z_projects/profiling/FINDINGS_v3.md`):

> "`nss_jit` is ~30–800× faster per eval than the others. That's the
> single biggest determinant of wall time, more than algorithmic
> efficiency."

The user-facing question in session-of-2026-05-11:

> "Do you think it is feasible to turn `nss_jit` into a first class
> PyAutoFit sampler so we can do some proper science runs?"

Answer (committed verbatim into this roadmap): yes, ~1–2 weeks of
focused work, **conditional on Phase 0** — without JAX-native priors,
`nss_jit`'s per-eval advantage doesn't survive PyAutoFit's
`Fitness.call` path. Phase 0 is the critical dependency; everything
else is mechanical.
