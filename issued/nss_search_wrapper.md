Add `af.NSS` — a first-class `NonLinearSearch` for Nested Slice
Sampling (JAX-native, yallup/nss) — so users can drop
`search = af.NSS(...)` into any production autolens/autogalaxy/autofit
script in place of `af.Nautilus(...)`.

This is **Phase 1** of `z_features/nss_first_class_sampler.md`.
**Blocked on Phase 0** (`autofit/priors_jax_native.md`) — without
JAX-native priors the cube → physical mapping has to use
`jax.pure_callback`, costing ~18 ms/eval and erasing the 30× per-eval
advantage that motivates this whole feature (see
`z_projects/profiling/FINDINGS_v3.md`).

__Why this matters__

The user-facing goal from the z_feature:

> "Be able to put `search = af.NSS(name=..., n_live=200,
> num_mcmc_steps=5, num_delete=50)` into any production
> `autolens_workspace` / `autogalaxy_workspace` script and have it
> work end-to-end — checkpointing, on-the-fly visualization,
> samples.csv / model.results / aggregator round-trip — just like
> `af.Nautilus`."

`z_projects/profiling/` already proved the numerical headline
(`nss_jit c3_big_delete`: 7 min wall, 0.57 ms/eval, recovered
`einstein_radius=1.5996` consistently across 4 of 6 sweep configs).
This prompt turns the bare `nss.ns.run_nested_sampling` call into a
production sampler.

__Reference implementation pattern__

Model the new class on `af.Nautilus`:

- `@PyAutoFit/autofit/non_linear/search/nest/nautilus/search.py` —
  class structure, `__init__` kwargs, `_fit` method, paths integration.
- `@PyAutoFit/autofit/non_linear/search/nest/nautilus/samples.py` —
  posterior extraction, log_Z, ESS.
- `@PyAutoFit/autofit/non_linear/search/nest/abstract.py` — the
  abstract `NonLinearSearch` interface every sampler inherits from.

The working bare-bones reference for the `nss_jit` numerical pipeline:
`@z_projects/profiling/scripts/nss_jit_one_config.py`. Lift the
`build_dataset`, model setup, log_likelihood, log_prior, and
`run_nested_sampling` call structure verbatim — those are already
validated on the HPC.

__What to build__

New directory:
`@PyAutoFit/autofit/non_linear/search/nest/nss/` with `__init__.py`,
`search.py`, `samples.py`, mirroring the Nautilus layout.

### `af.NSS` `__init__` signature (mirror Nautilus, prefer parameter names from yallup/nss)

```python
class NSS(AbstractNest):
    def __init__(
        self,
        name: str = "",
        path_prefix: str = "",
        unique_tag: str | None = None,
        # nss-specific knobs (production defaults from FINDINGS_v3 c3)
        n_live: int = 200,
        num_mcmc_steps: int = 5,
        num_delete: int = 50,
        termination: float = -3.0,
        # standard nest knobs
        iterations_per_quick_update: int = 10_000,
        number_of_cores: int = 1,           # nss is JAX; this is here for API parity, ignored
        seed: int = 42,
        session=None,                       # SQL session
        **kwargs,
    ):
```

`number_of_cores` is API parity only — `nss_jit` runs on whatever
device JAX is configured for. Document that explicitly in the
docstring.

### `_fit(model, analysis)` body

Outline:

1. Build the JAX-traceable `log_likelihood(unit_vector)` closure:
   `instance = model.instance_from_vector(model.vector_from_unit_vector(unit_vector, xp=jnp), xp=jnp)`
   then call `analysis.log_likelihood_function(instance=instance)`.
   Wrap with the NaN guard
   `jnp.where(jnp.isfinite(raw), raw, -1e30)`.

2. Build the JAX-traceable `log_prior(unit_vector)` closure: sum of
   `model.log_prior_list_from_vector(physical, xp=jnp)` mapped from
   the unit cube. **Phase 0 makes this possible.** Today's
   `@z_projects/profiling/scripts/nss_jit_one_config.py` uses a
   hard-bound box — replace that with the proper log-prior once
   Phase 0 lands.

3. Initial samples: `n_live` unit-cube draws mapped to physical via
   `model.vector_from_unit_vector(u, xp=jnp)`. Seed from
   `jax.random.PRNGKey(self.seed)`.

4. Call `nss.ns.run_nested_sampling(rng_key, loglikelihood_fn=...,
   prior_logprob=..., num_mcmc_steps=..., initial_samples=...,
   num_delete=..., termination=...)`.

5. After return, build an `NSSamples` object (new class in
   `samples.py`) from `final_state.particles` + `results.logZs`.

6. Write the standard PyAutoFit output schema (see "Output writing"
   below).

### Output writing

Match the file set Nautilus produces in `paths.samples_path`:

- `samples.csv` — every accepted sample's `(parameter_0, ...,
  parameter_N, log_likelihood, log_prior, weight)`. For nss the
  weights come from `logZs` increments per dead-point.
- `samples_summary.json` — max log L, log Z + error, max ESS, best fit.
- `samples_info.json` — sampler metadata (n_live, num_mcmc_steps,
  num_delete, termination, evals, sampling_time).
- `search.json` — sampler config (n_live, etc.) serialised so
  `Aggregator` can reproduce.
- `settings.json` — `iterations_per_quick_update`, seed, etc.
- `covariance.csv` — posterior covariance over the dead points.

Most of this is mechanical and the Nautilus implementation in
`@PyAutoFit/autofit/non_linear/search/nest/nautilus/samples.py:save_results`
is the template.

### `NSSamples` class

Inherit from `SamplesNest` (or whatever the existing nested-sampler
samples base class is). Override:

- `parameter_lists` — list of physical parameter vectors per dead point.
- `log_likelihood_list`, `weight_list`, `log_prior_list`.
- `log_evidence` — `results.logZs.mean()`.
- `log_evidence_err` — `results.logZs.std()` (nss returns an array
  of logZ estimates across the live ensemble).
- `total_samples` — `results.evals`.
- `total_accepted_samples` — `len(final_state.particles)`.

### Resumption + visualization

**Stub** — these are Phases 2 and 3. The Phase 1 implementation
should:

- Accept `iterations_per_quick_update` in `__init__` but log a
  warning that it's not yet wired up.
- Detect `paths.search_internal/state.json` existing and warn that
  resume isn't supported yet (then proceed with a fresh fit).

The Phase 2 prompt (TBD, depends on an upstream nss PR) wires both
properly.

### Plotting hooks

`AbstractNest` has standard `plot_results` methods. nss's posterior
is `final_state.particles` + per-point log-likelihood, mappable to
the `corner.py` interface the existing nested-sampling plot machinery
uses. Lift the Nautilus `plot_results` body verbatim if the API
matches.

__What to verify__

1. **Numerical parity smoke.** Run the `searches_minimal/nss_jit.py`
   case (or `z_projects/profiling/scripts/nss_jit_one_config.py`
   c3_big_delete) through the new `af.NSS(...)` and confirm
   `einstein_radius=1.5996 ± 0.0002`, max log L within 1 nat of
   the bare-script value (~31786). Wall time within 20% of the bare
   reference.

2. **Unit tests.** Test files under
   `@PyAutoFit/test_autofit/non_linear/search/nest/nss/`. Cover:
   - `NSS.__init__` accepts the documented kwargs and rejects unknown
     ones cleanly.
   - `NSS.fit` returns a `Result` with `result.samples` non-empty.
   - `samples_from(paths)` round-trip — write → load → re-extract
     max log L identically.
   - `Aggregator` round-trip: serialise a fit, load via `af.Aggregator`,
     check `agg.values("samples").log_evidence` returns the same number.

3. **`autolens_workspace_developer` smoke.** Add an opt-in script
   `searches_minimal/nss_first_class.py` that uses
   `search = af.NSS(...)` end-to-end on the HST MGE dataset. Compare
   the resulting `result.max_log_likelihood_instance` to the
   nautilus_jax baseline — should agree to <1 nat.

4. **Existing samplers unaffected.** Run `/smoke_test` on each PyAuto
   workspace_test. The change is additive (new module under
   `non_linear/search/nest/nss/`) so should not regress anything;
   verify anyway.

5. **`import autofit` does not require `nss`.** Phase 4 will ship a
   `pyautofit[nss]` extra; the Phase 1 implementation must guard the
   `nss` import so `import autofit` keeps working when nss is not
   installed:

   ```python
   try:
       from nss.ns import run_nested_sampling
       _HAS_NSS = True
   except ImportError:
       _HAS_NSS = False

   class NSS(AbstractNest):
       def __init__(self, ...):
           if not _HAS_NSS:
               raise ImportError(
                   "af.NSS requires nss. Install via 'pip install autofit[nss]' "
                   "or follow PyAutoPrompt/autofit/nss_install_simplification.md."
               )
           ...
   ```

__Out of scope__

- Resumption / checkpointing — Phase 2.
- `iterations_per_quick_update` visualization — Phase 3.
- Install simplification — Phase 4 (`autofit/nss_install_simplification.md`).
- Workspace tutorial scripts — Phase 5.
- HMC variant (`nss_grad`) — gated on the upstream NaN gradient fix
  from session-1's `probe_grad.py`.

__Risks / open questions__

1. **Sampling space.** nss currently samples in physical parameter
   space (initial samples are `model.vector_from_unit_vector(u)`
   per particle). The Nautilus and PocoMC convention is to sample in
   unit-cube space and apply the prior transform inside the
   likelihood. There's an argument either way:
   - Physical-space sampling (status quo): the slice walks have the
     natural metric of the problem, no remapping cost per step.
   - Unit-cube sampling: matches Nautilus/PocoMC mental model,
     simplifies `log_prior` to `jnp.where(in_unit_cube, 0, -inf)`.

   **Decision needed before writing the wrapper.** Recommendation:
   keep physical-space sampling (it's what nss is designed for) but
   document the convention in the `af.NSS` docstring so users know
   why the `model.priors_ordered_by_id` log-priors are summed inside
   the likelihood.

2. **`logZs` interpretation.** nss returns `results.logZs` as an
   array of log-evidence estimates per dead-point batch (the
   stochastic batch error of the NS estimator). PyAutoFit's
   `Samples.log_evidence` expects a single number. Use
   `logZs.mean()` and expose `logZs.std()` as
   `log_evidence_error`. Verify this matches how Nautilus reports
   it.

3. **JIT cache.** Every fit recompiles
   `run_nested_sampling`'s `while_loop` (~25–30 s cold). Production
   users running batched fits should not pay this each time —
   investigate `jax.persistent_cache` directory baked into PyAutoFit's
   `Paths` (probably `paths.search_internal/jax_cache/`). Worth a
   one-line `jax.config.update(...)` call in `_fit`'s setup; verify
   the cache directory survives across `python` invocations.

__Reference__

- `@PyAutoFit/autofit/non_linear/search/nest/nautilus/search.py` —
  reference structure
- `@PyAutoFit/autofit/non_linear/search/nest/abstract.py` — abstract
  base
- `@z_projects/profiling/scripts/nss_jit_one_config.py` — validated
  bare-bones reference for the JAX likelihood + nss call
- `@z_projects/profiling/FINDINGS_v3.md` — wall-time and ms/eval
  numbers
- `@PyAutoPrompt/z_features/nss_first_class_sampler.md` — Phase 1 in
  the sequenced roadmap
- `@PyAutoPrompt/autofit/priors_jax_native.md` — Phase 0 blocker
