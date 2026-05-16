Add `af.NSS` to the autofit workspace tutorial dispatch so production users
can adopt the Nested Slice Sampler from day one. Phase 5 of
`z_features/nss_first_class_sampler.md` — the workspace capstone.

Phases 0-4 made `af.NSS(...)` a first-class drop-in for `af.Nautilus(...)` with
single-command install (`pip install autofit[nss]`). Phase 5 surfaces that to
end users via the tutorial they actually copy-paste from.

**Initial scope: `autofit_workspace` only.** `autogalaxy_workspace` and
`autolens_workspace` use `af.Nautilus(...)` throughout their modeling scripts
but don't have a dedicated `scripts/searches/` directory — adopting NSS there
would mean reworking tutorial conventions per workspace, and is better
done as separate follow-ups once we see how users react to the NSS option
in autofit_workspace.

__What exists today__

`@autofit_workspace/scripts/searches/nest.py` is the canonical nested-sampler
tutorial. It walks through DynestyStatic, DynestyDynamic, and Nautilus on a
shared 1D-Gaussian dataset with shared model + analysis. The structure:

- Data load
- Model + Analysis setup (shared)
- Search: DynestyStatic
- Search: DynestyDynamic
- Search: Nautilus
- Search Internal: how to extract the underlying sampler object

Each search section configures the sampler, runs the fit, and shows how to
read the result. The Nautilus section is the natural template for what an
NSS section should look like.

__What to build__

### 1. Add an NSS section to `nest.py`

Insert between the Nautilus section and the "Search Internal" section.
Pattern matches Nautilus's section: brief prose intro, sampler instantiation,
fit, result summary.

Key prose points to hit (Opus-curated, not boilerplate):

- NSS is a JAX-native nested slice sampler — entirely different paradigm
  from boundary-based samplers like Nautilus/Dynesty. The slice walks
  happen inside `jax.jit` so per-evaluation cost on a JAX-traceable
  likelihood is dramatically lower (FINDINGS_v3 measured ~30× faster
  per-eval than Nautilus on the same HST MGE problem).
- Single-command install: `pip install autofit[nss]`. The extra pulls
  the right `handley-lab/blackjax` fork + pinned commit so the
  multi-step install saga from earlier nss-adopters doesn't apply.
- Three new kwargs worth surfacing in the example: `n_live`,
  `num_mcmc_steps`, `num_delete`. Brief one-line explanation of each;
  point users at the docstring for tuning advice.
- Mention `checkpoint_interval` and `iterations_per_quick_update`
  briefly — SLURM-friendly resume + on-the-fly viz, both shipped in
  Phases 2-3.
- Caveat: the per-eval speedup only manifests when the likelihood is
  JAX-traceable. For pure-Python `log_likelihood_function` bodies
  there's no benefit over Nautilus. Most production autolens /
  autogalaxy likelihoods can be JIT-compiled but the 1D Gaussian in
  this tutorial is so trivial that the speedup isn't visible at this
  scale.

Code template (adapt the Nautilus pattern, do NOT copy verbatim — the
docstring should be NSS-specific):

```python
"""
__Search: NSS__

NSS (Nested Slice Sampling, JAX-native) is the production-recommended sampler
when the likelihood can be JIT-compiled — it runs the full inner sampling
loop inside `jax.jit` for dramatic per-evaluation speedups versus boundary
samplers like Nautilus or Dynesty.

Install via `pip install autofit[nss]` — the extra pulls the required
`handley-lab/blackjax` fork at the pinned commit so no multi-step install
recipe is needed.
"""
search = af.NSS(
    path_prefix=path.join("searches"),
    name="NSS",
    n_live=200,           # live particles maintained throughout the run
    num_mcmc_steps=5,     # slice-MCMC inner steps per dead-point batch
    num_delete=50,        # particles removed per outer iteration
    termination=-3.0,     # delta-logZ stopping criterion
    # Phase 2: checkpoint every 100 iterations to allow SLURM resume
    checkpoint_interval=100,
    # Phase 3: visualize current best fit every 10 outer iterations
    # iterations_per_quick_update=10,  # uncomment when analysis.visualize is wired
)
result_nss = search.fit(model=model, analysis=analysis)

print(f"NSS log evidence: {result_nss.samples.log_evidence:.4f}")
print(f"NSS max log L:    {result_nss.samples.max_log_likelihood:.4f}")
```

### 2. Update the file's top-level `__Contents__` block

Add an "NSS" bullet between Nautilus and "Search Internal" in the contents
list. Update the intro prose's "the nested sampling algorithms supported by
**PyAutoFit**" list to include NSS.

### 3. Update `nest.py`'s "Relevant links" section

Add:
```
- NSS (Nested Slice Sampling): https://github.com/yallup/nss
```

### 4. Add a brief NSS mention to `scripts/searches/README.md`

Single-paragraph "When to use NSS" guidance:
- Use NSS if your likelihood is JAX-traceable and you want the per-eval
  speedup
- Use Nautilus / Dynesty if your likelihood is pure-NumPy / has scipy
  fallbacks that don't JIT cleanly
- All three samplers expose the same `result.samples` interface, so
  swapping is one line

__What to verify__

1. **`scripts/searches/nest.py` runs end-to-end** with all three sections
   (DynestyStatic, Nautilus, NSS) executing. The 1D-Gaussian model has 3
   parameters — NSS at `n_live=200` should complete in seconds even on
   CPU. Run via:

   ```bash
   cd autofit_workspace
   python scripts/searches/nest.py
   ```

   Add a check that the NSS section's `result_nss.samples.log_evidence`
   is finite and within ~1 nat of the Nautilus result on the same data
   (different samplers won't agree byte-for-byte, but should land in
   the same neighbourhood).

2. **Smoke pass on autofit_workspace** — the script is in
   `smoke_tests.txt` already (it's `searches/nest.py`), so the standard
   `/smoke_test autofit` invocation must continue to pass. Run after
   adding the NSS section.

3. **`pip install autofit[nss]` precondition** — the tutorial assumes the
   extra is installed. The `nss.py` import in the script would fail
   loudly with the Phase 4 ImportError if not. Document this
   precondition in the file's intro prose ("To run this script with the
   NSS section, `pip install autofit[nss]` first").

__Out of scope__

- `autogalaxy_workspace` and `autolens_workspace` NSS adoption — separate
  follow-up prompts once we see user reactions to the autofit_workspace
  version.
- `HowToFit` chapter additions for NSS — could be a Phase 5b once the
  searches/nest.py addition has been used by a few people.
- SLaM-pipeline NSS dispatch — production pipelines hard-code Nautilus
  via the SLaM functions; that's a deeper refactor than the tutorial
  surface.

__Risks / open questions__

1. **JIT compile time on first NSS run.** The 1D Gaussian is so small
   that the ~25-30s compile dominates the wall time. Users running
   the tutorial may think NSS is slow. Mention this explicitly in the
   prose ("first run includes a 25-30s JIT compile; the per-eval cost
   only beats Nautilus on more expensive likelihoods").

2. **`smoke_tests.txt` impact.** Adding NSS to nest.py means the smoke
   pulls in NSS install on every smoke run. CI envs without `[nss]`
   installed will fail. Either: (a) the smoke env always has [nss]
   installed (likely OK given Phase 4's CI workflow), or (b) wrap the
   NSS section in a try/except that skips if `_HAS_NSS` is False. Pick
   (a) — installing autofit with `[nss]` for smoke matches what users
   should do anyway.

3. **Result-comparison drift.** Asserting NSS and Nautilus log_evidence
   within 1 nat is fine on a 1D Gaussian but may be too tight on other
   tutorials. Keep the assertion local to `searches/nest.py`; don't
   propagate to other tutorial scripts in this PR.

__Reference__

- `@autofit_workspace/scripts/searches/nest.py` — file to extend
- `@autofit_workspace/scripts/searches/README.md` — add the "When to use NSS" paragraph
- `@PyAutoFit/autofit/non_linear/search/nest/nss/search.py` — the NSS class to demonstrate
- `@PyAutoPrompt/z_features/nss_first_class_sampler.md` — Phase 5 in the sequenced roadmap (last remaining)
- `@autolens_workspace_developer/searches_minimal/nss_first_class_gaussian.py` — working af.NSS example from Phases 1-3 (development-side, not user-facing)
