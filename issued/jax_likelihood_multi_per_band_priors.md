# Upgrade `jax_likelihood_functions/multi/` to per-band `ell_comps` priors (option B)

## Goal

The scripts in `autolens_workspace_test/scripts/jax_likelihood_functions/multi/`
currently use **option A** — the same `model` instance is passed to every
`af.AnalysisFactor`, so *all* model parameters are shared across both g and r
bands. This was chosen for the first parity sweep because it mirrors the
existing `multi/mge.py` wiring and keeps the test minimal.

The production tutorial at
`@autolens_workspace/scripts/multi/modeling.py` uses a richer pattern
(**option B**) where most parameters are shared but a small set of
wavelength-dependent parameters gets **per-band priors via `model.copy()` +
`af.GaussianPrior`**. In that example the source `bulge.ell_comps` are allowed
to vary per band (to capture chromatic shape differences), while the lens mass,
lens bulge, and all other source parameters remain shared.

This prompt is the option B upgrade: retrofit the seven existing multi JAX
scripts (`lp.py`, `mge.py`, `mge_group.py`, `rectangular.py`, `delaunay.py`,
`rectangular_mge.py`, `delaunay_mge.py`) so each band has its own
`ell_comps` priors for the source light (parametric/MGE scripts) or the
source-light/adapt-image equivalent for pixelized scripts.

## Reference pattern

From `autolens_workspace/scripts/multi/modeling.py`:

```python
analysis_factor_list = []

for dataset, analysis in zip(dataset_list, analysis_list):
    model_analysis = model.copy()
    model_analysis.galaxies.source.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
        mean=0.0, sigma=0.5, lower_limit=-1.0, upper_limit=1.0
    )
    model_analysis.galaxies.source.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
        mean=0.0, sigma=0.5, lower_limit=-1.0, upper_limit=1.0
    )
    analysis_factor_list.append(
        af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)
    )

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)
```

`model.copy()` returns an independent deep copy so per-factor prior edits do
not leak back into the shared template.

## Acceptance

For each of the seven scripts:

1. Replace the option-A factor construction (same `model` reused) with the
   option-B pattern above — `model.copy()` per factor, then override the
   appropriate per-band priors.
2. Re-run the script and capture the new `EXPECTED_VMAP_LOG_LIKELIHOOD`
   constant. The value WILL change (more free parameters → different
   prior-median vector → different log-likelihood).
3. Update the module docstring to note that the script uses option B and
   which parameters are per-band.
4. Keep everything else — auto-simulation, Path A, vmap + JIT parity — as-is.

## Scripts and which parameters to free per band

- `lp.py` — parametric Sersic source: free `source.bulge.ell_comps_0/1` per band.
- `mge.py` — MGE source: free the `ell_comps` that ties all source Gaussians
  together (the MGE `mge_model_from` helper keeps `ell_comps` as a single
  shared parameter pair per basis; free those per band).
- `mge_group.py` — MGE source + extra galaxies: free the source `ell_comps`
  shared parameter per band; leave the extra-galaxy parameters fully shared.
- `rectangular.py`, `delaunay.py`, `rectangular_mge.py`, `delaunay_mge.py` —
  pixelized sources have no `ell_comps`. Per band, free the
  **regularization coefficient** (`regularization.inner_coefficient` or
  `regularization.coefficient`, depending on the regularization class) so
  each band gets its own regularization strength. This is the closest
  pixelized analogue to "per-band source shape."

## Known library issue (do not try to fix in this task)

For **pixelized sources only**, `AnalysisImaging.log_likelihood_function(instance)`
returns different values under `use_jax=True` vs `use_jax=False` — the JAX
path only matches NumPy when routed through `fit_from(instance).log_likelihood`.
`FactorGraphModel` has no `fit_from` method, so the multi scripts assert
`vmap == JIT round-trip` rather than `NumPy == JIT`. Preserve that
pattern in the upgrade — do not try to add a NumPy reference assertion for
pixelized scripts.

The user knows the root cause of the rectangular-pixelization mismatch and
plans to fix it separately; do not investigate or patch it as part of this
task.

## Deliverables

- Seven edited scripts under
  `autolens_workspace_test/scripts/jax_likelihood_functions/multi/`.
- Seven fresh `EXPECTED_VMAP_LOG_LIKELIHOOD` values captured on first run.
- Updated `scripts/CLAUDE.md` table entries noting "(per-band `ell_comps`)"
  for rows that were pure option A before.
- PR title suggestion: `feat: per-band ell_comps priors in multi jax_likelihood scripts (option B)`.
- Branch suggestion: `feature/jax-likelihood-multi-per-band-priors`.

## Out of scope

- Library code changes.
- Fixing the pixelized NumPy/JAX parity divergence.
- Adding new scripts (the set of 7 is fixed).
- DSPL (double-source-plane) variants.
