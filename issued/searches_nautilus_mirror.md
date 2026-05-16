Phase 3 of the `autolens_profiling` z_feature
(see `z_features/autolens_profiling.md` for the full roadmap).

Stand up `autolens_profiling/searches/` with profiling for the Nautilus
sampler only, mirrored from
`autolens_workspace_developer/searches_minimal/`. `_developer` stays the
source of truth — do not delete originals.

Scripts to mirror (Nautilus-only first pass):

- `searches_minimal/nautilus_simple.py` → `searches/nautilus/simple.py`
- `searches_minimal/nautilus_jax.py` → `searches/nautilus/jax.py`
- `searches_minimal/_setup.py` → `searches/_setup.py` (shared)
- `searches_minimal/_metrics.py` → `searches/_metrics.py` (shared)

Other `searches_minimal/*` scripts (Dynesty, Emcee, BlackJAX-NUTS,
BlackJAX-SMC, NumPyro-ESS, PocoMC, NSS variants, LBFGS, sweep_*.py,
probe_grad.py) are explicitly out of scope for Phase 3 — design the
folder layout so they can slot in cleanly later under their own prompts.

Folder layout the implementer should commit:

```
searches/
    README.md
    _setup.py
    _metrics.py
    nautilus/
        README.md
        simple.py
        jax.py
    # future: dynesty/, emcee/, blackjax/, numpyro/, pocomc/, nss/, lbfgs/
```

Adjustments per-script:

- Update `_developer`-relative paths to the new layout.
- Confirm `_setup.py` / `_metrics.py` are import-clean from their new
  location and that the Nautilus scripts pick them up via relative import.
- Make sure run-time + sampler-stat JSON writes go to
  `autolens_profiling/results/searches/nautilus/` using the same versioned
  artifact pattern the rest of the repo uses.

Section README to author:

- `searches/README.md` — overview of the sampler-profiling section, what
  the shared `_setup.py` / `_metrics.py` provide, and a "supported
  samplers" table that today has only Nautilus filled in and explicit
  "planned" rows for the rest.
- `searches/nautilus/README.md` — what's profiled (chain time, n_live
  sensitivity, JIT vs NumPy if applicable), how to read the result JSON,
  and a table of the latest run-times that Phase 4 will keep refreshed.

Reference material: `_developer/searches_minimal/sweep_findings.md`
captures the user's prior observations from sweep runs — read it before
authoring the README so the narrative aligns with the existing
findings. Do not mirror `sweep_findings.md` itself; that's a workspace
note, not a public artifact.

Smoke check: run `python searches/nautilus/simple.py` end-to-end from
the new repo with a small problem size and confirm the result JSON
lands in the right place.

Out of scope: other samplers, sweep scripts, gradient probes.
