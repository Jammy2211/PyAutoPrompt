# Audit autogalaxy_workspace extra_galaxies feature, port modern API where stale

## Background

`feature/scaling-relation-update` (issue
[autolens_workspace#141](https://github.com/PyAutoLabs/autolens_workspace/issues/141)) refreshed
the autolens scaling-relation examples to use the modern API:

- MGE bulge via `al.model_util.mge_model_from(...)` instead of `Sersic` / `SersicSph`
- Centres loaded from `extra_galaxies_centres.json` via `al.from_json` (and an additional
  `scaling_galaxies_centres.json` for the group three-tier example)
- Final model collection structured as
  `af.Collection(galaxies=..., extra_galaxies=..., scaling_galaxies=...)`

A quick directory audit (2026-05-10) shows:

- `autogalaxy_workspace/scripts/imaging/features/extra_galaxies/` — exists, with `modeling.py` +
  `simulator.py` + `README.md`. Uses `Sersic` / `Exponential` light profiles in the modeling
  example header. Need to verify whether the API has been kept in lockstep with the autolens
  version.
- `autogalaxy_workspace/scripts/imaging/features/scaling_relation/` — **does not exist**.

## Goals

1. **Audit** the existing `autogalaxy_workspace` `extra_galaxies` example against the freshly-updated
   `autolens_workspace` version. In particular check:
   - Whether it uses `ag.model_util.mge_model_from` or still uses `Sersic` profiles in the lens light
   - Whether centres are loaded via `ag.from_json` / `Grid2DIrregular` from a `*_centres.json`, or
     hardcoded in Python
   - Whether the final model uses `extra_galaxies=...` as a top-level collection key

   If the autogalaxy version has drifted (older API), port the same modernisation that landed in
   the autolens example.

2. **Decide** whether a `scaling_relation` feature belongs in autogalaxy_workspace at all.

   Autogalaxy is **light-only** — there is no lensing, so the `einstein_radius = scaling_factor *
   luminosity ** scaling_exponent` relation does not transfer directly. A scaling relation could
   still apply to other quantities (e.g. tying the `effective_radius` or `intensity` of extra
   galaxies to a measured stellar-mass proxy), but whether that is a useful tutorial is a judgement
   call — confirm with the user before porting.

   If we decide a scaling_relation example **does** make sense in autogalaxy:
     - Add `scripts/imaging/features/scaling_relation/{__init__.py, simulator.py, modeling.py}` mirroring
       the autolens structure
     - The relation ties some light-profile quantity (e.g. `intensity = scaling_factor *
       luminosity ** scaling_exponent`, or stellar-mass proxy) — settle on what it ties before
       writing
     - Use `ag.lp_linear` / MGE for the light models

   If a light-only scaling-relation tutorial is **not** useful:
     - Document why in the autogalaxy `extra_galaxies/README.md` (one paragraph: "scaling
       relations apply to mass, not light, so the autolens version of this feature is not mirrored
       here")
     - Cross-link to the autolens example for users who arrive looking for it.

## Files likely to change

- `autogalaxy_workspace/scripts/imaging/features/extra_galaxies/modeling.py` (modernise if drifted)
- `autogalaxy_workspace/scripts/imaging/features/extra_galaxies/simulator.py` (only if API drift
  forces simulator changes — usually the simulator is fine)
- `autogalaxy_workspace/scripts/imaging/features/extra_galaxies/README.md` (cross-link or "not
  applicable" note)
- (optional) `autogalaxy_workspace/scripts/imaging/features/scaling_relation/{__init__,simulator,
  modeling}.py` if the answer to goal 2 is "yes, port it"

## Reference reads

- `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` (post issue #141) — the
  modernised pattern to port
- `autolens_workspace/scripts/imaging/features/extra_galaxies/modeling.py` — autolens version of
  the same feature, baseline for the autogalaxy comparison
- `autogalaxy_workspace/scripts/imaging/features/extra_galaxies/modeling.py` — current state in the
  autogalaxy workspace

## Out of scope

- HowToGalaxy tutorial updates (separate repo, separate task)
- Interferometer / multi-wavelength variants — start with imaging only
- CSV-driven loading — covered by the parallel `scaling_relation_csv_loader.md` follow-up
