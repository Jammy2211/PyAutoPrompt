Phase 5 of the `autolens_profiling` z_feature
(see `z_features/autolens_profiling.md` for the full roadmap).

Wire up GitHub Actions for the new `autolens_profiling` repo. There are
two distinct workflows here that should be designed and committed
together but kept in separate `.yml` files.

## Workflow 1 — Lint / format / smoke on PR

Standard "did the contributor break anything" check that runs on every
PR and on push to `main`. Should be cheap (CPU-only, no JAX needed) and
fast (<5 minutes).

Things to check:

- Python syntax + basic linting (`ruff check` is the obvious choice,
  matching other PyAutoLabs repos — confirm by looking at how
  `PyAutoLens/.github/workflows/` is set up and copy the same config).
- `black` / `ruff format --check` formatting parity.
- Markdown link-rot check on all `README.md` files (use `lychee` or
  similar; cheap to run).
- A *smoke* import / dry-run of one script per section to catch obvious
  breakage (the smallest pixelization profile script, the cheapest
  simulator, the smallest Nautilus run with `n_live=10`). Should not
  produce real result artifacts — set an `AUTOLENS_PROFILING_SMOKE=1`
  env var the scripts can read to short-circuit to a tiny problem size.

## Workflow 2 — Re-run profiles + refresh README dashboard

`workflow_dispatch` (manual trigger) and `release`-triggered workflow
that re-runs the real profile scripts, regenerates the JSON / PNG
artifacts under `results/`, runs the `scripts/build_readme.py` from
Phase 4 to refresh the tables in every README, and commits the result
back to `main`.

Open decisions the implementer needs to make:

- **CPU only on GitHub-hosted runners**, *or* **self-hosted runners for
  GPU runs?** GitHub-hosted = free + simple but CPU-only. Self-hosted
  GPU = expensive operationally but produces the laptop-GPU / HPC-GPU
  numbers the front-page table promises. Recommend: start with
  GitHub-hosted CPU-only; structure the workflow so a future
  self-hosted GPU job can append its numbers without touching the
  workflow shape.
- **Cadence**: every release? weekly cron? manual only? Lean toward
  "manual + on release tag" so we don't burn CI minutes on noise.
- **Bot identity**: which GitHub Actions identity commits the
  refreshed README back to `main`? Probably `github-actions[bot]` with
  a `[skip ci]` tag on the commit message to avoid loops.
- **Failure handling**: if one profile script crashes, does the whole
  workflow fail or do we mark that cell `ERR` in the tables and
  continue? Recommend the latter so a single regression doesn't block
  the dashboard refresh.

## Files to produce

- `.github/workflows/lint.yml` (Workflow 1)
- `.github/workflows/profile.yml` (Workflow 2)
- `pyproject.toml` (or `setup.cfg`) holding the `ruff` config that the
  lint workflow references — copy what PyAutoLens uses.
- A short `.github/workflows/README.md` documenting which workflow is
  for what and how to trigger Workflow 2 manually.

## Pre-flight

This phase depends on Phase 0 (repo exists) and benefits from Phase 4
(dashboard build script exists). If Phase 4 hasn't shipped yet,
Workflow 2's "refresh README" step can be a TODO stub that just
commits the new JSONs; flip it on once `build_readme.py` lands.

## Out of scope

- Cross-platform matrix (macOS, Windows) — lens-modeling profiling is
  Linux-only in practice.
- PyPI / conda packaging — the repo isn't a Python package.
- Coverage reporting — there are no unit tests in this repo, by design.
