Sub-task 2 of 3 following up on https://github.com/PyAutoLabs/autolens_workspace/issues/78.

The howtolens tutorial series has been extracted into its own repo at `PyAutoLabs/HowToLens`
(bootstrap PR: https://github.com/PyAutoLabs/HowToLens/pull/1, shipped 2026-04-21).

This sub-task removes the howtolens content from `PyAutoLabs/autolens_workspace` and updates
all internal cross-references that still point at `scripts/howtolens/` or `notebooks/howtolens/`.

## In scope

- Delete `scripts/howtolens/` and `notebooks/howtolens/` (wholesale).
- Fix the three `scripts/guides/` callers that `subprocess.run` into
  `scripts/howtolens/simulator/lens_sersic.py` — they'll break when howtolens goes:
    - `scripts/guides/data_structures.py:247`
    - `scripts/guides/plot/examples/plotters.py:70`
    - `scripts/guides/plot/advanced/plotters_pixelization.py:59`
  Pick one of: (a) copy the simulator out of howtolens into
  `scripts/imaging/simulators/` (or equivalent) and update the subprocess paths;
  (b) switch the guides to an already-shipped dataset like `simple` or
  `simple__no_lens_light` (simpler if feature parity allows).
- Slim the HowToLens sections in:
    - `autolens_workspace/README.rst` (~L12, L41-57, L79)
    - `autolens_workspace/start_here.py` (~L376-391) — regenerate matching `.ipynb`.
  Replace the embedded content with a short pointer paragraph + a link to
  `https://github.com/PyAutoLabs/HowToLens`.
- Update the `howtolens`-referencing prose in other workspace scripts/notebooks that
  the survey flagged (cross-refs in `scripts/imaging/modeling.py`,
  `scripts/interferometer/modeling.py`, `scripts/group/modeling.py`, etc.) to point at
  the new repo URL rather than `autolens_workspace/*/howtolens`.
- Remove the howtolens-keyed entries from `config/build/env_vars.yaml` and
  `config/build/no_run.yaml` (they have no targets to apply to anymore).

## Out of scope — sub-task 3

Changes to `PyAutoLens/docs/` (index.rst toctree, `howtolens/*.rst`, `overview/*.rst`,
`general/workspace.rst`, `README.rst`, `paper.md`). These live in the PyAutoLens library
repo, not the workspace — deferred to sub-task 3 (howtolens-docs-update).

## Validation

- Smoke list: after the edits, run `/smoke-test` on `autolens_workspace` (its existing
  `smoke_tests.txt` — no howtolens scripts are currently listed there, so no list edits
  expected, but confirm).
- Manually run the three `scripts/guides/` callers to confirm the subprocess fix works.

## Follow-up

Sub-task 3 (docs-update) will replace the PyAutoLens-side docs that still describe
howtolens as an `autolens_workspace/howtolens/` subtree. HowToGalaxy + HowToFit get the
same three-step treatment once HowToLens is fully bedded in.
