Phase 0 of the `autolens_profiling` z_feature
(see `z_features/autolens_profiling.md` for the full roadmap).

Create a new GitHub repository `PyAutoLabs/autolens_profiling` and scaffold it
so the later mirror phases (likelihood JIT, simulators, searches) have a clean
place to drop content.

What this task should produce on first commit:

- Empty public repo on GitHub (`PyAutoLabs/autolens_profiling`), description
  line along the lines of "Profiling and run-time tracking for PyAutoLens
  likelihood functions, simulators, and samplers across CPU, laptop GPU, and
  HPC GPU." Match the style of other PyAutoLabs repos.
- Local checkout at `/home/jammy/Code/PyAutoLabs/autolens_profiling/` so
  later /start_library calls can pick it up.
- Top-level `README.md` with:
  - One-paragraph vision / scope statement.
  - Section index linking to the planned `likelihood/`, `simulators/`,
    `searches/`, `results/` directories (some will still be empty stubs at
    this phase — that's fine).
  - A clear "JAX gradients are out of scope for now — see
    `PyAutoLabs/autolens_workspace_developer/jax_profiling/gradient/`" callout.
  - A "Related repos" section linking to `PyAutoLabs/PyAutoLens`,
    `PyAutoLabs/autolens_workspace`, `PyAutoLabs/autolens_workspace_developer`
    (source-of-truth during the migration), and the sibling
    `Jammy2211/autolens_colab_profiling` (Colab-specific scope, not yet
    migrated to PyAutoLabs).
  - A short "How to read this repo" guide that points readers at the
    versioned `results/*_v<version>.{json,png}` pattern that the existing
    `_developer` results folder already uses.
- `LICENSE` matching the other PyAutoLabs repos (almost certainly MIT — check
  PyAutoLens to confirm and copy).
- `.gitignore` mirrored from `autolens_workspace_developer/.gitignore` with
  any obvious additions (e.g. `results/**/*.tmp`, profiler trace files).
- Folder skeleton: `likelihood/`, `simulators/`, `searches/`, `results/`,
  each containing only a placeholder `README.md` saying "populated by Phase
  N of the z_feature".
- Decide: is this a Python *package* (with `pyproject.toml`) or just a
  collection of standalone scripts? Lean toward "scripts only" — the repo's
  job is to surface results, not to be importable. Note that decision in
  the top-level README.
- No code from `_developer` is moved yet — that's Phases 1–3.

Pre-existing context the implementer should look at first:

- `autolens_workspace_developer/jax_profiling/results/` — already uses the
  `*_summary_v2026.X.Y.Z.{json,png}` versioned-artifact pattern. Reuse it.
- `autolens_workspace_developer/CLAUDE.md` — describes the JIT profiling
  conventions (`.array` extraction, `xp` parameter) that the new repo's
  scripts will follow once Phase 1 starts.
- Other PyAutoLabs repo READMEs for tone / structure parity (e.g.
  `PyAutoLabs/PyAutoLens`, `PyAutoLabs/autolens_workspace`).

Out of scope for this task: any actual profiling scripts, any CI, any
result tables beyond placeholders. Those land in Phases 1–5.

When the repo is live and pushed to GitHub, the follow-up `/start_dev`
invocations for Phases 1–3 can begin.
