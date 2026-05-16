Phase 1 of the `autolens_profiling` z_feature
(see `z_features/autolens_profiling.md` for the full roadmap).

Mirror the JIT-profiling likelihood scripts currently in
`autolens_workspace_developer/jax_profiling/jit/` into the new
`autolens_profiling/likelihood/` directory. `_developer` remains the
source of truth — do not delete the originals.

Scripts to mirror (one-to-one, preserving file names):

- `jit/imaging/{mge,pixelization,delaunay}.py`
- `jit/interferometer/{mge,pixelization,delaunay}.py`
- `jit/point_source/{image_plane,source_plane}.py`
- `jit/datacube/delaunay.py` (+ its `__init__.py`)

Adjustments per-script:

- Update any path references that assume the `_developer` workspace layout
  (e.g. `Path("jax_profiling") / "imaging" / "dataset" / ...`) to the new
  repo's layout (e.g. `Path("likelihood") / "imaging" / "dataset" / ...`).
- Make sure `results/` writes use the same versioned-artifact naming the
  `_developer` repo already uses (`*_summary_v<version>.{json,png}`) so the
  Phase 4 README dashboard can pick them up.
- Keep the printed step-by-step profile narrative each script already
  produces — it is the user-facing documentation. The user has been
  explicit: "make sure you keep that".

Per-section READMEs the implementer should author at the same time:

- `likelihood/README.md` — overview of the section, what "JIT likelihood
  profiling" means, how to read the per-script output, and how the
  versioned result artifacts work.
- `likelihood/imaging/README.md`, `likelihood/interferometer/README.md`,
  `likelihood/point_source/README.md`, `likelihood/datacube/README.md` —
  each with a short narrative on what the profiled likelihood path looks
  like and a table of the most-recent run-times for the scripts in that
  folder. Table population can be a placeholder filled in by Phase 4's
  auto-generation, but the section structure should already be there.

Dataset handling: `_developer/jax_profiling/dataset/` provides the input
datasets these scripts depend on. Decide whether to mirror those files
into `likelihood/dataset/`, symlink, or document a one-time `make
dataset` step. Lean toward mirroring the small simulated datasets so the
new repo is fully self-contained — but check sizes first before
committing binary blobs.

Smoke check before declaring done: run one script from each subfolder
end-to-end from the new repo (`cd autolens_profiling && python
likelihood/imaging/mge.py` etc.) and confirm the JSON / PNG summaries
land under `autolens_profiling/results/`.

Out of scope: anything from `_developer/jax_profiling/gradient/` (still
gated on the JAX gradient story stabilising) or `_developer/jax_profiling/misc/`
(exploratory work, not part of the stable profiling surface).
