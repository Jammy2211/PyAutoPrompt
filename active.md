## ci-actions
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/7
- session: claude --resume "ci-actions"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/ci-actions
- repos:
- note: |
    Phase 5 of the autolens_profiling z_feature (tracker:
    PyAutoPrompt/z_features/autolens_profiling.md). Wires up GitHub
    Actions: lint.yml (PR + push, <5min CPU-only) + profile.yml (manual
    + release-triggered profile re-run + dashboard refresh). Also adds
    pyproject.toml with ruff config copied from PyAutoLens.

    Benefits from (but does not block on) Phase 4's
    scripts/build_readme.py — if Phase 4 hasn't shipped yet,
    Workflow 2's "refresh README" step is a TODO stub that just commits
    new JSONs.

    Touches every profile script to add AUTOLENS_PROFILING_SMOKE=1
    short-circuit (~10 scripts: likelihood/*, simulators/*,
    searches/nautilus/*).

    /start_workspace doesn't fit autolens_profiling. Manual worktree:
      source admin_jammy/software/worktree.sh
      worktree_create ci-actions autolens_profiling
      cd ~/Code/PyAutoLabs-wt/ci-actions/autolens_profiling
      git checkout -b feature/ci-actions

## instrument-dashboard
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/6
- session: claude --resume "instrument-dashboard"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/instrument-dashboard
- repos:
- note: |
    Phase 4 of the autolens_profiling z_feature (tracker:
    PyAutoPrompt/z_features/autolens_profiling.md). Builds the public-
    facing READMEs framed by astronomy instrument (HST, Euclid, JWST,
    ALMA, …) — the user's explicitly most-wanted deliverable.

    Adds scripts/build_readme.py that scans results/**/*_summary_v*.json,
    picks the latest per axis, regenerates markdown tables in every
    relevant README between sentinel comments. Idempotent.

    Depends on Phase 1 (shipped) + Phase 2 (in flight, #4). Best to wait
    until Phase 2 ships before starting so the dashboard has results to
    populate; Phase 3 (Nautilus, #5) is independent of the skeleton but
    rounds out section coverage.

    Open implementer decisions to make in scope:
      - CPU/laptop-GPU/HPC-GPU split: 3 cols per cell vs 3 stacked sub-tables
      - Versioning: keep last 6 minor releases? archive older?
      - "Cool extras" the user asked about — pick 1-2 to land
        (regression-watch badge, plotly timeline, flamegraphs).

    /start_workspace doesn't fit autolens_profiling. Manual worktree:
      source admin_jammy/software/worktree.sh
      worktree_create instrument-dashboard autolens_profiling
      cd ~/Code/PyAutoLabs-wt/instrument-dashboard/autolens_profiling
      git checkout -b feature/instrument-dashboard

## nautilus-mirror
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/5
- session: claude --resume "nautilus-mirror"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/nautilus-mirror
- repos:
- note: |
    Phase 3 of the autolens_profiling z_feature (tracker:
    PyAutoPrompt/z_features/autolens_profiling.md). Stands up
    autolens_profiling/searches/ with Nautilus-only profiling (4 files,
    ~20K), mirrored from _developer/searches_minimal/. Designs the folder
    layout so other samplers (Dynesty, Emcee, BlackJAX, NumPyro, PocoMC,
    NSS, LBFGS) can slot in cleanly under their own follow-up prompts.

    Folder layout: searches/{_setup,_metrics}.py + searches/nautilus/{simple,jax}.py
    + 2 READMEs.

    Phase 2 (simulators-mirror, issue #4) is queued but NOT yet started.
    Both touch disjoint dirs (simulators/ vs searches/), so they CAN run
    in parallel if needed. worktree_check_conflict currently treats two
    tasks claiming the same repo as a conflict regardless of file scope,
    so decide at execution time whether to serialise or use two worktrees.

    Apply F1 lesson: copy from worktree's clean origin/main of _developer,
    NOT the canonical.

    Manual worktree setup (autolens_profiling has no /start_workspace fit):
      source admin_jammy/software/worktree.sh
      worktree_create nautilus-mirror autolens_profiling
      cd ~/Code/PyAutoLabs-wt/nautilus-mirror/autolens_profiling
      git checkout -b feature/nautilus-mirror

## simulators-mirror
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/4
- session: claude --resume "simulators-mirror"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/simulators-mirror
- repos:
- note: |
    Phase 2 of the autolens_profiling z_feature (tracker:
    PyAutoPrompt/z_features/autolens_profiling.md). Mirrors 6 simulator
    scripts (~2040 LOC) from _developer/jax_profiling/simulators/ into
    autolens_profiling/simulators/ + writes section README + smokes each.

    Smaller scope than Phase 1 (no input-dataset mirroring decision —
    simulators PRODUCE datasets). Uses Phase 1's locked-in top-level
    `dataset/<type>/<name>/` layout for produced outputs.

    Apply F1 lesson: copy from the worktree's clean origin/main of
    _developer, NOT the canonical (which had ~36 modified files when
    Phase 1 was made and contaminated that mirror).

    /start_workspace does not fit autolens_profiling (no smoke harness,
    no activate.sh, no config/). Manual worktree:
      source admin_jammy/software/worktree.sh
      worktree_create simulators-mirror autolens_profiling
      cd ~/Code/PyAutoLabs-wt/simulators-mirror/autolens_profiling
      git checkout -b feature/simulators-mirror

## jax-phase3-adoption
- session: claude --resume "jax-phase3-adoption"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/jax-phase3-adoption
- repos:
  - autolens_workspace: feature/jax-phase3-adoption
  - autogalaxy_workspace: feature/jax-phase3-adoption
  - autofit_workspace: feature/jax-phase3-adoption
- summary: |
    Phase 3 of z_features/jax_visualization.md (now archived). Sweep
    adoption of `use_jax=True` across ~30 tutorial scripts in the three
    production workspaces, filling gaps in pixelization/MGE/group/
    advanced-feature modeling.py and chaining.py files, plus the
    slam-vs-modeling consistency mismatches surfaced by the 2026-05-16
    audit. Phase 2 baked viz into `use_jax=True`, so a single explicit
    flag is now sufficient — no `use_jax_for_visualization=True` calls
    added.

    Skip list (intentional opt-outs): cpu_fast_modeling.py,
    autogalaxy/ellipse/modeling.py, advanced/expectation_propagation.py +
    hierarchical.py (FactorGraphModel), autofit/searches/mle.py (LBFGS),
    simulators, fit.py/likelihood_function.py demos, aggregator scripts.

    Worktree-safe: no library worktree claimed. group-double-einstein-ring
    (PR #157) already shipped use_jax=True on its rewritten modeling.py,
    so that file dropped from scope; companion chaining.py kept.

## smoke-test-optimization
- issue: https://github.com/rhayes777/PyAutoFit/issues/1183
- session: claude --resume "profile-smoke-test-runtime"
- status: profiling-and-optimization
- location: cli-in-progress
- branch: main
- repos: all on main (previous PRs merged)
- summary: |
    Done: profiled imaging scripts, achieved 80-96% runtime reductions
    Next: investigate cosmology distance calc, profile interferometer scripts

### Completed optimizations
- `PYAUTO_WORKSPACE_SMALL_DATASETS=1` — caps grids/masks to 15x15, forces over_sample_size=2, skips radial bins (PyAutoArray)
- `PYAUTO_DISABLE_JAX=1` — forces use_jax=False in Analysis.__init__ (PyAutoFit)
- `PYAUTO_FAST_PLOTS=1` — skips tight_layout + savefig + critical curve/caustic overlays (PyAutoArray/Galaxy/Lens)
- Skip print_vram_use(), model.info, result_info, pre/post-fit I/O in test_mode >= 2 (PyAutoFit)
- Moved test_mode to autoconf (fixes PyAutoArray CI — no autofit dependency)

### Profiled scripts
- `imaging/simulator.py`: ~100s → 3.6s (96% reduction)
- `interferometer/simulator.py`: ~100s → 4.4s
- `imaging/modeling.py`: ~100s → 19.5s (80% reduction)

### Remaining work — next session
- Investigate cosmology distance calc (176 calls for 2-plane lens) in subplot_fit_imaging
- Investigate repeated ray-tracing in subplot panels
- Profile interferometer/modeling.py and other scripts
- Consider caching cosmology distances per redshift pair

## psf-oversampling
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/299
- session: claude --resume "psf-oversampling"
- status: parked
- parked: 2026-05-06
- classification: library (then workspace follow-up)
- suggested-branch: feature/psf-oversampling
- summary: |
    Parked — no resources claimed. Task worktree was created during /start_library
    but removed without edits; local feature/psf-oversampling branches deleted
    from PyAutoArray and PyAutoGalaxy. Both repos are free for other tasks.

    Affected repos (when resumed):
      - PyAutoArray (library, primary)
      - PyAutoGalaxy (library)
      - autolens_workspace_test (workspace follow-up)
      - autolens_workspace (workspace follow-up)

    To resume: run /start_library — it will recreate the worktree and the
    feature/psf-oversampling branches off origin/main. Then start with
    Phase 1 (over_sample_util helpers) per the agreed phasing below.

    Phasing (smaller tasks, agreed mid-session):
      1. over_sample_util: Mask2D upscale-by-N + fine->native sum-reduce helpers + tests
      2. Convolver: add convolve_over_sample_size kwarg (default 1, no behaviour change) + test
      3. Convolver: bin-down branch in all four conv paths, gated > 1 + brute-force test
      4. Imaging dataset: kwargs + 2 construction-time guards (adaptive over-sample, sparse)
      5. GridsDataset: expose oversampled grids when > 1
      6. OperateImage + FitImaging caller threading (PyAutoGalaxy)
      7. Inversion mapping audit + assertion (mapping.py / abstract.py)
      8. End-to-end library integration test
      (workspace) extend convolution.py + new convolution_oversampled.py + simulator.py

## knn-barycentric
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/317
- session: claude --resume "knn-barycentric"
- status: workspace-dev
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/318
- worktree: ~/Code/PyAutoLabs-wt/knn-barycentric
- repos:
  - PyAutoArray: feature/knn-barycentric
  - autolens_workspace_developer: feature/knn-barycentric
  - autolens_workspace_test: feature/knn-barycentric

