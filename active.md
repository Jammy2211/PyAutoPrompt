## disable-model-graph
- issue: none — ad-hoc cleanup
- session: claude --resume "disable-model-graph"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/disable-model-graph
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1264
- repos:
  - PyAutoFit: feature/disable-model-graph
  - autofit_workspace: feature/disable-model-graph
  - autogalaxy_workspace: feature/disable-model-graph
  - autolens_workspace: feature/disable-model-graph
  - autolens_workspace_test: feature/disable-model-graph
- summary: |
    Gate `model.graph` output in fit folders behind a new
    `output.model_graph` config key (default `false`). Fixes the bug
    where an empty `model.graph` was written for every non-graphical
    fit because the file was opened before `model.graph_info` raised
    AttributeError. Metadata file left alone — it's the sentinel
    Aggregator.from_directory() uses to detect search output dirs.

    Workspace follow-up: each workspace ships its own config/output.yaml
    with `default: true` and no `model_graph` key, so workspace users
    still see the file written. Add `model_graph: false` to:
      - autofit_workspace/config/output.yaml
      - autogalaxy_workspace/config/output.yaml
      - autolens_workspace/config/output.yaml
      - autolens_workspace_test/config/output.yaml

## analysis-ellipse-jax
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/411
- session: claude --resume "analysis-ellipse-jax"
- status: library-shipped, workspace-pending
- worktree: ~/Code/PyAutoLabs-wt/analysis-ellipse-jax
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/412
- repos:
  - PyAutoGalaxy: feature/analysis-ellipse-jax
- conflict-override: |
    url-check also holds PyAutoGalaxy via feature/url-check, but it's doc-only
    (URL string fixups across all PyAuto repos). This task touches ellipse/
    source files (model/analysis.py, fit_ellipse.py, analysis/jax_pytrees.py)
    — disjoint diffs. Parallel work explicitly cleared by user 2026-05-14.

## url-check
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/508
- session: claude --resume "url-check"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/url-check
- repos:
  - admin_jammy: feature/url-check
  - PyAutoConf: feature/url-check
  - PyAutoFit: feature/url-check
  - PyAutoArray: feature/url-check
  - PyAutoGalaxy: feature/url-check
  - PyAutoLens: feature/url-check
- summary: |
    Cross-repo doc URL audit and cleanup. Build a Python URL-checker in
    admin_jammy/software/url_check/ that scans every PyAuto repo, dedupes
    URLs, and validates them (HEAD/GET, with Colab URLs verified via the
    underlying raw.githubusercontent.com mapping). Then fix the broken
    URLs surfaced — known patterns: hhttps typo (2 sites), Jammy2211/
    workspace → PyAutoLabs/ rename (~113 sites), /blob/release/ → /blob/
    main/ — plus any case-by-case dead links the audit flags.

    Affected repos: admin_jammy (tool), PyAutoConf, PyAutoFit, PyAutoArray,
    PyAutoGalaxy, PyAutoLens, HowToFit, HowToGalaxy, HowToLens,
    autofit_workspace, autogalaxy_workspace, autolens_workspace.

    Conflict override: PyAutoFit and PyAutoGalaxy are claimed by
    priors-jax-native and fit-ellipse-jax respectively; user cleared
    parallel work since this task is doc-only and the others touch source.

## priors-jax-native
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1262
- session: claude --resume "priors-jax-native"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/priors-jax-native
- repos:
  - PyAutoFit: feature/priors-jax-native
  - autofit_workspace_test: feature/priors-jax-native

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
