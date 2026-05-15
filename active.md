## url-check
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/508
- session: claude --resume "url-check"
- status: library-merged, workspace-pending
- worktree: (library worktree removed; new worktree to be created by /start_workspace)
- library-prs-merged:
  - admin_jammy: https://github.com/Jammy2211/admin_jammy/pull/21
  - PyAutoConf:  https://github.com/PyAutoLabs/PyAutoConf/pull/105
  - PyAutoFit:   https://github.com/PyAutoLabs/PyAutoFit/pull/1265
  - PyAutoArray: https://github.com/PyAutoLabs/PyAutoArray/pull/309
  - PyAutoGalaxy: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/413
  - PyAutoLens:  https://github.com/PyAutoLabs/PyAutoLens/pull/509
- repos: (pending — workspace phase will claim HowToFit, HowToGalaxy, HowToLens,
    autofit_workspace, autogalaxy_workspace, autolens_workspace)
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

## visualize-combined-quick-update-kwarg
- issue: none — followup to priors-jax-native (#1262); deferred Bug B at #1266
- session: claude --resume "visualize-combined-quick-update-kwarg"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/visualize-combined-quick-update-kwarg
- repos:
  - PyAutoFit: feature/visualize-combined-quick-update-kwarg
  - PyAutoGalaxy: feature/visualize-combined-quick-update-kwarg
- summary: |
    Fix the missing `quick_update` kwarg on `VisualizerExample.visualize_combined`
    (PyAutoFit/autofit/example/visualize.py:177) and `VisualizerImaging.visualize_combined`
    (PyAutoGalaxy/autogalaxy/imaging/model/visualizer.py:144). Both inherit the
    `quick_update` plumbing added in PyAutoFit commit a1e360567 (5 May 2026) but
    didn't get updated. Today any graphical fit using these Analysis classes
    inside a FactorGraphModel crashes with TypeError on the first quick-update
    iteration. Pure additive — kwarg defaults to False, body unchanged.

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
