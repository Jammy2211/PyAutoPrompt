## likelihood-jit-mirror
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/1
- session: claude --resume "likelihood-jit-mirror"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/likelihood-jit-mirror
- repos:
- note: |
    Phase 1 of the autolens_profiling z_feature. Mirrors
    `_developer/jax_profiling/jit/` (10 likelihood scripts + tracked
    input datasets ~900K) into `autolens_profiling/likelihood/` and
    `autolens_profiling/dataset/`, plus writes 5 READMEs (section +
    4 per-data-type). `_developer` stays source of truth — no deletes.

    Dataset folder decision locked in: top-level `dataset/` (shared
    with Phase 2 simulators and Phase 3 searches), NOT `likelihood/dataset/`.

    This is the new `autolens_profiling` repo's first feature branch +
    PR — Phase 0 (issue #513 on PyAutoLens, now closed) committed
    direct-to-main since main didn't exist yet. Phase 1 uses the
    standard worktree + feature branch + PR flow.

    /start_workspace does NOT fit this repo cleanly — there's no
    `smoke_tests.txt`, no `activate.sh`, no `config/`. Set up the
    worktree manually:
      source admin_jammy/software/worktree.sh
      worktree_create likelihood-jit-mirror autolens_profiling
    Then create feature/likelihood-jit-mirror inside the worktree and
    work from there. Smoke step = manual per-subfolder script runs
    (see issue body step 8).

## nss-tutorial-dispatch
- issue: https://github.com/PyAutoLabs/autofit_workspace/issues/59
- session: claude --resume "nss-tutorial-dispatch"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/nss-tutorial-dispatch
- repos:
- summary: |
    Phase 5 of nss_first_class_sampler — workspace capstone. Add an
    "Search: NSS" section to autofit_workspace/scripts/searches/nest.py so
    end users discover af.NSS from the canonical nested-sampler tutorial.
    Scope is intentionally autofit_workspace only; autogalaxy / autolens
    follow-ups come once we see user reactions.

    Tutorial prose stays on Opus per feedback_tutorial_prose_opus.md.

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
