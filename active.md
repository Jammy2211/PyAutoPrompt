## cluster-c-point-source-rebaseline
- session: claude (current CLI)
- status: shipping
- classification: workspace
- suggested-branch: feature/cluster-c-point-source-rebaseline
- worktree: ~/Code/PyAutoLabs-wt/cluster-c-point-source-rebaseline
- repos: autolens_workspace_test
- summary: |
    Rebaseline JAX point-source likelihood literals after upstream
    autolens_workspace_test commit 931a381 changed positions_noise_map
    (0.2 → 0.005, PSF-centroiding precision). Regenerated
    dataset/point_source/simple/{point_dataset_positions_only,tracer}.json
    and updated three expected_likelihood literals across
    image_plane.py (1.313508 → -83.38049778), point.py (same), and
    source_plane.py (vmap & eager: -199.155… → -331481.26…). All three
    scripts pass on the new literals.

## subhalo-redshift-jax-repro
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/498
- session: claude (current CLI)
- status: shipping repro PR (fix work to follow on a separate branch)
- classification: workspace (then library follow-up)
- suggested-branch: feature/subhalo-redshift-jax-repro (workspace) / feature/subhalo-redshift-jax-fix (library, TBD)
- worktree: ~/Code/PyAutoLabs-wt/subhalo-redshift-jax-repro
- repos: autolens_workspace_test, PyAutoLens (later)
- summary: |
    Reported on Slack by @qiuhan96. Free-parameter subhalo redshift
    (af.UniformPrior) raises TracerBoolConversionError under JAX. Root
    cause is Python sorted()/<=/== on traced redshifts inside
    tracer_util.plane_redshifts_from + grid_2d_at_redshift_from.
    Reproducer added to autolens_workspace_test as
    scripts/jax_likelihood_functions/imaging/subhalo.py — runs two
    scenarios (fixed z=0.55 PASS, free UniformPrior FAIL with the
    expected TracerBoolConversionError). Exits 0 on the expected
    failure, 1 once the bug is fixed (so the same script becomes the
    regression test). Workaround for users today: use_jax=False.

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
