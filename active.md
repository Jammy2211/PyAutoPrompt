## mge-profiling-a100
- session: claude --resume "mge-profiling-a100"
- status: workspace-shipped, awaiting-merge
- worktree: ~/Code/PyAutoLabs-wt/mge-profiling-a100
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/56
- repos:
  - autolens_workspace_developer: feature/mge-profiling-a100
- notes: |
    Follow-up to fft-mixed-precision-fix: extends the profiling story
    from RTX 2060 (consumer) to NVIDIA A100 80GB (production) and
    consolidates 10 configs (4 post-fix local, 4 pre-fix local, 2 A100)
    into the canonical jax_profiling/results/jit/imaging/mge/ tracking
    dir. Headline: A100 fp64 5.7 ms full pipeline, 7.7x faster than
    RTX 2060 fp64. Mixed precision delivers ~zero benefit on A100
    (5% noise) — fp64 is not punitive on production hardware.

    Pure z_projects/profiling tooling (mge_profile.py + mge_aggregate.py
    + 2 SLURM submits) committed locally in profiling repo (no remote).
    autolens_workspace_developer PR #56 carries the result artifacts.

    Caveat noted: A100 JIT log_likelihood truncates to fp32 precision
    (-159734.59 vs eager fp64 -159736.355) — jax_enable_x64 likely not
    set in HPC PyAutoNSS venv. Worth a follow-up before quoting A100-
    served NSS / Nautilus log Z to high precision.

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
