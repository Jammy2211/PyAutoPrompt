## unpark-ellipse-scripts
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace/issues/72
- session: claude --resume "unpark-ellipse-scripts"
- status: library-shipped, workspace-pending
- worktree: ~/Code/PyAutoLabs-wt/unpark-ellipse-scripts
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1270
- repos:
  - PyAutoFit: feature/unpark-ellipse-scripts
  - autogalaxy_workspace: feature/unpark-ellipse-scripts
  - PyAutoBuild: feature/unpark-ellipse-scripts
- summary: |
    Unpark the five `ellipse/*` example scripts in autogalaxy_workspace
    after the JAX refactor (PyAutoGalaxy #408/#410/#412 merged 2026-05-14).
    All five pass under PYAUTO_TEST_MODE=2 once two unrelated bugs are
    fixed: (a) `database.py` had leftover `path.exists(...)` calls from
    the os.path→pathlib refactor on two lines, and (b) `Drawer.__init__`
    in PyAutoFit raised "got multiple values for keyword argument
    'number_of_cores'" when reconstructing a saved Drawer via `from_dict`
    (saved search.json carries number_of_cores at the top level, which
    collided with the hardcoded `super().__init__(number_of_cores=1, **kwargs)`).
    Fix: `kwargs.pop("number_of_cores", None)` before forwarding to super.
    Regression test added in test_drawer.py.

    Scope now: PyAutoFit library PR (Drawer fix + regression test) ships
    first; autogalaxy_workspace + PyAutoBuild workspace PRs (no_run.yaml
    cleanup + database.py pathlib patch) follow.

## log-prior-sign-convention
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1266
- session: claude --resume "log-prior-sign-convention"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/log-prior-sign-convention
- repos:
  - PyAutoFit: feature/log-prior-sign-convention
  - autofit_workspace_test: feature/log-prior-sign-convention
- summary: |
    Fix the sign-convention bug in `Prior.log_prior_from_value` (#1266).
    Today NormalMessage / LogGaussianPrior / TruncatedNormalMessage return cost form
    (positive for low-density); LogUniformPrior returns 1/value (Jacobian gradient,
    not a log at all). Fitness._call's `figure_of_merit = log_likelihood + sum(log_priors)`
    expects density form, so Emcee/Zeus/MLE-Drawer/LBFGS/BFGS posteriors are biased
    on every non-uniform prior — empirically confirmed by controlled experiments
    (Emcee samples diverge to 10^146, LBFGS converges to 8e143, both should give
    mean≈5 for a flat-likelihood + GaussianPrior(5,1) fit).

    Fix: density convention at the Prior layer. Sign-flip NormalMessage and
    LogGaussianPrior quadratics; replace LogUniformPrior body with -log(value).
    UniformPrior and TruncatedNormalMessage already correct (0.0 and density-form
    respectively). No Fitness changes. Update 3 test pins in test_prior.py.

    Workspace_test: promote /tmp/{emcee,lbfgs}_prior_bias_check.py to
    scripts/prior_correctness/ as permanent regression gates. Update parity
    assertions in scripts/jax_assertions/priors_xp_dispatch.py for the
    Gaussian-family + LogUniform new values.

    Migration warning: cached Emcee/Zeus/MLE-Drawer/LBFGS samples.csv with
    non-uniform priors are biased and should be re-run. Dynesty/Nautilus chains
    unaffected (priors via prior_transform); only their stored log_prior column
    is wrong-signed and auto-recovers on next log_prior_list_from_vector call.

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
