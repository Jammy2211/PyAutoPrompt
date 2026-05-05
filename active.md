## autogalaxy-wst-model-composition
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/26
- session: claude --resume "autogalaxy-wst-model-composition"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/autogalaxy-wst-model-composition
- repos:
  - autogalaxy_workspace_test: feature/autogalaxy-wst-model-composition

## alma-datacube
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/120
- session: claude --resume "alma-datacube"
- status: library-dev (steps 1-3 shipped/PR-open, library viz fix in flight)
- worktree: ~/Code/PyAutoLabs-wt/alma-datacube
- repos:
  - PyAutoFit: feature/alma-datacube (AnalysisFactor.visualize_combined fix)
  - PyAutoLens: feature/alma-datacube (VisualizerInterferometer combined plotter)
  - autolens_workspace: feature/alma-datacube (PR #122 open — gated on library PRs)
  - autolens_workspace_test: feature/alma-datacube (visualization dispatch tests)
  - autolens_workspace_developer: merged via PR #46
- shipped-prs:
  - https://github.com/PyAutoLabs/autolens_workspace_developer/pull/46 (merged 2026-05-04)
  - https://github.com/PyAutoLabs/autolens_workspace/pull/122 (open, library-gated)
- summary: |
    Done: dev workspace shipped (autolens_workspace_developer#46) and
    user-facing autolens_workspace scripts open as PR #122. While verifying
    visualization, found that FactorGraph.visualize_combined silently skipped
    VisualizerImaging.visualize_combined for multi-band fits (and there was no
    VisualizerInterferometer.visualize_combined at all).
    Fix: AnalysisFactor.visualize_combined / visualize_before_fit_combined
    forwarders in PyAutoFit; new subplot_fit_interferometer_combined +
    PlotterInterferometer.fit_interferometer_combined +
    VisualizerInterferometer.visualize_combined in PyAutoLens.
    Verification: scripts/multi/visualization_imaging.py +
    scripts/multi/visualization_interferometer.py in autolens_workspace_test
    (both PASS dispatch). Library PRs ship next; workspace + workspace_test
    PRs follow once libraries merge per the library-first gate.

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
