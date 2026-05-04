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

## rst-to-myst-md-pass2
- issue: none — direct followup to PyAutoFit#1245 (issue closed)
- session: claude --resume "rst-to-myst-md-pass2"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/rst-to-myst-md-pass2
- repos:
  - PyAutoConf: feature/rst-to-myst-md-pass2
  - PyAutoBuild: feature/rst-to-myst-md-pass2
  - PyAutoArray: feature/rst-to-myst-md-pass2
  - PyAutoFit: feature/rst-to-myst-md-pass2
  - PyAutoGalaxy: feature/rst-to-myst-md-pass2
  - PyAutoLens: feature/rst-to-myst-md-pass2
  - HowToFit: feature/rst-to-myst-md-pass2
  - HowToGalaxy: feature/rst-to-myst-md-pass2
  - HowToLens: feature/rst-to-myst-md-pass2
- summary: |
    Convert remaining prose .rst → MyST .md across the PyAuto ecosystem.
    Includes root README + CITATIONS, config/.../README files, and HowTo*
    notebooks/scripts READMEs. Keeps docs/api/*.rst (autosummary) and
    docs/_templates/*.rst (Sphinx Jinja templates) as native RST.
    Per-repo PR; library-first ordering.
