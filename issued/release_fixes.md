## Release Fix: 2026-04-12

Based on release run Apr 10 (run 24226814280). 345 failures, 808 passes, 90 skipped.

### Priority 1: Simulator Plot API Bugs (~200+ cascading failures)

These 2 bugs in workspace simulator scripts break data generation, cascading into ~220 workspace_issue + ~57 workspace_data failures.

1. **`autolens.plot` has no attribute `Output`**
   - Example: `scripts/imaging/simulator/no_lens_light.py:135`
   - `aplt.plot_array(array=dataset.data, output=aplt.Output(path=dataset_path, format="png"))`
   - `aplt.Output` was likely removed or renamed in a recent autolens/autogalaxy plot refactor
   - Fix: update all simulator scripts using `aplt.Output` to the current API

2. **`'numpy.ndarray' object has no attribute 'is_all_false'`**
   - Example: `scripts/interferometer/simulator/with_lens_light.py`
   - `autoarray/plot/utils.py:108` in `zoom_array` — calls `array.mask.is_all_false` on a plain ndarray
   - This is a **library bug** in autoarray's `zoom_array`, not just a workspace issue
   - Fix: `zoom_array` should check the array type before accessing `.mask.is_all_false`

### Priority 2: Source Code Bugs (60 failures)

These are genuine API breakages in library code or workspace scripts:

3. **`af.UltraNest` removed from autofit namespace**
   - `scripts/cookbooks/search.py:401`
   - `AttributeError: module 'autofit' has no attribute 'UltraNest'`
   - Fix: update script to use the correct import path, or add UltraNest back to `__init__.py`

4. **`interpolator.__getitem__` IndexError**
   - `scripts/features/interpolate.py:269`
   - `self.instances[0]` on empty list
   - Fix: investigate why interpolator query returns no instances

5. **`grid_search_result.log_evidences()` TypeError**
   - `scripts/features/search_grid_search.py:286`
   - `unsupported operand type(s) for -: 'NoneType' and 'float'`
   - Fix: `log_evidence` is None for some samples — handle in `result.py:306`

6. **Missing dataset JSON files for graphical model tutorials**
   - Multiple `howtofit/chapter_3_graphical_models/` scripts
   - `FileNotFoundError: dataset/example_1d/gaussian_x1__low_snr/dataset_0/data.json`
   - Fix: these tutorials need their dataset generator scripts run, or the datasets committed

7. **Various autolens/autogalaxy workspace script failures**
   - `howtogalaxy` chapter 2-4 tutorials, imaging features, guides
   - Mix of AttributeError, IndexError, TypeError from API changes
   - Fix: update workspace scripts to match current library APIs

### Priority 3: Environment Issues (6 failures, non-blocking)

8. **Missing `pytest` in CI for database scrape scripts** — add pytest to CI deps or skip these scripts
9. **Missing `util` module in simulator notebooks** — import path issue
10. **JAX `delaunay_mge.py` failure** — JAX-specific, likely not blocking

### Suggested Approach

**Phase 1 (library fix):** Fix bug #2 in PyAutoArray `plot/utils.py` — this is the only library-level code change needed. Ship via `/ship_library`.

**Phase 2 (workspace fixes):** Fix bugs #1, #3-7 in the workspace scripts. These are all workspace-level changes. Ship via `/ship_workspace`.

**Phase 3:** Re-trigger the release build and verify the failure count drops to ~60 or fewer (the remaining source_code_bugs that aren't cascade-related).

### Release Job Status

- test_pypi: all 5 succeeded (libraries are fine to release)
- release/release_workspaces/publish_release_notes: skipped (gated on test pass)
