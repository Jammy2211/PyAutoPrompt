- `autogalaxy_workspace/scripts/multi/start_here.py` fails in the mega-run under `PYAUTO_TEST_MODE=1`
  with a long `KeyError` listing every
  `N.galaxies.galaxy.bulge.profile_list.<i>.centre_0` (and `.centre_1`) key for a 20-Gaussian MGE
  across two analyses (`0.galaxies.galaxy.bulge.profile_list.0.centre_0` …
  `1.galaxies.galaxy.bulge.profile_list.19.centre_0`).

  The failure comes from the kwargs lookup that tries to resolve those paths against the sample —
  i.e. the multi-analysis model's bulge parameter path is no longer
  `profile_list.<i>.centre_0` but something else after recent API drift in the MGE composition.

  Investigation steps:

  1. Run `PYAUTO_TEST_MODE=1 python scripts/multi/start_here.py` from `autogalaxy_workspace/`
     to reproduce the traceback.
  2. Print `model.paths` (or the sample keys actually present) to discover the new parameter path
     naming convention after the MGE refactor.
  3. The script (or its underlying plot/aggregator helper) constructs these paths itself —
     locate the helper and update it to use the current parameter path convention.
  4. If the problem is in the library rather than the script, patch in `autogalaxy` and update the
     multi-analysis MGE examples accordingly.
  5. Confirm the fix by re-running the mega-run and checking `scripts/multi/start_here.py` passes.

  This is the only non-data-missing, non-API-drift failure from the 2026-04-24 mega-run that looks
  like a genuine library regression worth understanding, so treat it as a signal that the MGE
  parameter naming has diverged from what the examples assume.
