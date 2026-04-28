Once https://github.com/PyAutoLabs/PyAutoLens/issues/480 is fixed (PointSolver
magnification filter must use `plane_redshift`, not the tracer's last plane),
revisit `@autolens_workspace/scripts/point_source/features/multiple_sources/`.

The simulator and modeling scripts there are already written in their intended
"double Einstein cross" form — source_0 (z=1.0) has its own `Isothermal` mass
profile that lenses source_1 (z=2.0) on top of the foreground lens at z=0.5.
They were authored under autolens_workspace issue #97 but cannot run end-to-end
on the current PyAutoLens release because of #480, so both scripts were entered
into `config/build/no_run.yaml`.

This task closes the loop once #480 lands:

1. Remove these two entries from `autolens_workspace/config/build/no_run.yaml`:
   - `point_source/features/multiple_sources/simulator`
   - `point_source/features/multiple_sources/modeling`
2. Run `python scripts/point_source/features/multiple_sources/simulator.py`
   end-to-end (no `PYAUTO_*` overrides) and confirm both `point_dataset_0.json`
   and `point_dataset_1.json` are written with >=4 positions each.
3. Run `PYAUTO_TEST_MODE=2 python scripts/point_source/features/multiple_sources/modeling.py`
   end-to-end and confirm the likelihood evaluation succeeds without the
   "PointSolver finds 0 positions" failure mode.
4. Remove the `__Currently Blocked By PyAutoLens #480__` notice from both
   scripts' module docstrings.
5. Regenerate notebooks via `/generate_and_merge`.

If the simulator or modeling needs tweaks to match the post-fix solver behaviour
(e.g. slightly different prior bounds, repositioning source centres), that's an
acceptable scope expansion — but do not weaken the example back to a single-lens
configuration.
