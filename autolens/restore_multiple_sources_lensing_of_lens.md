Once https://github.com/PyAutoLabs/PyAutoLens/issues/480 is fixed (PointSolver
magnification filter must use `plane_redshift`, not the tracer's last plane),
revisit `@autolens_workspace/scripts/point_source/features/multiple_sources/`.

When that workspace example was first written (autolens_workspace issue #97),
the upstream PointSolver bug made it impossible to simulate or fit a configuration
where the intermediate source itself acts as a deflector for the further source.
To unblock the multi/factor-graph tutorial, the example was simplified so the
only deflector is the foreground lens — both source galaxies are point-only at
different redshifts, no source-plane mass profile.

This task restores the original "double Einstein cross" intent now that the
PointSolver bug is fixed:

1. Update `simulator.py` so source_0 (z=1.0) regains its `Isothermal` mass profile
   at (0.02, 0.03) with `einstein_radius=0.2` and a small ellipticity. Source_0
   should now genuinely lens source_1 in addition to the foreground lens.
2. Verify the simulator still runs end-to-end with a single tracer
   `[lens, source_0_with_mass, source_1]` and that `solver.solve(plane_redshift=1.0)`
   returns >=4 image-plane positions for source_0.
3. Update `modeling.py` so the model includes source_0's mass:
     - `source_0 = af.Model(al.Galaxy, redshift=1.0, mass=al.mp.Isothermal, point_0=al.ps.Point)`
   The lens model dimensionality goes from N=9 to N=14.
4. Decide whether the `AnalysisFactor` for source_0's dataset should fit using
   the full multi-plane model or a sub-model excluding source_1. With #480 fixed,
   the full multi-plane model should fit cleanly and is preferred — both factors
   share `lens` and `source_0.mass` priors, the factor graph sums log-likelihoods.
5. Re-run end-to-end with `PYAUTO_TEST_MODE=2` (no `PYAUTO_SMALL_DATASETS`) to
   confirm the simulator and modeling both work, then regenerate notebooks.
6. Update the script docstrings and the folder/feature README to mention the
   lensing-of-lens richness again, and remove any "simplified to work around
   PyAutoLens #480" comments.
