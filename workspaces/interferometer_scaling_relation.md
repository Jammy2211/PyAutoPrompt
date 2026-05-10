The imaging `features/scaling_relation` example needs improving and padding out before adapting to interferometer.

Once the imaging version is more complete, adapt it to the interferometer context in
`scripts/interferometer/features/scaling_relation/` (autolens_workspace).

Scaling relations let many extra galaxies share a luminosity-to-mass relation
(`einstein_radius = scaling_factor * luminosity^scaling_relation`), keeping model dimensionality low
even as galaxy count grows. This applies to interferometer modeling identically once visibility-
domain light profile fits are fast (now true via nufftax). The script should mirror the imaging
feature's API explanation in a standalone, beginner-friendly way and reference the nufftax
performance shift.
