The imaging `features/advanced/double_einstein_ring` example needs improving and padding out before adapting to interferometer.

Once the imaging version is more complete, adapt it to the interferometer context in
`scripts/interferometer/features/advanced/double_einstein_ring/` (autolens_workspace).

A double Einstein ring system has two source galaxies at different redshifts, both lensed by the
foreground galaxy with the intermediate source acting as a secondary lens for the more distant one.
The multi-plane ray-tracing is identical to imaging, but the model fit happens in the visibility
domain — both source-plane images must be Fourier-transformed via nufftax and combined before the
chi-squared comparison against the visibilities. The script should describe this two-source-plane
flow and credit nufftax for making it tractable.
