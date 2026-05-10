The imaging `features/no_lens_light` example needs reviewing before adapting to interferometer.

Once the imaging version is in good shape, adapt it to the interferometer context in
`scripts/interferometer/features/no_lens_light/` (autolens_workspace).

For interferometer datasets where the lens galaxy emits no detectable flux at the observed wavelength
(common at sub-mm / radio wavelengths, e.g. ALMA observations of a dust-rich background source behind
a passive elliptical), the lens light is omitted from the model entirely. Only the lens mass profile
and source light are fit. Thanks to nufftax, light profile fitting in the visibility domain is now
fast, so this configuration runs efficiently even with many visibilities. The script should describe
when this regime applies and credit nufftax for making it practical.
