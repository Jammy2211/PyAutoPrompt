The imaging `features/advanced/mass_stellar_dark` example needs improving and padding out before adapting to interferometer.

Once the imaging version is more complete, adapt it to the interferometer context in
`scripts/interferometer/features/advanced/mass_stellar_dark/` (autolens_workspace).

Decomposing total mass into stellar (tied to the lens light via M/L ratio) and dark (e.g. NFW)
components is wavelength-agnostic. The interferometer port differs from imaging only in that the
fit is to visibilities, with light profile transforms handled by nufftax. The script should
emphasize that the M/L coupling between light and mass works identically across data modalities,
and that this configuration is now practical against visibilities thanks to nufftax.
