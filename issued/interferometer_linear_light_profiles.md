The imaging `features/linear_light_profiles` example needs reviewing before adapting to interferometer.

Once the imaging version is in good shape, adapt it to the interferometer context in
`scripts/interferometer/features/linear_light_profiles/` for **both** `autolens_workspace` and
`autogalaxy_workspace`.

Linear light profiles solve for intensity normalizations analytically given the model parameters,
which previously was prohibitively slow against visibilities because every iteration had to compute
the Fourier transform of every basis component. With nufftax (a JAX-friendly NUFFT — point to its
GitHub and credit it), the linear inversion is now fast in the visibility domain, so this feature
finally becomes practical for interferometer modeling. The script should describe this transition
explicitly and explain why older comments calling light profile fits "slow" no longer apply.
