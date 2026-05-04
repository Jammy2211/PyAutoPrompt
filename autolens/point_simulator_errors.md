The errors on the positions in the point_source simulator are very large, for example the noise_map adds a value of 0.05"
which is the pixel scale of the image data (e.g. HST). However, I think much greater precision is possible via precisoin
astrometry, and these large errors drive model fits to have large errors.

Given the positions are basically where the point source hits image piels, its not even quite clear to me how exactly
we should simulate the noise, but maybe it should not be drawing from a Gaussian?

The errors on the time delay, which mutliply by 0.25 are also unrealistically large, a value of 0.05, or less, is more suitable.

I suspect the fluxes may be quite unrealistic too.

Can you read the point_source simulator.py script, and do some deep research on what errors should be and then updated
the script and its docstrings accordingly.