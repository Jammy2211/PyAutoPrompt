

I then want us to make datacube/likelihood_function.py more information, To be honest, the current script is more like a test script than a user facing description.

read interferometer/features/pixelization/likelihood_function.py, I want our likelihood function to be written
to this level of detail, but only at points which differ to it. Look at the style of imaging/features/multi_gaussian_expansion/likelihood_function.py,
which ahs statements like:

"The code below repeats that used in `light_profile/log_likelihood_function.py` to show how this was done."


The likelihood function for a DataCube is pretty much identical to that of an interferometer, the only differences are:

- It uses a list of Interferometer objects instead of just one.
- Each of these has their own uv_wavelength, meaning they each go into their own `Inversion` object, but everything that
happens in that object (e.g. the NUFFT) is identical. 
- We will illustrate the likelihood function the same as interferometer/likelihood_function.py, but keeps the notes like this which is how
pixelized likelihoods actually work "If the number of visibilities is large (e.g. 10^6) this matrix becomes extremely large and computationally expensive to 
store memory, meaning the sparse operator likelihood function must be used instead."

So make the likelihood_function more end to end, with the same level of detail like imaging/features/multi_gaussian_expansion/likelihood_function.py where
steps repeated in interferometer/features/pixelization/likelihood_function.py are not repeated but the specific stuff
using Interferometer lists and whatnot is fully explained.


The key point about Aris's optimization is that:

- Every interferometer has its own data, noise map and uv wavelengths.
- Every source reconstruction is unique and should be performed indepedently for each channel.
- The key point is that the same image-plane grid and source-plane grid and used, such that the same NUFFT 
can be performed once to convert the mapping matrix to a curvature matrix. 
- This single curvature matrix is then applied to every data_vector, all of which are unique, in order to perform
an indepednetl source reconstruction for every channel.

The problem here is the analysis list API does not currently share information across likelihood functions
or analysis objects. We therefore either need to make a DataCube data class, Inversion object and 
add bespoke source code or we need to have AnalysisCombined objects be able to share information in 
their likelihood functions.


the data_preparation.py script probably works as expected, so I should just simulate my own datacube
where I move a Sersic source over channels, simulate as a list of interferometer objects and at the end
map it to the 3d .fits file expected as input.