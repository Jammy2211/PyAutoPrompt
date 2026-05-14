From a liost of interferometer the likelihoodf unction is as I expect.

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