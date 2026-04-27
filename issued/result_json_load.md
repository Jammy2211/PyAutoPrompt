After a search has run, nearly everything a user could need is in the output folder and loadable via .json:

- Load the samples and do things with them.
- Load the model used to fit the data.
- Load the best fit tracer. 
- 
And so on.

Run modeling.py in both autogalaxy_workspace and autolens_workspace and make an assessment of the output folder's
loadable files (.json files, but also the .fits files in the image folder).

In guides/results, we now need to explain two different concepts in results/start_here.py:

- Loading from the output folder via the above methods, which loads things instantly and is quick and simple but uses memory instantl
- Loading things via the aggregator, which scrapes directories, but uses things like generators to minimize memory use.

I think we should split this in two, where results/start_here.py is the simple  .json / .fits loading, and the existing start_here.py
is moved to the folder "examples", which is renamed "aggregator", now has the start_here.py file. We should make sure
both start_here.py files refer to one another and explain the dfiference in their uses.

Finally, at the end of each modeling.py file we have code which is like:

"""
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` and `FitImaging` objects.
 
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
It also contains information on the posterior as estimated by the non-linear search (in this example `Nautilus`). 

Below, we make a corner plot of the "Probability Density Function" of every parameter in the model-fit.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).
"""
aplt.corner_anesthetic(samples=result.samples)

I think after this, we should add a small section called "__Loading From Output Folder__" which explains the
.json and .fits files in the output folder can be easily loaded, give an example of one of each (tracer.json and tracer.fits)
and then points to the new results start_here.py file to explain. 