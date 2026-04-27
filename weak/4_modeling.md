We are now going to add weak lensing modeling.

First, we need to create the analysis.py module, so inspect @autolens_workspace/scripts/weak and
@PyAutoLens/autolens/imaging/model . We are basically going to make everything weak does from here a "mirror" of
the imaging model API (and also interferoter.)

The log_likelihod_function is in particular important, we already have the codw which creates the shear field
show in the @autolens_workspace/scripts/weak/simulator.py file and we can turn it into a fit in 
@PyAutoLens/autolens/imaging/fit.py, so shouldnt be hard to work out the pattern and whats needed here.

I believe everything else (plotter.py, result.py, visualizer.py) should be relatively straight forward to work
out by simply copying the patterns and logic from the imaging module, but keep an eye out for unexpected tricky
parts.

Finally, read @autolens_workspace/scripts/imaging/modeling.py and then make an equivalent for
weak lensing.