PyAutoLens has a number of examples which perform source science, see:

@autolens_workspace/scripts/imaging/source_science.py
@autolens_workspace/scripts/imaging/features/pixelization/source_science.py

First, read this scripts and do an assessment on whether they are correct or could be improved.

However, on Euclid data we have found that the source magnitudes we estimate are unreliable,
they often vary depending on the the source model.

Theefore, in @autolens_workspace_developer/source_science I am building tools which simulate strong lenses,
output the source magnitude and magnification, and then perform simple and quick model fits to test this behaviour.

First, can you run the simulator, and then set up model fit using a Sersic source and lens light model
(see @autolens_workspace/scripts/imaging/modeling.py), perform the fit, and compare estimates of the magnification
and source magnificaition to the truth. For this sipmple example, I expect we will get acccurate results.

Then do the same thing but for a Multi Gaussian Expansion (See @autolens_workspace/scripts/imaging/features/multi_gaussian_expansion)
lens light model, but keep the Sersic sourcde. Then do another fit, with the source also an MGE. WE can then assess
the results and decide where to go next.w