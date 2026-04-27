The following example applies pixelizations to galaxies in autogalaxy:

@autogalaxy_workspace/scripts/imaging/features/pixelization

At the moment the applicaiton is a bit basic, and not really a good example of how one would want to use
these tools on galaxies for surface brightness fitting.

Can you update the example (which may include simulating a clump galaxy via a new simulator,
see @autogalaxy_workspace/scripts/imaging/simulator.py). The idea is that the galaxy would have a central bulge,
fitted for by a Sersic profiel, and then assymetric clumpy star formaiton that is hard to fit with a parametric profile, 
and so we would use a pixelization to fit this part of the galaxy.

Give me a plan for this, making sure to apply it to all codes in the pixelization package.

Once you have done this, can you look throughout the workspace as pixelization descriptions and look for more
upportunities in text to make it clear this is the example usage of the fucntion.