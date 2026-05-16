More and more work on autolens_workspace_developer is going towards profiling, including:

- Setting up realistic science cases for profiling on CPU, different GPUS via JAX (e.g. `jax_profiling`).
- Doing this across different datasets (e.g. `imaging`, `datacube`).
- Make graphs of these which are retained over version to some extent to track run time.

We are also beginning to add more and more of a searches aspect (e.g. `searches_minimal`) which track and profile
run time for different samplers.

I think we should begin to coordinate all this profiling effort in a single github repositroy
`autolens_profiling` whose scope is:

- To provide likelihood function profiling information for all supported science cases, dataset types and model composition (e.g. mge.py, pixelized, delaunay)
- To provide this information on CPU, laptop GPU and HPC GPU (e.g. A100).
- For the information to be easily visible navigating the GitHub pages (e.g. make the most up to date numbers tables and graphs in github README.md.
- For there to be the step-by-step guides of each likelihood function with all the profiling step times, which autolens_workspace_Developeer already does just making sure you keep that.
- For sampler profiling tests and their run times and stats to also be contained on this repo, likely in a separate high-level fodler to jax stuff. For the first pass, set up this up but only include Nautilus.
- Lets keep JAX gradient stuff in autolens_workspace_developer for now, and view this as out of scope but likely to come into the repo in the future. Put a note about JAX gradients on the front page.
- Lets include the autolens_workspace_developer/jax_profiling/simulators package in all this, and have first-class support and tracking for time simulators take to run, but remember that JAX jitting for these are not fully implemented everywhere so we may have placeholers and then follow that up.
- GitHub actions and CI in whatever way you think is useful.

Dont add any profiling scripts or whatnot that are missing from _develoeper (e.g. I will probably put a group one in sooner or later but the scope of this task is not to make any test examples).

And thus for this to have good easy to read README.md's through for people to inspect and reprodocue. There should be a focus on
the profiling examples using simulated datasets with values representative of real science cases (e.g. the imaging pckage already has options for "hst", "euclid"
so we should make sure this instrument-level profiling is expressed in the information people can rad on the GitHub. Framing it in terms of Astronomy
instruments and teelscopes, instead of just "number of pixels" or other such metrics is way more intuitive.

Furthermore, I am soon going to write the PyAutoLens-JAX JOSS paper and this will be a great way to substantiate
my claims about run times.

Do a bit of deep research and thinking about other cool ways we can enhance a repo thats all about profiling and run times!

Obviously this is likely to be a multi stage issue which we can put the high level prompt in z_features.