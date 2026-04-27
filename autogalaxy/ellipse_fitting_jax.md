- Ellipse fitting is defined in @PyAutoGalaxy/autogalaxy/ellipse, with scientific walk throughts in
@autogalaxy_workspace/scripts/ellipse .

It currently does not support JAX, can you add JAX support through the likelihood function, such that
@PyAutoGalaxy/autogalaxy/ellipse/model/analysis.py, Analysis.log_likelihood_function is JAX compatible.

You can refer to @PyAutoGalaxy/autogalaxy/imaging/model/analysis.py to see how JAX is supported in an
analysis class.

Before starting, can you make integration tests in @autogalaxy_workspace_test/scripts, in particular:

- visualization.py to test the ellipse/visualization of the ellipse fitting.
- jax_likelihood_functions/ellipse to test the JAX likelihood function of the ellipse fitting.


These can mirror @autolens_workspace_test/scripts

INCLUDE AOMETHING ABOUT DODGY LOOP, START THERE in PROMPT

- Use galaxies API, not separate ellipse / multipole_list, to unify API especially for aggregation?