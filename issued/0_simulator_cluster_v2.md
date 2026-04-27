-Read other simulator.py files (E.g. scripts/imaging/simulator.py) and copy language style,
amount of text to python code a bit more, its slightly short at times or too brief in descripton.

- Mention the option of multi plane stuff more clearly, in fact lets adopt the simulator for the
- soruces to be at different redshifts, 1.0 and 2.0. make sure both produce multiple images.

I think apply_over_sampling is applied twice redundantly?

The simulator is also slow to run, taknig over 5 minutes. This is probably because we are not JAX jitting
the slowest calculation (the point solver). Can you check if this is the case, and if so JAX.jit it so 
that it runs fast, adding a description and documentation for the JAX jit. In fact, we recently did a lot of
work on pytrees registration (see autolens_workspace_deveoper/jax_profiling/point_source) and thus
we should be aiming to use the same JAX jit API as there.

Can you update the script to only have 2 main lens galaxies and the host halo. We will update it to have lots
of galaxies on a scaling relation soon after, to make it more realistic.