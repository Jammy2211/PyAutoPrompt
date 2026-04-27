The files in @autolens_workspace_develoepr/jax_profiling give a clear run through of how long each step of the
JAX likelihood function takes.

However, the mge.py and other scripts have steps which I think do not go to the JAX jit level, due to the input
and outputs not having an easy translation.

Can you look at mge.py and explain what the issue is and how we make this fully have JAX jitted timings?