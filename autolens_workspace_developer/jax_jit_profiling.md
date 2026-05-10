We have lots of examples which profile how long JAX jitted functions take to run on various datasets.

However, we have not profiled the time it takes to jit functions thesmevles, which can also impact a 
users experience and how long it takes for claude to test things s it runs.

Do an assessment of this for all jax.jit's in the autolens_Workspace, autolens_workspace_test
and autolens_workspace_developer and do an assessment of if we need to put in some JAX jits
in the source code to speed this up, or if there are other placdes or options or things we can
do to make this aspect of JAX also run fast.