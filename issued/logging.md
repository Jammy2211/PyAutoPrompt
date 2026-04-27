Read @PyAutoFit/autofit/non_linear/search for context where this is.

The logging of a fit Can be improved as follows:

> 2026-04-01 18:46:35,500 - autofit.non_linear.search.abstract_search - INFO - Starting non-linear search with 1 cores.

This is confusing for JAX users, as JAX automatically uses all cores, so the above is not true. 

Instead, if JAX is being used it should be something like:

> 2026-04-01 18:46:35,500 - autofit.non_linear.search.abstract_search - INFO - Starting non-linear search with JAX.
> 
Can you make it so JAX displays if CPU or GPU is being used?

>Running search where parallelization is disabled.
> 
> Again, this is not true in JAX, just remove this.