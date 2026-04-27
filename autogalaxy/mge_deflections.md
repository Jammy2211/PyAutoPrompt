We are going to refactor the mass profile module to work with the following assumptions:

1) All mass profiles must have a deflections_yx_2d_from method defined, as they do now.
2) The best method they can use is an analytic function, however this could instead be an MGE decomposition
or CSE decomposition.
3) The potential_2d_from and convergence_2d_from methods can, optionally, be defined inside mass profiles. 
is common if they have analytic solutions which are quick to comptue.

However, a potential or convergence does not need to be defined naalytically. In this case, we will fall back to
an MGE or CSE method, which computes the potential via those funcgtions. These methods sitll need a dediciated
convergence function for the decomposition though, which not all functions have. Therefore, can you draw me up
a list of mass profiles, which currently need this MGE / CSE treatment applied to them. The red flag to catch this
is those where the potential_2d_from or convergence_2d_from method returns a numpy array of zeros. Once
we have this list, we'll address each profile one-by-one as best we can.