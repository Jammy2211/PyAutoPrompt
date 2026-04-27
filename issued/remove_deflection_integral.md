Throughout the package @PyAutoGalaxy/autogalaxy/profiles/mass there are methos which compute deflection
angles via a method deflections_2d_via_integral_from

These are slow, and often very large, I dont want them in the source code any more. But they do serve an important
purpose of providing numerical vlues to test gainsrt. Thus, first, can you move all methods to a dedicated package
in @autolens_workspace_test/scripts/ called mass_via_integral, assrting their values based on the autogalaxy
unit tests.

For the maojrity of these functions, the integral aws only used for testing anyway, with an MGE or CSE decompositon
forming the basis of the deflection angle. I therefore do not want you to remove or change any unit tests unless
they are directly against the integral. In most cases, because the MGE or CSE method is used nayway the
unit test should not change.

We also want to remove all long integral functions used for potentials or convergences, moving them to
the same mass_via_integral. If this does break unit tests, can you inform me and I'll look to see if
we can resolve this via some sort of calculation change on a case by case basis.
