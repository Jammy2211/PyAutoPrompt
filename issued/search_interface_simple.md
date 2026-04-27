Currently, searches are implemented in the autofit source code in the package @PyAutoFit/autofit/non_linear/search.

This puts a lot of demands on their interface and design, with them needing to support the wrapping to an Analysis class,
being called through stuff like visualiation and being subject to various calls for results, sampels and whatnot.

This means that if one has an existing model and Analysis in autofit, it can be quite difficult
to simply "plug" it into a new search and mess around with it, as the search has to be designed to support all of the above.

For this reason, I want to have an example whcih shows how to use a search which is not a autofit NonLinearSearch
at all, but simply interfaces the external libary's API with the Analysis and Model objects. We can use
Nautilus for this, which is already implemented in autofit so the interface is designed, read
through @@PyAutoFit/autofit/non_linear/search/nest/nautilus to see how it is implemented.

In @autofit_workspace_developer, create a minimal script which does this using the simple modeling example
shown in @autofit_workspace/scripts/overview/overview_1_the_basics.py,

This will contrast the existing examples in @autofit_workspace_developer/searches, which show the full
PyAutoFit search with output API. I guess we can make the folder called @autofit_workspace_developer/searches_minimal.

This will become a much easier way for us to try out searches and test their performance on problems
before doing the full autofit implementation, which is a lot of work and requires a lot of design to support all of the features of the package.

Make the simple example for Nautilus, Dynestty, Emcee and LBFGS.