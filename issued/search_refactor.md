The PyAutoFit non-linear search package allows us to perform modeling with a search.

A high level description of modeling is given in @autofit_workspace/scripts/overview/overview_1_the_basics.py

The full model API is given in the cookbook files @autofit_workspace/scripts/cookbooks/search.py
and their applicaiton in @autofit_workspace/scripts/cookbooks/analysis.py

Another guide is @autofit_workspace/scirpts/howtofit/chapter_1_introduction/tutorial_3_non_linear_search.py

The file @autofit_workspace/scirpts/howtofit/chapter_2_scientific_workflow shows many of the key outputs possible
with a search, which allows a user to inspect and judge results, which is a key feature of autofit.

Searches are defined in the source code mostly at @PyAutoFit/autofit/non_linear

There is a lot of scope to refactor and redesign this package, first, we can remove:

- ultranest (PyAutoFit/autofit/non_linear/search/nest/ultranest)
- pyswarms (PyAutoFit/autofit/non_linear/search/mle/pyswarms)

The following github describe some clean up and refactors:

https://github.com/rhayes777/PyAutoFit/issues/1003

https://github.com/rhayes777/PyAutoFit/issues/1002

First, after removing ultranest and pyswarms, can you review the unit tests, clean them up and add more test
coverage if you think it is needed? Then we can start to design the new API and implement it.

However, by remove, I dont want the code to be gone and lost forever. Can you
move their implmenetations to @autofit_workspace_developer,ensuring they can be run
with a test case locally. But such that most of the code which is a pain to keep ones head around
is in place.

