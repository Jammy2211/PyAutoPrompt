Issue Move search update to new class #1003 — Extract SearchUpdate class: perform_update and perform_visualization in abstract_search.py interact with many modules (saving samples, visualization, profiling, outputting results). Extracting to a SearchUpdate class would make each output task its own method.
https://github.com/PyAutoLabs/PyAutoFit/issues/1003

Do an assessment of if this is a good idea and if so come up with an example of th eimplementtation. I think its clear 
abstract_search needs to be simplified and have a better sense of separation of concerns.

The goal here is really to achieve seapration of concerns in abstract_search.py and move these different categories of output
results to dediciated modules. A benefit is that some of these, like samples, are tied to the type of
search (e.g. mcmc, nested sampling). So the type of output can mayb eexploit composition or stronger typing
to make the work more clear and explicit.