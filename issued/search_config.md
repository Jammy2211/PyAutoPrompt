All searches use config files to load theirn input parameters and interface with their respective
source code library.

Part of the design was so that a project like PyAutoLens could create a search, have the parameters set in a
way that was suitable for that model and hide from the user everything else:

search = af.Nautilus(
    path_prefix=Path("imaging"),  # The path where results and output are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see modeling examples for details.
    iterations_per_quick_update=1000,  # Every N iterations the max likelihood model is visualized in the Jupter Notebook and output to hard-disk.
)

Example of a modeling script:

@autolens_workspace/scripts/imaging/modeling.py

However, the downside of this approach is the autofit source code got very confusing with search config
loads, the config fiels which control this for a project are bloated and require energy to set up and
maintain and it inevitably leads to bug.

This issue shows part of the problem:

https://github.com/PyAutoLabs/PyAutoFit/issues/1001

Is there a solution which makes the source code interface a lot cleaner, doesnt have these config files
and makes it easier for neew samplers to be set up? I wonder if having a Python class input to each
search class is the solution, which in the past would be a pain to maintain for each specific sampler
but now AI agents can do this quickly isnt so bad. 

The biggest downside is a project like autolens_workspace may need to manually specify search inputs everywhere
the search is set up, albeit maybe theres a no-required-config trick or inheritance trick we could use?

Think hard and come up with an assessment of how we can make the source code a lot cleaner without 
losing the desired API.