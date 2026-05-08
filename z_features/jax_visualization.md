
autogalaxy/[visualizer_fit_for_visualization_dispatch.md](../autogalaxy/visualizer_fit_for_visualization_dispatch.md)
autogalaxy/[fit_pytree_registration_other_datasets.md](../autogalaxy/fit_pytree_registration_other_datasets.md)

issued/[fit_imaging_pytree.md](../issued/fit_imaging_pytree.md) (Path A — full jax.jit wrapping feasibility study)

Alongside these issues, its basically time to make it so all workspace eamples when use_jax=True go through
the full JAX visualization path, which will ensure really fast visualuzation and thus really fast udpates.

Remember we have examples already shwoing this works: 

- autolens_workspace_test/scritps/imaging/modeling_visualization_jit.py and similar scripts there in.
- autolens_workspace_test/scritps/imaging/visualization_jax.py

I want us to be at a point where all default runs do JAX visualization and the notion of it being a separate
things are not longer relevent (unless the user doesnt have JAX installed or has use_jax=False).

Remember also that ideally visualizaion (including via quick updates) would happens on a separate process
to the search so we dont "pause" to visualize. We also want this to be something where Juypter Notebooks
update the quick visuals on the fly during modeling, albeit these could be follow up tasks to the first
which si just getting it running.

However, before doing lots of work, maybe we should do an assessm,ent of autolens_workspace_test and
autogalaxy_workspace_test and assess ifw e are still missing JAX Visualizer coverage in order to implement
this seamlessly, my gut feeling is we actually want to build-out PyAutoPrompt/z_features/jax_visualizaiton.md
into a logic sequence of steps that fully covers this ste by step.


A follow up issue will then be once all these runs work, to make it so a user can "watch" lens modeling happen,
with the display updating every iterations_per_quick_update. Ideally, for a Python script this would be a matplotlib
window which just upates as it does, not disrupting the user from doing other stuff, and in a Jupyter Notebook itd
appear as a cell which updates. We may want to add some other images and info to this, but for now just having
subplot_fit display would probably be a good starting point.

I also want this to be the default behaviour when a user is running a notebook via Google Colab, as on GPU it should
run fast enough that they can thus easily watch the lens modeilng run. This should ideally be how it works
for all data types and Analysis class types.