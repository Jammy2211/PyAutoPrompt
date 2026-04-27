Visualization is performed during and after a model-fit in the method @PyAutoFit/autofit/non_linear/search/abstract_search.py:

if self.force_visualize_overwrite:
    self.perform_visualization(
        model=model,
        analysis=analysis,
        samples_summary=samples_summary,
        during_analysis=False,
    )

Currently, visualization does not use JAX, and does not use JAX jit to speed up calculations. 

In @PyAutoLens/autolens/imaging/model/visualizer.py, we can see an example of how visualization is performed. 
In particular, the method         fit = analysis.fit_from(instance=instance) is called, which does not use
JAX because only the log_likelihood_function is jitted in @PyAutoFit/autofit/non_linear/fitness.py

I want autofit to support JAX jitted visualization, and for this to be done when use_jax is passed to Analysis,
but for now lets use a use_jax_for_visualization flag to make it explicit that we are only using JAX for visualization.

Recent updatrs havr added pytrees registration to autofit and the source code, look up autofits recent PR on this
and the examples in @autolens_workspace_test/scripts/jax_likelihood_functions.py imaging.

Therefore, can you assess how feasible this is and in @autolens_workspace_test/scripts/imaging, read visualization.py
and produce an example visualization_jax.py which tries to achieve this, calling only the
VisualizerImaging's visualize method for now. Lets only do this for a MGE parametric source, for simplicitiy, 
I expect we'll first see some JAX issues due to certain  internal calls not support JAX (e.g. `tracer = fit.tracer_linear_light_profiles_to_light_profiles`).
this require pytree registration. 

That is fine, I want us to get to this point and then we can start to work through the issues one by one, 
and make sure that the visualization is working with JAX.

Do you foresee an issue with the combination of JAX and matplotlib?

We should also in this plan make sure we are fully confident of the interface between PyAutoFit JAX, visualization
in the search and this layer, if this can be polished before doing a lot of work please do.