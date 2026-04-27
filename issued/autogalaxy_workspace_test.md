The autogalaxy_workspace_test is very light compared to autolens_workspacd_test in terms of its script
and it does not have a .github actions continuous integration. Therefore, plan the following work as a series
of prompts in this folder, run them one after another as seaprate agent tasks:

- Set up contiuous integration server on autogalaxy_workspace_test
- Set up folder `scripts/model_composition` and set up autogalaxy version of existing scripts.
- Set up folder `scripts/jax_likelihood_functions/imaging` and set up autogalaxy versions of all examples (dont do dspl). This will likely require some spawn-off tasks which do pytree registration so make sure this prompt looks at our recent pytree work).
- Do same for `scripts/jax_likelihood_functions/interferometer`.
- Do same for `scripts/jax_likelihood_functions/multi`.
- Set up folder `scripts/jax_grad/imaging` and set up autogalaxy versions of all examples (dont do dspl). This will likely require some spawn-off tasks which do pytree registration so make sure this prompt looks at our recent pytree work).
- Do same for `scripts/jax_grad/interferometer`.
- Do same for `scripts/jax_grad/multi`.
- Set up a `scripts/imaging` folder like autolens_workspace_test's, but only include model_fit.py, modeling_visualization_jit.py, visualization.py, visualization_jax.py