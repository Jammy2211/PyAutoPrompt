I currently have source code repos (PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens) and
workspaces (autofit_workspace, autogalaxy_workspace, autolens_workspace).

These are buuilt into the claude dev cycle, including ship_library and ship_workspace.

I want to include @euclid_strong_lens_modeling_pipeline in the ship_workspace cycle, and in the PyAutoBuild deployment.
This thing should be maintained and up to date as the core workspasce repos.

Thus, can you first add a .github actions to it, which runs after ships like the workspace github actions do.

Can you also give it github actions following similar logic to things like autolens_workspace which now
uses github actions.