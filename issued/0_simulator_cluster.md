I am going to add cluster stong lens analysis to autolens, whcih autolens can already do but needs a series of 5 or so 
claude prompts to get us to a point where it good and mature.

First, we need to be able to simulate strong lens cluster lensing data. The good news is, the ability to generate 
the required point source images  from the source code is already possible. We basically just take the file
@autolens_workspace/scripts/point_source/simulator.py and make it have many more lens galaxies, reading through
@autolens_workspace/scripts/groups/simulator.py to see how the API For multiple lens gaalxies begins to add up
as well as how we output images of the cluster we simulator (but modeling will use point sources).

For now, can you assume the system has 5 main lens galaxies, 2 multiply imaged source galaxies, a host dark matter
halo which is not tied to any indivdiual lens gaalxy but included in the model "on top", with a mass of 1e14.5.
This is not many galaxies for a cluster, but we will build everything for a small cluster and then rework the
example to full complexity.

Use the .csv interface of the PointDataset when you write simulator.py and document it explaining how a user can
manually adjust the .csv tables to set up their data, which is more practical than python code or other options.

In cluster strong lensing, its common to use PIEP, so use the mass prpfile in [dual_pseudo_isothermal_mass.py](../../../PyAutoGalaxy/autogalaxy/profiles/mass/total/dual_pseudo_isothermal_mass.py)
and try to work out what good physical values for the system are.

The simulator script will output visualization, but they will not be optimal for a cluster which is has different
requirements for good viewing than the galaxy scale lenses visualization has been designed for. So output the visuals 
but we will then improve on them in the next claude prompt. In fact, make a follow up prompt cluster/1_visualizaiton.md,
that does some deep research to think abou the requiremnets of cluster visualization for us to work through
(for example, images are much larger in size, need individual multiple images plotted on in different colors,
Maybe needs functionality to plot the individual images on a subplot from the PointDataset list). Also have this
prompt make an autolens_workspace_test/scipts/cluster/visualization.py file.