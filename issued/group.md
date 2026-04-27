ORIGINAL PROMPPT WITH CHANGES IMPLEMENTED

In @autolens_workspace/scripts/imaging there are a list of file which illustrate all
the different tasks a user who is modeling CCD imaging data of a strong lens can perform.

These files are speciifc to strong lenses with just one lens galaxy, thus they are not GROUP scale lenses.

In @autolens_workspace/scripts/group the first scripts describing group scale modeling are available.

I want to put particular emphasis on @autolens_workspace/scripts/group/lam.psy, as this has the
group model composition API all examples will use in the group package from here in, where the lenses are
formed via lists:

Can you first update @autolens_workspace/scripts/group/start_here.py to use this new API,
where for the lens 102021990_NEG650312660474055399 we will trreat both its lens galaxies,
whose centres are (0.0, 0.0) and one in "extra_galaxies_centres.json", are both treated as main lens galaxies.
This is a small API shift, as before we treated the second lens galaxy as an "extra" galaxy. For now
write the tutorial without mention of extra galaxies and scaling galxies, just make it two main galaxies.
This will require an update to the folder @autolens_workspace/dataset/group/102021990_NEG650312660474055399).

When we are done, we'll then add a description of extra galaxies and scaling galaxies.

Now, can you compare: 

- @autolens_workspace/scripts/imaging/simulator.py to @autolens_workspace/scripts/group/simulator.py.
- @autolens_workspace/scripts/imaging/modeling.py to @autolens_workspace/scripts/group/modeling.py.

And update the group scripts to use the new API, where the lens is formed via lists of galaxies, rather than a single galaxy and a list of extra galaxies.

Now can you read all of the following which are only in imaging:

- @autolens_workspace/scripts/imaging/fit.py -> make group equivalent with main, extra and scaling galaxies.
- @autolens_workspace/scripts/imaging/likelihood_function -> make group equivalent with main, extra and scaling galaxies.
- @autolens_workspace/scripts/imaging/source_science.py -> make group equivalent with main, extra and scaling galaxies.
- @autolens_workspace/scripts/imaging/data_preparation -> make group equivalent with main, extra and scaling galaxies, noting many tools should document use of multiple galaxies.

We'll consider the folder @autolens_workspace/scripts/imaging/features once this is all done.


NEXT UPDATED PROMPOT:

You did not implement this change correct:

Can you first update @autolens_workspace/scripts/group/start_here.py to use this new API,
where for the lens 102021990_NEG650312660474055399 we will trreat both its lens galaxies,
whose centres are (0.0, 0.0) and one in "extra_galaxies_centres.json", are both treated as main lens galaxies.
This is a small API shift, as before we treated the second lens galaxy as an "extra" galaxy. For now
write the tutorial without mention of extra galaxies and scaling galxies, just make it two main galaxies.
This will require an update to the folder @autolens_workspace/dataset/group/102021990_NEG650312660474055399).

To be clear, I want the start_here.py example to not include extra galaxies at all in the fit to 102021990_NEG650312660474055399,
but insttead treat both lens galaxies as main lens galaxieis. This should use the model composition API give
in @autolens_workspace/scripts/group/slam.py.

The start hre guide should first show the single lens gaalxy model API found in @autolens_workspace/scripts/group/start_here.py,
for simple illustration, but then immediately showe how it extends to the multiple lens galaxies. E.g. show it
just for illustration before introducing the actual API used for a group.

Can you also think a bit hard about making sure the language and explanation is a natural explanation of strong lensing
and group scale lenses -- make sure the way descriptions pair to the single lens case in imaging and other
docs is clear.


- update  @autolens_workspace/scripts/group/simulator.py. so that it has two main lens galaxies and two extra galaxies.
- update @autolens_workspace/scripts/group/modeling.py, which models the output of the simulator, to have two main lens galaxies and two extra galaxies.
- update @autolens_workspace/scripts/group/fit.py, which fits the output of the simulator, to have two main lens galaxies and two extra galaxies.
- update @autolens_workspace/scripts/group/likelihood_function.py, to also be on the two main lens galaxy and two extra galaxy API.













THIS WORK:

- Make it so group/simulator.py has two main lens galaxies, not just one. Actually do it this time, and confirm to me
in your response you are doing it. Then do these:
- 
- update @autolens_workspace/scripts/group/modeling.py, which models the output of the simulator, to have two main lens galaxies and two extra galaxies.
- update @autolens_workspace/scripts/group/fit.py, which fits the output of the simulator, to have two main lens galaxies and two extra galaxies.
- update @autolens_workspace/scripts/group/likelihood_function.py, to also be on the two main lens galaxy and two extra galaxy API.

For modeling.py, I think the centre of every Isothermal should be fixed to its respective main lens galaxy
centre or extra galaxy centre. I think we should do the same for the centre of each of their respecitve MGEs.
We should explain this is not necessarily how one should do it for gneuine scientific study but its good for
illustration here. When you get to this, prompt me to write a paragraph or two explaining centre placement in group modeling.