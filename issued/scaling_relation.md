I want us to update the use of galaxies on a luminosity-mass scaling relation.

firstly, the correct API is the one provided in @z_projects/euclid_group/scripts/group.py, specfically the
use of the things like scaling_lens_centres = _load_centres(dataset_path / "scaling_galaxies_centres.json")
and:

    # --- scaling galaxy light models (same MGE profile; kept separate so the two
    #     populations remain distinguishable when building stage-1 mass models) ---
    scaling_light_list = []
    for centre in scaling_lens_centres:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre=(centre[0], centre[1]),
            centre_prior_is_uniform=True,
            ell_comps_prior_is_uniform=True,
        )
        scaling_light_list.appen    d(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge)
        )

    scaling_galaxies_free = (
        af.Collection(scaling_light_list) if scaling_light_list else None
    )

Note that we have three distinct concepts in terms of extra galaxies, main galaxies, extra galaxies whichare
see in group examples (e.g. group/modeling.py) and scaling galaxies which are given by the API above.

Old outdated documentation of scaling galaxies is given at @autolens_workspace/scripts/imaging/features/scaling_relation
which gives a good textual descriptipon of what this feature is for, providijng you context, which can be used
for this taks. However, the API should more closely follow the group.py example, in particular it should show how
to use extra galaxies but also then show how to use scaling galaxides, but in the same example. Thus, I think in
imaging I think it hsould be called extra_galaxies, whereas in group we need to add a features/scaling_relation example.

Firstly, can we move the API and other things from group.py to scaling_relation/modeling.py, so it basically
provides an example using the same quantities and variables (e.g. shows how to add extra galaxies to an imaging analysis). 
Then, when done, can we make scripts/group/features/scaling_relation  which shows the use of a scaling galaxie son a group scale lens.

Can you also use the csvable module in autoconf to extend each example with the API for using the CSV api to load
the galaxies. This means you need to design a scaling_galaxies.csv file and potentially extend autoconf to make this
suitable to loading extra galaixes. Do we have this csv stuff? I think this is its own follow up feature so dont
do this work here but write the prompt which extends the work based on what you learned doing all this.