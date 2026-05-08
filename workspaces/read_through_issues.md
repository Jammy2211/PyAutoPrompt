autolens_workspace:

scripts/group/fit.py: The script loads main_lens_centres but doesnt use them in fit.py, they are the right centres (But check they are otuput via simulator.py) and then make them used
- scripts/group_likelihood_function.py this should also use the .json file containing main lens centres.
- compare scripts/group/slam.py and scripts/group/features/pixelization/slam.py, I think the latter is just an older defunct version that can be replaced with the former. Do you agree? I only want a slam.py file in group/features/pixelization/slam.py so ultimately just reduce this to one there, copying its documentation and style from scripts/imaging/features/pixelization/slam.py. Basically the group slam should still rely on guides/slam_start_here.py, but will need more context for all the specific group steps, I think that is what the current group/slam.py is :).

- group/featurs/linear_light_profiles.py has a good over sampling section at the top:
"""
__Over Sampling__

Over sampling at each galaxy centre ensures that light profiles are accurately evaluated.
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

put this at the top of group/fit.py but with a slightly more expansive docstring like:

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

For group-scale lenses, we apply adaptive over-sampling at the centres of all galaxies in the group, including
both the main lens galaxies and the extra galaxies. This ensures that the light profiles of every galaxy
in the group are accurately evaluated.




scripts/imaging/modeling.py has this part at the end:

"""
__Loading From Output Folder__

Everything the `Result` object contains has also been written to hard-disk, inside the fit's output folder. Each
file loads back into a full Python object with a single line — much faster and simpler than re-running the fit.

For example, the maximum log likelihood `Tracer` is saved as a `.json` file and the tracer image-plane images as
a `.fits` file:
"""
from autoconf.dictable import from_json

result_path = search.paths.output_path  # Points at the fit's unique output folder.

if (result_path / "files" / "tracer.json").exists():
    tracer = from_json(file_path=result_path / "files" / "tracer.json")

    tracer_fits = al.Array2D.from_fits(
        file_path=result_path / "image" / "tracer.fits", hdu=0, pixel_scales=0.1
    )

Make sure other modeling.py examples at the base of packages (e.g. group, cluster, interferometeR) have the equivalent section.