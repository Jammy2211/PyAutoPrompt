This:

__Output Folder Layout__

Each completed fit lives at a path like::

    output/imaging/<dataset_name>/modeling/<unique_hash>/
        files/                     <- JSON + CSV: loadable Python objects
            tracer.json            <- max log likelihood Tracer
            model.json             <- fitted af.Collection model
            samples.csv            <- full Nautilus samples
            samples_summary.json   <- max log likelihood parameter values + errors
            samples_info.json      <- metadata about the samples
            search.json            <- non-linear search configuration
            settings.json          <- search settings
            cosmology.json         <- cosmology used for the fit
            covariance.csv         <- parameter covariance matrix
        image/                     <- FITS: imaging products
            dataset.fits           <- data, noise-map and PSF
            fit.fits               <- model image, residuals, chi-squared map
            tracer.fits            <- tracer image-plane images per galaxy
            source_plane_images.fits  <- source plane reconstructions
            model_galaxy_images.fits  <- per-galaxy model images
            galaxy_images.fits        <- per-galaxy images
            dataset.png, fit.png, tracer.png   <- visualisations
        model.info                 <- human-readable model summary
        model.results              <- human-readable fit summary
        search.summary             <- search run summary
        metadata                   <- run metadata

Is far superior to this in modeling.py files (and others, do a search):

__Output Folder__

Now this is running you should checkout the `autolens_workspace/output` folder. This is where the results of the 
search are written to hard-disk (in the `start_here` folder), where all outputs are human readable (e.g. as .json,
.csv or text files).

As the fit progresses, results are written to the `output` folder on the fly using the highest likelihood model found
by the non-linear search so far. This means you can inspect the results of the model-fit as it runs, without having to
wait for the non-linear search to terminate.
 
The `output` folder includes:

 - `model.info`: Summarizes the lens model, its parameters and their priors discussed in the next tutorial.
 
 - `model.results`: Summarizes the highest likelihood lens model inferred so far including errors.
 
 - `image`: Visualization of the highest likelihood model-fit to the dataset, (e.g. a fit subplot showing the lens 
 and source galaxies, model data and residuals) in .png and .fits formats.
 
 - `files`: A folder containing human-readable .json file describing the model, search and other aspects of the fit and 
   a `.csv` table of every non-linear search sample.
 
 - search.summary: A file providing summary statistics on the performance of the non-linear search.
 
 - `search_internal`: Internal files of the non-linear search (in this case Nautilus) used for resuming the fit and
  visualizing the search.

Update the modeling.py files to include the full thing you listed!



However, in results/start_here.py, you use this to load results:

result_path = (
    Path("output")
    / "imaging"
    / "simple"
    / "modeling"
    / "<unique_hash>"  # The 32-character identifier for the specific fit.
)

instead of the search path:

result_path = search.paths.output_path  # Points at the fit's unique output folder.

if (result_path / "files" / "tracer.json").exists():
    tracer = from_json(file_path=result_path / "files" / "tracer.json")

    tracer_fits = al.Array2D.from_fits(
        file_path=result_path / "image" / "tracer.fits", hdu=0, pixel_scales=0.1
    )


We do not want <unique_hash> to be something anywhere in these examples, can you make it so
results/start_here.py runs the same analysis and search as aggregator/start_here.py, and loads the results using
the seaerch path in the same way.