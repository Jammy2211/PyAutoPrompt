The PointDataset object in @PyAutoLens/autolens/point was designed with single lensed point sources in mind, which means
single inputs or a small  number of inputs are used to use it, see @autolens_workspace/scripts/point_source/simulator.py .

I am about to add strong lens cluster functionality, which means that it will be common for their to be tens or hundreds
of multiple image positions that required modeling. These should still use the PointDataset class, I think it
serves the right purpose and interfaces with the FitPoint objects correctly, but we need to design the API
to scale.

I think we need the following:

1) A .csv interface to load a PointDataset, which of course means it also needs to output to .csv, which is
what a simulator.py file may output to. The .json file outputs are not clean for a user to edit and update where a .csv
basically means they can edit things in a table.

2) However, remember that lists of PointDatasets are used for modeling and other tasks (see @autolens_workspace/scripts/point_source/)
see [INSERT EXAMPLE HERE]. So this .csv interface should really pair with lists of PointDatasets.

3) Also look at the FitPOint objects and assess if they are appropriate for examples where we have lots of POintDatasets
and if they would make sense to have a .csv interface too, or if thats too much complexity.