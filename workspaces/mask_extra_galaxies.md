The followin gfiles crash in smoke tests due to not having mask_extra_galaxies.fits:

autogalaxy_workspace:

  - scripts/imaging/start_here.py
  - scripts/imaging/features/extra_galaxies/modeling.py
  - scripts/imaging/features/pixelization/{fit,modeling}.py

I assume autolens_worksace has files trying to and failing to load mask_extra_galaxies.fits too

Can you expand their auto-dataset sections to run scripts/  imaging/data_preparation/optiona/examples/mask_extra_galaxies.py
so that the mask is auto simulate,d with docstring explaining why?

Can you aslo check the extra galaxies are in their simulator.py fiels and in the place the data preparation scritp
masks out?