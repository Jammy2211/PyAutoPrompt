The imaging `features/scaling_relation` example needs improving and padding out before adapting to group.

Once the imaging version is more complete, adapt it to the group context in `scripts/group/features/scaling_relation/`.

For group lenses, scaling relations are especially important: they allow many extra galaxies to share
a luminosity-to-mass relation (einstein_radius = scaling_factor * luminosity^scaling_relation),
keeping the model dimensionality low even as galaxy count grows. The group/slam.py already implements
scaling galaxies — the feature script should document this API in a standalone, beginner-friendly way.
