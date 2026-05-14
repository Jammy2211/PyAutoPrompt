Multi-dataset fits, for example that shown in autolens_workspace/scripts/multi/features/dataset_offsets/modeling.py,
use the DatasetModel object to implement shifts in the centre of datasets between one another in modeling.

However, the autogalaxy_workspace_test and autolens_workspace_test do not do any JAX testing of this feature,
even though both have jax_likelihood_functions/multi packages that would suit this.

Thus, to each jax_likelihood_functions/multi package, can you add a file dataset_model.py which ensures this
all works against JAX when there is more than one dataset?