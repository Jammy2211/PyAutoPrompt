- Ellipse fitting is defined in @PyAutoGalaxy/autogalaxy/ellipse, with scientific walk throughts in
@autogalaxy_workspace/scripts/ellipse .

It currently does not support JAX, the final goal is to add JAX support through the likelihood function, such that
@PyAutoGalaxy/autogalaxy/ellipse/model/analysis.py, Analysis.log_likelihood_function is JAX compatible.

You can refer to @PyAutoGalaxy/autogalaxy/imaging/model/analysis.py to see how JAX is supported in an
analysis class.

Before starting JAX work, can you make integration tests in @autogalaxy_workspace_test/scripts, in particular:

- visualization.py to test the ellipse/visualization of the ellipse fitting.
- jax_likelihood_functions/ellipse to test the JAX likelihood function of the ellipse fitting.
- Note how this will give a step by step guide of the code in numpy, which are then the steps we need to convert to support JAX.


There are a few nasty and poorly written loops which will ned careful conversion to JAX, these are unit tests
but lets make double sure we keep our numerics in the test workspace and so that when these are changed
we dont lose functionality:

        if self.interp.mask_interp is not None:

            i_total = 300

            total_points_required = points.shape[0]

            for i in range(1, i_total + 1):

                total_points = points.shape[0]
                total_points_masked = np.sum(self.interp.mask_interp(points) > 0)

                if total_points_required == total_points - total_points_masked:
                    continue

                if total_points_required < total_points - total_points_masked:

                    number_of_extra_points = (
                        total_points - total_points_masked - total_points_required
                    )

                    unmasked_indices = np.where(self.interp.mask_interp(points) == 0)[0]
                    unmasked_indices = unmasked_indices[number_of_extra_points:]

                    points = points[unmasked_indices]

                    continue

Adding specific unit tests on this before the conversion to something which supports JAX Is adviced.

Note that we will need the Drawer searchin autofit to support JAX jit.

Put this all together as a sequence of prompts which we run as a feature in z_features (analogous to [weak_shear.md](../z_features/weak_shear.md))

Do deep research and thikning on what else is required for this feature before makin the final plan.