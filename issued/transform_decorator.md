The transform decorator is used on nearly every mass and light prfoile in @PyAutoGalaxy/autogalaxy/profiles,
for example:


class Isothermal(PowerLaw):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope = 2.0.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
        )

    def axis_ratio(self, xp=np):
        axis_ratio = super().axis_ratio(xp=xp)
        return xp.minimum(axis_ratio, 0.99999)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        factor = (
            2.0
            * self.einstein_radius_rescaled(xp)
            * self.axis_ratio(xp)
            / xp.sqrt(1 - self.axis_ratio(xp) ** 2)
        )

        psi = psi_from(
            grid=grid, axis_ratio=self.axis_ratio(xp), core_radius=0.0, xp=xp
        )

        deflection_y = xp.arctanh(
            xp.divide(
                xp.multiply(xp.sqrt(1 - self.axis_ratio(xp) ** 2), grid.array[:, 0]),
                psi,
            )
        )
        deflection_x = xp.arctan(
            xp.divide(
                xp.multiply(xp.sqrt(1 - self.axis_ratio(xp) ** 2), grid.array[:, 1]),
                psi,
            )
        )
        return self.rotated_grid_from_reference_frame_from(
            grid=xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T),
            xp=xp,
            **kwargs,
        )

@aa.grid_dec.transform tells the code that the input grid comes in, is shifted to the centre of the mass or light profile,
rotated by its elliptical components (if its elliptical) such that this transformed grid enters the function.

This functionality is handled in @PyAutoGalaxy/autogalaxy/profiles/geometry_profiles.py and methods used within,
a description of the transform decorator is also given at @autolens_workspace/scripts/guides/advanced/add_a_profile.py

Note that the transform decroator has some "magic" which ensrues that if the grid is already transformed (e.g. it
was already passed to a function whcih has the decorator) it is not transformed a second time. 

This is where it happens:

        if not kwargs.get("is_transformed"):
            kwargs["is_transformed"] = True

This script shows how a user have to worry about and think about this when implementing a new profile:

 @autolens_workspace/scripts/guides/advanced/add_a_profile.py



So, with this implmenetation I am in two minds:

1) This massively reduced repeated code, for example you could have the transform at the top of every profile
function but this will bloat out the code. There are ways to make the API a bit more concise, but its still as 
concise as possible I think.

2) It has too much magic, and users have reported not quite understanding how and where the transformes were performed.
It doesnt help that many users dont really understand python decorators. Repeated code would make funtions more readable
and standalone which would help users understan them better, but the source code will bulk out as a result.

Can you give me your opinion on how to approach this?
