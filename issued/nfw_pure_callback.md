Certain dark matter profiles in mcr_util.py (autogalaxy/profiles/mass/dark) use a pure_callback to do a calculation:

def ludlow16_cosmology_jax(
    mass_at_200,
    redshift_object,
    redshift_source,
):
    """
    JAX-safe wrapper around Colossus + Astropy cosmology.
    """
    import jax
    import jax.numpy as jnp
    from jax import ShapeDtypeStruct

    return jax.pure_callback(
        _ludlow16_cosmology_callback,
        (
            ShapeDtypeStruct((), jnp.float64),  # concentration
            ShapeDtypeStruct((), jnp.float64),  # rho_crit(z)
            ShapeDtypeStruct((), jnp.float64),  # Sigma_crit
            ShapeDtypeStruct((), jnp.float64),  # kpc/arcsec
        ),
        mass_at_200,
        redshift_object,
        redshift_source,
        vmap_method="sequential",
    )

Thius is required because the library collosus is not JAX native, and it is not easy to make it JAX native.

Can you inspect the collosus source code and work out whether extracintg the specific functionality we need and making
it JAX native is feasible? Alternatively, can you assess if the same calculation can be done using a simpler Python
fcuntion does doesntly necesssrily use the collosus code? 