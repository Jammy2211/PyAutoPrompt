# Free-parameter subhalo redshift breaks under JAX (`TracerBoolConversionError`)

## Reporter

Reported on Slack by **@qiuhan96** (with an undergraduate at the University of Groningen). Running the public pip release of `autolens` on Python 3.13.

## Symptom

Setting the subhalo's redshift as a free parameter (a `af.UniformPrior`) raises a `jax.errors.TracerBoolConversionError` during model fitting. The fit only runs if `use_jax=False` is set, which is much slower.

## Reproduction

@RepoName: PyAutoLens

```python
import autofit as af
import autolens as al

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Subhalo
subhalo_mass = af.Model(al.mp.IsothermalSph)
subhalo_mass.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
subhalo_mass.centre_1 = af.UniformPrior(lower_limit=1.2, upper_limit=1.8)
subhalo_mass.einstein_radius = af.UniformPrior(lower_limit=0.01, upper_limit=0.4)

# Trigger: free-parameter redshift
redshift_subhalo = af.UniformPrior(lower_limit=0.2, upper_limit=0.9)
# redshift_subhalo = 0.6   # <-- works fine

subhalo_galaxy = af.Model(al.Galaxy, redshift=redshift_subhalo, mass=subhalo_mass)

# Source
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, subhalo=subhalo_galaxy, source=source))
```

When `redshift_subhalo` is a `UniformPrior`, the fit raises:

```
File autolens/analysis/analysis/lens.py:99, in AnalysisLens.tracer_via_instance_from
    subhalo_centre = tracer_util.grid_2d_at_redshift_from(
        galaxies=instance.galaxies,
        redshift=instance.galaxies.subhalo.redshift,
        ...
    )
File autolens/lens/tracer_util.py:247, in grid_2d_at_redshift_from
    plane_redshifts = plane_redshifts_from(galaxies=galaxies)
File autolens/lens/tracer_util.py:46, in plane_redshifts_from
    galaxies_ascending_redshift = sorted(galaxies, key=lambda galaxy: galaxy.redshift)
...
jax._src.core.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
```

Workaround: setting `use_jax=False` lets the fit run, but is much slower.

## Root cause

`@PyAutoLens/autolens/lens/tracer_util.py` performs several Python-level operations on the redshift values that fail when one of them is a JAX traced scalar:

- **Line 46** — `sorted(galaxies, key=lambda g: g.redshift)` does pairwise `<` comparisons on Python objects holding traced redshifts.
- **Line 49** — `[float(g.redshift) for g in ...]` calls `float()` on a traced scalar.
- **Line 249** — `if redshift <= plane_redshifts[0]:` is a Python branch on a traced boolean.
- **Line 257** — `[plane_index for ... if galaxies[0].redshift == redshift]` filters on a traced boolean.
- **Line 267-268** — `for ...: if redshift > plane_redshift: plane_index_insert = plane_index + 1` again branches on a traced boolean and uses a Python integer to index the inserted plane.

`grid_2d_at_redshift_from` is called from `@PyAutoLens/autolens/analysis/analysis/lens.py:99` whenever `instance.galaxies.subhalo` exists, with `redshift=instance.galaxies.subhalo.redshift`. When that redshift is free, the value passed in is a traced array and every comparison above is illegal under `jax.jit`.

The whole helper is structured around inserting the subhalo into a Python-level ordered plane list and returning the traced grid at a Python-indexed plane position. Because the insertion index depends on the traced redshift, this control-flow pattern cannot be naïvely lifted into JAX.

## Plan

The goal is to make `tracer_via_instance_from` work with a JAX-traced subhalo redshift while keeping the numpy path identical.

1. **Reproduce** with a minimal `pytest` test case that builds an analysis with a free-parameter subhalo redshift and calls `analysis.fit_from(instance)` under JAX, asserting it does not raise.
2. **Decide the JAX-compatible reformulation** of `grid_2d_at_redshift_from`. Two viable directions:
   - Compute traced grids at *all* candidate plane positions (subhalo before plane 0, between each pair of planes, after plane N) and select the correct one with `jnp.where` / `jax.lax.switch` based on redshift comparisons that are kept as `jnp.bool_` rather than Python `bool`.
   - Or constrain priors so the subhalo's plane index is fixed (e.g. always between lens and source) and only handle that single case under JAX, falling back to the general numpy implementation otherwise. This is less general but avoids `lax.switch`.
3. **Refactor `plane_redshifts_from` / `planes_from`** so they do not call `float()` on redshift values; carry redshifts through as the array type returned by the backend, sorting by the *concrete* redshifts of the *non-subhalo* galaxies (which are always Python floats — only the subhalo redshift is traced).
4. **Update `AnalysisLens.tracer_via_instance_from`** to stay on the JAX path and not mutate `instance.galaxies.subhalo.mass.centre` with `tuple(...)` (which forces Python conversion of a traced array).
5. **Add unit tests** in `test_autolens/lens/test_tracer_util.py` covering: subhalo redshift before lens plane, between lens and source, equal to lens redshift, equal to source redshift.
6. **Verify** with the user's reproduction script that the full Nautilus / Dynesty fit runs end-to-end with `use_jax=True`.

The fix should be implemented in PyAutoLens only — `autogalaxy` and `autoarray` are unaffected.
