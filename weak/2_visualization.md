Step 2 of the weak gravitational lensing series. Step 1 (`1_simulator.md`, shipped via PyAutoLens issue #472) added the `WeakDataset` data class and the `SimulatorShearYX` simulator, plus a workspace simulator script at `@autolens_workspace/scripts/weak/simulator.py` with `# TODO(2_visualization.md)` placeholders where the plots should go. This prompt adds the visualization layer.

The natural matplotlib primitive for shear fields is `plt.quiver` — each shear vector (gamma_2, gamma_1) at position (y, x) becomes an arrow whose length is the shear magnitude `|gamma| = sqrt(gamma_1**2 + gamma_2**2)` and whose orientation is the position angle `phi = 0.5 * arctan2(gamma_2, gamma_1)`. For weak-lensing science papers the convention is usually to draw "headless" line segments rather than arrows (since shear is a spin-2 quantity, not a vector — a 180-degree rotation maps shear to itself). Quiver supports this via `headwidth=0, headlength=0`.

The `ShearYX2D` data structures already expose what you need: `.ellipticities`, `.semi_major_axes`, `.semi_minor_axes`, `.phis`, and `.elliptical_patches`. Use these directly where useful.

Please:

1. Look at how plotters are structured in `@PyAutoLens/autolens/plot` and how `aplt.subplot_imaging_dataset`, `aplt.plot_array`, and similar helpers are wired through `autolens.plot` and `autoarray.plot`. The same module structure should host weak-lensing visualization.

2. Add a `WeakDatasetPlotter` (analogous to `ImagingPlotter`) which can produce:
   - A scatter / quiver plot of the shear field on top of the source-galaxy positions.
   - A subplot mosaic showing the shear, its noise map, and (if useful) the ellipticities and position angles.
   - A `subplot_weak_dataset` helper at the `aplt` namespace level mirroring `aplt.subplot_imaging_dataset`.

3. Replace the `# TODO(2_visualization.md)` placeholders in `@autolens_workspace/scripts/weak/simulator.py` with real calls to the new plotters, plus a `# %%` markdown cell explaining what's being shown.

4. Add unit tests for the plotter (just check it runs without raising under the workspace test mode flags, like the existing `ImagingPlotter` tests do).

5. The `ShearYX2D` `[:, 0] = gamma_2`, `[:, 1] = gamma_1` storage convention pinned by PyAutoGalaxy PR #366 must be respected by the plotters. Use `.phis` / `.ellipticities` rather than indexing directly, so the plotters keep working if downstream we ever swap the storage order.

Good prior art outside the codebase: most weak-lensing papers (e.g. KiDS, DES) draw shear fields as a regular grid of headless line segments scaled by `|gamma|`. The `quiver` arguments `pivot="middle", headwidth=0, headlength=0, headaxislength=0` produce exactly that.
