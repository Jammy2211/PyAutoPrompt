Cluster visualization: deep research + prototype script.

The new cluster simulator at `@autolens_workspace/scripts/cluster/simulator.py` outputs `.png` plots using the
default galaxy-scale plotters (`aplt.subplot_imaging_dataset`, `aplt.subplot_tracer`,
`aplt.subplot_galaxies_images`, `aplt.subplot_point_dataset`). These plots are known to be **suboptimal for
cluster-scale systems** — a 100"x100" field with 5 members, a DM halo and 2 sources looks crowded and
hard to parse with the defaults.

The goal of this prompt is **not** to build polished cluster plotters yet. It is to *research* what cluster
visualization should look like, and then commit a prototype script in
`autolens_workspace_test/scripts/cluster/visualization.py` that exercises the ideas against the output of
`cluster/simulator.py`. Any library-side plotter changes land in a later prompt, guided by what the prototype
proves out.

__Research phase — before writing any code__

Read the existing plotter stack so you know what's already available to build on:

- `@PyAutoGalaxy/autogalaxy/plot/` — the base `aplt` interface shared by autogalaxy and autolens, especially the
  `MatPlot2D` / `MatPlot1D` configuration classes and the `Include2D` / `Visuals2D` infrastructure for adding
  overlays (positions, critical curves, caustics).
- `@PyAutoLens/autolens/plot/point_dataset_plotters.py` (and sibling files) — current `PointDataset` plotters.
- `@autolens_workspace/scripts/point_source/start_here.py` — galaxy-scale point-source plotting, which is the
  current baseline.

Then do deep research on how cluster strong lensing is visualized in the literature and in other codebases.
Concrete things to consider:

- **Field size and panel layout**. Cluster fields are 100"x100" or larger. Single-panel imaging at that scale
  washes out fine structure. Think about zoom-in inserts (BCG, each source's image group), multi-panel layouts
  tiling one panel per source, and whether the critical-curve overlay should cover the whole field or a
  zoom-in around each source.
- **Per-source colouring**. With N sources, the current `aplt.subplot_point_dataset` plots each source on its
  own axes. For cluster visualization you often want *all* sources overlaid on a single image of the cluster,
  with each source's multiple images shown in a distinct colour. Think about palette choices (qualitative,
  colour-blind safe, ≤ 8 distinct colours before resorting to marker shape) and a legend.
- **Per-source subplots from a `List[PointDataset]`**. A utility that takes a list of datasets, a tracer, and
  optionally an imaging dataset, and produces a grid of one subplot per source (image + positions overlaid,
  zoomed to the image group) would be useful. Consider whether this belongs in `aplt` or as a workspace helper.
- **Critical curves and caustics at cluster scale**. The critical curve of a cluster is complex — a large
  outer curve plus smaller inner curves around each member. Think about line weight, colour, opacity (the
  default probably overwhelms the image), and whether to omit caustics entirely (they live in source plane,
  often not useful at cluster scale).
- **Host-halo and BCG markers**. Cluster plots often mark the BCG centre and the fitted halo centre with
  distinct markers. Consider whether this is a `Visuals2D` extension or a workspace helper.
- **Axis labels and scale bars**. Arcsec axes are fine, but also consider a physical kpc scale bar for clusters
  (requires cosmology + lens redshift — the halo carries both via `NFWMCRLudlowSph.redshift_object`).
- **Data ranges**. Cluster imaging has extreme dynamic range — bright BCG, faint diffuse ICL, bright arcs.
  `cmap="gnuplot2"` with a stretch (`Normalize` / `LogNorm`) often beats the default linear `Greys_r`.

Collect the research into a short "Design notes" section in the prototype script's top-level docstring. Do not
spin up a new design doc or add files outside `autolens_workspace_test/`.

__Prototype script — `autolens_workspace_test/scripts/cluster/visualization.py`__

Load the cluster simulator output and exercise each idea from the research phase:

1. Load `point_datasets.csv` via `al.list_from_csv` to get `List[PointDataset]`; load `tracer.json` and
   `data.fits` + `noise_map.fits` + `psf.fits` via the usual workspace loader.
2. Produce an **overlaid positions plot**: the cluster image with *every* source's positions overlaid, each in
   its own colour, with a legend. Prefer `matplotlib.pyplot` directly if the current `aplt` API cannot express
   this yet.
3. Produce a **per-source subplot grid**: one panel per source, each zoomed to its image group, with the
   positions overlaid on the zoomed cluster image. This is the hardest piece and is the main deliverable.
4. Produce a **critical-curve overlay** at cluster scale with tuned line weight / alpha — adjust the defaults
   so the curve informs rather than dominates.
5. For each plot, save to `dataset/cluster/simple/visualization_<name>.png`.

Each plot block should have a short docstring explaining *why* this view is useful at cluster scale and *what's
wrong* with the default `aplt` equivalent. The script's value is as a reference for later library work, not as
a polished figure dump.

__Scope guardrails__

- **Do not** modify `aplt` plotters, `Visuals2D`, or `Include2D` in `PyAutoLens`/`PyAutoGalaxy` in this prompt.
  Library-side plotter changes are a follow-up, informed by what the prototype surfaces.
- **Do not** write new cluster modeling scripts or touch `scripts/cluster/modeling.py` / `start_here.py` —
  those are parked in `no_run.yaml` and are outside this prompt's scope.
- **Do not** edit `autolens_workspace_test/scripts/cluster/` beyond creating `visualization.py` and (if needed)
  an `__init__.py`. Keep it small.
