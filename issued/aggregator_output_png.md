  Title: AggregateImages.output_to_folder — subplots from different source images with different grid sizes produce mismatched panel sizes

  Problem:

  When combining subplots from two different source images that have different grid layouts, the extracted panels end up at different physical sizes in the final composite PNG.

  Concrete example: I'm combining panels from rgb.png (a 2x2 grid) and subplot_fit.png (a 4x3 grid). The RGB panel is extracted at its native size from the 2x2 grid, while the fit panels are extracted from
  the larger 4x3 grid. Because each source image has a different total size and grid cell count, the individual panels end up at different pixel dimensions. When they're stitched side-by-side in the output,
  the RGB panel appears visibly smaller (roughly half the height/width) compared to the fit panels.

  Expected behaviour:

  All panels in the final composite should appear at the same size, regardless of which source image or grid layout they came from. Either:

  1. AggregateImages should automatically rescale all extracted panels to a common size before compositing, or
  2. There should be a user-facing parameter (e.g. panel_size=(width, height) or normalize_panel_size=True) that controls this.

  Where to look:

  - AggregateImages class and its output_to_folder method — this is where panels from different source images are stitched together.
  - The subplot extraction logic that crops panels from the source PNGs using the Enum grid positions — this is where the panel pixel dimensions diverge because each source image has a different grid.
  - The final compositing/concatenation step where extracted panels are placed side-by-side.

  Suggested fix:

  After extracting all panels (from whichever source images), resize them to a common target size before concatenation. The target could be the max height/width across all panels, or user-specified. PIL's
  Image.resize() with LANCZOS resampling would preserve quality.

  ---
  Want me to open this as a GitHub issue on the autofit repo?