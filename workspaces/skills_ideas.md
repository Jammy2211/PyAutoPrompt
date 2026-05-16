
- Euclid loading from output .fits with codex seemed to make mistakes, or at least it wasnt maybe plotting what it
thoguths and it didnt use the mask. I thinkw e need to update the result start_here.py scripts to have full hdu mapping information
for all .fits files, or at least a link to a script which provides this information explcitily. I as a user are sitll having to
instruct codex here. Codex got there with a small nudge so I think we just need to have it in the start_here (or follow up script) context.
It perhaps just need one example and the point to be more explicit that HDU information contains data type mappings and this needs to be carefully paired.

- Visualization needs dediciated skill or skills, including the plot module of the workspace being read more carefully and like having fully
mapltotlib code for customizeable figures in there. User then has option of using built in autolens visual library (simple and automated but less customization),
having the matplotlib code of these visuals being used directly for customization "on top" of autolens visuals or just to flat out do visualization themelves.
Make sure the skill knows that when two or more autolens plots are being plotted, and thus we are moving to a subplot, that it should use
- this stuff: • Yes. The built-in subplots create the full grid with figsize=conf_subplot_figsize(rows, cols) before any panel is drawn. Then each plot_array call sets the axes box aspect from the array extent, so the
  image fills its subplot cell cleanly. I’m going to mirror that: config-driven subplot size first, aplt.plot_array per axis second, then a small explicit wspace/hspace rather than constrained_layout.
- In order to avoid unecesary whitespace. • For multi-panel PyAutoLens/PyAutoArray figures, create the figure using autoarray.plot.utils.subplots(...) with figsize=conf_subplot_figsize(nrows, ncols), then pass each axis into aplt.plot_array(...)
  rather than using manually chosen Matplotlib spacing values. After all panels are plotted, call autoarray.plot.utils.tight_layout() so subplot sizing and spacing follow the same config-driven behavior as
  the built-in PyAutoLens subplot functions. There are also two use cases for using in built autolens plotting, whether you are doing lots of single arrays or a more concise inspection of things using higher level plotters (e.g. aplt.subplot_fit_imaging) that offer less customization
I guess we just ned to map out the different levels with which a user might be doing plotting and have the skill guide them -- perhaps we could have a second skill "plot_beginners_guide" which makes a single plot via aplt.plot_array, then a higher level plot via aplt_subplot_fit_imaging, the customizes aplt.plot_array, then adds a second plot to the subplot using it and then shows using your own
matplotlib code. So a user gets a vibe for the different options.



CODEX PROMPT FOR NOW:

• We are working in /home/jammy/Code/PyAutoLabs/z_projects/euclid and should build paper plotting scripts using the built-in PyAutoLens/PyAutoArray plotting API, especially autolens.plot as aplt and
  aplt.plot_array(...), rather than hand-rolled Matplotlib image plotting.
  Before making changes, read the plot_array source in /home/jammy/Code/PyAutoLabs/PyAutoArray/autoarray/plot/array.py to understand its defaults for masks, extents, colorbars, labels, and aspect handling.
  For multi-panel figures, mirror the PyAutoLens subplot setup by using autoarray.plot.utils.subplots(...) with figsize=conf_subplot_figsize(nrows, ncols), pass each axis into aplt.plot_array(...), and call
  autoarray.plot.utils.tight_layout() after all panels are plotted.
  Avoid manually chosen wspace, hspace, subplots_adjust, tick-label edits, or custom colormap/layout logic unless explicitly requested, because the goal is to preserve the same config-driven appearance as the
  built-in PyAutoLens subplot functions.