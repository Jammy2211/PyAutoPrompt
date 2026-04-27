The - RectangularMagnification light adaptive pixelization visualization show signs of issue.

In the file @autolens_workspace_develope/@plotting_alignment, I have made lots of visuals of the
rectangular mesh's source reconstruction. I see small offsets compared to plots of sources using other methods,
which I trust more.

Thereofre I think this plotting code @PyAutoArray/autoarray/plot/inversion.py:

        y_edges, x_edges = mapper.mesh_geometry.edges_transformed.T
        Y, X = np.meshgrid(y_edges, x_edges, indexing="ij")
        im = ax.pcolormesh(
            X,
            Y,
            pixel_values.reshape(shape_native),
            shading="flat",
            norm=norm,
            cmap=colormap,
        )
        _apply_colorbar(im, ax, is_subplot=is_subplot)


May be causing a small shift or misalignment?

Can you give me an assesment of of the code and if you think such an issue is feasible?