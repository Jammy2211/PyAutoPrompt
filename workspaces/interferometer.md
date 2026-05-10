PReviously, the interferometer examples assumed that it could not efficiently do modeling of light profiles
and linear light profiles, because TransformerNUFFT did not support JAX. However, a recent addition was to add
nufftax which eanbles this.

This means that statements like this in interferometer/start_here.py no long apply:

__Number of Visibilities__

This example fits a **low-resolution interferometric dataset** with a small number of visibilities (273). The
dataset is intentionally minimal so that the example runs quickly and allows you to become familiar with the API
and modeling workflow. The code demonstrated in this example can feasible fit datasets with up to around 10000
visibilities, above which computational time and VRAM use become significant for this modeling approach.

High-resolution datasets with many visibilities (e.g. high-quality ALMA observations
with **millions hundreds of millions of visibilities**) can be modeled efficiently. However, this requires
using the more advanced **pixelized source reconstructions** modeling approach. These large datasets fully
exploit **JAX acceleration**, enable lens modeling to run in **hours on a modern GPU**.

If your dataset contains many visibilities, you should start by working through this example and the other examples
in the `interferometer` folder. Once you are comfortable with the API, the `feature/pixelization` package provides a
guided path toward efficiently modeling large interferometric datasets.

The threshold between a dataset having many visibilities and therefore requiring pixelized source reconstructions, or
being small enough to be modeled with light profiles, is around **10,000 visibilities**.


Its still true that pixelizations do not use NUFFT, but the key points are:

- There is no limit to the number of visilbities and VRAM now we can use a NUFFT when we model light profiles.
- You do not need to use pixelizations to model large visiblities datasets.
- In scripts which illustrate using light profiles, descriptions about them being slow are no longer true.

Furthermore, there are certain design considerations whcihc can now be updated, for example interferometer/features/pixelization/slam.py
we have this:

"""
__SOURCE PIX PIPELINE 1__

Unlike `slam_start_here.py`, this pipeline does not use a `source_lp` pipeline before the pixelized source
pipeline. This is because fitting light profiles to interferometer datasets with many visibilities is slow.

The search therefore uses a `Constant` regularization (not adaptive) as there is no adapt image available.
"""

We can now re add SOURCE LP PIPELINE 1 (see imaging slam.py for example) as the light profile fitting is now fast again
thanks to the NUFFT.

Do a scan of both interferometer packages (autolens_workspace and autogalaxy_workspace), update all this context
which says its slow which is no longer true, high up in start_here.py detail and describe nufftax pointing
to its github and giving it credit and generally flesh out the interferomeer packages in light of this update.