# Optimize Python Import Times

## Motivation

Smoke test profiling (issue rhayes777/PyAutoFit#1183) has reduced per-script runtime from ~100s to 3.6-5.3s. The remaining breakdown for `imaging/modeling.py` (5.3s total):

| Component | Time | % |
|-----------|------|---|
| **Python imports** | **2.6s** | **49%** |
| Simple model composition | 0.6s | 11% |
| search.fit (1 likelihood) | 0.6s | 11% |
| Subplot rendering (matplotlib) | 1.0s | 19% |
| Everything else | 0.5s | 10% |

Import time is now the single largest cost — nearly half the total. For `imaging/simulator.py` (3.6s total), imports are 2.3s (64%).

The import breakdown (from earlier profiling):

| Package | Time |
|---------|------|
| autofit | 0.8s |
| autoarray | 0.3s |
| autogalaxy | 1.0s |
| autolens | 0.02s |
| autoconf.jax_wrapper | 0.08s |

autogalaxy (1.0s) and autofit (0.8s) dominate. These likely pull in heavy dependencies at module level (scipy, numba, astropy, matplotlib).

## Approach

Investigate lazy imports for heavy dependencies that aren't needed at module load time. Candidates:
- `scipy` — only needed for specific operations (convolution, interpolation)
- `numba` — only needed when JIT functions are first called
- `astropy` — only needed for FITS I/O and cosmology
- `matplotlib` — only needed when plotting functions are called

The goal is to defer these imports until first use, reducing the ~2.6s import floor. Even a 50% reduction would bring modeling scripts under 4s and simulator scripts under 2.5s.

## Context

All other smoke test optimizations have been implemented:
- `PYAUTO_WORKSPACE_SMALL_DATASETS=1` — 15x15 grids, over_sample_size=2, 2 MGE gaussians
- `PYAUTO_FAST_PLOTS=1` — skip tight_layout + savefig
- `PYAUTO_DISABLE_CRITICAL_CAUSTICS=1` — skip curve overlays
- `PYAUTO_DISABLE_JAX=1` — skip JAX JIT compilation
- `PYAUTOFIT_TEST_MODE=2` — skip VRAM, model.info, result.info, pre/post-fit I/O
- Cosmology distance caching, FitImaging cached_property

Import optimization is the last remaining lever for further speedup.
