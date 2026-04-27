Continuation of the smoke test speed-up work. The first phase shipped env vars
for small datasets and disabled critical curves (autolens_workspace#34). This
phase targets the remaining PyAutoFit-level bottlenecks identified by profiling.

With PYAUTO_WORKSPACE_SMALL_DATASETS=1 and PYAUTO_DISABLE_CRITICAL_CAUSTICS=1,
the imaging and interferometer modeling scripts still take ~40-50s each under
PYAUTOFIT_TEST_MODE=2. The remaining time is dominated by:

1. **VRAM estimation: ~16s** — `analysis.print_vram_use()` does full JAX JIT
   compilation + memory analysis. Should be skipped when PYAUTOFIT_TEST_MODE >= 2.
   Located in @PyAutoFit/autofit/non_linear/analysis/analysis.py:319

2. **search.fit pre-fit I/O: ~10s** — `_fit_bypass_test_mode()` writes model.info,
   visualization, saves samples, saves results. Much of this can be skipped in
   test mode 2+. Located in @PyAutoFit/autofit/non_linear/search/abstract_search.py:790

3. **model.info formatting: ~7s** — The `info` property on MGE models with ~40
   gaussians takes 7s to format. Can be cached or stubbed in test mode.
   Located in @PyAutoFit/autofit/mapper/prior_model/abstract.py:1775

4. **Result access: ~5s** — `max_log_likelihood_instance` takes ~5s to
   deserialize. The `result_info_from()` function calls expensive
   `samples_text.summary()` twice.
   Located in @PyAutoFit/autofit/non_linear/result.py and
   @PyAutoFit/autofit/text/text_util.py:53

Target: get both scripts under 10s total with all env vars set.
