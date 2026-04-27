  Prompt: Refactor PYAUTOFIT_TEST_MODE bypass to handle all model compositions reliably

  Background

  rhayes777/PyAutoFit#1179 added PYAUTOFIT_TEST_MODE=2 and 3 which bypass the sampler entirely and produce fake SamplesPDF results using Sample.from_lists(). This works for simple models but breaks on certain
   model compositions.

  The Core Problem

  _fit_bypass_test_mode() in abstract_search.py generates a fake parameter vector at the prior median and constructs Sample objects via Sample.from_lists(). This method uses model.unique_prior_paths to build
  the kwargs dict with tuple keys. When the result is later accessed (e.g. result.max_log_likelihood_instance), the Sample.parameter_lists_for_paths() method tries to look up each entry in model.all_paths
  within kwargs.

  This fails for certain model structures:

  1. Ellipse models (ag.Ellipse in a list): unique_prior_paths produces ('ellipses', '0', 'ell_comps', 'ell_comps_0') but the lookup path from all_paths doesn't match when the sample is loaded from the
  SamplesSummary (the second fit in a sequential chain loads from disk and the path mapping breaks).
  2. Point source / Isothermal models (al.mp.Isothermal nested in al.Galaxy): Similar tuple-path KeyError: ('galaxies', 'lens', 'mass', 'centre', 'centre_0').
  3. LBFGS/MLE searches: The bypass code writes a .completed marker and post_fit_output() tries to clean up search_internal which doesn't exist in bypass mode, causing FileNotFoundError.
  4. Sequential fits (search chaining): When a script runs multiple fits sequentially, later fits may load results from earlier bypass runs via result_via_completed_fit() → load_samples_summary(). The JSON
  roundtrip of tuple-keyed kwargs works correctly via autoconf.dictable, but the model structure at load time may differ from the model at save time (e.g. prior linking changes the model between fits).

  What Works vs What Breaks

  - Works: Simple models like af.Model(Gaussian), shared-prior MGE models (40 Gaussians, 4 unique priors), basic galaxy models with Sersic + Isothermal.
  - Breaks: Ellipse lists, point source models, any model where all_paths / all_names grouping doesn't align with unique_prior_paths keys after going through the SamplesSummary save/load cycle or search
  chaining result access.

  Affected Files

  - PyAutoFit/autofit/non_linear/search/abstract_search.py — _fit_bypass_test_mode(), _build_fake_samples(), post_fit_output()
  - PyAutoFit/autofit/non_linear/samples/sample.py — Sample.from_lists(), parameter_lists_for_paths(), is_path_kwargs
  - PyAutoFit/autofit/non_linear/samples/summary.py — SamplesSummary and how it stores/retrieves samples
  - PyAutoFit/autofit/non_linear/samples/pdf.py — SamplesPDF.summary property
  - PyAutoFit/autofit/non_linear/samples/interface.py — max_log_likelihood() path dispatch logic

  Suggested Investigation Areas

  1. How do real samplers (Nautilus, Dynesty, Emcee) construct their samples? They all use Sample.from_lists() too — so why do they work? The difference may be that real samplers go through perform_update()
  which calls SamplesPDF.summary (which reconstructs samples correctly), while the bypass constructs SamplesPDF directly and the summary property may behave differently.
  2. Should bypass mode use name-based kwargs instead of path-based? If the bypass created samples with model.all_names-style string keys instead of tuple keys, it would avoid the path mismatch. But this
  would change is_path_kwargs behavior.
  3. Should _build_fake_samples() delegate to the same SamplesPDF.summary path that real runs use? Instead of manually constructing samples and then a summary, the bypass could create a SamplesPDF and call
  .summary on it, ensuring the same code path as real runs.
  4. Is post_fit_output() correctly guarded for bypass mode? The search_internal=None guard exists but may not cover all code paths (e.g. LBFGS-specific cleanup).
  5. Consider a Model.fake_result() or AbstractSearch.fake_result() method that the model/search knows how to produce correctly for any model composition, rather than having the bypass code manually assemble
  samples.

  Currently Disabled Smoke Tests

  These 5 scripts are disabled in smoke_tests.txt pending this fix:
  - autofit_workspace: searches/mle/LBFGS.py
  - autogalaxy_workspace: ellipse/modeling.py
  - autolens_workspace: point_source/start_here.py
  - autolens_workspace_test: imaging/visualization.py, interferometer/visualization.py

  Test Commands

  # Reproduce ellipse failure
  cd autogalaxy_workspace
  rm -rf output/ellipse
  PYAUTOFIT_TEST_MODE=2 NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/ellipse/modeling.py

  # Reproduce point_source failure
  cd autolens_workspace
  rm -rf output/point_source
  PYAUTOFIT_TEST_MODE=2 NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/point_source/start_here.py