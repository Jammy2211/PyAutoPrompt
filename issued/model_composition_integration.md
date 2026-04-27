There are unit tests again model composition, but we dont have anything which tests against the full level of complex
of full complex models we typically use in science runs.

For example, in PyAutoLens we compose models made of multiple galaxies and made with a multi gaussina
expansion (e.g. @autolens_workspace/scripts/imaging/modeling.py).

We also have a script which gives a run through of the autofit model composition aPI at the lower level
@autofit_workspace/scripts/cookbooks/model_internal.py

I would like some integrations tests in autofit_workspace_test/scripts and autolens_workspace_test/scripts
which basically compose models of realistic complexity, make asertions against their number of 
parameters, aspects of their composition and their random unique_id which defines the folder
results are output into. We can basically assume the source code is ok, but this will
mean I can maintain confidence that upcoming large refactors of autofit do not impact the actual
model composition.

Another benefit specific of the unique_id is if this changes, it means a source code change has change
how a model is composed and maybe break backwards compatibiltiy for autolens users. So its a good
sanity check on if my dev work has an unexpected impact.



There was a bug which actually would of got caught if i had had these integration tests in place:


● ---
  I want to add integration test coverage for model composition helpers in
  PyAutoGalaxy — specifically `model_util.py` — using a recent bug as a
  motivating example. This is a testing / test-infrastructure task, not a bug
  fix. Your job is to think through and propose (then implement, once approved)
  a testing strategy that would have caught the bug below and will catch
  similar issues in the future. Do not attempt to fix the bug.

  ## The motivating bug (case study)

  File: `/home/jammy/Code/PyAutoLabs/PyAutoGalaxy/autogalaxy/analysis/model_util.py`
  Function: `mge_model_from(mask_radius, total_gaussians, gaussian_per_basis, ...)`

  This helper composes a Multi-Gaussian Expansion model as a `Basis` of linear
  `Gaussian` profiles. `gaussian_per_basis` is meant to add model flexibility
  by giving each basis an independent ellipticity (representing e.g. isophotal
  twists where ellipticity varies with radius).

  The bug is a model-composition regression. In PyAutoFit, parameter tying is
  implemented through prior *identity*: assigning the same prior instance to
  two attributes ties them into a single free parameter. The helper originally
  relied on this by constructing centre and ell_comps priors ONCE, then
  assigning them across Gaussians, so every Gaussian within a basis shared a
  centre and an orientation. To give each basis its own orientation, the
  ell_comps prior construction had to live inside the per-basis loop — a new
  instance per basis.

  At some point during a refactor, prior construction for `ell_comps` was
  lifted out of (or never placed inside) the `for j in range(gaussian_per_basis)`
  loop. The code still compiles, the helper still returns an `af.Model` with
  the correct number of Gaussians, and `type()` / `isinstance` checks still
  pass. But every Gaussian across every basis is now assigned the same
  ell_comps prior instance, so `gaussian_per_basis=2` produces a model that is
  structurally identical to `gaussian_per_basis=1`. The extra 30 Gaussians are
  perfectly tied duplicates of the first 30.

  This is the generic shape of the bug: a code motion within a composition
  helper changed which priors are distinct objects and which are shared
  instances. None of the visible return-value properties changed. The helper
  silently produces a degenerate model, and every downstream fit runs
  normally with degraded results.

  The helper is re-exported as `al.model_util.mge_model_from` and used across
  ~30+ workspace scripts. Anyone calling it with `gaussian_per_basis>1` has
  been silently getting the wrong model structure since the regression
  landed. No test caught it.

  ## Why this slipped through

  Think about what kinds of tests would and would not have caught this:

  - Unit tests on return type: no. The function returns the correct class
    (`af.Model(Basis, profile_list=[...])`).
  - Tests on the number of Gaussians in the basis: no. The count is correct;
    it's the prior coupling that's wrong.
  - Assertions on `.prior_count` for each `gaussian_per_basis` value: yes,
    if someone thought to write them.
  - Downstream likelihood comparisons between `gaussian_per_basis=1` and
    `gaussian_per_basis=2` on the same synthetic image: yes — the bug forces
    identical behaviour where there should be a measurable difference.
  - A docstring-vs-behaviour comparison: yes — the docstring claims each
    basis adds flexibility; the post-regression behaviour does not.

  The root issue is that model composition helpers sit in a testing blind
  spot. They are not pure functions (they return mutable `af.Model` graphs
  built out of shared-identity prior objects), they are not runnable
  end-to-end (workspace smoke tests don't exercise them directly), and they
  are not simple enough for a one-line return-type assertion to be
  meaningful. Refactors that move code across a loop boundary can silently
  change the parameter structure without changing anything that a
  conventional test would notice.

  ## Your task

  Design and implement a layer of **integration tests** for model composition
  helpers in `PyAutoGalaxy/autogalaxy/analysis/model_util.py`. The tests
  should sit between pure unit tests and end-to-end fit tests: structural
  assertions on the composed `af.Model` graph that go beyond "does it return
  the right type".

  Start by investigating:

  1. Read `PyAutoGalaxy/autogalaxy/analysis/model_util.py` and enumerate every
     public helper that composes an `af.Model`. `mge_model_from` is one;
     `mge_point_model_from` is another; there may be more.

  2. For each helper, identify what the caller actually cares about — not
     just "did this return an `af.Model`" but "is the parameter structure
     what the docstring says it is". Examples:
       - prior_count for each non-default argument combination
       - which priors are tied (by object identity) vs independent
       - which priors are fixed scalars vs free parameters
       - the sigma / scale / width values of fixed parameters
       - type of prior (Uniform vs Gaussian vs TruncatedGaussian) matches kwargs
       - edge cases: `centre_fixed`, `use_spherical`, `PYAUTO_WORKSPACE_SMALL_DATASETS`

  3. Find the existing test directory for `model_util`:
         /home/jammy/Code/PyAutoLabs/PyAutoGalaxy/test_autogalaxy/analysis/
     Report what's there (if anything) and what's missing. Do not assume a
     test file exists — check.

  ## Proposed test structure

  Then propose (and, after approval, implement) integration tests organised
  roughly like this:

  1. **Structural assertions** — for each helper, at each meaningful kwarg
     combination, assert the prior structure directly. Not just `.prior_count`
     but the identity of which priors are tied. Concrete pattern:

         model = mge_model_from(mask_radius=1.0, total_gaussians=30, gaussian_per_basis=2)
         gaussians = model.profile_list
         assert len(gaussians) == 60
         # centres tied across every Gaussian
         assert all(g.centre.centre_0 is gaussians[0].centre.centre_0 for g in gaussians)
         # ell_comps tied within a basis but NOT across bases
         assert gaussians[0].ell_comps.ell_comps_0 is gaussians[29].ell_comps.ell_comps_0
         assert gaussians[0].ell_comps.ell_comps_0 is not gaussians[30].ell_comps.ell_comps_0

     The `is` checks are the key part — they catch exactly the class of
     regression described above. `.prior_count` alone can be offset by other
     structural errors; identity checks cannot.

  2. **Behavioural differentiation** — for `gaussian_per_basis=1` vs `=2`,
     confirm the composed models produce DIFFERENT behaviours, not just
     different counts. One concrete approach: take a small synthetic image,
     call `al.FitImaging(...).log_likelihood` on an instance of each model
     at the same dummy parameter values, and assert the likelihoods differ
     in the expected way. If that's too heavy for the test suite, at minimum
     assert `prior_count` differs across `gaussian_per_basis` values.

  3. **Docstring ↔ behaviour** — for each documented claim in the helper's
     docstring (e.g. "each basis adds flexibility"), write one assertion
     that would fail if the claim stops being true. Treat the docstring as a
     machine-verifiable contract where possible.

  4. **Env-var branches** — test that `PYAUTO_WORKSPACE_SMALL_DATASETS=1`
     actually caps `total_gaussians=2, gaussian_per_basis=1` regardless of
     what was passed.

  5. **Parametrise across kwargs** — use `pytest.mark.parametrize` to cover:
       - `gaussian_per_basis in (1, 2, 3)`
       - `use_spherical in (True, False)`
       - `centre_prior_is_uniform in (True, False)`
       - `ell_comps_prior_is_uniform in (True, False)`
       - `centre_fixed in (None, (0.0, 0.0))`
     Expect O(30-50) test cases. They should run in well under a second each
     because there is no fitting.

  ## Why this matters (put this in the commit / PR message)

  - Model composition helpers are the main API surface users touch before a
    fit. They define the model's degrees of freedom, which is the most
    consequential decision in the entire workflow.
  - A broken composition helper is worse than a crash: it silently produces
    a wrong model with the correct `type()` and the correct number of
    components, and every downstream fit proceeds normally with degraded
    results. Users have no signal that anything is wrong.
  - End-to-end workspace smoke tests cannot catch this class of bug because
    they only assert "the fit ran" — not "the model had the intended
    parameter structure".
  - Pure unit tests on prior count alone are brittle: they catch the gross
    case but miss subtle structural regressions (e.g. "tied across the wrong
    axis").
  - The right layer is *integration tests on the composed `af.Model` graph
    itself*: assertions about which priors are tied by identity, which are
    free, which are fixed, and how that structure responds to kwargs. This
    is what catches refactors that move code across a loop boundary or
    across a helper boundary without changing anything observable from the
    outside.
  - `model_util.py` is a particularly high-value place to invest test effort
    because every SLaM pipeline, every `start_here.py`, and every advanced
    modeling script routes through it. A bug here has enormous blast radius.

  ## Deliverables

  1. A short audit of what's currently tested in `model_util` (probably very
     little or nothing) and where that test file lives / should live.
  2. A proposed test module layout and naming convention, consistent with
     the rest of the PyAutoGalaxy test suite.
  3. Once approved: the test module itself, with parametrised cases
     covering `mge_model_from` at minimum, and `mge_point_model_from` and
     any other composition helpers in the same file if time permits.
  4. The tests must currently FAIL against the buggy `mge_model_from`
     implementation if the bug has not yet been fixed — that is the test's
     credibility. If the bug is already fixed, the tests must be written in
     a form that would have failed against the pre-fix version; use
     `git log` on `model_util.py` to locate the fix commit and confirm the
     assertion would have tripped.

  ## Planning + branch survey

  Follow the project's CLAUDE.md planning requirement: produce a plan, get
  explicit approval, run `/plan_branches`, then implement. Report the git
  repo root and current branch before editing. Do NOT start writing test
  code until the audit and proposed structure have been reviewed.

  ---
  Let me know if you want any of the sections tightened further.