Profile and speed up unit tests in @PyAutoLens.

## Goal

Reduce the test suite runtime by identifying and optimizing slow tests. The source code is stable, so safe changes to test parameters are acceptable — but prompt the user before changing any numerical assertion values.

## Approach (proven on PyAutoFit)

1. **Profile**: Run `pytest <test_dir> --durations=0 -q` to get wall-clock times for all tests
2. **Identify**: Find the top 30+ slowest tests and read their source to understand bottlenecks
3. **Categorize**: Group by cause — large data arrays, excessive iterations/max_steps, expensive fixtures, real sampler initialization (Dynesty)
4. **Optimize** (in order of impact):
   - Reduce `n_obs`, data array sizes, `range(N)` loop counts where tests use more data than needed for their assertion tolerances
   - Reduce `max_steps`, `n_iters`, `maxcall` values in EP/sampler tests that validate plumbing not convergence
   - Reduce `number_of_steps` in grid search fixtures to shrink the job count (e.g. 10->5 gives 25 jobs instead of 100)
   - Adjust query/assertion values if data range changes (e.g. if range(100)->range(50), update assertions that referenced value 50 to value 25)
   - Leave tests with inherent sampler initialization overhead (DynestyStatic with maxcall<=5) as-is
5. **Verify**: Run full suite, confirm all tests pass, compare before/after total time

## What NOT to change

- Don't change numerical reference values in assertions without user approval
- Don't change fixture scoping to module/session — this conflicts with function-scoped `set_config_path` autouse fixtures
- Don't mock real sampler tests — they test actual sampling behavior
- Don't change `test_messages.py::test_normal_simplex` — the slowness is numerical quadrature integration

## Known pattern: FactorValue hash collision

If you encounter a `KeyError: <GaussianPrior id=0>` in `AbstractJacobian.grad()`, this is a known bug where `FactorValue` (id=0) hash-collides with a GaussianPrior that gets id=0. This was fixed in PyAutoFit PR #1182 — check that the fix is present in the installed autofit version.

## Target

25-35% reduction in total test runtime. On PyAutoFit this took 61s -> 41s.
