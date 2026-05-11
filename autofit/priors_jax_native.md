Make PyAutoFit's `Prior.value_for(unit)` and
`Prior.log_prior_from_value(value)` JAX-traceable on every concrete
`Prior` subclass, so cube → physical and the prior log-density can run
inside a `jax.jit` boundary without a `jax.pure_callback` host
roundtrip.

This is **Phase 0** of `z_features/nss_first_class_sampler.md` — the
critical prerequisite. Without it, the cube → physical mapping inside
an `af.NSS` likelihood would have to go through `jax.pure_callback`
which costs ~18 ms/eval (we measured this on the
`numpyro_ess_one_config.py` profiling sweep — `pure_callback` brings
ESS up to 412–680 ms/eval). That would erase the ~30× per-eval
advantage `nss_jit` currently has over Nautilus on this likelihood
(see `z_projects/profiling/FINDINGS_v3.md`).

__Why this matters__

The profiling project on the RAL HPC measured:

| Sampler | ms/eval | Why |
|---|---|---|
| **`nss_jit`** | **0.52** | Pure JAX, no boundary, JIT'd end-to-end. |
| nautilus | 17.9 | NumPy↔JAX boundary on every call. |
| bjsmc | 32.6 | `pure_callback` inside JIT. |
| npess | 412 | `pure_callback` × multiple slice steps per sample. |

`nss_jit`'s 0.5 ms/eval **requires** the entire likelihood (including
cube → physical and `log_prior`) to be JIT-traceable. Everything else
(Nautilus, PocoMC) pays the boundary cost on every call.

`af.NSS` (Phase 1) intends to call `Fitness.call(unit_vector)` from
inside the JIT'd `loglikelihood_fn`. Today that path goes through
`model.vector_from_unit_vector(unit_vector)` which is NumPy-only.
Phase 0 changes that.

__What exists today__

`@PyAutoFit/autofit/mapper/prior/uniform.py:135` —
`UniformPrior.value_for(self, unit: float) -> float`:

```python
def value_for(self, unit: float) -> float:
    return float(
        round(super().value_for(unit), 14)
    )
```

`super().value_for` lives in `@PyAutoFit/autofit/mapper/prior/abstract.py:140`.
The `float(round(...))` is the JIT-breaking line — `round` produces a
Python float, which forces materialisation during tracing.

`@PyAutoFit/autofit/mapper/prior/uniform.py:160` —
`log_prior_from_value(self, value, xp=np)` already takes an `xp` kwarg
**but** all five Prior subclasses thread `xp` inconsistently (some
return Python scalars regardless, some use `xp.where`, some import
`scipy.stats` directly).

`Model.vector_from_unit_vector(unit_vector)` (and the related
`log_prior_list_from_vector`) iterate `prior.value_for(u)` /
`prior.log_prior_from_value(v)` per parameter and return a Python
`list[float]`. The list-construction also breaks JIT tracing.

__What to change__

Concrete files (use `git grep "def value_for\|def log_prior_from_value"`
under `PyAutoFit/autofit/mapper/prior/` to confirm exhaustive coverage):

- `@PyAutoFit/autofit/mapper/prior/uniform.py` — `UniformPrior`
- `@PyAutoFit/autofit/mapper/prior/gaussian.py` — `GaussianPrior`
- `@PyAutoFit/autofit/mapper/prior/truncated_gaussian.py` — `TruncatedGaussianPrior`
- `@PyAutoFit/autofit/mapper/prior/log_uniform.py` — `LogUniformPrior`
- `@PyAutoFit/autofit/mapper/prior/log_gaussian.py` — `LogGaussianPrior`
- `@PyAutoFit/autofit/mapper/prior/abstract.py:140` — `Prior.value_for`
  base method
- `@PyAutoFit/autofit/mapper/model.py` — `Model.vector_from_unit_vector`
  and `Model.log_prior_list_from_vector`
- `@PyAutoFit/autofit/mapper/prior/tuple_prior.py` and `constant.py`
  must keep working but probably don't need JAX paths (TuplePrior is
  a collection, ConstantPrior is fixed) — confirm.

### 1. Add `xp` kwarg to every `Prior.value_for`

Signature change:

```python
def value_for(self, unit, xp=np) -> float | jnp.ndarray:
```

Inside each implementation:

- `UniformPrior`: `xp.where(...)` if needed for limits, no `round`,
  return `xp.float64` not Python `float`.
- `GaussianPrior`: replace `scipy.stats.norm.ppf` with
  `xp.scipy.stats.norm.ppf` (works for both `numpy` and
  `jax.numpy`). `xp.scipy.stats.norm.ppf` exists in JAX 0.4.x — verify
  available in the pinned versions of JAX used by `autoarray`'s tests.
- `TruncatedGaussianPrior`: `jax.scipy.stats.truncnorm.ppf` does NOT
  exist as of JAX 0.4.38. Either roll our own using
  `jax.scipy.special.ndtri` + bounds, OR add a `jax_wrapper` shim that
  branches `truncnorm = jax.scipy.stats.truncnorm if hasattr else _manual_truncnorm`.
  See `jax.scipy.stats` docs and confirm at implementation time.
- `LogUniformPrior`: `lower * (upper / lower) ** unit` — straight
  math, no special functions, works in any `xp`.
- `LogGaussianPrior`: `xp.exp(GaussianPrior-style ppf)`.

Critical: **do not call Python `float()`, `round()`, or `.item()`**
when `xp` is `jax.numpy` — those force tracing materialisation and
break the JIT. Keep operations symbolic.

### 2. Update every `log_prior_from_value` to use the existing `xp` kwarg

`UniformPrior.log_prior_from_value` returns `0.0` today —
that's a Python scalar. Change to `xp.zeros_like(value)` (so it
broadcasts when `value` is a JAX array). Returning `-xp.inf` outside
bounds requires `xp.where(in_bounds, 0.0, -xp.inf)`.

For Gaussian-family priors, the current implementations probably
already use `xp.log`, `xp.square`, etc. — audit that they don't use
`np.log` literally where `xp.log` should be.

### 3. Make `Model.vector_from_unit_vector` JAX-traceable

`@PyAutoFit/autofit/mapper/model.py` — find the method. Currently:

```python
def vector_from_unit_vector(self, unit_vector):
    return [
        prior.value_for(u)
        for prior, u in zip(self.priors_ordered_by_id, unit_vector)
    ]
```

Change to accept `xp=np`:

```python
def vector_from_unit_vector(self, unit_vector, xp=np):
    return xp.stack([
        prior.value_for(u, xp=xp)
        for prior, u in zip(self.priors_ordered_by_id, unit_vector)
    ])
```

(`xp.stack` produces a `(ndim,)` 1-D array of the right backend.)
`Model.log_prior_list_from_vector` analogously.

Both methods are exercised today as Python-only paths — Nautilus
calls them with NumPy. Default `xp=np` keeps existing callers
working.

### 4. Confirm downstream consumers tolerate the array return

Two existing consumers call `vector_from_unit_vector`:

- `Fitness.call` in `@PyAutoFit/autofit/non_linear/fitness.py` — needs
  to thread `xp` if the Fitness is being called inside a JIT trace.
  Audit `Fitness.__init__` for an existing `use_jax_jit` flag (likely
  exists per the ellipse-fitting work) — wire `xp=jnp` when that flag
  is set.
- `model.instance_from_vector(vector)` in
  `@PyAutoFit/autofit/mapper/model.py` — already accepts `xp=jnp` per
  the autolens MGE work. Should already work with JAX arrays.

__What to verify__

1. **Unit tests for each Prior subclass.** Pattern:

   ```python
   import numpy as np
   import jax.numpy as jnp
   from autofit.mapper.prior import UniformPrior

   def test_uniform_value_for_jax_matches_numpy():
       prior = UniformPrior(lower_limit=0.5, upper_limit=2.5)
       u = 0.3
       np_v = prior.value_for(u, xp=np)
       jnp_v = float(prior.value_for(jnp.float64(u), xp=jnp))
       assert np.isclose(np_v, jnp_v, atol=1e-12)
   ```

   Add per subclass × per method (`value_for` + `log_prior_from_value`).
   Test files probably under
   `@PyAutoFit/test_autofit/mapper/prior/test_<subclass>.py`.

2. **Round-trip test.** A model with 5 priors (one of each type),
   `unit_vector = jnp.linspace(0.1, 0.9, 5)`, run
   `model.vector_from_unit_vector(unit_vector, xp=jnp)`, then
   `model.log_prior_list_from_vector(physical, xp=jnp)`, check no
   tracing errors and finite results.

3. **JIT trace test.** The full motivation:

   ```python
   @jax.jit
   def cube_to_phys(unit):
       return model.vector_from_unit_vector(unit, xp=jnp)

   _ = cube_to_phys(jnp.array([0.5] * model.prior_count))
   ```

   This must succeed without an `XlaRuntimeError` or
   `TracerArrayConversionError`. Add this as a separate test file
   under `test_autofit/mapper/test_model_jax.py` (gated on `import jax`
   so the test suite still passes without JAX installed).

4. **Existing Nautilus + Dynesty smoke.** Their NumPy paths must
   still work — default `xp=np` everywhere. Run `/smoke_test` on
   `autofit_workspace_test`, `autolens_workspace_test`, and the
   `searches_minimal` reference suite under
   `autolens_workspace_developer/`.

5. **`autoconf.jax_wrapper` interaction.** Importing autoconf flips
   `JAX_ENABLE_X64=True`. The new tests must not assume x64; default
   to whatever JAX has when the test runs. The Prior JAX paths
   themselves must produce `float64` outputs when x64 is enabled
   (consistent with the autolens MGE pipeline).

__Out of scope__

- `af.NSS` itself — separate prompt (Phase 1,
  `autofit/nss_search_wrapper.md`).
- Removing the `round(..., 14)` precision-snapping in
  `UniformPrior.value_for(unit)` for the **NumPy** path. That snap is
  load-bearing for some existing unit tests (rounded values are used
  as hash keys in `model.priors`); keep it for `xp=np`, drop it for
  `xp=jnp`.
- New Prior types (e.g. Beta, Exponential). Stick with the existing
  five.
- The `width_modifier` and `vectorized` Prior helpers — they're
  meta-priors and don't have `value_for` of their own.

__Risks / open questions__

1. **`jax.scipy.stats.truncnorm.ppf` may not exist in JAX 0.4.38.**
   Verify before starting and either:
   - Manual implementation using `jax.scipy.special.ndtri` and clipping.
   - Conditional import + helper function in a new
     `@PyAutoFit/autofit/mapper/prior/_jax_helpers.py` module.

2. **Performance.** `Model.vector_from_unit_vector` iterates priors
   in Python. Inside `jax.jit` the loop unrolls at trace time, so
   trace cost grows with the number of priors. For a 15-parameter
   MGE model this is ~15 trace iterations — negligible. For a
   1000-parameter MGE problem it could matter. Consider whether
   `jax.lax.scan` over a homogeneous prior-type batch is worth it —
   probably not until we hit a model that complains.

3. **Compatibility with `paths.parquet` / `paths.csv` serialisation.**
   The Sample.parameter_lists output should remain Python `float`s
   when written to disk. JAX-array prior outputs need to be converted
   to NumPy before the existing serialisers see them. The Phase 1
   `af.NSS` wrapper will handle this at the sampler boundary.

__Reference__

- `@PyAutoFit/autofit/mapper/prior/uniform.py:135` — current
  `UniformPrior.value_for`
- `@PyAutoFit/autofit/mapper/prior/abstract.py:140` — base `Prior.value_for`
- `@PyAutoFit/autofit/mapper/model.py` — `Model.vector_from_unit_vector`
- `@autolens_workspace_developer/searches_minimal/nss_jit.py` — original
  smoke reference: today uses a hard-bound `log_prior` because of this
  exact gap
- `@z_projects/profiling/FINDINGS_v3.md` — measured per-eval cost
  table that motivates Phase 0
- `@PyAutoPrompt/z_features/nss_first_class_sampler.md` — Phase 0 in
  the sequenced roadmap
