# Census of priors and messages — confirmed bugs + redesign ideas

## Why this exists

After the LogUniformPrior sign-convention bug was fixed in PyAutoFit
commit `e95295b83` ("fix: log_prior_from_value sign convention — density
form across Prior subclasses"), we did a full audit of every prior and
message to check for similar latent math bugs. This file is the parked
result so the cleanup can be picked up later — it is **not** a plan to
start work on now.

The audit covered every file in:

- `PyAutoFit/autofit/mapper/prior/`
- `PyAutoFit/autofit/messages/`

## Confirmed bugs (real, sitting in `main`)

These are ordered roughly from "would crash in normal use" to "subtle
math wrong but rarely exercised". File:line refs are PyAutoFit at the
audit time (post-`e95295b83`).

### A1. `LogGaussianPrior.with_limits` and `_new_for_base_message` will crash on first call
`autofit/mapper/prior/log_gaussian.py:95-114`

```python
return cls(
    mean=(lower_limit + upper_limit) / 2,
    sigma=upper_limit - lower_limit,
    lower_limit=lower_limit,    # ctor does not accept this
    upper_limit=upper_limit,    # ctor does not accept this
)
```

`LogGaussianPrior.__init__` only accepts `(mean, sigma, id_)`. Any
prior-passing pipeline that calls `LogGaussianPrior.with_limits(...)`
(the standard prior-passing entry point) will raise `TypeError`. Same
root cause in `_new_for_base_message` — also references
`self.lower_limit` which is never set.

Same shape as the original log_uniform bug: this branch of prior passing
has clearly never run end-to-end.

### A2. `inv_beta_suffstats` clamp is a no-op
`autofit/messages/beta.py:96-102`

```python
if np.any(ab < 0):
    warnings.warn(...)
    b = np.clip(ab, 0.5, None)   # writes to local b, then b is discarded
```

Clamped value is bound to local `b` and never assigned back to `ab`, so
the returned `ab[:, 0]` / `ab[:, 1]` are still negative. Either
Newton-Raphson must always converge to valid `(α, β)` or the fallback is
silently lying.

### A3. `GammaMessage.from_mode` formula is dimensionally wrong
`autofit/messages/gamma.py:79`

```python
alpha = 1 + m ** 2 * V   # m=mode, V=variance → units: variance²
beta  = alpha / m
```

For Gamma with rate `β`: `mode = (α-1)/β` (α≥1) and `var = α/β²`. The
two-equation solution is quadratic in α; `1 + m²·V` doesn't satisfy
either equation. Probably meant `α = 1 + m²/V` (mean-matching closed
form when mean ≈ mode). Untested anywhere in the suite.

### A4. `TruncatedNormalMessage.log_partition` is incomplete
`autofit/messages/truncated_normal.py:29-49`

Returns only `log Z` (the truncation correction), not
`log Z + A_gauss(η)`. Consequences:

- `MessageInterface.logpdf` (the generic exponential-family path,
  inherited and not overridden) returns
  `log_base_measure + η·T(x) − log_partition`, which for the truncated
  normal is short by `−log σ − μ²/(2σ²)`. So `truncated_message.pdf(x)`
  does **not** integrate to 1.
- The specific paths (`_normal_gradient_hessian`,
  `log_prior_from_value`) compute the correct density. So sampling and
  `log_prior` are fine; only the generic interface path is wrong.
- Latent because TruncatedNormalMessage is rarely fed through EP (where
  `log_partition` actually matters). Still, anything calling `.pdf()`
  on a truncated normal is silently wrong.

### A5. Inconsistent constant convention in `log_prior_from_value`

After `e95295b83`:

- `NormalMessage` → `-(x-μ)²/(2σ²)` (drops `-log σ - 0.5·log(2π)`)
- `UniformPrior` → `0.0` (drops `-log(b-a)`)
- `LogUniformPrior` → `-log x` (drops `-log log(b/a)`)
- `LogGaussianPrior` → `-(log x − μ)²/(2σ²) − log x` (drops `-log σ - 0.5·log(2π)`)
- `TruncatedNormalMessage` → full density including all constants and `−log Z`

For MCMC / MLE / LBFGS this is fine (constants don't affect shape). But
it's a foot-gun for marginal-likelihood / evidence calculations,
cross-prior-family chain comparison, and any downstream code that
assumes log-priors are normalised. Pick one convention.

### A6. `assert_sigma_non_negative` is silently disabled
`autofit/messages/normal.py:103` — call site is commented out. The JAX
branch (`(_ for _ in ()).throw(...)` inside `jax.lax.cond`) wouldn't
behave as intended anyway — exceptions raised inside a `cond` branch
under `jit` get suppressed/rewritten. Currently `NormalMessage(mean,
sigma=-1)` constructs without complaint, while `TruncatedNormalMessage`
*does* enforce non-negative sigma at line 85. The two classes disagree
on validity.

### A7. `UniformPrior.logpdf` does not handle arrays
`autofit/mapper/prior/uniform.py:111-116` — `if x == self.lower_limit:`
is scalar-only. The `# TODO` comment acknowledges this. Calling
`prior.logpdf(np.array([…]))` triggers the well-known "truth value is
ambiguous" error.

### A8. `FixedMessage.logpdf_cache` is an unbounded class-level dict
`autofit/messages/fixed.py:57-62`. Every distinct shape seen during a
long run is memoised forever on the class. Not a math bug; a slow leak
in long-running searches.

### A9. `RelativeWidthModifier` silently degenerates near zero/negative means
`autofit/mapper/prior/width_modifier.py:78-81` — `value * mean`. For
prior passing on a parameter centred near 0 the new Gaussian's sigma
collapses to ~0. For negative means sigma goes negative — and
`NormalMessage` won't complain (see A6).

## Things audited that look fine

End-to-end (closed-form vs message stack vs natural-parameter path):

- `UniformPrior.value_for` — closed-form and message-stack agree
- `LogUniformPrior.value_for` and `log_prior_from_value` — match
  `e95295b83` fix
- `NormalMessage` — natural params / log-partition / suff stats / KL /
  `_normal_gradient_hessian` / `from_mode` / `value_for` all consistent
- `BetaMessage` — natural params, log-partition, mean/variance,
  `logpdf_gradient_hessian` derivatives all check out (modulo A2)
- `GammaMessage` — same (modulo A3)
- `log_transform` / `log_10_transform` / `phi_transform` /
  `LinearShiftTransform` — forward/inverse/grad/hess/log_det all
  consistent; composition order in `TransformedMessage._transform`
  (reversed) vs `_inverse_transform` (forward) is correct
- `TruncatedNormalMessage.value_for` and `log_prior_from_value` (the
  direct paths, not via the generic interface) — correct
- `PriorVectorized` — batched paths match the per-prior `value_for`

## Redesign / refactor ideas

The bugs above all come from the same structural issue: **the same
density is encoded in three places, each with its own conventions**.

### C1. Single source of truth for the density (highest impact)

Each `Prior` subclass has its own `value_for`, the `Message` it wraps
has its own `value_for` and `logpdf`, and `log_prior_from_value` is a
fourth implementation (a quadratic by hand) that doesn't share code
with either. That's why the sign convention drifted: nothing forced
the three to be consistent.

Proposal: each distribution defines exactly one scalar function
`log_density(x, *params)` and one `value_for(unit)` (inverse CDF).
`log_prior_from_value` becomes `log_density - constants_we_choose_to_drop`.
`logpdf` becomes `log_density`. `_normal_gradient_hessian` becomes
`jax.grad(log_density)` (or hand-written but tested against `jax.grad`).
The three callers stay, but the body lives in one place.

### C2. Make the dropped-constants convention explicit and uniform

Either every `log_prior_from_value` keeps full normalisation (consistent
with `TruncatedNormalMessage` today, friendly to evidence calculations)
or every one returns density-up-to-additive-constant (consistent with
`NormalMessage`/`UniformPrior` today). Pick one, write it in the
`Prior` base-class docstring, add a test asserting `exp(log_prior_from_value(x))`
integrates to 1 over the support (or to the documented fixed normaliser).

### C3. Property-based tests that would have caught A1–A4 in CI

Hypothesis-style sweeps inherited by every `Prior` subclass:

1. `pdf` integrates to 1 over `_support` (`scipy.integrate.quad`) —
   catches A4
2. `cdf(value_for(u)) ≈ u` for `u ∈ (0, 1)` — catches inverse-CDF errors
3. `jax.grad(log_prior_from_value)(x)` matches numerical gradient —
   catches sign-convention slips like the original LogUniform bug
4. `with_limits(a, b)` returns an instance whose `value_for(0) ≈ a`
   and `value_for(1) ≈ b` — catches A1
5. `from_mode(m, v)` then `.variance ≈ v` and `.mean ≈ m` (or whatever
   invariant the function claims) — catches A3

The existing `test_message_norm` does (1) for
`NormalMessage`/`BetaMessage`/`GammaMessage`/`LogNormalMessage` but
**not** `TruncatedNormalMessage`, `LogGaussianPrior`, or anything
`TransformedMessage`-wrapped — that's why A4 has survived.

### C4. Collapse `Prior` and `Message`

The split between `LogUniformPrior` (in `mapper/prior/`) and
`UniformNormalMessage` (in `messages/`) carries no semantic weight:
every prior holds a message, and the prior delegates almost everything
to it via `__getattr__`. The indirection is why `GaussianPrior`
"inherits" `log_prior_from_value` from `NormalMessage` without anyone
realising — and why the recent fix had to touch both `normal.py` and
the prior-side overrides separately. A single hierarchy
(`Distribution` → `Gaussian`/`LogGaussian`/`Uniform`/...) combining both
responsibilities would shrink the surface area roughly in half.

### C5. Replace the hand-rolled transform stack with `tfp` bijectors or `numpyro.distributions.transforms`

`AbstractDensityTransform` (forward/inverse/log_det/log_det_grad/jacobian/transform_jac/transform_det_jac/transform_func_grad_hess)
reinvents `tensorflow_probability.bijectors` /
`numpyro.distributions.transforms`. Those libraries have hundreds of
contributor-years on this exact math, JAX-native gradients for free,
and are battle-tested. `phi_transform`/`log_transform`/`log_10_transform`
have near-1:1 equivalents (`tfb.NormalCDF`, `tfb.Exp.inverse`, etc.).
Bigger lift, eliminates the entire transform-correctness category of bug.

### C6. Tighten `TransformedMessage` semantics

- Constructor "unwraps" nested `TransformedMessage` into a flat
  `transforms` tuple (lines 84-86). The reversal convention
  (`_transform` reverses, `_inverse_transform` does not) works but is
  easy to get wrong — a 6-line docstring with a worked example would
  pay for itself.
- `LinearShiftTransform.__init__` stores `DiagonalMatrix(1/scale)` as
  the Jacobian, but the API takes `scale` directly. The
  off-by-reciprocal was a contributing factor in the LogUniform bug.
  Naming the kwarg `inv_scale=` or storing `scale` directly and
  computing the reciprocal in `log_det` would be clearer.

### C7. Width-modifier rethink

`RelativeWidthModifier(0.5)` × `mean=0` → `sigma=0`, then
`NormalMessage` constructs silently (A6) and the next search
degenerates. Safer: width = `max(absolute_floor, relative * |mean|)`.
Even better: YAML `prior_config` could let users specify `min_sigma`
per attribute.

## Suggested next steps (smallest unit of work first)

1. **Fix A1, A2, A3** — three small commits, all currently latent
   crashes / silent wrong answers, no API change.
2. **Decide and document A5** — pick a convention, add
   `assert_log_prior_normalised` test for each prior.
3. **Add the C3 property tests** — would have caught the original
   LogUniform bug + A1 + A4 in one stroke.
4. C1 / C4 (single-source density, merge Prior+Message) is the right
   long-term refactor but a much bigger change; design doc first.

## Repos touched (when this is picked up)

- `PyAutoFit` — bugfixes + tests
- `PyAutoFit/test_autofit` — property tests
- (Possibly) `autofit_workspace_test` — regression chains comparing
  pre/post-fix `from_mode` and prior-passing behaviour
