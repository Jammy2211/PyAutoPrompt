# Implement TruncatedNormalMessage._normal_gradient_hessian

## Problem

`PyAutoFit/autofit/messages/truncated_normal.py:349-352` defines:

```python
def _normal_gradient_hessian(self, x):
    raise NotImplementedError
```

This means **EP / Laplace optimisation cannot use any `TruncatedGaussianPrior`** (i.e.
any `GaussianPrior` with finite `lower_limit` and/or `upper_limit`). The Laplace
optimiser walks the call chain
`grad_condition → state.gradient → factor_gradient → cavity_dist.logpdf_gradient`,
which calls `m.logpdf_gradient(x)` on every cavity-message component. For
`TruncatedNormalMessage` that immediately raises `NotImplementedError`.

The EP outer loop catches the exception per-factor, so the search prints a result
block and looks like it succeeded. In reality the truncated parameters never get
their cavity update — the posterior means stay glued to the prior means. Concretely
this was caught running `z_projects/concr/scripts/toy/ep.py` with `PYAUTO_TEST_MODE=1`:
the shared (untruncated) `centre` `GaussianPrior` updates correctly, but the
truncated `normalization` and `sigma` per-dataset stay clamped to their priors.

## Requested change

Implement `_normal_gradient_hessian` on `TruncatedNormalMessage` so that
`logpdf_gradient(x)` and `logpdf_gradient_hessian(x)` return the right values.

### Math

For the truncated normal with underlying Gaussian (μ, σ) and bounds [a, b]:

```
log p(x) = -log Z - log σ - 0.5 log(2π) - 0.5 ((x-μ)/σ)²        for a ≤ x ≤ b
        = -∞                                                    otherwise
```

where `Z = Φ((b-μ)/σ) - Φ((a-μ)/σ)` is the truncation normalisation.

The gradient and Hessian **with respect to x** are independent of `Z`, because `Z`
and `log σ` are constants in `x`:

```
∂/∂x  log p(x) = -(x-μ)/σ²
∂²/∂x²log p(x) = -1/σ²
```

So inside the support these are identical to the untruncated case
(`NormalMessage._normal_gradient_hessian` at
`autofit/messages/normal.py:329-366`). The differences are:

1. The `logl` value gets the extra `-log Z` correction.
2. For `x` outside `[lower_limit, upper_limit]`, `logl = -inf` and gradient/Hessian
   should be `0` (so the optimiser sees a flat region rather than NaNs that crash
   the linesearch).

### Implementation sketch

`autofit/messages/truncated_normal.py`, replace the `raise NotImplementedError`
body with something like (preserving the existing scalar / vector branching used
by `NormalMessage`):

```python
def _normal_gradient_hessian(self, x):
    from scipy.stats import norm

    a = (self.lower_limit - self.mean) / self.sigma
    b = (self.upper_limit - self.mean) / self.sigma
    Z = norm.cdf(b) - norm.cdf(a)
    log_Z = np.log(Z) if Z > 0 else -np.inf

    shape = np.shape(x)
    if shape:
        x = np.asanyarray(x)
        deltax = x - self.mean
        hess_logl = -self.sigma ** -2
        grad_logl = deltax * hess_logl
        eta_t = 0.5 * grad_logl * deltax
        logl = self.log_base_measure + eta_t - np.log(self.sigma) - log_Z

        in_bounds = (x >= self.lower_limit) & (x <= self.upper_limit)
        logl = np.where(in_bounds, logl, -np.inf)
        grad_logl = np.where(in_bounds, grad_logl, 0.0)

        if shape[1:] == self.shape:
            hess_logl = np.repeat(
                np.reshape(hess_logl, (1,) + np.shape(hess_logl)), shape[0], 0
            )
        # Note: hess_logl is shape-broadcast and may be scalar or array.
        # If you want it masked too, do `np.where(in_bounds, hess_logl, 0.0)`
        # *after* the broadcast.

    else:
        deltax = x - self.mean
        hess_logl = -self.sigma ** -2
        grad_logl = deltax * hess_logl
        eta_t = 0.5 * grad_logl * deltax
        logl = self.log_base_measure + eta_t - np.log(self.sigma) - log_Z

        if not (self.lower_limit <= x <= self.upper_limit):
            logl = -np.inf
            grad_logl = 0.0

    return logl, grad_logl, hess_logl
```

Reference the untruncated implementation at `autofit/messages/normal.py:329-366`
to keep shape handling consistent.

### Watch out for `TruncatedNaturalNormal`

`TruncatedNaturalNormal` (same file, line ~512) inherits from
`TruncatedNormalMessage`, so it would inherit the new implementation. **But** its
`mean` and `sigma` `cached_property` overrides return the *truncated* moments
(via `scipy.stats.truncnorm.mean / .std`), not the underlying Gaussian's
parameters. The gradient formula above assumes `self.mean` and `self.sigma` are
the *underlying* Gaussian. Two options:

(a) Inside `_normal_gradient_hessian`, compute the underlying Gaussian's
parameters from the natural ones when the instance is `TruncatedNaturalNormal`:

```python
mu_underlying = -self.parameters[0] / (2 * self.parameters[1])
sigma_underlying = (-2 * self.parameters[1]) ** -0.5
```

(b) Override `_normal_gradient_hessian` on `TruncatedNaturalNormal` itself.

Pick whichever feels cleaner. Probably (b) — keeps the base class clean and
mirrors how `mean`/`sigma` are already overridden there.

## Verification

1. **Unit test**: add tests in `test_autofit/messages/test_truncated_normal.py`
   (create the file if it doesn't exist) covering:
   - Inside the support, gradient/Hessian match `NormalMessage` *and* logl differs
     by exactly `-log Z`.
   - Outside the support, logl is `-inf` and gradient is `0`.
   - Scalar vs array `x` paths both return correct shapes.
   - `TruncatedNaturalNormal` returns finite gradients (regression test for option
     (a) vs (b) above).

2. **Integration test (the original reproducer)**:
   ```bash
   cd /home/jammy/Code/PyAutoLabs/z_projects/concr
   PYAUTO_TEST_MODE=1 NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
     python3 scripts/toy/ep.py --sample=toy__gaussian_x1__low_snr --total_datasets=5
   ```
   Should run with **no `NotImplementedError`** in the traceback, and the
   per-dataset `TruncatedGaussianPrior` posteriors for `normalization` / `sigma`
   should differ visibly from their priors (currently they're glued to
   `mean=3.0, sigma=5.0` and `mean=10.0, sigma=10.0`).

3. **Full `pytest test_autofit`** — make sure no existing message / EP /
   Laplace tests regress.

## Out of scope

- Don't touch `z_projects/concr/scripts/toy/ep.py`. The whole point is that the
  library should support the existing model spec.
- Don't add a new prior class. Fix the existing message.
- No JAX gradient implementation needed for the first pass — Laplace runs on
  numpy. If `logpdf_gradient` is later called from a JAX path, that'd be a
  follow-up.

## Branch / PR

Single PR against PyAutoFit `main`. Suggested branch:
`feature/truncated-normal-gradient-hessian`. Title:
`Implement gradient/Hessian for TruncatedNormalMessage so EP+Laplace works with truncated priors`.
