# Rectangular Interpolator — Replace JITTER with Safe-Division Interp

## Context

PR [PyAutoArray #281](https://github.com/PyAutoLabs/PyAutoArray/pull/281) fixed an
O(1e24) gradient explosion inside
`autoarray/inversion/mesh/interpolator/rectangular.py::create_transforms`,
caused by `jnp.interp` dividing by zero knot gaps in ray-traced source
grids (~50% of sorted adjacent pairs are exact duplicates for realistic
Isothermal + circular-mask setups).

The shipping fix has two parts:

1. `jax.lax.stop_gradient(sort_points)` — blocks the knot-gradient
   failure mode.
2. `sort_points += jnp.arange(N) * 1e-7` (JITTER) — guarantees non-zero
   knot gaps so the query-gradient slope `(Δt)/(Δxp)` is bounded.

Part (2) is the clean-but-imperfect half: it perturbs the forward
interpolation value by up to `N * JITTER ~ 1.5e-3` in scaled
source-plane units, which drifts integration-test reference
likelihoods by ~1e-4 relative. This tripped
`autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular.py`,
whose `np.testing.assert_allclose(..., -650633.057031, rtol=1e-4)`
now sees `-650730.037` (diff = 96.98, rel diff = 1.5e-4, just over
tolerance).

## The follow-up

A reference implementation of the alternative fix already sits in
`autoarray/inversion/mesh/interpolator/rectangular.py::forward_interp_safe`,
**unused and well-commented**. It replaces `jnp.interp` with a hand-
written `searchsorted` + slope computation that uses the "double-where"
pattern to floor the denominator:

```python
safe_gap = jnp.where(gap > eps, gap, 1.0)
t        = jnp.where(gap > eps, (x - x0) / safe_gap, 0.0)
```

This gives **finite backward gradients without perturbing the forward
value** at well-separated knots. At exact-duplicate knots it returns
`yp[i]` (the left-duplicate's value), which may or may not match
`jnp.interp`'s implementation-defined behaviour at that zero-measure
set of query points.

## Task

1. Swap `forward_interp` for `forward_interp_safe` in the JAX branch of
   `create_transforms`, and remove the JITTER offset.
2. Verify that:
   - `autolens_workspace_test/.../rectangular.py` passes its
     `rtol=1e-4` reference check against `-650633.057031`.
   - `mapper_grad_probe.py` and `mapper_grad_isolate.py` still produce
     finite, FD-agreeing gradients at every stage.
   - `pixelization_gradients.py` steps 4–6 still PASS.
   - `test_autoarray/inversion/` (all 155 tests) still pass.
3. Decide on the value of `eps`: current default `1e-30` effectively
   "no floor" (relies entirely on the double-where), but an explicit
   floor like `1e-12` bounds the worst-case gradient at `1/N * 1e12 ~
   1e8` for `N ~ 1e4`. Pick a principled value, document why.
4. Consider whether the clean approach is actually `jax.custom_vjp`
   wrapping the *real* `jnp.interp`, so the forward stays bit-identical
   and only the backward uses safe division. Tradeoff: more code, no
   duplicate-bin return-value ambiguity.
5. If the approach works, delete the JITTER path in `create_transforms`
   and update the inline comment block.
6. Open a follow-up PR against `PyAutoArray` crediting PR #281 and this
   prompt.

## Why not done in PR #281

The gradient-fix PR had to ship to unblock pixelization JAX gradients
for Sersic + RectangularAdaptDensity models. The JITTER approach was
the smallest, easiest-to-review change that fixed both failure modes
in one place. Switching to `forward_interp_safe` needs independent
review of the `searchsorted` bin convention vs `jnp.interp`'s behaviour
at boundaries, and a discussion about whether `custom_vjp` is the
cleaner long-term design — hence this follow-up.

## Files touched (expected)

- `PyAutoArray/autoarray/inversion/mesh/interpolator/rectangular.py` —
  swap `forward_interp` → `forward_interp_safe`, delete JITTER, rewrite
  the comment block.
- `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular.py` —
  no change expected (reference value should now match again).
