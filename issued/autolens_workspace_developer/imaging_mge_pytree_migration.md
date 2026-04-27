# Imaging MGE JAX JIT Profiling — Migrate to Pytree Inputs

## Context

`autolens_workspace_developer/jax_profiling/imaging/mge.py` is the
flagship JAX JIT profiling script for the MGE source model. It
builds `Fitness.call` around a 1D parameter vector
(`jnp_params`, shape `(N,)`) and JITs that closure.

Passing parameters as a flat 1D vector has two long-term downsides:

1. **`model.instance_from_vector` is the JIT boundary**. That call
   has to unpack the flat vector into every named structured
   parameter (centres, ell_comps, intensities, sigmas, redshifts,
   …) inside the trace. For a model with many parameters this is
   pure Python shape-juggling that runs once per compile but
   obscures the parameter identity under `jax.jit`.
2. **Batching with `vmap`** requires the caller to stack parameter
   vectors into a 2D `(batch, N)` array. The structured parameter
   names are lost along the way — any diagnostic that wants to
   plot likelihood vs `einstein_radius` has to know the positional
   index of `einstein_radius` in the flat vector, which is
   fragile.

The rest of the codebase (tracers, galaxies, profiles) has moved
toward **pytree-native** inputs: nested dicts / `ModelInstance`
objects registered as JAX pytrees, so `jax.jit` flattens and
unflattens them automatically, and `vmap` batches over the pytree
leaves without the caller having to reshape anything.

## Task

Update `autolens_workspace_developer/jax_profiling/imaging/mge.py`
so that:

1. The JIT'd closure takes a **pytree** of parameters (e.g. a
   nested dict keyed by galaxy name → profile name → parameter
   name, or the `af.ModelInstance` object itself if it's now
   pytree-registered) rather than a flat 1D array.
2. `jax.jit` and `jax.vmap` are applied to that pytree-accepting
   closure directly — no flat-vector shim.
3. The profiling output (compile time, steady-state runtime, vmap
   batch throughput) should be unchanged within noise — this is a
   clarity / API-ergonomics change, not a performance change.
4. Update the inline commentary to point at the pytree inputs as
   the recommended pattern for new profiling scripts.

Once `mge.py` is updated, treat it as the new reference style and
propagate the same pattern to `pixelization.py` and `delaunay.py`
as a follow-up.

## Why this matters

- Downstream users reading the script will more easily see *which*
  parameter is which, because the pytree preserves names.
- `jax.jit` compile caches are keyed on the abstract pytree
  structure; this change makes the cache key human-readable.
- Eventually we will want to build user-facing `jax.jit`'d
  likelihood functions that accept a `ModelInstance` directly —
  the profiling script should demonstrate that end-state, not an
  older flat-vector style.

## Risks / blockers

- If `af.ModelInstance` is not yet registered as a JAX pytree,
  this change either (a) needs that registration to happen first,
  or (b) uses a plain nested dict as an interim representation.
  Check `autofit`'s pytree registration status before starting;
  file a blocker issue if registration is missing.
- `model.instance_from_vector(..., xp=jnp)` is currently relied on
  by several other profiling scripts and by the production
  `Fitness.call`. This task *only* changes the profiling script,
  not the production path — do not touch `Fitness` or `model.instance_from_vector`.

## Pytree Registration Old

Older versions (12ish months ago) of PyAutoFit, PyAutoArray and PyAutoGalaxy had working Pytree circulations.
Dig through their git history first and see if you can find source code there to help you as a reference.