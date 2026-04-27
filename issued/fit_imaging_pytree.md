Can we register @PyAutoLens/autolens/imaging/fit_imaging.py `FitImaging` (and the autoarray /
autogalaxy types it transitively contains) as JAX pytrees, so that a function that returns
`FitImaging` can be wrapped in `jax.jit`?

__Why this matters__

In the JAX-visualization pilot (#1227 on PyAutoFit, shipped via `use_jax_for_visualization`)
we intentionally took the "Path C" route: `fit_from` runs on the eager JAX path
(`use_jax=True` makes `self._xp` = `jnp`, operations run eagerly under JAX) and the
matplotlib plotters materialize arrays to NumPy at the boundary. No `jax.jit` wrapping.

The endgame — "Path A" — is full `jax.jit` wrapping of `analysis.fit_from`, so visualization
gets the same compile-time speedup the likelihood function does. This requires the
`FitImaging` return type (and every autoarray / autogalaxy / autolens type reachable from
it) to be a JAX pytree. PyAutoLens's CLAUDE.md currently documents the reverse:

> Autoarray types (`Array2D`, `ArrayIrregular`, `VectorYX2DIrregular`, etc.) are **not
> registered as JAX pytrees**. They can be constructed inside a JIT trace, but **cannot
> be returned** as the output of a `jax.jit`-compiled function.

This task is the feasibility study for lifting that restriction.

__What's already in place__

Recent autofit work added `autofit.jax.pytrees` (see `@PyAutoFit/autofit/jax/pytrees.py`)
which registers `Model`, `Collection`, `ModelInstance`, and user classes found by walking
a `Model` tree. The pattern:

- `children` = dynamic (traced) arrays / sub-pytrees
- `aux` = static Python objects (concrete constants, class references, etc.)
- `flatten` / `unflatten` lift/restore the instance across the JIT boundary

The same pattern needs to extend down through the autoarray / autogalaxy layers.

__Assess, don't implement (yet)__

The deliverable is a **feasibility assessment**, not a fully-registered FitImaging. Produce
the assessment as a markdown document (e.g. in this prompt file's `issued/` location, or
as a draft PR) answering:

1. **Type inventory.** Walk `FitImaging` (`@PyAutoLens/autolens/imaging/fit_imaging.py`) and
   list every distinct class reachable from a populated instance. For each class:
   - Where it's defined (autoarray / autogalaxy / autolens)
   - Whether it currently carries `tree_flatten` / `tree_unflatten` methods
   - Whether its constructor is compatible with pytree unflatten (takes the children as
     positional / keyword args) or whether a `_build_*_pytree_funcs` helper (like
     `autofit.jax.pytrees._build_instance_pytree_funcs`) would be needed

2. **Dynamic vs static classification.** For each class, which attributes are dynamic
   (JAX arrays — get traced) vs static (numpy arrays that mustn't change, masks, shapes,
   pixel scales, redshifts, config objects)? The CLAUDE.md rule for autoarray is:
   `grid.array[:, 0]` is traced, `grid.mask` is static. That rule needs to be restated
   per type here.

3. **Structural blockers.** Are there types that *cannot* be registered cleanly? Examples
   to look for:
   - Classes whose `__init__` does non-trivial work that can't be replayed during
     unflatten (e.g. reading config, triggering a numba compilation)
   - Classes that hold live references to an `Inversion` whose solver state isn't
     pickleable / is stateful under JAX
   - `Tracer` / `Galaxies` — these are `List[List[Galaxy]]` nested containers; are they
     already pytree-friendly via Python list recursion, or do they need explicit
     registration?
   - `Mapper` / `LinearEqn` / NNLS solver state inside `Inversion` — likely the hardest

4. **Interaction with the existing "xp is np" guard pattern.** PyAutoLens has an established
   convention (see `@PyAutoGalaxy/autogalaxy/operate/lens_calc.py`): functions that return
   autoarray wrappers guard with `if xp is np: return Array2D(...)` else return raw
   `jax.Array`. Registering `Array2D` as a pytree means these guards may become
   unnecessary. Is removing them desirable, or should they stay for the non-JIT path?

5. **Proof-of-concept.** Register **one** concrete type (suggest `Array2D` — it's the
   simplest leaf) and demonstrate that a toy function `def f(x): return Array2D(values=x*2,
   mask=static_mask)` runs under `jax.jit` and returns an `Array2D` backed by
   `jax.Array`. This both validates the approach and surfaces any wrinkles that the static
   analysis missed.

6. **Estimate.** Given the inventory and the proof-of-concept, roughly how many types need
   registration to make `FitImaging` a pytree? Which ones are cheap (static dataclass-ish),
   which are hairy (inversion, mapper)?

__Scope boundary__

- Do **not** change `use_jax_for_visualization` behaviour. That flag currently dispatches
  to the eager-JAX path; once this task produces a green light + registration PR, a
  follow-up task will flip the dispatch to `jax.jit`.
- Do **not** start mass pytree-registration in this task. One-type PoC only.
- Do **not** register `FitImaging` itself yet — that's the goal *after* every type it
  reaches is registered.

__Starting points__

- `@PyAutoLens/autolens/imaging/fit_imaging.py` — the `FitImaging` subclass
- `@PyAutoGalaxy/autogalaxy/imaging/fit_imaging.py` — base `FitImaging`
- `@PyAutoArray/autoarray/structures/arrays/uniform_2d.py` — `Array2D` (suggested PoC type)
- `@PyAutoFit/autofit/jax/pytrees.py` — the existing autofit pytree machinery; mirror its
  structure when proposing new registrations
- `@PyAutoLens/CLAUDE.md` — the "xp is np" guard rule this task's outcome will revise

__Deliverables__

1. Type inventory + classification table (markdown)
2. List of structural blockers with suggested workarounds
3. Working PoC registering `Array2D` with a minimal test in
   `@PyAutoArray/test_autoarray/jax/` following the three-step pattern in
   `autolens_workspace_test/scripts/hessian_jax.py`
4. Effort estimate + recommended ordering for the follow-up implementation task
