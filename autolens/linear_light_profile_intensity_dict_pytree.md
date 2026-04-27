Make `fit.linear_light_profile_intensity_dict` survive a `jax.jit` round-trip so the
visualization path works for any model containing a linear light profile (e.g. MGE via
`al.lmp_linear.GaussianGradient` or `al.lp_linear.Gaussian`).

__Why this prompt exists__

Path A (jax.jit-wrapped `analysis.fit_from`) is landed for `FitImaging` — see
@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py. That PoC only
asserts on `fit.log_likelihood` (a scalar leaf), so it never walks the code path that consumes
`fit.linear_light_profile_intensity_dict`. The end-to-end validator
@autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py added Part 2 — a live
Nautilus run with `use_jax_for_visualization=True` — to exercise the visualizer. That run only
survives because Part 2 uses a **parametric** `Sersic` source; any model with a linear light
profile dies inside the visualization callback with:

```
KeyError: GaussianGradient   # or Gaussian, SersicOperated, …any LightProfileLinear
```

__Where it breaks__

The visualizer rebuilds a displayable tracer via
@PyAutoGalaxy/autogalaxy/abstract_fit.py `model_obj_linear_light_profiles_to_light_profiles`,
which eventually calls
@PyAutoGalaxy/autogalaxy/profiles/light/linear/abstract.py:148
`intensity = linear_light_profile_intensity_dict[self]`. The dict is built at
@PyAutoGalaxy/autogalaxy/abstract_fit.py:122 by iterating
`linear_obj_func.light_profile_list` and keying on each `light_profile` **instance**.

After `jax.jit(self.fit_from)(instance)` returns, the `FitImaging` (and therefore the dict) has
been through a pytree unflatten cycle — `cls.__new__(cls)` + `setattr(...)` — so every profile
key is a **fresh object** whose `id()` does not match any profile reachable from the *outside*
`ModelInstance` the visualizer walks. The dict lookup is identity-keyed; the lookup fails.

This is not a `GaussianGradient` bug. Any `LightProfileLinear` hits it — including
`al.lp_linear.Gaussian`, confirmed by inspecting the MRO (both extend `LightProfileLinear` and
share the same `lp_instance_from` at `abstract.py:132`).

__What needs to change__

Re-key `linear_light_profile_intensity_dict` on something that survives pytree unflatten:

- **Option A — structural path key.** Key by the linear profile's position in the `Tracer`
  (e.g. `("galaxies", "lens", "bulge", "profile_list", 0)`) or by its `af.ModelInstance` path.
  The visualizer walks the instance by path too, so both ends agree on keys.
- **Option B — stable identifier token.** Attach a monotonic id to each `LightProfileLinear` at
  registration time (or lazily on first access), route it through `__dict__` as a dynamic leaf,
  and key the dict by that id. Adds a field but avoids path bookkeeping.
- **Option C — eq/hash on value.** Give `LightProfileLinear` value-based `__eq__` / `__hash__`
  so identity is replaced by structural equality. Risk: surprise collisions if two linear
  profiles have identical params.

Option A is the most honest — it matches how `af.ModelInstance` navigates — but requires
threading a path through the `linear_obj_func.light_profile_list` construction. Option B is the
smallest diff. Option C is the scariest but needs the least plumbing; evaluate once you know
the blast radius.

Whichever option lands, the change has two consumers: the dict **constructor** at
`abstract_fit.py:122` and the dict **reader** at `abstract.py:148` (plus any other
`_dict[self]`-style reads — grep for them).

__Validation__

1. Revert @autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py Part 2 to its
   pre-split MGE model (parametric MGE basis of `GaussianGradient`, NFWSph mass, MGE source).
   Git log on that file shows the full MGE version before it was split; copy it back.
2. Keep `use_jax_for_visualization=True` and `iterations_per_quick_update=500`.
3. Expected: Nautilus runs to completion, `subplot_fit.png` lands on disk, no `KeyError`.

__Scope boundary__

- This prompt is **only** about the identity-keyed dict. It does **not** own broader linear-
  light-profile pytree work (that lives under the per-variant `fit_*_pytree_*.md` prompts).
- Do **not** regress the NumPy path. Whatever re-keying scheme lands must behave identically
  when `use_jax=False`.
- Do **not** change the public shape of `linear_light_profile_intensity_dict` — downstream code
  treats it as `Dict[LightProfile, float]`. If the internal key changes, preserve the external
  contract (profile instance → intensity) via a view or wrapper.

__Starting points__

- @PyAutoGalaxy/autogalaxy/abstract_fit.py — `linear_light_profile_intensity_dict` property
  (construction site) and `model_obj_linear_light_profiles_to_light_profiles` (consumer)
- @PyAutoGalaxy/autogalaxy/profiles/light/linear/abstract.py:148 — `lp_instance_from` (lookup
  site; where the `KeyError` is raised)
- @PyAutoArray/autoarray/abstract_ndarray.py — `register_instance_pytree` and `no_flatten`
  mechanism, if Option B needs a dynamic leaf to carry the id token
- @autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py — end-to-end
  validator that will flip green once this ships
- @PyAutoLens/autolens/imaging/model/analysis.py — `AnalysisImaging.fit_for_visualization` /
  `_register_fit_imaging_pytrees` (wiring end)

__Deliverables__

1. Library PR(s) re-keying `linear_light_profile_intensity_dict` under the chosen option, with
   matching updates to every call site that reads the dict.
2. `modeling_visualization_jit.py` reverted to full-MGE for Part 2 and passing end-to-end under
   `use_jax_for_visualization=True`.
3. PR body `## API Changes` section noting the re-keying option chosen and the rationale, plus
   a one-line note under each sibling `fit_*_pytree_*.md` variant that ships a linear-light-
   profile model clarifying that this prerequisite now lands.
