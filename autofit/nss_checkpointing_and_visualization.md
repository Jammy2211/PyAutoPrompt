Add checkpointing + on-the-fly visualization to `af.NSS`.

This is **Phases 2 and 3** of `z_features/nss_first_class_sampler.md`. Both
phases share the same architectural hook — the outer loop in
`af.NSS._fit` — so they ship together in one PR.

__Critical finding from Phase 1 follow-up audit__

The z_features roadmap claimed `nss.ns.run_nested_sampling` is "a one-shot
JIT'd `jax.lax.while_loop`" requiring an upstream `yallup/nss` PR to add
a checkpoint hook. **This is wrong.** Inspecting the actual upstream
source (`/home/jammy/venv/PyAuto/lib/python3.12/site-packages/nss/ns.py`):

```python
@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

dead = []

while not state.integrator.logZ_live - state.integrator.logZ < termination:
    (state, rng_key), dead_info = one_step((state, rng_key), None)
    dead.append(dead_info)
```

The JIT boundary is `one_step` (one outer iteration = `num_delete` deaths
processed in one batch). The outer `while` loop is plain Python. This
means both checkpointing and on-the-fly visualization can be implemented
**entirely inside `af.NSS._fit`** — no upstream PR needed.

__What to build__

Replace the single `nss.ns.run_nested_sampling(...)` call inside
`@PyAutoFit/autofit/non_linear/search/nest/nss/search.py::NSS._fit` with
an inlined equivalent that:

1. Imports `blackjax` and `nss.ns` helpers directly (`finalise`,
   `log_weights`, `Results`, `safe_ess`).
2. Builds `algo = blackjax.nss(logprior_fn, loglikelihood_fn, num_delete,
   num_inner_steps)` and a JIT'd `one_step` closure — mirroring the
   upstream pattern.
3. Detects a checkpoint at
   `paths.search_internal_path / "nss_checkpoint.pkl"`; if present,
   loads `(state, dead, rng_key, iteration)` and resumes; otherwise
   initialises fresh from the unit-cube draws.
4. Runs the outer loop with the standard termination criterion, with
   two new hooks per iteration:
   - **Phase 2 — checkpoint:** every `self.checkpoint_interval` outer
     iterations, pickle `(state, dead, rng_key, iteration)` to disk.
   - **Phase 3 — quick-update visualization:** every
     `self.iterations_per_quick_update` outer iterations (when set
     non-None), pick the current best live point and call
     `analysis.visualize(paths=..., instance=..., during_analysis=True)`.
5. On loop exit, `finalise(state, dead)` + `log_weights` + repackage
   into `_NSSInternal` exactly as Phase 1 does.
6. On success, delete the checkpoint file (or leave it for the
   aggregator to record — see open question #2 below).

### New `__init__` kwarg

```python
checkpoint_interval: int = 100,
```

Default 100 outer iterations ≈ 5000-10000 evals at typical `num_delete=50`
— a reasonable trade-off between disk-write cost and restart granularity.

`iterations_per_quick_update` already exists in the Phase 1 signature
(accepted with a no-op log). Phase 3 turns it into the actual viz hook.

### Pickle format for checkpoints

`state` is a blackjax pytree of JAX arrays. Use `jax.tree_util.tree_map(np.asarray, ...)`
before pickle and `jax.tree_util.tree_map(jnp.asarray, ...)` on load —
standard JAX pytree-to-NumPy round-trip. `dead` is a Python list of
`NSInfo` pytrees, handled the same way (`tree_map(np.asarray, dead_item)`
per element).

Wrap in a small helper:

```python
def _save_checkpoint(path, state, dead, rng_key, iteration):
    to_numpy = lambda x: jax.tree_util.tree_map(np.asarray, x)
    blob = {
        "state": to_numpy(state),
        "dead": [to_numpy(d) for d in dead],
        "rng_key": np.asarray(rng_key),
        "iteration": int(iteration),
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)

def _load_checkpoint(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    to_jax = lambda x: jax.tree_util.tree_map(jnp.asarray, x)
    return (
        to_jax(blob["state"]),
        [to_jax(d) for d in blob["dead"]],
        jnp.asarray(blob["rng_key"]),
        int(blob["iteration"]),
    )
```

### Quick-update visualization

Mirror how Nautilus / Dynesty / Emcee do it — pull the current best live
point at iteration boundaries and call `analysis.visualize`. The
`Fitness.manage_quick_update` machinery from Phase 1 is the existing
plumbing; we just need to fire `analysis.visualize` from the outer loop
since `af.NSS` bypasses `Fitness._call`.

```python
if (
    self.iterations_per_quick_update is not None
    and iteration % self.iterations_per_quick_update == 0
):
    best_idx = int(state.particles.loglikelihood.argmax())
    best_params = np.asarray(state.particles.position[best_idx])
    instance = model.instance_from_vector(vector=best_params.tolist())
    analysis.visualize(
        paths=self.paths,
        instance=instance,
        during_analysis=True,
    )
```

The Phase 1 `iterations_per_quick_update` warning log goes away when
this lands.

__What to verify__

1. **Unit tests.**
   - `_save_checkpoint` + `_load_checkpoint` round-trip preserves a small
     pytree (synthetic `state`, `dead`, `rng_key`).
   - `NSS.__init__` accepts `checkpoint_interval` kwarg.
   - Resume detection — given an existing checkpoint file, `_fit` loads
     it instead of re-initialising (mock the loop body to avoid running
     real nss).

2. **Workspace_developer integration smoke** — new script
   `autolens_workspace_developer/searches_minimal/nss_checkpoint_resume.py`:
   - Run `af.NSS` for N iterations with `checkpoint_interval=10`.
   - Confirm `nss_checkpoint.pkl` exists at the expected path mid-run.
   - Kill the process (simulate SLURM timeout), restart with the same
     paths, confirm the run resumes from the saved state and continues
     to convergence.
   - Final `einstein_radius` and `log_evidence` should be within the
     same tolerance band as a single-shot run.

3. **Quick-update smoke** — extend
   `autolens_workspace_developer/searches_minimal/nss_first_class_gaussian.py`
   (or a sibling) to set `iterations_per_quick_update=5`, run, and
   confirm that the `paths.image_path` directory contains visualization
   PNGs that were updated during the run (not just at the end).

4. **Full pytest test_autofit** unchanged — change is additive at the
   `_fit` boundary.

__Out of scope__

- JIT persistent cache (`paths.search_internal/jax_cache/`) — separate
  follow-up. Each cold + resumed fit currently pays a 25-30 s compile.
- Install simplification — Phase 4 (`autofit/nss_install_simplification.md`).
- Workspace tutorial scripts — Phase 5.
- Replacing the Phase 1 stubbed `iterations_per_full_update` kwarg —
  it's API-parity-only for nss (nss doesn't have a full-update concept
  separate from outer-iteration cadence).

__Risks / open questions__

1. **Aggregator + checkpoint persistence.** After a successful run, do
   we keep `nss_checkpoint.pkl` on disk for forensic inspection, or
   delete it? Nautilus deletes its `checkpoint.hdf5` after a successful
   completion (`output_search_internal` calls `os.remove`). Mirror that
   pattern: delete on success, leave intact on interrupted runs.

2. **`dead` list memory growth.** For a 15-dim MGE problem at
   `num_delete=10, n_live=200`, a typical run accumulates ~5000 outer
   iterations × `num_delete=10` particles = 50000 dead-point pytrees in
   memory. Each is small (~15 floats + bookkeeping), but pickling 50000
   pytrees per checkpoint is wasteful disk I/O. Consider:
   - Streaming `dead` to a separate file via append-mode (e.g.
     `dead_iter_NNNN.pkl`) on each iteration, and only pickling `state`
     + `rng_key` + `iteration` to the main checkpoint.
   - Pickling the concatenated dead pytree (via `jax.tree_util.tree_map(jnp.concatenate, ...)`)
     periodically — one big pytree is cheaper than 50000 small ones.

   Defer this optimisation; first land the naïve `pickle.dump(dead)`
   path and measure the disk cost on a real run. The simpler scheme
   may be fast enough.

3. **Resume reproducibility.** A resumed run rebuilds `algo` + `one_step`
   from the same `loglikelihood_fn`, `prior_logprob`, and `num_delete` —
   the JIT recompiles but the math is identical to a fresh run that
   landed at the same `state`. RNG advancement is deterministic via the
   stored `rng_key`. Add a parity test: run for 50 iterations
   single-shot vs (run for 25, save, resume, run for 25 more) and
   compare `state.particles.position` byte-for-byte at iteration 50.

4. **Visualization frequency vs cost.** Each `analysis.visualize` call
   on the autolens HST MGE problem takes a few seconds (plot the model,
   fit residuals, etc.). At `iterations_per_quick_update=10` with
   `num_delete=50`, that's a viz every 500 evals — fine. Document this
   so users don't set `iterations_per_quick_update=1` and tank
   performance.

__Reference__

- `@PyAutoFit/autofit/non_linear/search/nest/nss/search.py` — current
  Phase 1 implementation
- `@PyAutoFit/autofit/non_linear/search/nest/nautilus/search.py` —
  reference checkpoint pattern (`checkpoint_file`,
  `output_search_internal`)
- `/home/jammy/venv/PyAuto/lib/python3.12/site-packages/nss/ns.py` —
  upstream `run_nested_sampling` source (the outer-loop pattern we copy)
- `@PyAutoPrompt/z_features/nss_first_class_sampler.md` — Phases 2 and
  3 in the sequenced roadmap
