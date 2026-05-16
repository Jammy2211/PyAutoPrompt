__Phase 4 of `z_features/jax_visualization.md` (archived under
`z_features/complete/jax_visualization.md`). Feasibility prototype — NOT
a production implementation.__

Today PyAutoFit's `iterations_per_quick_update` mechanism triggers
visualization **synchronously** inside the search process:

```
search loop iteration N
  → fitness evaluation
  → if N % iterations_per_quick_update == 0:
       analysis.visualize(fit)       # <— search blocks here
  → continue loop
```

For a fast Nautilus fit on a pixelization model the synchronous render
adds seconds of wall-clock per quick-update — and as Phase 1-3 made JAX
viz the default path, the gap between "search ready for next iteration"
and "viz finished writing PNGs" only widens.

The user-facing goal stated in the archived tracker is:

> "Visualization (including via quick updates) would happen on a separate
> process to the search so we don't 'pause' to visualize."

This task is the **architectural decision** that unlocks that goal. It
does NOT ship the production implementation. It ships:

1. A prototype branch in PyAutoFit demonstrating the chosen approach
   working end-to-end on one workflow.
2. A short Architecture Decision Record (ADR) — written as
   `@PyAutoFit/docs/adr/0001-subprocess-visualization.md` (new docs
   subfolder is OK) — answering the four open questions below with
   measurements where applicable.

The production implementation (Phase 4 proper) is a follow-up prompt
authored once this feasibility lands.

__Why this matters__

Phase 5 of the JAX visualization roadmap is "live Jupyter / Colab cell
that updates in place every `iterations_per_quick_update`". That
explicitly depends on Phase 4 — the cell-update has to happen without
blocking the kernel, which requires the renderer to live outside the
search process. So Phase 4's design choices constrain Phase 5's IPC
shape. Getting Phase 4 right is the load-bearing decision.

__Four open architectural questions__

Each must be answered in the ADR with reasoning and (where the
prototype gives a signal) measurements.

**Q1 — IPC mechanism.** Three plausible options:

| Option | Pros | Cons |
|--------|------|------|
| `multiprocessing.Process` + `Queue` | First-class Python primitive, no disk I/O on hot path, queue gives natural backpressure handle | Forking JAX-initialised processes has known caveats (`os.fork()` warnings already appear in our smoke logs — see `feedback_jax_pure_callback_const_fold` for context). Worker startup cost. |
| `concurrent.futures.ProcessPoolExecutor` | Cleaner API, future-based cancellation, persistent worker pool | Same fork caveats. Less control over per-task lifecycle. |
| `subprocess.Popen` with disk-staged state | Worker is a separate Python process from cold start — full failure isolation from JAX runtime in parent. Easy to debug (inspect on-disk staging dir). | Disk I/O on every quick-update. Need a stable serialisation format. Startup cost per render unless we keep the worker alive. |

Bias-disclosure: the tracker's framing prefers IPC option 1 or 2, but
the prototype should measure both Python-native options on at least one
realistic workflow (a pixelization imaging fit) and only fall back to
disk-staged subprocess if the in-process options crash or stall under
JAX. Don't pre-commit.

**Q2 — `FitImaging` picklability.** After Phase 0a-0c (pytree
registration shipped in PR #364 / #376 / #401), `FitImaging`,
`FitInterferometer`, `FitEllipse`, and `FitQuantity` are JAX
pytree-registered. That makes them flattenable to leaves + aux data,
but does NOT automatically make them `pickle`-able. The prototype
must determine:

- Does a populated `FitImaging` round-trip cleanly through
  `pickle.dumps(...)` / `pickle.loads(...)` today?
- If not, where does it break — closures over the `Analysis`, captured
  PyAutoLens model instances, cosmology distance caches, NUFFT
  operators (interferometer specifically), JAX-cached `_jitted_fit_from`
  attributes set by `fit_for_visualization`?
- Is the fix "strip JAX-cached attrs in `__getstate__`" (analogous to
  `Fitness.__getstate__` in `@PyAutoFit/autofit/non_linear/fitness.py`
  — see `assert_pickle_strips_jax_cached_attrs` in
  `autofit_workspace_test/scripts/jax_assertions/fitness_dispatch.py`),
  or is the right shape to send raw arrays + reconstruct the
  `FitImaging` in the worker from `(dataset, instance, settings)`?

This is the highest-risk question. If `FitImaging` can't pickle and
can't be made to pickle cheaply, the IPC choice falls back to "send
arrays + reconstruct in worker", which constrains the worker design.

**Q3 — Backpressure.** What happens when the search produces
quick-update events faster than the worker can render? The prototype
must pick **one** policy and justify it:

- **Drop**: discard the in-flight render request, send the new one.
  Simplest. Users always see the latest state. Stale render frames
  lost.
- **Queue**: buffer up to N pending renders. Smoothest visual cadence
  if the renderer briefly stalls. Risks unbounded memory + lag if the
  search is much faster than the renderer for a sustained period.
- **Coalesce**: merge consecutive requests, render only the latest
  when the worker becomes free. Functionally similar to drop but
  preserves the "render this one" semantics.

Drop is probably the right default for `iterations_per_quick_update`
(by definition it's a status snapshot, not a frame). The prototype
should confirm by measuring on the chosen workflow: how often does
the producer outpace the consumer?

**Q4 — Failure isolation.** A render crash MUST NOT take the search
down. The prototype must demonstrate (e.g. by deliberately raising
inside the renderer):

- Worker death is detected within one quick-update cycle.
- The search continues; subsequent quick-updates either restart the
  worker or log-and-skip.
- A user-visible warning fires once (not on every subsequent
  iteration) so the user knows visualization stopped.

__What to build (prototype scope)__

Branch off `main` in PyAutoFit. The prototype touches:

1. **`@PyAutoFit/autofit/non_linear/analysis/analysis.py`** — extend
   `fit_for_visualization` (or add a sibling `dispatch_visualization`)
   to optionally route through the worker. Keep the synchronous path
   as the default; gate the new path on a boolean kwarg or env var.

2. **`@PyAutoFit/autofit/non_linear/visualization/` (new module)** —
   the worker harness. One Python file containing:
   - `WorkerHandle` — start/stop lifecycle, send-render, health check.
   - The chosen IPC primitive's wiring (queue / pool / Popen).
   - The chosen backpressure policy.
   - Failure-isolation glue.

3. **`@PyAutoFit/autofit/non_linear/fitness.py`** or wherever
   `iterations_per_quick_update` triggers visualization — wire the new
   dispatch in behind the feature gate.

4. **One end-to-end demo** in
   `@autolens_workspace_test/scripts/jax_assertions/subprocess_viz_prototype.py`
   that runs a pixelization imaging fit with the new dispatch and
   measures: search wall-clock with vs without subprocess viz, render
   latency, dropped-frame count.

Resist the urge to make this production-clean. The prototype's job is
to answer the four questions, not ship the final API.

__What success looks like__

- ADR at `@PyAutoFit/docs/adr/0001-subprocess-visualization.md`
  picking one IPC mechanism + one backpressure policy + the
  picklability resolution, each with a one-paragraph rationale and (for
  IPC) the prototype measurement that justified it.
- A merged prototype branch on PyAutoFit (or kept as a long-lived
  feature branch — user's call at ship time) with the worker harness
  and the demo script working end-to-end.
- A short follow-up prompt at
  `PyAutoPrompt/autofit/visualization_subprocess_production.md`
  authored before this prompt closes, with the production scope
  derived from the ADR's recommendations.

__What is OUT of scope__

- Production-quality error reporting, logging configuration, or user
  config plumbing (`config/visualize.yaml` etc.). Hardcode for the
  prototype.
- Phase 5 (Jupyter/Colab live-cell rendering). The worker design must
  not preclude Phase 5 — but actually wiring up `IPython.display` with
  a stable `display_id` is the next prompt.
- GPU resource sharing between search and renderer. The prototype can
  assume CPU rendering or that the GPU is large enough for both.
- Cleaning up `autolens_workspace_test`'s ~12 redundant
  `use_jax_for_visualization=True` call sites (separate trivial
  cleanup; not gated on this).
- Production migration of the existing synchronous `analysis.visualize`
  call sites in PyAutoGalaxy / PyAutoLens visualizers. The dispatch
  change lives in PyAutoFit's base `Analysis`; downstream visualizers
  are not touched by the prototype.

__References__

- `@PyAutoFit/autofit/non_linear/analysis/analysis.py:36-138` — `Analysis.__init__` and `fit_for_visualization` after the Phase 2 sentinel change (PR #1278)
- `@PyAutoFit/autofit/non_linear/fitness.py` — `iterations_per_quick_update` plumbing + `Fitness.__getstate__` pickle precedent
- `complete.md` entry `use-jax-for-vis-default` — Phase 2 ship notes, the immediate predecessor
- `complete.md` entries `fit-imaging-pytree*`, `ag-ellipse-quantity-pytree`, `autogalaxy-viz-dispatch-swap` — Phase 0 pytree foundations
- `autofit_workspace_test/scripts/jax_assertions/fitness_dispatch.py::assert_pickle_strips_jax_cached_attrs` — existing precedent for stripping JAX-cached attrs during pickle
- `z_features/complete/jax_visualization.md` — archived tracker, Phase 4 + Phase 5 stubs preserved verbatim
- `feedback_jax_pure_callback_const_fold` (memory) — context on JAX + multiprocessing fork interactions seen during smoke runs
