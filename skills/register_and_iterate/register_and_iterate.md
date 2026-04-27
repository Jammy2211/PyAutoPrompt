# Register and Iterate: Autonomous Pytree PoC Loop

Autonomously work through a queue of `fit_*_pytree_*.md` prompts: scaffold each variant's
`_pytree.py`, run it, register offending types until the `jax.jit(analysis.fit_from)` round-trip
matches NumPy, then hand off to `/ship_library` + `/ship_workspace`. Pause only at hard judgment
gates.

## Usage

```
/register_and_iterate <prompt1>[,<prompt2>,...]
/register_and_iterate --queue          # reads PyAutoPrompt/queue.md
```

Prompts are paths relative to `PyAutoPrompt/` (e.g. `autolens/fit_imaging_pytree_rectangular.md`).

## Autonomy Contract

The skill runs without intervention **except** at these gates, where it writes a clear
question and stops:

1. **Aux/dynamic judgment** on an offending type whose classification is not obvious
   (non-Array attributes, callable state, known-gotcha class).
2. **Ship PR approval** — `/ship_library` and `/ship_workspace` always require user sign-off
   on the `## API Changes` / `## Scripts Changed` sections. Never bypass.
3. **Hard blockers** — a type that cannot be registered (e.g. an iterative solver with
   Python-level state). Stop and write up the blocker per the prompt's fallback clause.

Between tasks, the skill auto-advances to the next queued prompt. All iteration, scaffolding,
testing, and post-merge cleanup runs unattended.

## Per-Prompt Flow

### 1. Resume or start the task

Read `PyAutoPrompt/active.md`. If the prompt's derived task name is already an active
entry, resume it — source the worktree's `activate.sh`, verify the feature branch, continue.
Otherwise invoke `/start_dev <prompt>` followed by `/start_library` logic inline (create
issue, create worktree, register repos in `active.md`).

The derived task name is the filename stem (e.g. `fit_imaging_pytree_rectangular` → task name
`fit-imaging-pytree-rectangular`).

### 2. Check dependencies

Parse the prompt's `__Depends on__` section. For each dependency, verify it appears in
`PyAutoPrompt/complete.md`. If a dependency is missing, stop and tell the user:

```
Cannot start <this task>: dependency <dep task> has not shipped.
  Dependency prompt: PyAutoPrompt/<dep path>
  Run /register_and_iterate on the dependency first, or remove it from this queue.
```

Do **not** re-order the queue automatically — the user chose the order.

### 3. Scaffold `<variant>_pytree.py`

Copy `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py` as the
template. From the prompt's `__Reference script__` line, locate the reference model file, and
splice the reference's dataset + mask + model construction into the template. Replace the
assertion body with the three-step pattern:

```python
# 1. NumPy reference scalar
ref = analysis.fit_from(instance).log_likelihood

# 2. jit-wrapped round-trip
jit_ll = jax.jit(lambda i: analysis.fit_from(i).log_likelihood)(instance)

# 3. Assert
assert jnp.allclose(ref, jit_ll, rtol=1e-4), f"divergence: {ref} vs {jit_ll}"
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
```

Target path: `autolens_workspace_test/scripts/jax_likelihood_functions/<path>/<variant>_pytree.py`.

### 4. Iterate the registration loop

Set `max_iters = 8`. Each iteration:

1. Run the script inside the worktree with `activate.sh` sourced.
2. If `PASS` prints, break the loop.
3. If it errors, parse the traceback. Identify the offending type — usually the frame
   immediately above the JAX tracer error, or a `TypeError: unhashable type` / `NotImplementedError`
   from `jax.tree_util`.
4. **Classify the offender** using the heuristic below.
5. Register via `register_instance_pytree(<Cls>, no_flatten=<aux_fields>)` at the appropriate
   registration site (`_register_fit_imaging_pytrees` in PyAutoLens, or the equivalent for
   the variant — add a new site if the prompt says so). Include a one-line comment: the
   variant task name that introduced the registration.
6. Re-run.

If `max_iters` is exhausted with no `PASS`, stop and write the full iteration log to
`~/Code/PyAutoLabs-wt/<task>/register_iterate.log` and ask the user to take over.

### 5. Classification heuristic

Read the offending class's `__init__` and `__dict__`. Apply this decision tree:

- **All attributes are `jax.Array`, `np.ndarray`, `AbstractNDArray`, or primitives** (int, float,
  bool, str, None, tuple-of-primitives): auto-register with `no_flatten=()` — all dynamic.
- **Has attributes matching known-aux patterns**: `cosmology`, `settings`, `config`, `dataset`,
  `psf`, `mask`, `_cache`, `_compiled`, any `scipy.spatial.*`, any `Transformer*`, any
  `PointSolver*`: auto-register with those fields in `no_flatten`.
- **Has a callable attribute** (function, method, class reference): **pause**. Ask the user:

  ```
  Classification needed for <ClassName>:

    File: <path/to/class.py>
    Attributes: <list of __dict__ keys with types>

    Suggested no_flatten: <your best guess>

    Reply with:
      aux      — register with all attrs in no_flatten (entire object rides as static)
      dyn      — register with no_flatten=() (entire object is traced)
      split    — specify which attrs are aux: e.g. "split: settings, _cache"
      skip     — don't register; investigate manually
      blocker  — this class can't be registered (explain why in follow-up)
  ```

- **On the prompt's `__What's likely to surface__` hit-list**: proceed per the prompt's
  guidance (e.g. `TransformerNUFFT` → aux, `NNLS solver state` → blocker).

### 6. After PASS

1. Run the affected library repo's test suite under the worktree (`pytest test_<repo>/ -x`).
   If red, stop and show failures.
2. Also run the new `_pytree.py` script one more time as a smoke check.
3. Stage changes in each touched repo. Do **not** commit yet.
4. Move to step 7.

### 7. Gate: ship approval

Write a summary to the conversation:

```
<task-name> — ready to ship.

Registrations added:
  - <Cls1> in <repo/file>  no_flatten=<fields>
  - <Cls2> in <repo/file>  no_flatten=<fields>

Workspace script: <path>
Test suite: PASS (<N> tests)

Run /ship_library and /ship_workspace, or reply "abort <reason>" to stop the queue.
```

Wait for user confirmation. **Do not auto-invoke `/ship_library`.** The API Changes body is
user-facing and must be reviewed. When the user runs the ship skills, they handle tests,
commit, push, PR, merge, and post-merge cleanup.

### 8. Advance

After both PRs for this task are merged and post-merge cleanup completes, read the queue
and start the next prompt. If queue is empty, print a final summary and stop.

## Queue Mode

When invoked with `--queue`, read the queue from `PyAutoPrompt/queue.md`:

```markdown
# Pytree variant queue

- autolens/linear_light_profile_intensity_dict_pytree.md
- autolens/fit_imaging_pytree_lp.md
- autolens/fit_imaging_pytree_rectangular.md
# ... etc
```

Skip lines that are blank or start with `#`. Process in order. When a task ships successfully,
prepend `# DONE <date> ` to its line (don't delete — preserves ordering history).

If the user hasn't created `queue.md`, write the recommended order from the mge-jit-visualization
session into it and ask them to confirm before proceeding:

```
1. autolens/linear_light_profile_intensity_dict_pytree.md
2. autolens/fit_imaging_pytree_lp.md
3. autolens/fit_imaging_pytree_rectangular.md
4. autolens/fit_imaging_pytree_mge_group.md
5. autolens/fit_imaging_pytree_delaunay.md
6. autolens/fit_interferometer_pytree_mge.md
7. autolens/fit_point_pytree.md
8. autolens/fit_imaging_pytree_rectangular_mge.md
9. autolens/fit_imaging_pytree_rectangular_dspl.md
10. autolens/fit_imaging_pytree_delaunay_mge.md
11. autolens/fit_interferometer_pytree_mge_group.md
12. autolens/fit_interferometer_pytree_rectangular.md
```

## Delegation

Inside each task's iteration loop, delegate the mechanical run-and-register cycle to a Sonnet
subagent to save tokens. The outer skill (on Opus, or whichever model is running the skill)
keeps context across tasks and handles the judgment gates.

```
Agent(
  model="sonnet",
  subagent_type="general-purpose",
  prompt="""
  Inside ~/Code/PyAutoLabs-wt/<task>/, source activate.sh and run
  <task>/autolens_workspace_test/scripts/jax_likelihood_functions/<path>/<variant>_pytree.py.

  If it errors: identify the offending type from the traceback. Read its class definition.
  Classify using this heuristic: <paste step 5>.

  If classification is unambiguous, register it via register_instance_pytree at the
  appropriate site (<repo>/<path>). Re-run.

  Max 8 iterations. Stop if:
    - PASS prints (success)
    - An ambiguous classification is needed (pause, report the class + attrs)
    - A blocker is found (pause, report why)
    - Iters exhausted (pause, report full log)

  Report: a table of (Class, Registration Site, no_flatten fields, iteration number).
  """
)
```

The outer skill reads the subagent's report, handles any pause reasons by querying the user,
then either resumes the subagent with the decision or advances.

## Scope Boundaries

- **Do not** modify `use_jax` dispatch, the `xp is np` guard, or any non-JAX code path.
- **Do not** auto-run `/ship_library` or `/ship_workspace` — those are human gates.
- **Do not** re-order the queue. If a dependency is missing, stop and ask.
- **Do not** commit or push — `/ship_library` handles that.
- **Do not** delete or modify existing registrations, even if they look wrong. Flag them.

## Failure Modes and Recovery

- **Iteration limit reached**: dump log to `register_iterate.log` in the worktree, pause queue.
- **Dependency not shipped**: pause, tell user which prompt to run first.
- **Test suite regression after PASS**: stop, show failing tests. Do not ship.
- **Worktree already dirty on resume**: warn, ask whether to discard or continue.
- **Classification ambiguity**: pause with the structured prompt from step 5.
- **Subagent reports a blocker**: stop the queue, write up the blocker per the prompt's
  fallback clause (usually a markdown file describing the obstruction). Advance only when
  the user explicitly confirms.

## Reporting

At the end of the queue (or when paused), print:

```
Queue progress
==============

Shipped (N):
  - <task> → <library PR> + <workspace PR>
  - ...

Paused (M):
  - <task> — <reason>
  - ...

Pending (K):
  - <task>
  - ...
```
