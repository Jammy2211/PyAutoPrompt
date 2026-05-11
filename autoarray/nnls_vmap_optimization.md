## Speed up the JAX NNLS solver under vmap at production-fiducial size

This **supersedes** `autoarray/nnls_gpu_bottleneck.md` — that earlier prompt
was written from MGE-only profiling at a 40-element problem and prioritised
single-JIT GPU performance. Production runs use vmap (nested-sampling
proposes batches of parameter vectors) at much larger problem sizes, and
PR #60 (autolens_workspace_developer) makes the new bottleneck picture
concrete.

The earlier prompt's 5-lever structure (`MAX_ITER`, Cholesky vectorisation,
gradient-infrastructure check, K≤N closed-form bypass, vendor jaxnnls) is
all still valid — keep it as the "how to attack" reference. This prompt
re-frames "what to attack first" given the new vmap-relevant data.

### What the latest profiling found (PR #60)

Production-fiducial setup:
- Imaging dataset: HST, 0.05″/px, mask radius 3.5″
- Lens light: **MGE-60** (60 linear Gaussians) — the production-realistic
  setup, not single-Sersic
- Source: rectangular pixelization (35×35 = 1225 px) OR Delaunay
  (39×39 overlay → 1231 vertices)
- Combined K (NNLS dimension) = ~1285 for rectangular, ~1291 for Delaunay

A100 fp64 timings at this fiducial:

| Likelihood | Single-JIT full | vmap (batch=3) per call | vmap regress |
|---|---:|---:|---:|
| Rectangular | 25.1 ms | 34.8 ms | 1.39× |
| **Delaunay** | **49.6 ms** | **438 ms** | **8.84×** |

The Delaunay-vmap regression is the science-throughput wall. RTX 2060
shows the same pattern less dramatically (Delaunay vmap 590 → 1217 ms mp).

### Per-step vmap probes confirmed scipy is NOT the issue

Earlier rounds suspected `scipy.spatial.Delaunay` (inside `jax.pure_callback`
with `vmap_method="sequential"`) was the problem. Per-step vmap probes
on RTX 2060 (committed in `z_projects/profiling/scripts/{delaunay,
rectangular}_vmap_probe.py`) flip the story:

| Step | single (ms) | vmap (ms/call) | ratio |
|---|---:|---:|---:|
| Inversion setup (incl. scipy pure_callback) | 127 | **108** | **0.85×** |
| **NNLS reconstruction** | **111** | **141** | **1.27×** |
| Mapped + log_ev (slogdet) | 33 | 32 | 0.98× |

`pure_callback` actually scales **sublinearly** under vmap — its dispatch
overhead amortises across batch elements. NNLS scales **superlinearly**
because its iterations are inherently serial per batch element.

So: NNLS is the universal vmap target. Pure-JAX-Delaunay (the previous
"long-term win") is no longer the highest-value lever.

### Cross-likelihood comparison (where NNLS sits in each)

The three imaging-family likelihoods all use jaxnnls for the
`reconstruction_positive_only_from` step. Their setups differ in
problem size K and in what other steps share the budget.

**Production-fiducial summary (all from PR #56 / PR #60 measurements):**

| | MGE (#56) | Rectangular (#60) | Delaunay (#60) |
|---|---:|---:|---:|
| Lens light | MGE-30 source-only setup* | MGE-60 lens + rect source | MGE-60 lens + Delaunay source |
| Source type | MGE-20 source bulge | RectangularAdaptDensity 35×35 | Delaunay 39×39 overlay |
| Inversion K (matrix dim) | ~40 | 1225 + 60 = **1285** | 1231 + 60 = **1291** |
| Has scipy `pure_callback`? | No | No | Yes (Delaunay triangulation) |

*MGE PR #56 did not separate "lens light" from "source MGE" — both galaxies
use MGE bulges (20 Gaussians each = 40 total linear obj). It's the small-K
reference but not directly comparable to the rect/Delaunay production setup.

**A100 fp64 timings (latest measurements per PR):**

| Config | MGE (K=40) | Rectangular (K=1285) | Delaunay (K=1291) |
|---|---:|---:|---:|
| Full pipeline single | 5.7 ms | 25.1 ms | 49.6 ms |
| Full pipeline vmap (per call) | 2.4 ms | 34.8 ms | **438 ms** |
| **NNLS step single** | **2.0 ms** | ~16 ms | ~16 ms |
| NNLS share of single | 35% | 64% | 31% |
| vmap regress vs single | **0.4× (faster!)** | 1.39× | **8.84×** |

**RTX 2060 fp64 timings:**

| Config | MGE | Rectangular | Delaunay |
|---|---:|---:|---:|
| Full pipeline single | 43.7 ms | 537 ms | 590 ms |
| Full pipeline vmap | 23.9 ms | 567 ms | 954 ms |
| NNLS step single | ~12 ms | ~125 ms | ~110 ms |
| vmap regress | 0.55× | 1.06× | 1.62× |

**The pattern:**
- **MGE benefits from vmap** (small K, kernel-launch-bound NNLS — see prior
  prompt's analysis). Lever 1 (MAX_ITER) maps to "fewer launches".
- **Rectangular at production K barely regresses under vmap** (1.39× A100,
  1.06× RTX 2060). NNLS dominates the budget but it's compute-bound.
- **Delaunay at production K explodes under vmap** (8.84× A100). The
  combined K=1291 with the linear lens MGE columns hits something nasty
  in NNLS-under-vmap that doesn't appear at smaller K. **This is the
  unexplained 7× factor that needs diagnosis** (see "Open question"
  section below).

If the optimisation work successfully fixes NNLS under vmap, all three
likelihoods benefit: MGE keeps its existing "vmap is faster" status,
rectangular drops vmap regress to ~1.0×, and Delaunay hopefully drops
toward 2× or better (still won't beat single-JIT due to the inherent
batch-3 overhead, but 438 ms → 100 ms would be a 4× production speedup).

### What to measure (updated regression baselines)

Use the canonical references at:
- `autolens_workspace_developer/jax_profiling/jit/imaging/pixelization.py`
- `autolens_workspace_developer/jax_profiling/jit/imaging/delaunay.py`

Both now ship at the MGE-60 fiducial (PR #60). Their
`EXPECTED_LOG_EVIDENCE_HST` constants:
- Rectangular: `24746.105672366088`
- Delaunay:    `26288.321397232066`

Track these on every NNLS variant — log-evidence drift > rtol=1e-4 means
the optimisation broke something.

For wall-clock targets, the per-config profiles in
`z_projects/profiling/scripts/{pixelization,delaunay}_profile.py` give
side-by-side comparisons. Specific targets:

| Config | Current | Target | Rationale |
|---|---:|---:|---|
| Rect A100 fp64 vmap-per-call | 34.8 ms | **≤ 25 ms** | Within 1.0× of single-JIT (currently 1.39×) |
| Delaunay A100 fp64 vmap-per-call | 438 ms | **≤ 100 ms** | Within 2.0× of single-JIT (currently 8.84×) |
| Rect RTX 2060 fp64 vmap | 567 ms | **≤ 540 ms** | Within 1.0× of single-JIT |
| Delaunay RTX 2060 fp64 vmap | 954 ms | **≤ 700 ms** | NNLS share is biggest regress vector |

The Delaunay A100 vmap target is the headline science improvement —
4.4× speedup on production hot path. Anything that hits 200 ms (2× from
current) is already meaningful.

### What's likely to work (priority order, post-PR-#60 evidence)

1. **MAX_ITER reduction** (still the obvious first lever). The previous
   prompt's analysis of `pdip.py:MAX_ITER = 50 → ~5 kernel launches/iter
   → 20 ms` was at K=40. At K=1285 each Cholesky kernel actually does
   real work, so the kernel-launch model is less dominant. But fewer
   iterations still helps — both at the launch level AND at the linalg
   level. Test MAX_ITER ∈ {10, 15, 20, 30} against the rectangular and
   Delaunay log-evidence. Look for the inflection where Δlog-evidence
   exceeds 1e-3.

2. **Audit the gradient infrastructure** (lever 3 from prior prompt).
   For sampler use we don't need NNLS gradients — only the forward
   solve. Confirm `solve_nnls` isn't dragging `pdip_relaxed.py` machinery
   through the JIT trace. If it is, splitting forward/backward at the
   API level should give a clean speedup with no numerical change.

3. **The vmap pathology specifically.** This is the new highest-value
   item, not in the prior prompt. The 8.8× Delaunay vmap regression vs
   1.27× per-step probe ratio means the full-pipeline vmap is hitting
   something worse than just NNLS×3-batch. Hypotheses worth testing:
   - PDIP's per-iter Cholesky on (1291, 1291) under vmap might be
     causing XLA to materialise (3, 1291, 1291, 1291) workspace tensors
     that don't fit nicely. Use `jax.jit(jax.vmap(...)).lower(...).compile()
     .memory_analysis()` to compare.
   - The pure_callback for Delaunay triangulation runs sequentially
     (3 calls), but XLA may stall on the pure_callback's host
     synchronisation. Try removing the callback's `vmap_method`
     constraint, or bundling all 3 mesh grids into a single callback
     call.
   - Profile the full vmap path with `jax.profiler.start_trace(...)` to
     see where the wall-time actually goes — the 8.8× factor over the
     per-step decomposition is unexplained.

4. **Pallas kernel for the per-iter Cholesky** (lever 2 from prior).
   At K=1285 a `jax.lax.linalg.cholesky` is real work (not launch-bound),
   but a fused triangular factorisation could still help. Lower-priority
   than items 1-3 because it's invasive and only nibbles at the cost.

5. **Vendor jaxnnls** (lever 5 from prior). Last resort if items 1-4
   produce code changes worth maintaining.

### What to NOT spend time on (deprioritised by PR #60)

- **Pure-JAX Delaunay triangulation.** This was previously framed as
  the long-term win for unblocking A100. Per-step probes show the
  pure_callback already scales sublinearly under vmap — it's not the
  vmap bottleneck. The single-JIT speedup it would provide is small
  relative to the NNLS opportunity.
- **Mixed-precision tuning.** Three sweeps (PR #56/#57/#60) all show
  mp is essentially a no-op on A100 and a marginal-to-no-op effect on
  RTX 2060 at production scale. Not a useful lever.
- **vmap=8 batching tuning.** PR #60's vmap regression measurements
  use batch=3. Larger batch sizes are the sampler's lever, not NNLS's
  — sampler concern (PyAutoFit), not PyAutoArray.

### How to validate (full operational runbook)

#### Local-only validation (RTX 2060 + CPU)

1. Apply the NNLS change. If you're testing in-place: edit
   `/home/jammy/venv/PyAutoGPU/lib/python3.10/site-packages/jaxnnls/pdip.py`
   directly. If you're testing a vendored copy: import `from autoarray.inversion.nnls import solve_nnls`
   in `inversion_util.py`. Either way, `pip show jaxnnls` confirms which
   version is loaded.
2. Activate venv + worktree (or canonical), in this order:
   ```bash
   source /home/jammy/venv/PyAutoGPU/bin/activate
   source /home/jammy/Code/PyAutoLabs-wt/<your-task>/activate.sh   # if worktreed
   ```
   The PyAutoGPU venv has Python 3.10 + JAX-CUDA12 (the default `python`
   may resolve to the CPU-only PyAuto venv — explicit GPU venv first).
3. Run the per-config profilers from anywhere (they take absolute paths):
   ```bash
   # GPU fp64 — the headline config
   PYTHONUNBUFFERED=1 python -u \
     /home/jammy/Code/PyAutoLabs/z_projects/profiling/scripts/pixelization_profile.py \
     --config-name local_gpu_fp64 \
     --output-dir "$PYAUTO_ROOT/autolens_workspace_developer/jax_profiling/results/jit/imaging/pixelization"
   PYTHONUNBUFFERED=1 python -u \
     /home/jammy/Code/PyAutoLabs/z_projects/profiling/scripts/delaunay_profile.py \
     --config-name local_gpu_fp64 \
     --output-dir "$PYAUTO_ROOT/autolens_workspace_developer/jax_profiling/results/jit/imaging/delaunay"

   # mp variants
   ... add --use-mixed-precision --config-name local_gpu_mp ...

   # CPU variants — GOTCHA: use JAX_PLATFORM_NAME=cpu (NOT JAX_PLATFORMS=cpu;
   # JAX 0.4.38 has a bug with the new env var that errors with
   # "Unknown backend cuda" because pre-existing CUDA arrays from
   # register_model can't move).
   PYTHONUNBUFFERED=1 JAX_PLATFORM_NAME=cpu \
     NUMBA_CACHE_DIR=/tmp/numba_cache_cpu_$$ \
     python -u .../pixelization_profile.py --config-name local_cpu_fp64 ...
   ```
4. Per-step vmap probe (the diagnostic that lets you see WHERE NNLS
   speedup lands):
   ```bash
   PYTHONUNBUFFERED=1 python -u \
     /home/jammy/Code/PyAutoLabs/z_projects/profiling/scripts/delaunay_vmap_probe.py \
     --config-name local_gpu_fp64 --output-dir /tmp/nnls_test
   PYTHONUNBUFFERED=1 python -u \
     /home/jammy/Code/PyAutoLabs/z_projects/profiling/scripts/rectangular_vmap_probe.py \
     --config-name local_gpu_fp64 --output-dir /tmp/nnls_test
   ```
   Each probe times inversion-setup, NNLS, and log-ev steps both
   single and under vmap=3, and prints per-step vmap/single ratios.
   For a useful NNLS speedup you want the NNLS ratio to drop from
   1.27× toward 1.0×.
5. Confirm `EXPECTED_LOG_EVIDENCE_HST` assertion still passes — every
   per-config profiler asserts against the constant on every config
   that runs.

#### HPC validation (A100, where production lives)

The HPC profiling harness is a self-contained `z_projects/profiling/`
project. The `hpc/sync` tool wraps push/submit/pull/jobs. Key paths:

- HPC project dir: `/mnt/ral/jnightin/profiling`
- HPC venv: `/mnt/ral/jnightin/PyAuto/PyAuto/` (sourced by the project's
  `activate.sh` at submit time)
- Local working dir: `/home/jammy/Code/PyAutoLabs/z_projects/profiling`
- HPC sync config: `z_projects/profiling/hpc/sync.conf` (already
  configured: `HPC_HOST=euclid_jump`, etc.)

Operate from `/home/jammy/Code/PyAutoLabs/z_projects/profiling/`.

**Push code + check connection** (does NOT need the worktree, sync
operates on the canonical z_projects/profiling/):
```bash
cd /home/jammy/Code/PyAutoLabs/z_projects/profiling
hpc/sync check       # verify SSH + remote dir + sbatch availability
hpc/sync push        # rsync code + dataset (data uses --ignore-existing)
```

**Submit jobs** — one per (likelihood × precision) pair:
```bash
hpc/sync submit gpu submit_pixelization_profile_fp64
hpc/sync submit gpu submit_pixelization_profile_mp
hpc/sync submit gpu submit_delaunay_profile_fp64
hpc/sync submit gpu submit_delaunay_profile_mp
hpc/sync jobs        # show queue
```

Each prints `Submitted batch job <ID>`. Job IDs are sequential. Each
takes ~1–4 min A100 wall + queue wait (often instant if no other
user's array is in the queue, occasionally 15–30 min if there's
competition).

**Wait for completion in the background** (the sleep pattern works
even if you exit the shell — jobs run on the cluster regardless):
```bash
# Foreground if you're going to wait actively
until ! hpc/sync jobs 2>/dev/null | grep -qE "(<ID1>|<ID2>|<ID3>|<ID4>)"; do
  sleep 60
done; echo "ALL DONE"
```

**Pull results + consolidate**:
```bash
hpc/sync pull        # rsync down output/ + hpc/batch_gpu/{output,error}/
# Move HPC json+png pairs into the canonical worktree dir:
python scripts/pixelization_aggregate.py \
  --consolidate-from output/imaging/pixelization
python scripts/delaunay_aggregate.py \
  --consolidate-from output/imaging/delaunay
# Generate comparison.json + comparison.png across all 4–6 configs:
python scripts/pixelization_aggregate.py
python scripts/delaunay_aggregate.py
```

The aggregator honours `PYAUTO_ROOT` from the worktree's `activate.sh` —
canonical results land on the feature branch's worktree, not on
canonical-main. Without `PYAUTO_ROOT`, results go to canonical-main
(wrong if you're working on a branch).

**Common gotchas observed during PR #60:**
- After modifying any script in `z_projects/profiling/scripts/`, you
  MUST re-run `hpc/sync push` before resubmitting — otherwise HPC
  runs the OLD script and you get stale results.
- SLURM jobs report exit `0:0` even if Python crashes inside, because
  the bash submit script's epilogue (`echo "Finished."; date`) always
  runs. Verify success by reading `hpc/batch_gpu/output/output.<ID>.out`
  AND checking the JSON file mtime in `output/imaging/<lik>/`.
- `hpc/sync push` skips dataset files that already exist on HPC
  (`--ignore-existing`). If the local dataset was regenerated and
  HPC needs the new version, manual force-push the dataset.

#### A complete "is the optimisation working?" loop

```bash
# 0. Set up
cd /home/jammy/Code/PyAutoLabs/z_projects/profiling
source /home/jammy/venv/PyAutoGPU/bin/activate
source /home/jammy/Code/PyAutoLabs-wt/<task>/activate.sh

# 1. Local first — fast feedback (~5 min total for 4 GPU configs)
WORKTREE_OUT_PIX="$PYAUTO_ROOT/autolens_workspace_developer/jax_profiling/results/jit/imaging/pixelization"
WORKTREE_OUT_DEL="$PYAUTO_ROOT/autolens_workspace_developer/jax_profiling/results/jit/imaging/delaunay"
PYTHONUNBUFFERED=1 python -u scripts/pixelization_profile.py --config-name local_gpu_fp64 --output-dir "$WORKTREE_OUT_PIX"
PYTHONUNBUFFERED=1 python -u scripts/pixelization_profile.py --use-mixed-precision --config-name local_gpu_mp --output-dir "$WORKTREE_OUT_PIX"
PYTHONUNBUFFERED=1 python -u scripts/delaunay_profile.py --config-name local_gpu_fp64 --output-dir "$WORKTREE_OUT_DEL"
PYTHONUNBUFFERED=1 python -u scripts/delaunay_profile.py --use-mixed-precision --config-name local_gpu_mp --output-dir "$WORKTREE_OUT_DEL"

# 2. Probe shows per-step vmap-vs-single — confirms NNLS share dropped
PYTHONUNBUFFERED=1 python -u scripts/delaunay_vmap_probe.py --config-name local_gpu_fp64 --output-dir /tmp/probe
PYTHONUNBUFFERED=1 python -u scripts/rectangular_vmap_probe.py --config-name local_gpu_fp64 --output-dir /tmp/probe

# 3. HPC for the production-fiducial number
hpc/sync push
hpc/sync submit gpu submit_pixelization_profile_fp64
hpc/sync submit gpu submit_pixelization_profile_mp
hpc/sync submit gpu submit_delaunay_profile_fp64
hpc/sync submit gpu submit_delaunay_profile_mp
# (wait for completion — see watcher snippet above)
hpc/sync pull
python scripts/pixelization_aggregate.py --consolidate-from output/imaging/pixelization
python scripts/delaunay_aggregate.py --consolidate-from output/imaging/delaunay
python scripts/pixelization_aggregate.py
python scripts/delaunay_aggregate.py

# 4. The tables in the comparison.json's headline section + the printed
#    stdout summary are what you compare against the targets in this
#    prompt's "What to measure" section.
```

Decision point: if MAX_ITER reduction alone gets us to the targets,
ship as a config change (no vendor). If it requires structural
changes, vendor `jaxnnls` under `autoarray/inversion/nnls/`.

### Open question: the 8.8× vmap-regress mystery

The per-step probes on RTX 2060 show NNLS scaling at 1.27× under vmap.
The full-pipeline Delaunay vmap on A100 shows 8.84× regress. Even if
NNLS is the bottleneck, 1.27× × 1.0 (for the rest) ≠ 8.84×. There's a
~7× factor unaccounted for.

Possible explanations:
- A100 NNLS vmap behaves much worse than RTX 2060 NNLS vmap (test:
  port `delaunay_vmap_probe.py` to HPC and run on A100)
- XLA scheduler pessimisation at this matrix size only kicks in for
  the full-pipeline graph, not for per-step sub-graphs
- Some other step (slogdet? pure_callback?) regresses much harder
  under the full graph than in isolation

Resolving this is a precondition for the optimisation work — without
it, we can fix NNLS and still see most of the 8.8× regression remaining.

### Files to touch / read

Read-only first:
- `/home/jammy/venv/PyAutoGPU/lib/python3.10/site-packages/jaxnnls/pdip.py`
  (the PDIP solver — `MAX_ITER = 50` is at module scope)
- `/home/jammy/venv/PyAutoGPU/lib/python3.10/site-packages/jaxnnls/pdip_relaxed.py`
  (gradient path; check whether it's needed)
- `@PyAutoArray/autoarray/inversion/inversion/inversion_util.py`
  (`reconstruction_positive_only_from` — incorrect docstring claims
  fnnls/Bro-Jong, actually calls jaxnnls PDIP)
- `@autolens_workspace_developer/jax_profiling/jit/imaging/pixelization.py`
  + `delaunay.py` (canonical refs at MGE-60 fiducial; the regression
  benchmarks with `EXPECTED_LOG_EVIDENCE_HST` constants)
- `@z_projects/profiling/scripts/{pixelization,delaunay,delaunay_vmap_probe,
  rectangular_vmap_probe}_*.py` (the per-config profilers + vmap probes)

If changes needed (in roughly this order):
- Fork `jaxnnls`, change `MAX_ITER`, install
- `@PyAutoArray/autoarray/inversion/inversion/inversion_util.py`
  (fix the docstring lie at minimum)
- Vendor `jaxnnls` to `@PyAutoArray/autoarray/inversion/nnls/`
  (only if MAX_ITER reduction isn't enough)

### Reference precedent

- PR #60 (autolens_workspace_developer): production fiducial profiling
  that produced the headline 8.8× Delaunay vmap regression.
  https://github.com/PyAutoLabs/autolens_workspace_developer/pull/60
  See its body for the headline numbers and the bottleneck-shift table
  across the three iterations of "make it production-realistic".
- PR #57 + #58 (same repo): the smaller-scale predecessors.
- `autoarray/nnls_gpu_bottleneck.md` (this repo): the prior framing,
  superseded but its 5-lever structure is still valid as a how-to.
- `z_projects/profiling/scripts/delaunay_vmap_probe.py` and
  `rectangular_vmap_probe.py`: the per-step decomposition probes that
  produced the "scipy pure_callback is fine, NNLS is the wall"
  finding.

### Out of scope

Same as the prior prompt:
- Replacing PDIP with active-set NNLS (much bigger change with its own
  gradient story).
- Anything that silently reduces correctness (clipping negatives, etc.).
- vmap-batch sizing on the sampler side (PyAutoFit concern).

Plus newly:
- Pure-JAX Delaunay triangulation (deprioritised — see "What to NOT
  spend time on").
- Mixed-precision micro-optimisation (not a lever at production scale).

### Definition of done

PR to a feature branch on PyAutoArray (and possibly a small companion
to autolens_workspace_developer to bump the regression artifacts at
the new NNLS implementation), passing:

1. `EXPECTED_LOG_EVIDENCE_HST` assertions for both rectangular + Delaunay
   at MGE-60 fiducial (rtol=1e-4 fp64, rtol=1e-2 mp)
2. Delaunay A100 vmap_per_call ≤ 200 ms (2× improvement, conservative
   target — the headline is ≤ 100 ms but anything significant counts)
3. Rect A100 vmap_per_call ≤ 25 ms (within 1.0× of single-JIT)
4. No regression on the existing autolens_workspace_test JAX likelihood
   functions (rtol=1e-4 vs current main)
