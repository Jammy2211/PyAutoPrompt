## Speed up NNLS on consumer GPUs by hacking jaxnnls itself

This is a follow-up to the FFT precision / mixed-precision audit work
(see `autolens_workspace_developer/jax_profiling/jit/imaging/mge.py`). After
fixing the FFT bugs, the remaining single-JIT bottleneck on GPU for the
MGE imaging likelihood is the inversion's NNLS step. Profiling on an RTX 2060
laptop GPU showed:

| Step                       | GPU fp64 single JIT |
|----------------------------|---------------------|
| Mapping matrix             | 22 ms               |
| Blurred mapping matrix     | 19 ms (→ ~10 ms post-FFT-fix) |
| **Reconstruction (NNLS)**  | **20 ms**           |
| Full pipeline              | 37 ms               |

NNLS is ~54% of the single-JIT GPU pipeline cost, on a *40-element* problem.
On the same problem on CPU, NNLS takes 0.6 ms — the GPU is 30× *slower* than
CPU for this step. That's a kernel-launch-overhead signature, not a compute
problem.

### Why the obvious fixes don't work

We already considered and rejected two approaches in the upstream conversation:

- **`jax.pure_callback` to scipy.optimize.nnls.** Inside vmap, pure_callback is
  per-batch-element (no batched-callback semantics in current JAX), so on a
  vmap=8 sampler it adds ~40 ms of host transfers — a regression for the
  production hot path.
- **Cholesky-only fast-path with `lax.cond` fallback to NNLS.** Under vmap,
  `lax.cond` is rewritten as `lax.select` and *both* branches run, so we'd
  pay both Cholesky AND NNLS cost on every call. Worse, the empirical
  positivity-hit-rate during sampling is low (most lens models drawn from the
  prior produce negative MGE coefficients somewhere), so even the no-vmap
  case doesn't win on average.

Neither is the right tool. The actual answer is to make the in-JIT NNLS
itself cheaper.

### The library is small enough to hack

`jaxnnls` (the package PyAutoArray imports for the JAX NNLS path) is ~520
lines total, forked from `kevin-tracy/qpax`. It's installed at
`/home/jammy/venv/PyAutoGPU/lib/python3.10/site-packages/jaxnnls/` — read it
from there, and cross-reference https://github.com/kevin-tracy/qpax for the
upstream context.

The structure is:

- `jaxnnls/__init__.py` — re-exports `solve_nnls` and `solve_nnls_primal`
- `jaxnnls/pdip.py` — the primal-dual interior-point solver (used in the
  forward pass)
- `jaxnnls/pdip_relaxed.py` — the relaxed-KKT variant used for the backward
  pass (the gradient through the linear system)
- `jaxnnls/diff_qp.py` — the diff-through-QP wrapper

Note: `reconstruction_positive_only_from` in
@PyAutoArray/autoarray/inversion/inversion/inversion_util.py
documents itself as using "fnnls / Bro-Jong (1997) active-set NNLS" — that
is incorrect. The current implementation calls `jaxnnls.solve_nnls` which is
PDIP, not active-set fnnls. Fix the docstring as part of this work.

### Where the 20 ms goes

`pdip.py` declares `MAX_ITER = 50` at module scope. PDIP iterations under JIT
cannot terminate early without unrolling the loop, so every NNLS solve runs
the full 50 iterations regardless of true convergence. Each iteration does
a Cholesky factor + cho_solve + line search — a handful of GPU kernel
launches per iter. So:

  50 iter × ~5 kernel launches × ~80 µs/launch ≈ 20 ms

That's the actual cost model. It's not the math, it's the kernel-launch
multiplier.

### Things to investigate

These are roughly in increasing order of invasiveness. The aim is GPU speedup
for consumer-laptop hardware (RTX 2060-class, 6 GB VRAM, where typical
PyAutoLens users from less-resourced departments live). Anything that helps
CPU users too is a bonus, but NNLS is already cheap on CPU so the bar is GPU.

1. **Reduce `MAX_ITER`.** PDIP converges in ~5–15 iters for well-conditioned
   small problems. 50 is conservative. Run the existing JAX likelihood
   functions in `@autolens_workspace_test/scripts/jax_likelihood_functions/`
   with a forked jaxnnls at MAX_ITER = 20, 15, 10 and compare log-likelihoods
   against the MAX_ITER=50 reference. If 15 iters gives Δlog-likelihood << 1,
   that's a 3.3× NNLS speedup with no API change.

2. **Vectorise the inner Cholesky update.** PDIP's per-iter Cholesky factor
   is on a (K, K) matrix where K ≤ 40 for MGE. On GPU that's a single Cholesky
   kernel with significant launch overhead vs actual work. Look at whether
   `jax.lax.linalg.cholesky` can be replaced with an explicit fused
   triangular factorisation that runs as one kernel for K ≤ 64. Possibly
   a `jax.experimental.pallas` kernel.

3. **Check if `pdip_relaxed.py` is being used unnecessarily.** PyAutoArray's
   `reconstruction_positive_only_from` calls `solve_nnls` which dispatches
   between forward (`pdip.py`) and gradient (`pdip_relaxed.py`) paths. For the
   sampler hot path we don't need gradients — we just want the forward
   result. Make sure `solve_nnls` isn't carrying gradient infrastructure
   through the trace when only the forward pass is evaluated.

4. **Bypass jaxnnls for K ≤ N_THRESHOLD.** Below some K, the kernel-launch
   tax of PDIP dominates the actual NNLS work. For very small K (1–4)
   there are direct closed-form solutions (e.g., Lawson-Hanson on tiny
   systems can be unrolled). For MGE with N_THRESHOLD ≈ 40 this might be
   one big Pallas kernel rather than 50 launches.

5. **Vendor jaxnnls into PyAutoArray.** It's 520 lines. Upstream activity is
   minimal. If we end up making PyAutoLens-specific changes (lower MAX_ITER,
   skipped backward path, custom small-K kernel), it's cleaner to vendor it
   under @PyAutoArray/autoarray/inversion/nnls/ than to maintain a fork. Drop
   the runtime dependency. This is the last-resort move once we know we
   actually want code changes.

### What to measure

Use `@autolens_workspace_developer/jax_profiling/jit/imaging/mge.py` (with
USE_MIXED_PRECISION already supported via the working-tree changes from the
parent task) as the GPU baseline. Key numbers to track:

- Reconstruction (NNLS) per-call time, GPU single JIT — current 20 ms, target ≤ 10 ms
- Reconstruction (NNLS) per-call time, GPU vmap=3 — current ~6.5 ms, target ≤ 4 ms
- Full pipeline GPU single JIT — currently 37 ms, target ≤ 20 ms (combined
  with FFT precision fix from parent task)
- Δlog-likelihood vs MAX_ITER=50 reference — must stay below 1e-3 absolute
  on the HST regression dataset (log-likelihood ≈ 27,379)

Also worth running on the pixelization paths
(@autolens_workspace_developer/jax_profiling/jit/imaging/pixelization.py and
delaunay.py) since those genuinely need NNLS positivity (unlike MGE) and
have larger K (often 1000+ source pixels). The kernel-launch tax matters
less there because compute dominates, but iteration count still matters.

### Numerical safety

For MGE this is empirically benign: 40 well-conditioned columns, near-orthogonal
basis. For pixelization paths with K = 1000+ and condition numbers >1e6,
reducing PDIP iterations is riskier — verify on the rectangular and Delaunay
pipelines, not just MGE.

### Files to touch

- `/home/jammy/venv/PyAutoGPU/lib/python3.10/site-packages/jaxnnls/pdip.py`
  (read-only initially; if changes are needed, vendor under
  @PyAutoArray/autoarray/inversion/nnls/)
- @PyAutoArray/autoarray/inversion/inversion/inversion_util.py (the
  `reconstruction_positive_only_from` function and its incorrect docstring)
- @autolens_workspace_developer/jax_profiling/jit/imaging/mge.py (regression
  benchmark; do not modify, just run)

### Out of scope

- Replacing PDIP with active-set NNLS. The original docstring claims
  fnnls/Bro-Jong but the implementation is PDIP. Switching would be a much
  larger change with its own gradient story; if PDIP can be made fast enough,
  there's no reason to swap algorithms.
- Anything that reduces correctness silently (e.g., clipping negative
  coefficients to zero). The output must be NNLS-equivalent within
  numerical tolerance.
- vmap-batch sizing on the sampler side. That's a PyAutoFit concern, not
  PyAutoArray.
