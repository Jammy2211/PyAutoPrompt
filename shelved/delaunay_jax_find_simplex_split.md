# SHELVED — Split scipy.Delaunay callback: JAX find_simplex + optional process pool

**Status**: ready-to-ship code sitting on a feature branch. Paused
2026-05-11 because the measured speedup at production batch=20
(~1.19×) was deemed not worth the source-code change risk vs other
levers the user wants to pursue first.

**If you decide to resume and ship this:** the code is complete and
validated; the work below is roughly half a day of mechanical effort
(end-to-end log-evidence validation, default-flag decision, PR opening,
review iteration).

## Where the code lives

**Branch (pushed):** `feature/delaunay-jax-find-simplex` on
`PyAutoLabs/PyAutoArray`. One commit `eda747c2` adds:

- `autoarray/inversion/mesh/interpolator/delaunay_jax_locate.py` (new)
- `autoarray/inversion/mesh/interpolator/delaunay_scipy_pool.py` (new)
- `autoarray/inversion/mesh/interpolator/delaunay.py` (modified)
- `autoarray/config/general.yaml` (two new flags, both default false)

Pull it back with:
```bash
git -C ~/Code/PyAutoLabs/PyAutoArray fetch origin feature/delaunay-jax-find-simplex
git -C ~/Code/PyAutoLabs/PyAutoArray checkout feature/delaunay-jax-find-simplex
# OR via worktree:
source ~/Code/PyAutoLabs/admin_jammy/software/worktree.sh
worktree_create delaunay-jax-find-simplex PyAutoArray
git -C ~/Code/PyAutoLabs-wt/delaunay-jax-find-simplex/PyAutoArray reset --hard origin/feature/delaunay-jax-find-simplex
source ~/Code/PyAutoLabs-wt/delaunay-jax-find-simplex/activate.sh
```

The HPC's pip-editable PyAutoArray install was restored to canonical
main when the work was shelved, so no orphaned modified library on HPC.

## What the change does

`jax_delaunay()` in `autoarray/inversion/mesh/interpolator/delaunay.py`
gains a config-flag dispatch:

- **flag off (default):** behaviour unchanged — monolithic
  `scipy_delaunay` runs in one `jax.pure_callback`. Identical to main.
- **flag on:** scipy callback runs ONLY the Delaunay triangulation.
  `find_simplex` (point location for the data grid and for the
  split-cross-points), the barycentric dual area computation and the
  split-points construction all run as JAX-native code under vmap.

The new `delaunay_jax_locate.py` implements
`jax_find_simplex_and_gather()` — a chunked brute-force barycentric
point-in-triangle search with NN-1 fallback for outside-hull queries.
The chunking (default `chunk_size=128`) keeps peak VRAM bounded under
vmap so the user's prior memory-blow-up doesn't recur.

The new `delaunay_scipy_pool.py` exposes a persistent
`ProcessPoolExecutor` whose workers each pre-import scipy. When
`delaunay_scipy_pool_workers > 0`, the scipy triangulation callback
runs via `vmap_method='legacy_vectorized'` with all batch elements
dispatched to the pool concurrently.

The `barycentric_dual_area_from()` function in the same file was
broken for `xp=jnp` (used the numpy-only `xp.add.at` mutation idiom);
it now branches on `xp.__name__.startswith('jax')` and uses
`jax.ops.segment_sum` for the JAX path.

## Measured speedup at production batch=20 on A100 fp64

```
Config                          inversion_setup    full_pipeline    batch_time   vs baseline
                                vmap (ms/elem)    vmap (ms/elem)    (ms)
─────────────────────────────────────────────────────────────────────────────────────────────
Baseline (flag off)                     51.3              69.5         1390        1.00x
Split (flag on)                         39.86             58.27        1165        1.19x
Split + pool_workers=4                  45.82             56.67        1133        1.23x
Split + pool_workers=8                  42.12             60.35        1207        1.15x
```

Key reading:
- **Split alone gives the bulk of the win** (1.19×). The pool is
  marginal at b=20 because scipy.Delaunay per call (~3-5 ms) is small
  relative to IPC overhead (~1-2 ms per dispatch).
- **pool=8 is slower than pool=4** at b=20 — diminishing returns.
- **NNLS and log_ev are unaffected** by the change — only
  inversion_setup moves (51.3 → 39.86 ms per element).

Speed measurement files on HPC:
```
output/imaging/delaunay/hpc_a100_fp64_b20.json                       # baseline (existing main)
output/imaging/delaunay/hpc_a100_fp64_b20_split_vmap_probe.json      # split, pool=0
output/imaging/delaunay/hpc_a100_fp64_b20_pool4_split_pool_vmap_probe.json
output/imaging/delaunay/hpc_a100_fp64_b20_pool8_split_pool_vmap_probe.json
```

## Validation done

1. **Algorithm correctness vs scipy.find_simplex**: 0 disagreements on
   4 test cases including production scale (1231 mesh × 15361 queries,
   ~5000 outside-hull queries). See
   `z_projects/profiling/scripts/nnls_prototypes/delaunay_jax_locate_check.py`.
2. **Memory under vmap**: no OOM at vmap=20 on either RTX 2060 6GB or
   A100 80GB. The chunking strategy works as designed.
3. **Runtime parity**: NNLS and log_ev steps produced identical timings
   with flag on vs off (within bench noise).

## Validation NOT yet done — must complete before shipping

1. **End-to-end log-evidence at rtol=1e-4** on the cross-pipeline
   matrix. The autolens_workspace_developer canonical regression at
   `jax_profiling/jit/imaging/delaunay.py` (constant
   `EXPECTED_LOG_EVIDENCE_HST = 26288.321397232066`) must hold with the
   flag on. Note: that canonical currently has an unrelated
   curvature/regularization matrix shape bug — the regression run may
   need to be done via a small custom script (or wait for that bug to
   be fixed).
2. **autolens_workspace_test/scripts/jax_assertions/nnls.py** still
   passes with the flag on. This tests the well-conditioned-matches and
   ill-conditioned-finite-gradient cases.
3. **The interferometer Delaunay pipeline** at
   `autolens_workspace_developer/jax_profiling/jit/interferometer/delaunay.py`
   also holds at rtol=1e-4 with the flag on.

If validation passes at rtol=1e-4 → default the flag to `true`. If only
rtol=1e-3 passes → ship as opt-in (default `false`, document as
"available, may shift log-evidence by ~1e-4 — opt in if speed matters
more than reproducibility").

## Steps to resume and ship

### 1. Recreate the worktree and activate

```bash
source ~/Code/PyAutoLabs/admin_jammy/software/worktree.sh
worktree_create delaunay-jax-find-simplex PyAutoArray
cd ~/Code/PyAutoLabs-wt/delaunay-jax-find-simplex/PyAutoArray
git fetch origin feature/delaunay-jax-find-simplex
git reset --hard origin/feature/delaunay-jax-find-simplex
source ~/Code/PyAutoLabs-wt/delaunay-jax-find-simplex/activate.sh
```

### 2. Re-deploy to HPC for benchmark runs (if needed)

```bash
WT=~/Code/PyAutoLabs-wt/delaunay-jax-find-simplex/PyAutoArray
for f in \
    autoarray/inversion/mesh/interpolator/delaunay.py \
    autoarray/inversion/mesh/interpolator/delaunay_jax_locate.py \
    autoarray/inversion/mesh/interpolator/delaunay_scipy_pool.py \
    autoarray/config/general.yaml; do
  rsync "$WT/$f" "euclid_jump:/mnt/ral/jnightin/PyAuto/PyAutoArray/$f"
done
```

### 3. End-to-end log-evidence validation

Cleanest: write a small script that builds the same fiducial as
`autolens_workspace_developer/jax_profiling/jit/imaging/delaunay.py`
(MGE-60 + Delaunay source), evaluates `analysis.log_likelihood_function`
under vmap once with flag off and once with flag on, asserts both
within rtol=1e-4 of `EXPECTED_LOG_EVIDENCE_HST = 26288.321397232066`.

Reference setup module: `z_projects/profiling/scripts/_setup_delaunay.py`.

Then the same for interferometer Delaunay
(`autolens_workspace_developer/jax_profiling/jit/interferometer/delaunay.py`,
its `EXPECTED_LOG_EVIDENCE_SMA`).

Finally rerun
`autolens_workspace_test/scripts/jax_assertions/nnls.py` with the flag
on.

### 4. Decide on default

- All three pass rtol=1e-4 → set `delaunay_jax_find_simplex: true` in
  `autoarray/config/general.yaml`. Production picks up the 1.19×
  automatically.
- Only rtol=1e-3 passes → keep default `false`, document carefully.
- Any test fails → investigate. The most likely cause is a tolerance
  difference in barycentric vs scipy's exact Qhull predicates. May
  need to tighten `bary_tolerance` in `jax_find_simplex_and_gather()`
  or add tie-breaking against the largest-min-bary triangle.

### 5. Open the PR

```bash
gh pr create --repo PyAutoLabs/PyAutoArray \
  --title "perf: split scipy.Delaunay callback — JAX find_simplex (1.19x on production Delaunay)" \
  --body "$(cat <<'EOF'
## Summary

- Splits `jax_delaunay()` so scipy.Delaunay handles only the
  triangulation; find_simplex point location, barycentric dual areas
  and split-points run as JAX-native code under vmap.
- Adds a new `delaunay_jax_locate.jax_find_simplex_and_gather` helper
  with chunked memory-bounded point-in-triangle search.
- Fixes `barycentric_dual_area_from` for `xp=jnp` (the prior
  `xp.add.at` idiom is numpy-only and silently broken for JAX).
- Adds an opt-in `delaunay_scipy_pool_workers` config flag that
  parallelises the scipy triangulation across batch elements via a
  persistent `ProcessPoolExecutor`. Default 0 = no pool (sequential
  callback, current behaviour).

## Speedup at production batch=20 on A100 fp64

| Config | full_pipeline ms/elem | speedup |
|---|---:|---:|
| Baseline (current main) | 69.5 | 1.00x |
| `delaunay_jax_find_simplex=true` | 58.27 | **1.19x** |
| + `delaunay_scipy_pool_workers=4` | 56.67 | 1.23x |

Bench: `z_projects/profiling/scripts/nnls_prototypes/delaunay_vmap_probe_split.py`,
HPC A100 fp64, MGE-60 fiducial (1231 mesh vertices, 15361 over-sampled
data pixels), batch_size=20.

## Test plan

- [ ] EXPECTED_LOG_EVIDENCE_HST (Delaunay imaging) holds at rtol=1e-4
- [ ] EXPECTED_LOG_EVIDENCE_SMA (Delaunay interferometer) holds at rtol=1e-4
- [ ] autolens_workspace_test/scripts/jax_assertions/nnls.py passes
- [ ] No VRAM blow-up under vmap=20 on RTX 2060 6GB or A100 80GB

🤖 Generated with Claude Code

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### 6. Post-merge cleanup (per CLAUDE.md)

After PR merges:
```bash
source ~/Code/PyAutoLabs/admin_jammy/software/worktree.sh
worktree_remove delaunay-jax-find-simplex
git -C ~/Code/PyAutoLabs/PyAutoArray checkout main
git -C ~/Code/PyAutoLabs/PyAutoArray pull --ff-only
git -C ~/Code/PyAutoLabs/PyAutoArray branch -d feature/delaunay-jax-find-simplex
git -C ~/Code/PyAutoLabs/PyAutoArray push origin --delete feature/delaunay-jax-find-simplex
# Move this prompt from shelved/ to issued/ (or delete)
rm ~/Code/PyAutoLabs/PyAutoPrompt/shelved/delaunay_jax_find_simplex_split.md
```

## Why this was shelved

The user wants to pursue other Delaunay-speedup options before
committing to this 1.19× change. Specifically:

1. **InterpolatorKNNBarycentric wildcard** — prompt at
   `PyAutoPrompt/autoarray/knn_barycentric.md`. If that experiment
   passes log-evidence rtol=1e-4 against true Delaunay, it eliminates
   scipy entirely AND the JAX find_simplex cost, obsoleting this work.
2. **Other JAX-traced inversion_setup work** (~25 ms per element
   beyond find_simplex) is a separate, potentially larger lever.

If those options don't pan out, this branch is ready to ship.

## File-level summary

```
PyAutoArray/
├── autoarray/inversion/mesh/interpolator/
│   ├── delaunay.py            (modified: dispatch + barycentric_dual_area_from fix)
│   ├── delaunay_jax_locate.py (new: chunked find_simplex + gather + NN fallback)
│   └── delaunay_scipy_pool.py (new: persistent ProcessPoolExecutor)
└── autoarray/config/
    └── general.yaml           (modified: two new flags, both default false)
```

Diff stat:
```
 autoarray/config/general.yaml                                   |   2 +
 autoarray/inversion/mesh/interpolator/delaunay.py               | 161 +++++++++++++++++++++-
 autoarray/inversion/mesh/interpolator/delaunay_jax_locate.py    | 178 +++++++++++++++++++++++++
 autoarray/inversion/mesh/interpolator/delaunay_scipy_pool.py    | 147 ++++++++++++++++++++++
 4 files changed, 488 insertions(+), 3 deletions(-)
```

Probe scripts in `z_projects/profiling/scripts/nnls_prototypes/`
(canonical, untracked) that supported this work:
- `delaunay_jax_locate_check.py` — algorithm validation vs scipy
- `find_simplex_vmap_test.py` — RTX 2060 vmap memory check
- `delaunay_vmap_probe_split.py` — A100 benchmark, split path only
- `delaunay_vmap_probe_split_pool.py` — A100 benchmark, split + pool

HPC submit scripts in `z_projects/profiling/hpc/batch_gpu/`:
- `submit_delaunay_vmap_probe_split_b20`
- `submit_delaunay_vmap_probe_split_pool4_b20`
- `submit_delaunay_vmap_probe_split_pool8_b20`

## Reference findings

- `z_projects/profiling/FINDINGS_nnls_v2.md` — the parent
  nnls-vmap-speedup investigation findings (closed issue #307).
- `PyAutoPrompt/autoarray/delaunay_research.md` — the deep-research
  doc that proposed the split + pool approach.
