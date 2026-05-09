---
name: profile_likelihood
description: Run a multi-config JAX likelihood profiling sweep (CPU/GPU × fp64/mp, optional A100 on RAL HPC) for a given likelihood function. Produces structured JSON+PNG outputs with comparison tables under autolens_workspace_developer/jax_profiling/results/jit/.
user-invocable: true
---

Profile a JAX likelihood function across multiple device + precision configurations and consolidate results into the canonical long-term tracking directory `autolens_workspace_developer/jax_profiling/results/jit/<dataset_type>/<likelihood_type>/`. Mirrors the workflow that produced the `mge-profiling-a100` PR (autolens_workspace_developer #56) and reuses the tooling at `z_projects/profiling/scripts/`.

## Usage

```
/profile_likelihood <dataset_type> <likelihood_type>
```

Examples:

```
/profile_likelihood imaging mge                       # what mge-profiling-a100 ran
/profile_likelihood imaging delaunay                  # Delaunay pixelization
/profile_likelihood imaging rectangular               # Rectangular pixelization
/profile_likelihood imaging delaunay_mge              # MGE light + Delaunay source
/profile_likelihood interferometer rectangular_mge    # Interferometer with both
```

`<dataset_type>` is one of `imaging` / `interferometer` / `point_source`. `<likelihood_type>` is one of the canonical reference scripts present at `autolens_workspace_developer/jax_profiling/jit/<dataset_type>/<likelihood_type>.py` (mge, delaunay, rectangular, delaunay_mge, rectangular_mge, etc.).

## What this produces

After completion, `autolens_workspace_developer/jax_profiling/results/jit/<dataset_type>/<likelihood_type>/` contains:

```
local_cpu_fp64.{json,png}
local_cpu_mp.{json,png}
local_gpu_fp64.{json,png}
local_gpu_mp.{json,png}
hpc_a100_fp64.{json,png}
hpc_a100_mp.{json,png}
comparison.{json,png}
```

Each `<config>.json` carries: per-step JIT timings for the 8 likelihood steps (ray-trace, mapping matrix, blurred mapping matrix, data vector, curvature, NNLS, mapped reconstructed image, chi-squared), full-pipeline single-JIT time, vmap batch=3 time, log-likelihood values from eager/JIT/vmap, device info, and static memory analysis.

`comparison.{json,png}` is the aggregated cross-config view (log-scale grouped bar chart by step, table of full pipeline + vmap + log-likelihood per config).

## Prerequisites — verify before starting

1. **Hardware**: a CUDA-capable GPU is needed for the `local_gpu_*` configs. If running on a CPU-only machine, skip to a 4-config sweep without GPU rows (the aggregator handles missing configs gracefully).

2. **HPC access**: the A100 sweep requires `z_projects/profiling/hpc/sync.conf` configured (HPC_HOST, HPC_BASE, PROJECT_NAME). Run `hpc/sync check` to verify SSH + remote project dir + sbatch availability. If HPC is unavailable, the local sweep alone is still valuable.

3. **Venvs**:
   - Local: `/home/jammy/venv/PyAutoGPU/bin/activate` (Python 3.10 + JAX-CUDA12). The default `python` may resolve to the CPU-only PyAuto venv — explicitly source `/home/jammy/venv/PyAutoGPU/bin/activate` BEFORE the worktree's `activate.sh`.
   - HPC: `/mnt/ral/jnightin/PyAutoNSS/PyAutoNSS/bin/activate`, sourced by `z_projects/profiling/activate.sh` automatically when SLURM jobs run.

4. **Canonical reference exists**: confirm `autolens_workspace_developer/jax_profiling/jit/<dataset_type>/<likelihood_type>.py` exists. The new profiling script is a simplified, argparse-driven version of this canonical reference.

## Steps

### 1. Identify the likelihood and check existing tooling

Verify the canonical reference and the per-likelihood profiling script:

```bash
ls autolens_workspace_developer/jax_profiling/jit/<dataset_type>/<likelihood_type>.py
ls z_projects/profiling/scripts/<likelihood_type>_profile.py 2>/dev/null
```

If the profile script already exists, skip to step 4. Otherwise, scaffold it in step 2.

### 2. Scaffold per-likelihood profiling script (if first time for this likelihood)

Use `z_projects/profiling/scripts/mge_profile.py` as the template — it imports `_setup.build_dataset` / `_setup.build_model` / `_setup.build_analysis` to produce dataset/model/analysis byte-identical to the existing nss_jit / nautilus runs.

For MGE (already templated), the existing `_setup.py` works. For other likelihoods you may need:

- A new `_setup_<likelihood_type>.py` module exporting `build_dataset / build_model / build_analysis / gpu_info` for that specific likelihood. Mirror `_setup.py` but adjust `build_model` to use the right pixelization / profile / regularization classes from the canonical reference at `autolens_workspace_developer/jax_profiling/jit/<dataset_type>/<likelihood_type>.py`.
- A new `<likelihood_type>_profile.py` mirroring `mge_profile.py` but:
  - imports the new `_setup_<likelihood_type>` instead of `_setup`
  - adapts the per-step JIT calls to whatever step structure the canonical reference defines (some likelihoods have extra steps like regularization-matrix construction or border relocation)
  - keeps the same JSON schema so the aggregator works unchanged

For likelihood types where the canonical reference is identical-structure to MGE (just different `build_model`), only `_setup_<likelihood_type>.py` is needed and the profile script can stay generic.

### 3. Create SLURM submit scripts (if first time for this likelihood)

Clone `z_projects/profiling/hpc/batch_gpu/submit_mge_profile_{fp64,mp}` to `submit_<likelihood_type>_profile_{fp64,mp}`, updating:

- The `#SBATCH -J` job name
- The python invocation: `--config-name hpc_a100_<fp64|mp>` and the script path
- The `--output-dir $PROJECT_PATH/output/<dataset_type>/<likelihood_type>` argument

Make both executable: `chmod +x z_projects/profiling/hpc/batch_gpu/submit_<likelihood_type>_profile_*`.

### 4. Plan + branch survey

Before any commits, run:

```
/plan_branches
```

Affected repos: `z_projects/profiling` (new tooling, local-only — no remote, commits to local main directly) and `autolens_workspace_developer` (result artifacts, has remote — feature branch + PR). Suggested branch name: `feature/<likelihood_type>-profiling-a100`.

Worktree the autolens_workspace_developer feature branch — z_projects/profiling has no remote so it gets symlinked back to canonical:

```bash
source admin_jammy/software/worktree.sh
worktree_create <likelihood_type>-profiling-a100 autolens_workspace_developer
```

### 5. Local sweep — 4 configs

Activate venvs in the right order (PyAutoGPU first, then worktree):

```bash
source /home/jammy/venv/PyAutoGPU/bin/activate
source /home/jammy/Code/PyAutoLabs-wt/<likelihood_type>-profiling-a100/activate.sh
```

The worktree's `activate.sh` exports `PYAUTO_ROOT` — both `mge_profile.py` and `mge_aggregate.py` honor it so canonical writes land on the feature branch (not on canonical-main). If you skip the worktree activate, results land on canonical-main, which is wrong.

Run the 4 configs:

```bash
WORKTREE_OUTPUT="$PYAUTO_ROOT/autolens_workspace_developer/jax_profiling/results/jit/<dataset_type>/<likelihood_type>"

# Local GPU
python z_projects/profiling/scripts/<likelihood_type>_profile.py \
  --config-name local_gpu_fp64 --output-dir "$WORKTREE_OUTPUT"
python z_projects/profiling/scripts/<likelihood_type>_profile.py \
  --use-mixed-precision --config-name local_gpu_mp --output-dir "$WORKTREE_OUTPUT"

# Local CPU
JAX_PLATFORM_NAME=cpu python z_projects/profiling/scripts/<likelihood_type>_profile.py \
  --config-name local_cpu_fp64 --output-dir "$WORKTREE_OUTPUT"
JAX_PLATFORM_NAME=cpu python z_projects/profiling/scripts/<likelihood_type>_profile.py \
  --use-mixed-precision --config-name local_cpu_mp --output-dir "$WORKTREE_OUTPUT"
```

Note: use `JAX_PLATFORM_NAME=cpu` (not `JAX_PLATFORMS=cpu` — the newer env var has a config bug in JAX 0.4.38 where existing CUDA arrays from the registered model can't be moved to CPU).

Each run takes ~1–3 minutes on a consumer laptop (most of it JIT compilation). Spot-check the JSON: `device.backend` should be `gpu` for the GPU rows and `cpu` for the CPU rows. If GPU runs accidentally fall back to CPU, the venv activation is wrong — re-source `/home/jammy/venv/PyAutoGPU/bin/activate` explicitly.

### 6. (Optional) Pre-fix data ingestion

If this is the first profiling of this likelihood AND there are old session artifacts in `/tmp` (e.g. the user has run the canonical reference manually before), the aggregator can ingest them as `*_pre_fix` configs:

```bash
python z_projects/profiling/scripts/<likelihood_type>_aggregate.py \
  --ingest-pre-fix /tmp
```

The aggregator looks for `/tmp/<likelihood_type>_<config>_summary.json` and `/tmp/<likelihood_type>_<config>.log` (`config` ∈ `{cpu_fp64, cpu_mp, gpu_fp64, gpu_mp}`). It parses both formats and writes normalised `local_*_pre_fix.json` files into the canonical dir. Skip this step if no /tmp artifacts exist.

### 7. HPC sweep — A100 fp64 + mp

Push the project (skipping data, since it's already on HPC):

```bash
cd z_projects/profiling
hpc/sync push
```

Submit both jobs and queue:

```bash
hpc/sync submit gpu submit_<likelihood_type>_profile_fp64
hpc/sync submit gpu submit_<likelihood_type>_profile_mp
hpc/sync jobs
```

Each A100 job takes ~5 minutes wall time. Use `/loop` to schedule a wakeup if you want to detach:

```
/loop check job <fp64_job_id> and <mp_job_id> via hpc/sync jobs; if both gone, hpc/sync pull and consolidate.
```

When both jobs are gone from `hpc/sync jobs`, pull:

```bash
hpc/sync pull
```

### 8. Consolidate HPC results into canonical tracking dir

```bash
python z_projects/profiling/scripts/<likelihood_type>_aggregate.py \
  --consolidate-from z_projects/profiling/output/<dataset_type>/<likelihood_type>
```

This copies `hpc_a100_*.{json,png}` from the staging dir (which `hpc/sync pull` populated) into the canonical worktree path resolved via `$PYAUTO_ROOT`.

### 9. Aggregate

```bash
python z_projects/profiling/scripts/<likelihood_type>_aggregate.py
```

Prints the comparison table to stdout and writes `comparison.json` + `comparison.png` (log-scale grouped bar chart) into the canonical dir.

Spot-check the table: A100 rows should be O(1)–O(10) ms per call; consumer GPU is O(10)–O(100) ms; CPU is O(100)–O(1000) ms. If any row is wildly off (e.g. CPU run accidentally went through GPU because `JAX_PLATFORM_NAME` wasn't honored), re-run that single config.

### 10. Commit + open PR

In z_projects/profiling (no remote, local main):

```bash
cd /home/jammy/Code/PyAutoLabs/z_projects/profiling
git add scripts/<likelihood_type>_profile.py scripts/<likelihood_type>_aggregate.py \
        hpc/batch_gpu/submit_<likelihood_type>_profile_fp64 \
        hpc/batch_gpu/submit_<likelihood_type>_profile_mp \
        scripts/_setup_<likelihood_type>.py 2>/dev/null  # if scaffolded
git commit -m "add <likelihood_type> profiling: A100 + local sweep, with comparison aggregator"
```

In the worktree's autolens_workspace_developer:

```bash
cd /home/jammy/Code/PyAutoLabs-wt/<likelihood_type>-profiling-a100/autolens_workspace_developer
git add jax_profiling/results/jit/<dataset_type>/<likelihood_type>/
git commit -m "add jax_profiling/results/jit/<dataset_type>/<likelihood_type>/ — <likelihood_type> A100 + RTX 2060 + CPU sweep"
git push -u origin feature/<likelihood_type>-profiling-a100
gh pr create --title "Add <likelihood_type> profiling: A100 + RTX 2060 + CPU sweep" \
             --body-file <pre-drafted PR body>
```

PR body should include:

- Headline timings table (full pipeline + vmap per call across all configs)
- Key findings (fp64 vs mp gap on each device class, A100 vs RTX 2060 ratio)
- Caveats (jax_enable_x64 status on HPC, JAX cache state warning for cross-session deltas)
- Test plan checklist

Reference the `mge-profiling-a100` PR (#56) as the template if helpful.

### 11. Post-merge cleanup

After merge:

```bash
source admin_jammy/software/worktree.sh
worktree_remove <likelihood_type>-profiling-a100

git -C /home/jammy/Code/PyAutoLabs/autolens_workspace_developer fetch origin
git -C /home/jammy/Code/PyAutoLabs/autolens_workspace_developer checkout main
git -C /home/jammy/Code/PyAutoLabs/autolens_workspace_developer pull --ff-only
git -C /home/jammy/Code/PyAutoLabs/autolens_workspace_developer branch -d feature/<likelihood_type>-profiling-a100
```

Move the active.md entry to complete.md and `prompt_sync_push` per the standard post-merge cleanup in CLAUDE.md.

## Gotchas worth knowing

- **`JAX_PLATFORMS=cpu` doesn't work** on JAX 0.4.38 — it errors with "Unknown backend cuda" because pre-existing CUDA arrays from `register_model` can't move. Use `JAX_PLATFORM_NAME=cpu` (older API).
- **GPU venv is PyAutoGPU, not PyAuto.** The default `python` may resolve to the CPU-only PyAuto venv. Always activate `/home/jammy/venv/PyAutoGPU/bin/activate` BEFORE the worktree's `activate.sh` for GPU runs. Verify with `python -c "import jax; print(jax.default_backend())"` — should print `gpu`.
- **`PYAUTO_ROOT` for canonical-vs-worktree path resolution.** The aggregator and profile scripts look at this env var (set by worktree `activate.sh`) so writes land on the feature branch. Without it, writes go to canonical main and the result artifacts won't be on your PR.
- **A100 jax_enable_x64**: the HPC `PyAutoNSS` venv may not have `jax_enable_x64=True` set by default. JIT-path log-likelihood will truncate to fp32 (~7 digits). Eager numpy reference is always fp64 and is the trustworthy value for correctness audits. See the caveat in `mge-profiling-a100` PR (#56) for context.
- **Single-machine cross-session deltas are unreliable.** JAX cache state and GPU thermal state vary. Cross-platform comparisons (A100 vs RTX 2060 vs CPU) are robust; pre-fix vs post-fix on the same machine across different sessions less so.
- **HPC dataset can be stale.** `hpc/sync push` skips dataset files that already exist on HPC (`--ignore-existing`). If the simulator regenerated the dataset locally, run `hpc/sync push-data-init` to force a re-upload.
- **Pre-existing dirty files**: don't `git add -A` or `prompt_sync_push` without first checking `git status` — workspace_developer often has unrelated dirty files from other in-progress work (e.g. euclid_bug/). Stage only the new `<likelihood_type>/` subdir.

## Reference precedent

- `mge-profiling-a100` (autolens_workspace_developer #56, merged 2026-05-09) is the first run of this workflow. See its PR body and `complete.md` entry for the full numerical story.
- The skill leans on `z_projects/profiling/scripts/mge_profile.py` and `mge_aggregate.py` as the canonical implementations. They live at https://github.com/PyAutoLabs/z_projects (no remote, local-only) and were committed as commit `781cb76` on local main.
