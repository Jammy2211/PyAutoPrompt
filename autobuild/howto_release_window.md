# Incorporate HowToFit / HowToGalaxy / HowToLens into PyAutoBuild's release window

While shipping `welcome-start-here-fixes` (autolens_workspace#108) we discovered
the HowTo* repos are only **partially** wired into the release infrastructure.
The release pipeline already releases them — but several pieces of the
surrounding admin (labels, version pinning, docs) treat them as second-class.
This task is to close those gaps so HowToFit/HowToGalaxy/HowToLens are
first-class members of every release window, on the same footing as
`autofit_workspace` / `autogalaxy_workspace` / `autolens_workspace`.

## What's already wired in (don't redo)

PyAutoBuild's release.yml already handles the HowTos:

- `pre_build.sh:64-66` — runs `black` + `generate.py` + commit/push for HowToGalaxy, HowToLens, HowToFit.
- `release.yml:178-199` (`find_scripts`) — checks out all three HowTos and feeds them into `script_matrix.py`.
- `release.yml:222-230` (`generate_notebooks`) — regenerates notebooks for `howtogalaxy` / `howtolens` / `howtofit`.
- `release.yml:703-717` (`release_workspaces`) — bumps Colab URLs, **writes `version.txt`**, tags, and pushes to `main` for all three. This is what *will* drag HowToLens's `version.txt` from `2026.4.21.0` back to whatever the next release tags it as.
- `bump_colab_urls.sh:12,30` — regex already covers HowToGalaxy/HowToLens/HowToFit.

So the release pipeline itself doesn't need new stages. The work is around it.

## Gaps to close

### 1. `pending-release` label drift

Across all PyAutoLabs workspace + test + HowTo + euclid repos:

| Repo | Label | Color | Description |
|------|-------|-------|-------------|
| autolens_workspace | yes | `ededed` (default gray) | _(empty)_ |
| autogalaxy_workspace | yes | `0E8A16` | "PR queued for the next release build" |
| autofit_workspace | yes | `0E8A16` | "PR queued for the next release build" |
| **HowToLens** | **no** | — | — |
| **HowToGalaxy** | **no** | — | — |
| HowToFit | yes | `0075ca` | "Awaiting release" |
| autolens_workspace_test | yes | `0075ca` | "Merged but awaiting library release" |
| autogalaxy_workspace_test | yes | `0075ca` | "Awaiting next release to go live" |
| autofit_workspace_test | yes | `FBCA04` | "Pending next release build" |
| **euclid_strong_lens_modeling_pipeline** | **no** | — | — |

Three repos are missing the label, four colors and five descriptions across the
ones that do have it. The label is what `gh pr create --label pending-release`
attaches inside `/ship_workspace`; when it's missing the PR ships unlabeled and
falls outside any "PRs awaiting release" sweep.

Pick a canonical color/description (suggest `0E8A16` + `"PR queued for the next
release build"` since two repos already match), then bootstrap+standardize all
ten via `gh api`:

```bash
# Create where missing (POST):
gh api -X POST repos/PyAutoLabs/HowToLens/labels                          -f name=pending-release -f color=0E8A16 -f description="PR queued for the next release build"
gh api -X POST repos/PyAutoLabs/HowToGalaxy/labels                        -f name=pending-release -f color=0E8A16 -f description="PR queued for the next release build"
gh api -X POST repos/PyAutoLabs/euclid_strong_lens_modeling_pipeline/labels -f name=pending-release -f color=0E8A16 -f description="PR queued for the next release build"

# Standardize the seven that already have it (PATCH):
for repo in autolens_workspace autogalaxy_workspace autofit_workspace HowToFit \
            autolens_workspace_test autogalaxy_workspace_test autofit_workspace_test; do
  gh api -X PATCH "repos/PyAutoLabs/$repo/labels/pending-release" \
    -f color=0E8A16 -f description="PR queued for the next release build"
done
```

Better still, write `admin_jammy/software/ensure_workspace_labels.sh` (or a Python
script) that idempotently asserts the label exists with the right config across
a hardcoded list of repos. Run it manually after creating any new workspace-style
repo. Could also be invoked at the top of `pre_build.sh` so each release run
self-heals the label state.

### 2. CLAUDE.md `pre_build.sh` table is out of date

`PyAutoBuild/CLAUDE.md` describes the `pre_build.sh` table but lists only
HowToGalaxy and HowToLens — **HowToFit is missing from the table** even though
it's at `pre_build.sh:66`. Update the table to include HowToFit (and
`euclid_strong_lens_modeling_pipeline` as a `false` entry in the generate
column, matching `pre_build.sh:63`).

### 3. Aspirational `version.txt` at bootstrap

The HowToLens bootstrap commit (`62a5f3a`, 2026-04-21) set `version.txt` to
`2026.4.21.0` — a release that didn't exist. Every `welcome.py` run between
bootstrap and the next release dispatch fails with
`WorkspaceVersionMismatchError`. We just patched it to match the current
library (`2026.4.13.6`) — but the next release will overwrite that anyway via
the `Write workspace version.txt` step in `release.yml:763-770`.

Two ways to prevent the recurrence:

- **Process:** when bootstrapping a new workspace-style repo from a template,
  initialise `version.txt` with the **currently-installed** library version,
  not the date the bootstrap commit lands. Could codify this in
  `start-new-project` / template scaffolding.
- **Tooling:** add a step in `pre_build.sh` (or a separate
  `verify_workspace_versions.sh`) that fails fast if any workspace's
  `version.txt` is ahead of the installed library, before the release dispatch
  runs.

### 4. Wire `/ship_workspace` and `/start_workspace` to assert the label

When `/ship_workspace`'s subagent calls `gh pr create --label pending-release`
and the label doesn't exist, GitHub returns an error and the PR is created
unlabeled — silently. Two fixes worth considering:

- Have `/start_workspace` ensure the label exists in each workspace repo it
  touches, before any PR is opened. Cheap idempotent API call.
- Have `/ship_workspace` *fail loudly* (not silently) if the label apply
  rejected. Right now the failure was buried in stderr and only surfaced when
  I cross-checked label state manually post-merge.

## Test plan

- `release.yml` already supports `skip_release=true` and `skip_scripts=true`.
  After making any change to release.yml itself, dispatch with
  `--field skip_release=true --field skip_scripts=true --field skip_notebooks=true`
  to dry-run the pipeline (only `find_scripts` / `version_number` / setup jobs
  run). Confirm matrix entries for howtogalaxy/howtolens/howtofit are still
  present.
- After running the label-bootstrap snippet, verify with:
  ```bash
  for repo in PyAutoLabs/{autolens_workspace,autogalaxy_workspace,autofit_workspace,HowToLens,HowToGalaxy,HowToFit,autolens_workspace_test,autogalaxy_workspace_test,autofit_workspace_test,euclid_strong_lens_modeling_pipeline}; do
    gh api "repos/$repo/labels/pending-release" --jq '"\(.color) \(.description)"'
  done
  ```
  Every line should print `0E8A16 PR queued for the next release build`.

## Files likely to change

- `admin_jammy/software/ensure_workspace_labels.sh` — new helper
- `PyAutoBuild/pre_build.sh` — optionally invoke the helper at the top
- `PyAutoBuild/CLAUDE.md` — add HowToFit to the table, and euclid_pipeline as a
  `false` row
- `PyAutoLabs/<workspace>` repos — label state via API (no file changes)
- `claude/skills/start_workspace.md` (or wherever the skill lives) — label
  assertion step
- `claude/skills/ship_workspace.md` — convert silent label-apply failure into a
  hard failure
