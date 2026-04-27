# Start Workspace: Set Up Workspace Development

Set up the development environment for workspace repository changes (autofit_workspace, autogalaxy_workspace, autolens_workspace, autolens_workspace_test, euclid_strong_lens_modeling_pipeline, HowToLens). Handles two entry modes: linked to an upstream library PR, or standalone.

## Conflict Guard

Before doing anything, check whether any workspace repo this task needs is already held by another task's worktree.

```bash
source admin_jammy/software/worktree.sh
worktree_check_conflict <task-name> <workspace_repo1> [workspace_repo2 ...]
```

A conflict only fires when another entry in `active.md` already claims one of the target workspace repos via its `worktree:` field. Two tasks that touch different workspace repos can run in parallel — that is the whole purpose of worktrees.

If `worktree_check_conflict` exits non-zero, **block and display**:

```
CONFLICT: <repo> is currently in use

  Active task: <other-task-name>
  Worktree:    ~/Code/PyAutoLabs-wt/<other-task-name>
  Branch:      feature/<other-task-name>

  This repo cannot be used for a new task until the existing
  work is shipped.

  Options:
  (a) Switch to that session to finish the work first
  (c) Abort this skill
```

Option (b) — "abandon the other task's work" — is intentionally gone. Because the other task lives in its own worktree, you do not need to free up the main checkout; the two workspaces coexist on disk. The only sensible paths are finish the other work, or abort.

If the repo appears under the **same** task (resuming work), that's fine — proceed.

## Mode Detection

Read `PyAutoPrompt/active.md` to find the current issue URL.

If no matching issue is found in `active.md`, check `PyAutoPrompt/planned.md` — the task may have been queued there because of a conflict when `/start_dev` ran. If found in `planned.md`:

1. Re-run the conflict guard against the task's `affected-repos`
2. If the conflict is **resolved**: move the entry from `planned.md` to `active.md` (using the active.md format with `status: workspace-dev`) and proceed
3. If the conflict **still exists**: report it and stop — the task can't start yet

Fetch the issue comments:

```bash
gh issue view <number> --repo <owner/repo> --json comments --jq '.comments[].body'
```

- If a comment contains **"Library PR Created"** → **linked mode** (the workspace work follows a library change)
- Otherwise → **standalone mode** (workspace-only task)

## Linked Mode (follows /ship_library)

### L1. Find the upstream library PR

From the "Library PR Created" comment on the issue, extract the library PR URL(s).

### L2. Fetch API Changes from the library PR

```bash
gh pr view <number> --repo <owner/repo> --json body --jq '.body'
```

Parse the `## API Changes` summary and the `<details>` block containing the full structured API changes (Removed, Added, Renamed, Changed Signature, Changed Behaviour, Migration sections).

### L3. Search workspace scripts for affected symbols

For each removed, renamed, or changed symbol:

```bash
grep -rn "<old_symbol>" \
  autofit_workspace/scripts/ \
  autogalaxy_workspace/scripts/ \
  autolens_workspace/scripts/ \
  autolens_workspace_test/scripts/
```

### L4. Present impact report

```
Upstream API Changes Impact
===========================

Upstream PR: <PR URL>
API Changes: <brief summary>

Affected scripts:
  autofit_workspace:
    scripts/overview/overview_1_the_basics.py:47 — uses OldClass
  autogalaxy_workspace:
    (none found)
  autolens_workspace:
    scripts/imaging/start_here.py:89 — uses old_function
  autolens_workspace_test:
    (none found)

Migration (from upstream PR):
  - Before: `obj = module.OldClass(x)`
  - After: `module.new_function(x)`
```

If grep finds **no affected scripts**, tell the user: "No scripts use the changed API symbols. This might be option (iii) from /ship_library. Want to run `/smoke-test` to confirm and merge the library PR instead?"

### L5. Attach workspace repos to the existing task worktree

The task already has a worktree root from `/start_library` — e.g. `~/Code/PyAutoLabs-wt/<task-name>/` — with the library repos checked out as real worktrees and every workspace repo sitting there as a symlink back to the main checkout. Replace each affected workspace symlink with a real worktree on the same `feature/<task-name>` branch:

```bash
source admin_jammy/software/worktree.sh
worktree_add_repo <task-name> autofit_workspace
worktree_add_repo <task-name> autolens_workspace
# ...one call per affected workspace repo
```

`worktree_add_repo` is idempotent: if a repo is already a real worktree it no-ops. The feature branch is created from `origin/main` the first time a given workspace repo is attached. All subsequent edits, commits, and script runs MUST happen inside `$WT_ROOT/<workspace_repo>`, never the main checkout.

### L6. Register repos in active.md

Update the task's entry in `PyAutoPrompt/active.md` to add workspace repos and update status. The `worktree:` field is already present from `/start_library` — keep it as-is and just extend the `repos:` list:

```markdown
## <task-name>
- issue: <issue-url>
- session: claude --resume <session-id>
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/<task-name>
- library-pr: <PR URL>
- repos:
  - PyAutoFit: feature/<task-name>
  - PyAutoArray: feature/<task-name>
  - autofit_workspace: feature/<task-name>
  - autolens_workspace: feature/<task-name>
```

The library repos from the earlier phase remain listed. The workspace repos are added alongside them. This is what `worktree_check_conflict` reads to detect collisions.

After updating active.md, push the change:

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_push "prompt: register <task-name> workspace repos in active.md"
```

### L7. Present "ready to develop" summary

```
Workspace Development Environment Ready
========================================

Issue: <issue URL>
Upstream PR: <library PR URL>

Worktree root: ~/Code/PyAutoLabs-wt/<task-name>

>>> ACTIVATE THIS SESSION BEFORE RUNNING ANY PYTHON <<<
  source ~/Code/PyAutoLabs-wt/<task-name>/activate.sh

Workspace repos attached (edit these — not the main checkout):
  ~/Code/PyAutoLabs-wt/<task-name>/autofit_workspace  (feature/<task-name>)
  ~/Code/PyAutoLabs-wt/<task-name>/autolens_workspace (feature/<task-name>)

Scripts to update:
  autofit_workspace/scripts/overview/overview_1_the_basics.py:47
    Before: obj = module.OldClass(x)
    After:  module.new_function(x)
  autolens_workspace/scripts/imaging/start_here.py:89
    Before: obj = module.OldClass(x)
    After:  module.new_function(x)

Rules:
  - Only edit files in scripts/ — never edit notebooks/ directly
  - Notebooks are regenerated automatically by /ship_workspace

When done, run /ship_workspace to validate and create PRs.
```

## Standalone Mode (workspace-only task)

### S1. Read the issue

Fetch the issue body from `active.md`:

```bash
gh issue view <number> --repo <owner/repo> --json body,title --jq '.body'
```

If the issue doesn't contain a detailed plan, read the description and formulate a plan by exploring the relevant workspace scripts.

### S2. Identify affected workspace repos

From the issue plan, determine which workspace repos need changes:
- `autofit_workspace`
- `autogalaxy_workspace`
- `autolens_workspace`
- `autolens_workspace_test`
- `euclid_strong_lens_modeling_pipeline`
- `HowToLens`

### S3. Create the task worktree

Workspace-only tasks get their own task worktree root, just like library tasks do. Create it with every affected workspace repo in one call:

```bash
source admin_jammy/software/worktree.sh
worktree_create <task-name> <workspace_repo1> [workspace_repo2 ...]
```

This creates `~/Code/PyAutoLabs-wt/<task-name>/` with each listed workspace repo checked out as a real worktree on `feature/<task-name>` (branched from `origin/main`), symlinks every other top-level PyAutoLabs entry back to the canonical checkout, and writes `activate.sh` with the per-task `PYTHONPATH`, `NUMBA_CACHE_DIR`, and `MPLCONFIGDIR` overrides.

If a branch with the same name already exists (e.g. resuming work), `worktree_create` checks it out instead of creating a new one.

### S4. Register repos in active.md

Update the task's entry in `PyAutoPrompt/active.md` to add workspace repos and record the worktree path:

```markdown
## <task-name>
- issue: <issue-url>
- session: claude --resume <session-id>
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/<task-name>
- repos:
  - autolens_workspace: feature/<task-name>
```

The `worktree:` field is what `worktree_check_conflict` reads to detect collisions from other Claude sessions.

After updating active.md, push the change:

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_push "prompt: register <task-name> standalone-workspace repos in active.md"
```

### S5. Explore relevant scripts

Read the scripts that will be modified or created inside the worktree (e.g. `~/Code/PyAutoLabs-wt/<task-name>/autolens_workspace/scripts/...`). Understand the existing directory structure and naming conventions.

### S6. Present "ready to develop" summary

```
Workspace Development Environment Ready
========================================

Issue: <issue URL>

Worktree root: ~/Code/PyAutoLabs-wt/<task-name>

>>> ACTIVATE THIS SESSION BEFORE RUNNING ANY PYTHON <<<
  source ~/Code/PyAutoLabs-wt/<task-name>/activate.sh

Workspace repos checked out in worktree (edit these — not the main checkout):
  ~/Code/PyAutoLabs-wt/<task-name>/autolens_workspace  (feature/<task-name>)

Scripts to modify/create:
  <list based on the plan, paths relative to the worktree>

Rules:
  - Only edit files in scripts/ — never edit notebooks/ directly
  - Notebooks are regenerated automatically by /ship_workspace

When done, run /ship_workspace to validate and create PRs.
```

## Notes

- If `active.md` has multiple issues, ask the user which one to work on.
- If the repo is already on the correct feature branch, skip branch creation and note it.
- If the repo has uncommitted changes, warn the user before switching branches.

## Remote / mobile mode

**Environment detection:**
- If `~/Code/PyAutoLabs` exists and contains repo subdirectories → **laptop mode** (use all steps above)
- Otherwise → **mobile mode** (follow this section)

**Required repos:** The workspace repos listed in the task's affected repos, plus any library repos if linked mode.
**GitHub orgs:** Jammy2211 (all workspaces), rhayes777 (PyAutoConf, PyAutoFit), Jammy2211 (PyAutoArray, PyAutoGalaxy, PyAutoLens)

**Mobile behavior:**
1. Skip the conflict guard (no worktrees on mobile)
2. Read the active issue and detect linked vs standalone mode as normal
3. Instead of creating/attaching a worktree, clone each required repo:
   ```bash
   WORK_DIR="$(pwd)"
   for repo in <repos>; do
     case "$repo" in
       PyAutoConf|PyAutoFit) ORG="rhayes777" ;;
       *) ORG="Jammy2211" ;;
     esac
     [ -d "$WORK_DIR/$repo" ] || git clone "https://github.com/$ORG/$repo.git" "$WORK_DIR/$repo"
     git -C "$WORK_DIR/$repo" checkout feature/<task-name> 2>/dev/null || \
       git -C "$WORK_DIR/$repo" checkout -b feature/<task-name>
   done
   ```
4. Set PYTHONPATH manually (same as start_library mobile mode)
5. Register repos in `active.md` (omit `worktree:` field)
6. Set `location: mobile-in-progress`
7. Present the "ready to develop" summary with mobile paths
