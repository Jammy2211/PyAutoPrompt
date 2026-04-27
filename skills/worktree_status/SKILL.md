---
name: worktree_status
description: Show the state of every task-scoped git worktree under ~/Code/PyAutoLabs-wt/, cross-referenced with PyAutoPrompt/active.md. Use this to see which parallel tasks are in flight and which repos each one is holding.
---

# Worktree Status

Diagnostic skill. Lists every worktree root under `$PYAUTO_WT_ROOT` (default `~/Code/PyAutoLabs-wt`), the task it belongs to per `active.md`, and the branch and dirty state of every real worktree inside it.

This skill is **read-only**. It never creates, removes, or modifies worktrees. Use `start_library` / `ship_library` / the post-merge cleanup flow for mutations.

## Steps

### 1. Source the worktree helper

```bash
source admin_jammy/software/worktree.sh
```

This provides `worktree_list_claimed`, `worktree_root_path`, and the `PYAUTO_WT_ROOT` variable used below.

### 2. Enumerate worktree roots on disk

```bash
ls -1 "$PYAUTO_WT_ROOT" 2>/dev/null
```

If the directory doesn't exist or is empty, report "No active worktrees." and stop.

### 3. For each worktree root, inspect its contents

For every directory `$PYAUTO_WT_ROOT/<task>/`, identify:

- **Real worktrees** (entries that are a directory with a `.git` file — note: linked worktrees have `.git` as a FILE, not a dir):
  ```bash
  for entry in "$PYAUTO_WT_ROOT/<task>"/*; do
    [[ -L "$entry" ]] && continue
    [[ -e "$entry/.git" ]] || continue
    # $entry is a real worktree
  done
  ```
- For each real worktree, capture:
  - Branch: `git -C "$entry" branch --show-current`
  - Dirty? `git -C "$entry" status --porcelain | head -1` (non-empty = dirty)
  - Unpushed? `git -C "$entry" log @{u}.. --oneline 2>/dev/null | wc -l` (0 = up to date with origin, -1/error = no upstream)
- The activate script: does `$PYAUTO_WT_ROOT/<task>/activate.sh` exist?

### 4. Cross-reference with active.md

Run `worktree_list_claimed` and build a lookup of `task → list of claimed repos`.

For each worktree root on disk, find the matching task in the lookup. Flag mismatches:

- **Orphan worktree root:** directory exists but no entry in `active.md`. Might be leftover from a crashed session.
- **Missing worktree:** `active.md` lists a task with a `worktree:` field, but the directory is gone. Task state is stale.
- **Repo mismatch:** `active.md` says task X claims PyAutoFit, but the worktree root contains no real PyAutoFit worktree (or vice versa).

### 5. Display the dashboard

Group output by task. Example format:

```
Worktree Status
===============

Active Worktrees
----------------

jax-search-logging  (~/Code/PyAutoLabs-wt/jax-search-logging)
  issue: https://github.com/rhayes777/PyAutoFit/issues/42
  activate: source ~/Code/PyAutoLabs-wt/jax-search-logging/activate.sh
  repos:
    PyAutoFit    feature/jax-search-logging  clean     (3 commits unpushed)
    PyAutoArray  feature/jax-search-logging  2 dirty   (pushed)

psf-oversampling  (~/Code/PyAutoLabs-wt/psf-oversampling)
  issue: https://github.com/Jammy2211/PyAutoArray/issues/50
  activate: source ~/Code/PyAutoLabs-wt/psf-oversampling/activate.sh
  repos:
    PyAutoArray  feature/psf-oversampling  clean  (up to date)

Warnings
--------
ORPHAN: ~/Code/PyAutoLabs-wt/old-experiment exists but is not in active.md
  → inspect and remove with `worktree_remove old-experiment` if stale.

MISSING: active.md lists 'abandoned-task' with worktree ~/Code/PyAutoLabs-wt/abandoned-task, but the directory does not exist.
  → remove the stale entry from active.md.
```

If there are no warnings, omit the Warnings section entirely.

### 6. Suggest next actions

Only if warnings are present:

- **Orphan:** "Run `worktree_remove <task>` if this work is abandoned. Otherwise, re-register it in `active.md`."
- **Missing:** "Edit `PyAutoPrompt/active.md` to remove the stale entry, or re-create the worktree with `worktree_create <task> <repos...>` if the work is resuming."
- **Repo mismatch:** "Either the worktree needs more repos (`git worktree add` inside the root) or `active.md` needs correcting. Investigate before the next `ship_library` run."

## Notes

- This skill is read-only.
- It assumes `admin_jammy/software/worktree.sh` is sourceable from the current directory (run from the PyAutoLabs root, or pass `PYAUTO_MAIN` explicitly).
- "Dirty" means any uncommitted change, including untracked files.
- "Unpushed" is counted against the branch's upstream; a branch with no upstream is reported as "no upstream" rather than a commit count.

## Remote / mobile mode

**Environment detection:**
- If `~/Code/PyAutoLabs` exists and contains repo subdirectories → **laptop mode** (use all steps above)
- Otherwise → **mobile mode** (follow this section)

**Mobile behavior:**

On mobile/server there are no worktrees — this skill degrades gracefully:

1. Read `PyAutoPrompt/active.md` and display task entries with their `location:`, `status:`, and `repos:` fields
2. For each task with a `worktree:` field, note that the worktree is on the laptop (not accessible from here)
3. Check branch state via GitHub API for each repo:
   ```bash
   gh api repos/<owner>/<repo>/branches/<branch> --jq '.name' 2>/dev/null
   ```
4. Display a simplified dashboard:
   ```
   Worktree Status (mobile view — no local worktrees)
   ===================================================

   task-name (#339) — location: ready-for-mobile
     Worktree (on laptop): ~/Code/PyAutoLabs-wt/task-name
     Branches (via GitHub):
       PyAutoGalaxy: feature/task-name exists ✓
       autolens_workspace: feature/task-name exists ✓
   ```
5. Omit dirty/unpushed checks (those only apply to local worktrees)
6. Suggest `/handoff resume` for tasks marked `ready-for-mobile`
