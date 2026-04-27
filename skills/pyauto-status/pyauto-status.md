# PyAuto Status: Active Work Dashboard

Show a dashboard of all active work across all repositories. Use this to check for conflicts before starting new work, or when resuming a session to understand what's in flight.

## Usage

```
/pyauto-status
```

## Steps

### 1. Read the work registry

Read all three registry files:

**`PyAutoPrompt/planned.md`** — tasks with issues created but blocked from starting:

```markdown
## <task-name>
- issue: <issue-url>
- planned: <YYYY-MM-DD>
- classification: <library | workspace | both>
- suggested-branch: feature/<name>
- blocked-by: <conflicting-task-name> (using <repo>)
- affected-repos:
  - <repo1>
  - <repo2>
```

**`PyAutoPrompt/active.md`** — tasks currently in progress:

```markdown
## <task-name>
- issue: <issue-url>
- session: <session-id>
- status: <status>
- library-pr: <PR URL — optional>
- repos:
  - <repo>: <branch>
```

**`PyAutoPrompt/complete.md`** — recently completed tasks (show last 5 for context).

### 2. Live scan all repositories and worktrees

First, scan every canonical checkout:

```bash
git -C <repo_path> branch --show-current
git -C <repo_path> status --short
git -C <repo_path> log --oneline -1
```

**All repos to scan:**
- PyAutoConf
- PyAutoFit
- PyAutoArray
- PyAutoGalaxy
- PyAutoLens
- autofit_workspace
- autogalaxy_workspace
- autolens_workspace
- autolens_workspace_test
- euclid_strong_lens_modeling_pipeline
- HowToLens

Then also scan every task worktree root:

```bash
source admin_jammy/software/worktree.sh
ls -1 "$PYAUTO_WT_ROOT" 2>/dev/null
```

For each `<task>` under `$PYAUTO_WT_ROOT`, iterate its children and collect the same branch/status/log information for every entry that is a real worktree (not a symlink; linked worktrees use a `.git` **file**, so test with `[[ -e "$entry/.git" ]]` rather than `-d`):

```bash
for entry in "$PYAUTO_WT_ROOT/<task>"/*; do
  [[ -L "$entry" ]] && continue
  [[ -e "$entry/.git" ]] || continue
  git -C "$entry" branch --show-current
  git -C "$entry" status --short
done
```

Record both the task name and the repo name for each worktree hit — these feed into the Active Work section below.

### 3. Cross-reference and display

Compare the live git state with the registries. Display a dashboard:

```
Planned (queued, not yet started)
=================================
psf-oversampling (#50) — planned 2026-04-03
  Classification: library
  Affected repos: PyAutoArray, PyAutoGalaxy
  Blocked by: jax-search-logging (using PyAutoArray)
  → Conflict still active. Cannot start yet.

dark-matter-sightlines (#51) — planned 2026-04-04
  Classification: both
  Affected repos: PyAutoLens, autolens_workspace
  → No conflict. Ready to start — run /start_library.


Active Work
===========
jax-search-logging (#42) — library-shipped, workspace-pending
  Worktree: ~/Code/PyAutoLabs-wt/jax-search-logging
  PyAutoFit:         feature/jax-search-logging  (pushed, PR #87 open)
  PyAutoArray:       feature/jax-search-logging  (pushed, PR #88 open)
  autolens_workspace: feature/jax-search-logging (dirty, 3 uncommitted files)

psf-oversampling (#50) — library-dev
  Worktree: ~/Code/PyAutoLabs-wt/psf-oversampling
  PyAutoArray:       feature/psf-oversampling  (2 commits ahead of origin)


Recently Completed
==================
grid-refactor (#38) — completed 2026-04-01
  library-pr: https://github.com/Jammy2211/PyAutoArray/pull/82


Warnings
========
UNREGISTERED: PyAutoGalaxy is on feature/old-experiment (not in active.md)
  This may be leftover work from a previous session.


Idle Repos (on main, clean)
===========================
PyAutoConf, PyAutoGalaxy, PyAutoLens, autofit_workspace,
autogalaxy_workspace, autolens_workspace_test
```

### 4. Re-check planned tasks

For each task in `planned.md`, check whether its `blocked-by` conflict still exists:

- If the blocking task has been shipped (no longer in `active.md`) → mark as **"Ready to start"** in the dashboard
- If the conflict still exists → mark as **"Still blocked"**

This is the key value of `/pyauto-status` for planned tasks — it tells you which queued work can now begin.

### 5. Flag issues

Detect and prominently flag:

- **Unblocked planned tasks:** Tasks in `planned.md` whose conflicts have resolved — these are ready to start
- **Conflicts:** A repo claimed by multiple tasks in `active.md` (only possible if two `worktree:` entries list the same repo under different tasks)
- **Unregistered main-checkout work:** A canonical checkout on a feature branch when no active task lists that checkout in its `worktree:` tree. Cross-reference with every worktree root's branches *before* warning — a worktree on `feature/foo` is expected, not stale. Only warn about branches sitting on the **main checkout** without any registered task.
- **Orphan worktree roots:** A directory under `$PYAUTO_WT_ROOT` that does not match any entry in `active.md`.
- **Missing worktrees:** An `active.md` entry with a `worktree:` field pointing at a directory that no longer exists.
- **Stale entries:** A task in `active.md` whose repos are actually on `main` in both the worktree and the main checkout (work was abandoned or completed without cleanup)
- **Dirty repos:** Repos with uncommitted changes, whether in the main checkout or any worktree (highlight which task they belong to)

### 6. Suggest actions

Based on the flags, suggest next steps:

- For unblocked planned tasks: "Task `<name>` is ready to start — run `/start_library` or `/start_workspace`."
- For conflicts: "Resolve by finishing one task before starting another, or abandon one with `/start_library` option (b)."
- For unregistered work: "This branch isn't tracked. Either register it in `active.md` or reset to main if the work is abandoned."
- For stale entries: "This task's repos are back on main. Remove from `active.md` or re-run `/start_library` to resume."

## Notes

- This skill is read-only — it never modifies files or branches.
- Run this at the start of any resumed session to understand the current state.
- The dashboard is designed to be scannable at a glance — active work first, warnings second, idle repos last.

## Remote / mobile mode

**Environment detection:**
- If `~/Code/PyAutoLabs` exists and contains repo subdirectories → **laptop mode** (use all steps above)
- Otherwise → **mobile mode** (follow this section)

**Mobile behavior:**
1. Pull latest `PyAutoPrompt` to get current registry files
2. Read `planned.md`, `active.md`, and `complete.md` as normal
3. For each repo listed in active tasks, check branch state via GitHub API instead of local git:
   ```bash
   gh api repos/<owner>/<repo>/branches/<branch> --jq '.name' 2>/dev/null
   gh pr list --repo <owner>/<repo> --head <branch> --json number,state,title
   ```
4. Skip worktree scanning entirely (no worktrees on mobile)
5. Skip local dirty/uncommitted checks (no local checkouts)
6. Display the dashboard with GitHub API data instead of local git data
7. Flag tasks marked `ready-for-mobile` prominently — these are actionable now
