# Start Library: Set Up Library Development

Set up the development environment for library source code work (PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens, PyAutoConf). Assumes a GitHub issue already exists in `active.md` (created by `/start_dev`).

## Steps

### 1. Conflict Guard

Before doing anything, check whether any repo this task needs is already held by another task's worktree.

```bash
source admin_jammy/software/worktree.sh
worktree_check_conflict <task-name> <repo1> [repo2 ...]
```

A "conflict" only fires when another entry in `active.md` already claims one of the target repos via its `worktree:` field. Two tasks that touch different repos can run in parallel — that's the whole purpose of worktrees. Two tasks that both want the same library still have to serialise.

If `worktree_check_conflict` exits non-zero, **block and display**:

```
CONFLICT: <repo> is currently in use

  Active task: <other-task-name> (issue <URL>)
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

### 2. Read the active issue

Read `PyAutoPrompt/active.md` to find the current issue URL.

If no matching issue is found in `active.md`, check `PyAutoPrompt/planned.md` — the task may have been queued there because of a conflict when `/start_dev` ran. If found in `planned.md`:

1. Re-run the conflict check (step 1) against the task's `affected-repos`
2. If the conflict is **resolved**: move the entry from `planned.md` to `active.md` (using the active.md format with `status: library-dev`) and proceed
3. If the conflict **still exists**: report it and stop — the task can't start yet

If no issue is found in either file, tell the user to run `/start_dev` first.

Fetch the issue body:

```bash
gh issue view <number> --repo <owner/repo> --json body,title --jq '.body'
```

### 3. Parse the implementation plan

Extract from the issue body:
- The **Affected Repositories** list
- The **Suggested branch** name
- The **Implementation Steps**
- The **Key Files** list

If the issue doesn't contain a detailed plan (e.g., it was created manually rather than by `/start_dev`), read the issue description and formulate a plan by exploring the relevant code.

### 4. Verify repos are library repos

Confirm the affected repos are library repositories:
- PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens

If any workspace repos appear, warn the user: "This issue involves workspace repos too. Complete library work first with `/ship_library`, then workspace work follows."

### 5. Create the task worktree

Instead of creating branches in the main checkout, create a task-scoped worktree root containing a real worktree for every affected library repo. Symlinks are created automatically for everything else in PyAutoLabs so relative paths keep working.

```bash
source admin_jammy/software/worktree.sh
worktree_create <task-name> <repo1> [repo2 ...]
```

The helper:
- creates `~/Code/PyAutoLabs-wt/<task-name>/`
- runs `git worktree add -b feature/<task-name>` for each listed repo, branched from `origin/main`
- symlinks every other top-level PyAutoLabs entry back to the canonical checkout
- writes `activate.sh` inside the worktree root with the per-task `PYTHONPATH`, `NUMBA_CACHE_DIR`, and `MPLCONFIGDIR` overrides

If a branch with the same name already exists (e.g., resuming work), `worktree_create` checks it out instead of creating a new one.

**Critical:** after this step, Python must be run with the activate script sourced, otherwise `import autofit` will still resolve to the main checkout, not the worktree. Print this reminder prominently:

```
Before running Python, pytest, or /smoke_test in this session, run:
  source ~/Code/PyAutoLabs-wt/<task-name>/activate.sh
```

All subsequent skills (`ship_library`, `smoke_test`, etc.) source this script automatically, but any manual Python you run also needs it.

### 6. Register repos in active.md

After the worktree is created, update the task's entry in `PyAutoPrompt/active.md` to record the worktree path and claimed repos:

```markdown
## <task-name>
- issue: <issue-url>
- session: claude --resume <session-id>
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/<task-name>
- repos:
  - PyAutoFit: feature/<task-name>
  - PyAutoArray: feature/<task-name>
```

The `worktree:` field is what `worktree_check_conflict` reads to detect collisions from other Claude sessions.

After updating active.md, push the change so other machines see the registered repos:

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_push "prompt: register <task-name> library repos in active.md"
```

### 7. Explore key files

Read the key files listed in the implementation plan. Focus on:
- The specific classes, functions, and methods that will be modified
- Their callers and dependents (to understand blast radius)
- Related test files (to understand what tests exist and where)
- Import chains across repos (changes in PyAutoFit may affect PyAutoArray, etc.)

### 8. Present "ready to develop" summary

Display:

```
Development Environment Ready
==============================

Issue: <issue URL>
Task: <issue title>

Worktree root: ~/Code/PyAutoLabs-wt/<task-name>

>>> ACTIVATE THIS SESSION BEFORE RUNNING ANY PYTHON <<<
  source ~/Code/PyAutoLabs-wt/<task-name>/activate.sh

Repos checked out in worktree:
  PyAutoFit:   feature/<task-name>  (from origin/main)
  PyAutoArray: feature/<task-name>  (from origin/main)

Key files to modify (edit inside the worktree — not the main checkout):
  ~/Code/PyAutoLabs-wt/<task-name>/PyAutoFit/autofit/non_linear/search/nest/nautilus.py
  ~/Code/PyAutoLabs-wt/<task-name>/PyAutoArray/autoarray/operators/convolver.py

Test directories:
  PyAutoFit:   test_autofit/
  PyAutoArray: test_autoarray/

Implementation steps:
  1. <first step from plan>
  2. <second step from plan>
  ...

When done, run /ship_library to test, commit, and create PRs.
```

## Notes

- If `active.md` has multiple issues, ask the user which one to work on.
- If the repo is already on the correct feature branch, skip branch creation and note it.
- If the repo has uncommitted changes, warn the user before switching branches.

## Remote / mobile mode

**Environment detection:**
- If `~/Code/PyAutoLabs` exists and contains repo subdirectories → **laptop mode** (use all steps above)
- Otherwise → **mobile mode** (follow this section)

**Required repos:** The library repos listed in the task's affected repos (from the issue plan).
**GitHub orgs:** rhayes777 (PyAutoConf, PyAutoFit), Jammy2211 (PyAutoArray, PyAutoGalaxy, PyAutoLens)

**Mobile behavior:**
1. Skip the conflict guard (no worktrees on mobile — conflicts are only relevant for parallel local work)
2. Read the active issue from `active.md` as normal
3. Parse the implementation plan as normal
4. Instead of creating a worktree, clone each required repo into the current working directory:
   ```bash
   WORK_DIR="$(pwd)"
   for repo in <affected-repos>; do
     case "$repo" in
       PyAutoConf|PyAutoFit) ORG="rhayes777" ;;
       *) ORG="Jammy2211" ;;
     esac
     git clone "https://github.com/$ORG/$repo.git" "$WORK_DIR/$repo"
     git -C "$WORK_DIR/$repo" checkout -b feature/<task-name>
   done
   ```
5. Skip `activate.sh` — set PYTHONPATH manually:
   ```bash
   export PYTHONPATH="$WORK_DIR/PyAutoConf:$WORK_DIR/PyAutoFit:$WORK_DIR/PyAutoArray:$WORK_DIR/PyAutoGalaxy:$WORK_DIR/PyAutoLens:$PYTHONPATH"
   export NUMBA_CACHE_DIR=/tmp/numba_cache
   export MPLCONFIGDIR=/tmp/matplotlib
   ```
6. Register repos in `active.md` as normal (omit `worktree:` field)
7. Set `location: mobile-in-progress` in `active.md`
8. Present the "ready to develop" summary with mobile paths
