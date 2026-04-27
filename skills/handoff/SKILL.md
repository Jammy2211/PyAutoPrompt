---
name: handoff
description: Manage work transitions between mobile/web and CLI/laptop environments. Park, resume, and track task status across machines using active.md and git branches.
user-invocable: true
---

# Handoff: Cross-Environment Work Management

Automate the commit-push-status cycle when transitioning work between laptop CLI and mobile/server environments. Uses `PyAutoPrompt/active.md` as the single source of truth and git branches as the unit of work.

## Usage

```
/handoff park              # park current work for the other machine
/handoff park --complete   # mark work as done
/handoff resume            # pick up work parked by the other machine
/handoff status            # show all tasks and their locations
```

## Environment Detection

Detect current environment before every operation:

- **Laptop/CLI:** `~/Code/PyAutoLabs` exists AND contains repo subdirectories (e.g. `PyAutoFit/`, `PyAutoArray/`)
- **Mobile/Server:** the above path doesn't exist — repos are cloned on demand

```bash
if [ -d "$HOME/Code/PyAutoLabs/PyAutoFit" ]; then
  ENV="cli"
else
  ENV="mobile"
fi
```

## Status Lifecycle

Each task in `active.md` has a `location:` field tracking where the work currently lives:

| Value | Meaning |
|---|---|
| `cli-in-progress` | Currently being worked on from laptop CLI |
| `mobile-in-progress` | Currently being worked on from server/web |
| `ready-for-cli` | Mobile finished a chunk, laptop should pick up |
| `ready-for-mobile` | CLI finished a chunk, mobile should pick up |
| `complete` | Done (moved to complete.md) |

State transitions:

```
(new task)
    │
    ├──[start on CLI]──→ cli-in-progress
    │                         │
    │                    [handoff park]
    │                         │
    │                         ▼
    │                   ready-for-mobile
    │                         │
    │                    [handoff resume on mobile]
    │                         │
    │                         ▼
    │                   mobile-in-progress
    │                         │
    │                    [handoff park]
    │                         │
    │                         ▼
    │                    ready-for-cli
    │                         │
    │                    [handoff resume on CLI]
    │                         │
    │                         ▼
    │                   cli-in-progress  ← (cycle repeats)
    │
    └──[handoff park --complete]──→ complete (→ complete.md)
```

## `handoff park` — Park Current Work

"I'm done for now on this machine."

### Steps

1. **Detect environment** (cli or mobile)

2. **Identify the active task.** Read `PyAutoPrompt/active.md` and find the task whose `location:` is `<env>-in-progress`. If multiple tasks match, ask the user which one to park. If no task has a `location:` field yet (legacy entry), infer from context — it's likely `cli-in-progress` if on laptop.

3. **For each repo in the task's `repos:` list**, commit and push working changes:

   ```bash
   # Determine repo path based on environment
   if [ "$ENV" = "cli" ]; then
     # Use worktree path if available
     REPO_PATH="$WT_ROOT/<repo>"
     # Fall back to main checkout if no worktree
     [ -d "$REPO_PATH" ] || REPO_PATH="./<repo>"
   else
     # Mobile: repos are in current working directory
     REPO_PATH="./<repo>"
   fi

   git -C "$REPO_PATH" add -A
   git -C "$REPO_PATH" diff --cached --quiet || \
     git -C "$REPO_PATH" commit -m "wip: handoff park from $ENV"
   git -C "$REPO_PATH" push -u origin "$(git -C "$REPO_PATH" branch --show-current)"
   ```

   If there are no changes to commit, just ensure the branch is pushed.

4. **Ask the user for a handoff summary:**

   > What was accomplished and what's left to do?

   Store the response as the `summary:` field.

5. **Update the task entry in `active.md`:**

   - Set `location:` to `ready-for-mobile` (if on CLI) or `ready-for-cli` (if on mobile)
   - Set `summary:` with the user's response (use `|` block scalar for multiline)

   ```markdown
   ## task-name
   - issue: https://github.com/...
   - session: claude --resume "..."
   - status: library-dev
   - location: ready-for-mobile
   - branch: feature/task-name
   - worktree: ~/Code/PyAutoLabs-wt/task-name
   - repos:
     - PyAutoFit: feature/task-name
   - summary: |
       Done: implemented new API for FitImaging
       Next: update workspace scripts to use new API
   ```

6. **Commit and push PyAutoPrompt** so active.md syncs across machines:

   ```bash
   git -C PyAutoPrompt add active.md
   git -C PyAutoPrompt commit -m "handoff: park <task-name> → ready-for-<other>"
   git -C PyAutoPrompt push
   ```

### `--complete` flag

If `handoff park --complete` is used:
- Set `location: complete`
- Move the task entry from `active.md` to `complete.md`
- Follow the same format as `/ship_workspace` uses for completed entries

## `handoff resume` — Pick Up Parked Work

"Pick up where the other machine left off."

### Steps

1. **Detect environment** (cli or mobile)

2. **Pull latest PyAutoPrompt** to get current active.md:

   ```bash
   git -C PyAutoPrompt pull --ff-only
   ```

3. **Read `active.md`** and filter tasks where `location:` matches `ready-for-<current-env>`:
   - On CLI: show tasks with `location: ready-for-cli`
   - On mobile: show tasks with `location: ready-for-mobile`

4. **Present matching tasks.** If only one, auto-select it. If multiple, ask the user to pick:

   ```
   Tasks ready for this environment:

   1. merge-fast-plot-env-vars (#339)
      Branch: feature/merge-fast-plot-env-vars
      Summary: Done: merged library PRs. Next: update workspace scripts.

   2. smoke-test-optimization (#1183)
      Branch: feature/smoke-test-optimization
      Summary: Done: profiled imaging scripts. Next: investigate cosmology calc.

   Which task? [1]
   ```

5. **Set up the working environment for the selected task:**

   **On CLI (laptop):**

   Source the worktree helper and set up worktrees:

   ```bash
   source admin_jammy/software/worktree.sh
   ```

   - If a worktree root already exists (`$PYAUTO_WT_ROOT/<task>`), verify branches and report status
   - If no worktree exists, create one:
     ```bash
     worktree_create <task-name> <repo1> [repo2 ...]
     ```
   - For each repo, fetch and checkout the task branch:
     ```bash
     git -C "$WT_ROOT/<repo>" fetch origin
     git -C "$WT_ROOT/<repo>" checkout feature/<task-name>
     git -C "$WT_ROOT/<repo>" pull --ff-only
     ```

   **On mobile/server:**

   Clone repos into the working directory:

   ```bash
   WORK_DIR="$(pwd)"
   for repo in <repo-list>; do
     if [ ! -d "$WORK_DIR/$repo" ]; then
       # Determine GitHub org
       case "$repo" in
         PyAutoConf|PyAutoFit) ORG="rhayes777" ;;
         *) ORG="Jammy2211" ;;
       esac
       git clone "https://github.com/$ORG/$repo.git" "$WORK_DIR/$repo"
     fi
     git -C "$WORK_DIR/$repo" fetch origin
     git -C "$WORK_DIR/$repo" checkout feature/<task-name>
     git -C "$WORK_DIR/$repo" pull --ff-only
   done
   ```

6. **Show the handoff summary:**

   ```
   Resumed: merge-fast-plot-env-vars
   ===================================

   Issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/339
   Branch: feature/merge-fast-plot-env-vars

   What was done:
     Merged library PRs for env var support

   What's next:
     Update workspace scripts to use new API

   Repos ready:
     PyAutoGalaxy:       feature/merge-fast-plot-env-vars (up to date)
     autolens_workspace: feature/merge-fast-plot-env-vars (up to date)
   ```

   On CLI, also remind:
   ```
   >>> source ~/Code/PyAutoLabs-wt/<task>/activate.sh
   ```

7. **Update `active.md`:**
   - Set `location:` to `<current-env>-in-progress`
   - Commit and push PyAutoPrompt

## `handoff status` — Show All Task Locations

### Steps

1. **Pull latest PyAutoPrompt:**
   ```bash
   git -C PyAutoPrompt pull --ff-only
   ```

2. **Read `active.md`** and display all tasks with their handoff state:

   ```
   Handoff Status
   ==============

   Current environment: CLI (laptop)

   Ready for THIS machine:
     merge-fast-plot-env-vars (#339) — ready-for-cli
       Branch: feature/merge-fast-plot-env-vars
       Summary: Done: merged PRs. Next: workspace scripts.

   In progress elsewhere:
     smoke-test-optimization (#1183) — mobile-in-progress
       Branch: feature/smoke-test-optimization
       Summary: Profiling interferometer scripts.

   In progress HERE:
     (none)

   No location set (legacy):
     old-task (#100) — no location field
       → Consider running /handoff park to set initial state
   ```

3. **Suggest actions:**
   - Tasks ready for current env: "Run `/handoff resume` to pick up."
   - Tasks in progress elsewhere: "Wait for handoff park from the other machine."
   - Legacy tasks: "Run `/handoff park` to register location, or add `location:` manually."

## Branch Naming Convention

Follow the existing convention from `/start_dev` and `/start_library`:
- `feature/<task-name>` for feature work
- The task name is lowercase, kebab-case
- Matches what GitHub/Claude Code already uses

## GitHub Org Mapping

When cloning repos on mobile/server, use these orgs:

| Repository | GitHub Org |
|---|---|
| PyAutoConf | rhayes777 |
| PyAutoFit | rhayes777 |
| PyAutoArray | Jammy2211 |
| PyAutoGalaxy | Jammy2211 |
| PyAutoLens | Jammy2211 |
| autofit_workspace | Jammy2211 |
| autogalaxy_workspace | Jammy2211 |
| autolens_workspace | Jammy2211 |
| autolens_workspace_test | Jammy2211 |
| euclid_strong_lens_modeling_pipeline | Jammy2211 |
| HowToLens | Jammy2211 |
| PyAutoBuild | Jammy2211 |

## active.md Format

The handoff skill expects and maintains this format in `PyAutoPrompt/active.md`:

```markdown
## task-name
- issue: https://github.com/...
- session: claude --resume "task-name"
- status: library-dev | workspace-dev | profiling-and-optimization | ...
- location: cli-in-progress | mobile-in-progress | ready-for-cli | ready-for-mobile
- branch: feature/task-name
- worktree: ~/Code/PyAutoLabs-wt/task-name
- library-pr:
  - https://github.com/...
- repos:
  - RepoName: feature/task-name
- summary: |
    Done: what was accomplished
    Next: what remains to do
```

Fields added by handoff:
- `location:` — where the work currently lives
- `branch:` — canonical branch name for quick lookup
- `summary:` — what's done and what's next (written by `handoff park`)

All other fields are maintained by existing skills (start_dev, start_library, ship_library, etc.) and are left untouched by handoff.

## Notes

- The handoff skill never modifies code — it only commits existing changes, pushes, and updates active.md.
- On CLI, it respects the worktree system — repos are accessed via `$WT_ROOT/<repo>`, never the main checkout.
- On mobile, repos are expected to be in the current working directory (cloned fresh per session).
- `PyAutoPrompt` must be a git repo with a remote for the push/pull to work. If it's not, warn the user.
- The `summary:` field is free-form text written by the user. Keep it concise — 2-3 lines max.
