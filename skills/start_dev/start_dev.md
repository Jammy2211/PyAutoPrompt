# Start Dev: Create Issue and Route

Universal entry point for starting work on a new task. Reads a prompt file, creates a tracked GitHub issue, and routes to the appropriate development skill.

## Usage

```
/start_dev <prompt-file-path>
```

The path is relative to `PyAutoPrompt/`. Examples:
- `/start_dev autofit/logging.md`
- `/start_dev autoarray/psf_oversampling.md`
- `/start_dev autolens/dark_matter_sight_lines.md`

## Steps

### 1. Read the prompt file

**Normalize the argument first.** PyCharm and similar editors copy markdown file paths as a markdown link, e.g. `autogalaxy/[adapt_images_pytree_fix.md](autogalaxy/adapt_images_pytree_fix.md)` or just `[adapt_images_pytree_fix.md](autogalaxy/adapt_images_pytree_fix.md)`. Before reading, extract the real path:

- If the argument contains `](...)` — a markdown link — use the path inside the parentheses (the last `](...)` group wins).
- Otherwise, use the argument as-is.
- Strip surrounding whitespace, backticks, and angle brackets.

Both of these resolve to `autogalaxy/adapt_images_pytree_fix.md`:
- `autogalaxy/adapt_images_pytree_fix.md`
- `autogalaxy/[adapt_images_pytree_fix.md](autogalaxy/adapt_images_pytree_fix.md)`

Read the file at `PyAutoPrompt/<normalized-argument>`. If the file doesn't exist, report the error and list available prompt files in that subdirectory.

### 2. Identify target repositories

Scan the prompt content for repository references. The user's prompts use the `@RepoName` convention (e.g., `@PyAutoFit`, `@PyAutoArray/autoarray/operators/convolver.py`) and sometimes bare path references.

Count references per repo to determine the primary (most-referenced) repository. If it's ambiguous (equal counts or no references), ask the user which repo should own the issue.

**Local directory to GitHub repo mapping:**

| Local Directory | GitHub Repo |
|----------------|-------------|
| `PyAutoConf` | `rhayes777/PyAutoConf` |
| `PyAutoFit` | `rhayes777/PyAutoFit` |
| `PyAutoArray` | `Jammy2211/PyAutoArray` |
| `PyAutoGalaxy` | `Jammy2211/PyAutoGalaxy` |
| `PyAutoLens` | `Jammy2211/PyAutoLens` |
| `autofit_workspace` | `Jammy2211/autofit_workspace` |
| `autogalaxy_workspace` | `Jammy2211/autogalaxy_workspace` |
| `autolens_workspace` | `Jammy2211/autolens_workspace` |
| `HowToLens` | `Jammy2211/HowToLens` |

### 3. Classify the work

Based on the target repositories, classify the task:

**Library repos:** PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens
**Workspace repos:** autofit_workspace, autogalaxy_workspace, autolens_workspace, autolens_workspace_test, euclid_strong_lens_modeling_pipeline, HowToLens

- If primary repos are **libraries** → library work
- If primary repos are **workspaces** → workspace work
- If **both** → starts as library work, workspace follows after shipping
- If **ambiguous** → ask the user

### 4. Explore the referenced code

Read the key files and classes mentioned in the prompt to build context. Focus on:
- Classes and functions referenced directly
- The surrounding architecture (imports, inheritance, callers)
- Existing tests related to the change

Do not spend more than a few minutes on this — enough to produce an informed plan, not an exhaustive audit.

### 5. Generate the plan

Produce **two levels of plan**:

**High-level plan** (3-8 bullet points):
- Written for a human reader (no code, no file paths)
- Describes what will change and why, in plain English
- Covers the logical sequence of work

**Detailed plan** (step-by-step):
- File paths and function/class names
- Specific changes to make at each step
- Key decisions and trade-offs
- Testing approach

### 6. Run plan_branches analysis

For each affected repository, run:

```bash
git -C <repo_path> branch --show-current
git -C <repo_path> status --short
git -C <repo_path> branch --sort=-committerdate | head -5
```

Display a summary table and suggest a branch name in the format `feature/<short-description>` (lowercase, kebab-case, under 50 chars).

Also compute the task name (same kebab-case stem as the branch, without the `feature/` prefix) and derive the worktree root path: `~/Code/PyAutoLabs-wt/<task-name>/`. This path will be handed to `/start_library` in step 11 and is also recorded in the issue body below.

### 7. Generate the issue title

Create a concise issue title (under 70 characters) that describes the task. Use conventional prefixes where appropriate:
- `feat:` for new features
- `fix:` for bug fixes
- `refactor:` for refactoring
- `docs:` for documentation
- `perf:` for performance improvements

### 8. Create the GitHub issue

Create the issue on the primary repo using `gh issue create`. Present the issue body to the user for review before creating it. The issue body must follow this structure:

```markdown
## Overview

<2-4 sentence summary of what this task is and why it matters>

## Plan

<High-level bullet-point plan — human readable, no code>

<details>
<summary>Detailed implementation plan</summary>

### Affected Repositories
- repo1 (primary)
- repo2

### Work Classification
<Library / Workspace / Both>

### Branch Survey

| Repository | Current Branch | Dirty? |
|-----------|---------------|--------|
| ./RepoName | main | clean |

**Suggested branch:** `feature/<name>`
**Worktree root:** `~/Code/PyAutoLabs-wt/<name>/` (created later by `/start_library`)

### Implementation Steps

1. Detailed step with file paths and specifics...
2. ...

### Key Files
- `path/to/file.py` — description of changes

</details>

## Original Prompt

<details>
<summary>Click to expand starting prompt</summary>

<original prompt content copied verbatim>

</details>
```

Use a HEREDOC to pass the body to `gh issue create`:

```bash
gh issue create --repo <owner/repo> --title "<title>" --body "$(cat <<'ISSUE_EOF'
<body content>
ISSUE_EOF
)"
```

### 9. Check for repo conflicts

Source the worktree helper and check whether any repo this task needs is already held by another active task's worktree:

```bash
source admin_jammy/software/worktree.sh
worktree_check_conflict <task-name> <repo1> [repo2 ...]
```

A "conflict" means another entry in `active.md` already claims one of the target repos via its `worktree:` field. Two tasks that edit **different** repos can run in parallel — that's the whole point of worktrees. Two tasks that both want PyAutoFit still collide, and must be serialised.

**If `worktree_check_conflict` exits non-zero**, this task cannot start yet. Go to step 9a.

**If it exits zero**, go to step 9b.

Legacy check (for tasks still on the pre-worktree flow): if the target repo's main checkout is on a feature branch *not* referenced by any `worktree:` field in `active.md`, warn the user — it's unregistered work from a previous session. Do not treat it as an automatic conflict; ask whether to proceed.

#### 9a. Route to planned.md (conflict — task is queued)

Add the entry to `PyAutoPrompt/planned.md` instead of `active.md`:

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

Tell the user:

```
Task planned but cannot start yet.

  Blocked by: <conflicting-task-name> — using <repo>
  Issue created: <issue URL>
  Entry added to: planned.md

  Once the blocking task ships, run /status to check availability,
  then /start_library or /start_workspace to begin this task.
```

Skip to step 10 (move prompt file).

#### 9b. Route to active.md (no conflict — task can start)

Add the entry to `PyAutoPrompt/active.md`:

```markdown
## <task-name>
- issue: <issue-url>
- session: claude --resume <session-id>
- status: <library-dev | workspace-dev>
- worktree: ~/Code/PyAutoLabs-wt/<task-name>
- repos:
```

The `status` field is set based on the classification from step 3:
- Library work or both → `library-dev`
- Workspace work → `workspace-dev`

The `worktree:` field records where the task's worktree root will live. The directory does not yet exist — `/start_library` creates it when the user is ready to develop. The `repos:` list also starts empty and is populated when branches are created.

### 10. Move the prompt file

Move the original prompt file to `PyAutoPrompt/issued/`:

```bash
mv PyAutoPrompt/<path> PyAutoPrompt/issued/<filename>
```

If a file with the same name already exists in `issued/`, append a timestamp suffix.

### 11. Route to the next skill

**If the task went to active.md (no conflict):**

Based on the classification from step 3, tell the user what to run next:

- **Library work:** "Run `/start_library` to set up your development environment."
- **Workspace work:** "Run `/start_workspace` to set up your development environment."
- **Both:** "Run `/start_library` first. After shipping the library changes, `/start_workspace` will follow."

Display:
- The issue URL
- The primary repo it was created on
- The suggested branch name
- The work classification
- The next skill to run

**If the task went to planned.md (blocked):**

Display:
- The issue URL
- What's blocking it and when it might clear
- Reminder to run `/status` to check when repos become available

### 12. Push PyAutoPrompt

After active.md / planned.md and the prompt-file move are settled, push the PyAutoPrompt state so the new entry is visible from any other machine:

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_push "prompt: route <task-name> (#<issue>) → <next-skill>"
```

Substitute the actual task name, issue number, and the routing destination (`/start_library`, `/start_workspace`, or `planned` if the task was blocked). If a Step 0a sync already pushed earlier in this run, this push only carries the active.md / planned.md / `issued/<file>` changes added by the routing steps.

## Step 0a — Sync new prompt ideas

Before reading the requested prompt, sweep up any other ideas the user has dropped into `PyAutoPrompt/` since the last task — these accumulate locally and would otherwise be lost in the next merge.

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_new_prompts
```

`prompt_sync_new_prompts` is a no-op when nothing is untracked. When it finds new files it commits them as a single `prompt: sync new task ideas` commit (each new file listed in the body) and pushes to `origin main`.

## Step 0 — Check for parked handoff tasks

Before creating a new task, check `PyAutoPrompt/active.md` for any tasks with a `location:` field matching `ready-for-<current-env>`:

- On CLI (laptop): look for `location: ready-for-cli`
- On mobile/server: look for `location: ready-for-mobile`

Detect environment:
```bash
if [ -d "$HOME/Code/PyAutoLabs/PyAutoFit" ]; then ENV="cli"; else ENV="mobile"; fi
```

If matching tasks are found, present them before proceeding:

```
A task is ready to resume on this machine:

  merge-fast-plot-env-vars (#339) — ready-for-cli
    Summary: Done: merged library PRs. Next: update workspace scripts.

Resume this task, or start a fresh one?
  (r) Resume — runs /handoff resume
  (n) New task — continue with /start_dev as normal
```

If the user picks (r), invoke the `/handoff resume` logic and stop — do not create a new issue.
If the user picks (n), proceed with step 1 as normal.

## Notes

- Always present the issue body to the user for review before creating it. Ask for confirmation.
- If `gh auth status` fails, tell the user to run `! gh auth login` to authenticate.
- The detailed plan should be thorough enough that a new Claude CLI session could pick it up and start working from the issue alone.
