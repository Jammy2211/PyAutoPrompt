---
name: create-issue
description: Convert a prompt file from PyAutoPrompt/ into a GitHub issue with overview, plan, and starting AI prompt.
user-invocable: true
---

Convert a prompt file into a tracked GitHub issue with a human-readable plan and detailed implementation spec.

## Usage

```
/create-issue <prompt-file-path>
```

The path is relative to `PyAutoPrompt/`. Examples:
- `/create-issue autofit/logging.md`
- `/create-issue autoarray/psf_oversampling.md`
- `/create-issue autolens/dark_matter_sight_lines.md`

## Steps

### 0. Sync new prompt ideas

Before reading the requested prompt, sweep up any other ideas the user has dropped into `PyAutoPrompt/` since the last task — these accumulate locally and would otherwise be lost in the next merge.

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_new_prompts
```

`prompt_sync_new_prompts` is a no-op when nothing is untracked. When it finds new files it commits them as a single `prompt: sync new task ideas` commit (each new file listed in the body) and pushes to `origin main`.

### 1. Read the prompt file

Read the file at `PyAutoPrompt/<argument>`. If the file doesn't exist, report the error and list available prompt files in that subdirectory.

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

### 3. Explore the referenced code

Read the key files and classes mentioned in the prompt to build context. Focus on:
- Classes and functions referenced directly
- The surrounding architecture (imports, inheritance, callers)
- Existing tests related to the change

Do not spend more than a few minutes on this — enough to produce an informed plan, not an exhaustive audit.

### 4. Generate the plan

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

### 5. Run plan_branches analysis

For each affected repository, run:

```bash
git -C <repo_path> branch --show-current
git -C <repo_path> status --short
git -C <repo_path> branch --sort=-committerdate | head -5
```

Display a summary table and suggest a branch name in the format `feature/<short-description>` (lowercase, kebab-case, under 50 chars).

### 6. Generate the issue title

Create a concise issue title (under 70 characters) that describes the task. Use conventional prefixes where appropriate:
- `feat:` for new features
- `fix:` for bug fixes
- `refactor:` for refactoring
- `docs:` for documentation
- `perf:` for performance improvements

### 7. Create the GitHub issue

Create the issue on the primary repo using `gh issue create`. The issue body must follow this structure:

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

### Branch Survey

| Repository | Current Branch | Dirty? |
|-----------|---------------|--------|
| ./RepoName | main | clean |

**Suggested branch:** `feature/<name>`

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

### 8. Show the user the result

Display:
- The issue URL
- The primary repo it was created on
- The suggested branch name
- A reminder to run `/update-issue <url>` during CLI work to post progress

### 9. Update active.md

Add a line to `PyAutoPrompt/active.md` with the issue URL and the original prompt file:

```
<task-name>: <issue-url> | claude --resume <session-id>
```

### 10. Move the prompt file

Move the original prompt file to `PyAutoPrompt/issued/`:

```bash
mv PyAutoPrompt/<path> PyAutoPrompt/issued/<filename>
```

If a file with the same name already exists in `issued/`, append a timestamp suffix.

### 11. Push PyAutoPrompt

After active.md is updated and the prompt file has been moved into `issued/`, push the PyAutoPrompt state so the new entry is visible from any other machine:

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_push "prompt: file issue for <task-name> (#<issue>)"
```

Substitute the actual task name and issue number. If the Step 0 sync already pushed earlier in this run, this push only carries the active.md and `issued/<file>` changes added by steps 9–10.

## Notes

- Always present the issue body to the user for review before creating it. Ask for confirmation.
- If `gh auth status` fails, tell the user to run `! gh auth login` to authenticate.
- The detailed plan should be thorough enough that a new Claude CLI session could pick it up and start working from the issue alone.
