# Start Dev For User: Pick Up a User-Filed GitHub Issue

Variant of `/start_dev` whose starting point is a GitHub issue **opened by an external user** rather than a `PyAutoPrompt/` prompt file. Same downstream routing (classify → branch survey → register → `/start_library` or `/start_workspace`), but with two extra responsibilities:

1. **Conversational, milestone-driven updates posted back to the issue** so the reporter can follow progress without reading code or commits.
2. **An explicit clarification gate** — user-filed issues are often incomplete, so the skill stops and asks for missing info before producing a plan.

## Usage

```
/start_dev_for_user <issue-url-or-number>
```

Examples:
- `/start_dev_for_user https://github.com/Jammy2211/PyAutoArray/issues/512`
- `/start_dev_for_user Jammy2211/PyAutoArray#512`
- `/start_dev_for_user 512` (only valid when the cwd is inside a checkout of the relevant repo — the skill resolves `owner/repo` via that repo's origin)

## Tone rules

These apply to every issue comment this skill (and its downstream successors) post on a `user-facing: true` task:

- First and second person, contractions allowed. Write like a polite teammate, not a status bot.
- Lead with what the reporter cares about: status, plan, what's next. File paths and class names go inside `<details>` blocks.
- No emojis (matches the wider project convention; only on explicit request).
- Length budget: receipt comment ≈ 2 lines, plan comment ≈ 15 visible lines + collapsible detail, routing comment ≈ 3 lines.
- When asking for clarification, group every question into **one** comment. Explain in one sentence why each piece of info would help.

## Steps

### 0. Check for parked handoff tasks

Identical to `/start_dev` step 0 — check `PyAutoPrompt/active.md` for any task with a `location:` field matching the current environment (`ready-for-cli` on the laptop, `ready-for-mobile` on mobile/server). If one exists, offer to resume it before starting fresh. If the user picks resume, hand off to `/handoff resume` and stop.

### 1. Resolve and read the issue

**Normalize the argument.** Accept any of:
- Full URL: `https://github.com/<owner>/<repo>/issues/<n>`
- Slug: `<owner>/<repo>#<n>`
- Bare number: `<n>` (only valid when cwd is inside a known PyAuto repo checkout — derive `<owner>/<repo>` from `git -C . remote get-url origin` and the mapping table in step 2; abort with a friendly error otherwise)

Strip surrounding whitespace, backticks, and angle brackets.

**Fetch the issue:**

```bash
gh issue view <n> --repo <owner>/<repo> \
  --json number,title,body,author,labels,state,url,comments
```

- If `.state == "CLOSED"`: abort. Tell the developer the issue is closed and won't be picked up.
- If `.comments` already contains a comment from the bot account whose body includes the marker `<!-- start_dev_for_user:claimed -->`, this issue has been claimed before. Warn the developer and offer to **resume** instead of re-claim:
  - "Resume" → skip step 1a, treat as a re-run; jump to whichever step is appropriate based on the existing `active.md` entry (e.g. resume at clarification gate, or at plan posting).
  - "Re-claim" → only allowed if the previous claim was abandoned. Confirm before posting a fresh receipt.

### 1a. Post receipt comment (milestone #1)

Post a short acknowledgment so the reporter sees activity within seconds of the developer running the skill:

```bash
gh issue comment <n> --repo <owner>/<repo> --body "$(cat <<'RECEIPT_EOF'
Hi @<author> — thanks for the report. I'm Jammy's CLI assistant; I'll take a look and post back shortly with a plan, or follow-up questions if I need more detail.

<!-- start_dev_for_user:claimed -->
RECEIPT_EOF
)"
```

Substitute the reporter's actual GitHub handle for `<author>`. The HTML comment marker is invisible to the reporter but lets step 1 detect re-runs.

### 2. Identify target repositories

Scan the issue **title + body + reporter-supplied details** for repository references. The convention is `@RepoName` (e.g. `@PyAutoFit`, `@PyAutoArray/autoarray/operators/convolver.py`); bare path references also count.

**Local directory to GitHub repo mapping:**

| Local Directory | GitHub Repo |
|---|---|
| `PyAutoConf` | `rhayes777/PyAutoConf` |
| `PyAutoFit` | `rhayes777/PyAutoFit` |
| `PyAutoArray` | `Jammy2211/PyAutoArray` |
| `PyAutoGalaxy` | `Jammy2211/PyAutoGalaxy` |
| `PyAutoLens` | `Jammy2211/PyAutoLens` |
| `autofit_workspace` | `Jammy2211/autofit_workspace` |
| `autogalaxy_workspace` | `Jammy2211/autogalaxy_workspace` |
| `autolens_workspace` | `Jammy2211/autolens_workspace` |
| `HowToLens` | `Jammy2211/HowToLens` |

Count references per repo to determine the primary. **If the issue body has no repo references at all, default the primary repo to the repo the issue lives on.** If counts are tied across multiple repos, ask the developer.

### 3. Classify the work

Same as `/start_dev` step 3:

- **Library repos**: PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens
- **Workspace repos**: autofit_workspace, autogalaxy_workspace, autolens_workspace, autolens_workspace_test, euclid_strong_lens_modeling_pipeline, HowToLens

- Primary in libraries → **library work**
- Primary in workspaces → **workspace work**
- Both → **starts library, workspace follows after shipping**
- Ambiguous → ask the developer

### 4. Explore the referenced code

Read the key files and classes mentioned in the issue to build context. Focus on classes/functions referenced directly, the surrounding architecture (imports, inheritance, callers), and existing tests. Spend a few minutes — enough for an informed plan, not an exhaustive audit.

### 5. Clarification gate

Before producing a plan, decide whether the issue is **actionable**. Heuristics for "unclear":

- Bug report with no reproduction steps
- No version / no error trace where one is needed
- Ambiguous scope ("make it faster" with no target metric, "fix the plotting" without saying which plot)
- Conflicting signals between title and body
- Asked-for behavior that contradicts a documented design choice (worth confirming intent rather than guessing)

**If the issue is clear**, proceed to step 6.

**If the issue is unclear**, run the clarification subroutine:

1. Use `AskUserQuestion` to surface the gaps **to the developer (Jammy) first**. Present each gap as a question option ("missing repro? missing version? both? something else?") so the developer can curate, add, or drop questions before they go public. The skill's job here is to triage — the developer decides what's actually worth asking the reporter.

2. Compose **one** consolidated clarifying comment in the conversational tone described above. Never post a numbered legalistic checklist; write it like a teammate. Tag the reporter and explain in one sentence per question why the info would help.

3. Post via:
   ```bash
   gh issue comment <n> --repo <owner>/<repo> --body "$(cat <<'CLARIFY_EOF'
   <comment body>
   CLARIFY_EOF
   )"
   ```

4. Apply a `needs-info` label if the repo has one:
   ```bash
   gh label list --repo <owner>/<repo> | grep -q '^needs-info' && \
     gh issue edit <n> --repo <owner>/<repo> --add-label needs-info
   ```

5. Add a partial entry to `PyAutoPrompt/active.md`:
   ```markdown
   ## <task-name>
   - issue: <issue-url>
   - user-facing: true
   - status: awaiting-info
   ```
   Skip `worktree:`, `repos:`, and `session:` fields — the task isn't actually started, just registered so it shows up in `/status`.

6. Tell the developer:
   ```
   Posted clarifying questions on the issue. Re-run /start_dev_for_user <issue-url>
   once the reporter replies.
   ```

7. **Stop.** Do not run plan generation, branch survey, or routing. Do not push PyAutoPrompt — the partial registration will be picked up by the next push or by the resumed run.

### 6. Generate the plan

Produce **two levels of plan** — same format as `/start_dev` step 5:

**High-level plan** (3–8 bullet points): plain-English, no code, no file paths, describes what will change and why and the logical sequence of work.

**Detailed plan** (step-by-step): file paths and function/class names, specific changes per step, key decisions and trade-offs, testing approach.

### 7. Run plan_branches analysis

Same as `/start_dev` step 6. For each affected repo:

```bash
git -C <repo_path> branch --show-current
git -C <repo_path> status --short
git -C <repo_path> branch --sort=-committerdate | head -5
```

Display a summary table and suggest a branch name in the format `feature/<short-description>` (lowercase, kebab-case, under 50 chars).

Compute the task name (the kebab-case stem without the `feature/` prefix) and the worktree root path: `~/Code/PyAutoLabs-wt/<task-name>/`. Pass this to `/start_library` later.

### 8. Generate a working title

Internal task name only — we **do not rename the issue**. Use the standard conventional prefixes (`feat:`, `fix:`, `refactor:`, `docs:`, `perf:`) for the branch suffix.

### 9. Post the plan as an issue comment (milestone #2)

We do **not** create a new issue. We comment on the existing one.

The comment reuses the *content* of the `/start_dev` issue-body template, but reframed so it reads like an update from a teammate, not a code dump. Structure:

```markdown
Here's what I'm planning to do — let me know if anything looks off.

<one-paragraph plain-English summary>

**Plan**
- <high-level bullet 1>
- <high-level bullet 2>
- <…>

<details>
<summary>Detailed implementation plan</summary>

### Affected repositories
- repo1 (primary)
- repo2

### Work classification
<Library / Workspace / Both>

### Branch survey

| Repository | Current Branch | Dirty? |
|---|---|---|
| ./RepoName | main | clean |

**Suggested branch:** `feature/<name>`
**Worktree root:** `~/Code/PyAutoLabs-wt/<name>/` (created later by `/start_library`)

### Implementation steps

1. <detailed step with file paths>
2. <…>

### Key files
- `path/to/file.py` — description of changes

</details>

I'll post back when the work is in progress and again when there's a PR. If anything in the plan looks off, just reply and I'll adjust.
```

**Present the comment body to the developer for review before posting.** Same review gate as `/start_dev` step 8 — never post without explicit approval.

Post via:

```bash
gh issue comment <n> --repo <owner>/<repo> --body "$(cat <<'PLAN_EOF'
<comment body>
PLAN_EOF
)"
```

### 10. Check for repo conflicts

Identical to `/start_dev` step 9, including the 9a/9b sub-paths:

```bash
source admin_jammy/software/worktree.sh
worktree_check_conflict <task-name> <repo1> [repo2 ...]
```

If `worktree_check_conflict` exits **non-zero**, this task cannot start yet → step 10a.
If it exits **zero**, the task can start → step 10b.

The legacy unregistered-feature-branch warning from `/start_dev` step 9 also applies here.

#### 10a. Route to planned.md (conflict — task is queued)

Add the entry to `PyAutoPrompt/planned.md`:

```markdown
## <task-name>
- issue: <issue-url>
- user-facing: true
- planned: <YYYY-MM-DD>
- classification: <library | workspace | both>
- suggested-branch: feature/<name>
- blocked-by: <conflicting-task-name> (using <repo>)
- affected-repos:
  - <repo1>
  - <repo2>
```

The `user-facing: true` field carries through so the eventual unblocked session keeps the conversational mode on.

Tell the developer:

```
Task planned but cannot start yet.

  Blocked by: <conflicting-task-name> — using <repo>
  Issue: <issue URL>
  Entry added to: planned.md

  Once the blocking task ships, run /status to check availability,
  then /start_library or /start_workspace to begin this task.
```

Skip to step 13.

#### 10b. Route to active.md (no conflict — task can start)

Add the entry to `PyAutoPrompt/active.md`:

```markdown
## <task-name>
- issue: <issue-url>
- user-facing: true
- session: claude --resume <session-id>
- status: <library-dev | workspace-dev>
- worktree: ~/Code/PyAutoLabs-wt/<task-name>
- repos:
```

The `user-facing: true` field is the **mode signal** for downstream skills (see "Forward compatibility" below). The `status` field follows the same rule as `/start_dev`:

- Library work or both → `library-dev`
- Workspace work → `workspace-dev`

The `worktree:` directory does not yet exist — `/start_library` creates it. The `repos:` list starts empty and is populated when branches are created.

### 11. (No prompt-file move)

Removed. There is no prompt file in this flow.

### 12. Route to the next skill (milestone #3, optional)

Same routing message as `/start_dev` step 11:

- **Library work**: "Run `/start_library` to set up your development environment."
- **Workspace work**: "Run `/start_workspace` to set up your development environment."
- **Both**: "Run `/start_library` first. After shipping the library changes, `/start_workspace` will follow."

Display:
- The issue URL
- The primary repo
- The suggested branch name
- The work classification
- The next skill to run

**Then add this reminder, specific to this skill:**

> This task is `user-facing: true`. Use `/update-issue` between milestones with a conversational summary; downstream skills will eventually do this automatically (see "Forward compatibility" in `start_dev_for_user.md`).

**Optional third milestone comment.** Ask the developer via `AskUserQuestion` whether to post a brief "Plan looks good — starting work now" comment to the issue. Default to **not** posting unless they confirm — if they're moving straight into `/start_library`, that skill (once updated for `user-facing: true` mode) can post the "starting work" milestone instead. Posting both would be a noisy double comment.

### 13. Push PyAutoPrompt

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_push "prompt: route <task-name> (#<issue>) → <next-skill> [user-facing]"
```

Substitute the actual task name, issue number, and routing destination (`/start_library`, `/start_workspace`, or `planned`). The `[user-facing]` tag in the commit message makes it trivial to grep the log for issues that originated from external reporters.

If the clarification gate fired in step 5 instead, the push uses:

```bash
prompt_sync_push "prompt: register <task-name> (#<issue>) awaiting reporter info [user-facing]"
```

## Forward compatibility — `user-facing: true`

This skill introduces the `user-facing: true` field on `active.md` / `planned.md` entries. **Downstream skills do not yet read it.** This is intentional: the field is cheap to add, costs nothing if ignored, and lets `/start_dev_for_user` ship standalone.

A follow-up task should teach the downstream skills to honor the flag:

- **`/update-issue`**: when `user-facing: true`, default to conversational tone (no commit-SHA tables in the visible body — fold them inside `<details>`), shorter summaries, and address the reporter directly.
- **`/ship_library` and `/ship_workspace`**: post a milestone comment when the PR opens ("PR is up: <link> — I'll let you know when it merges.") and again on merge ("Merged into main. Closing this issue; reopen if anything regressed.").
- **`/handoff`**: surface user-facing tasks distinctly in the parked-task list so a fresh session knows they have a reporter waiting.

Until those updates land, the developer should manually call `/update-issue` (or post comments by hand) at PR-open and merge milestones to keep the cadence intact.

## Notes

- Always present every comment body to the developer for review before posting. The clarification gate, the plan comment, and the optional "starting work" comment all go through the same approval step.
- If `gh auth status` fails, tell the developer to run `! gh auth login` to authenticate.
- Reporter handles need to be quoted carefully when injected into HEREDOCs — they always start with `@` and won't contain anything weird, but check the JSON `.author.login` field rather than parsing the rendered body.
- The detailed plan inside the `<details>` block should be thorough enough that a fresh Claude CLI session resuming via `claude --resume` can pick up from the issue + `active.md` alone.
