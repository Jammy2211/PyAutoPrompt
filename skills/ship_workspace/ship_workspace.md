# Ship Workspace: Smoke Test, Branch, Push, PR

Ship workspace repository changes (autofit_workspace, autogalaxy_workspace, autolens_workspace, autolens_workspace_test, euclid_strong_lens_modeling_pipeline, HowToLens) by regenerating notebooks, validating with smoke tests, ensuring a feature branch, committing, pushing, and raising a PR — for every workspace repository touched by this task.

This skill ships **scripts, notebooks, and configs only**. It must never touch library source code (PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens, PyAutoConf).

## Steps

### 1. Identify Affected Workspace Repositories

Read `PyAutoPrompt/active.md` to find the current task's `worktree:` field and its `repos:` list. The workspace repos listed under this task are the ones to be shipped.

```bash
source admin_jammy/software/worktree.sh
WT_ROOT=~/Code/PyAutoLabs-wt/<task-name>
source "$WT_ROOT/activate.sh"
```

All `git`, `pytest`, and script-runner commands in the steps below must target `$WT_ROOT/<workspace>`, never the main checkout. Running without `activate.sh` sourced will import from the main checkout and silently bypass the library code that this workspace PR depends on.

Only these repositories are in scope:
- `autofit_workspace`
- `autogalaxy_workspace`
- `autolens_workspace`
- `autolens_workspace_test`
- `euclid_strong_lens_modeling_pipeline` (no `notebooks/` mirror — skip notebook regeneration for this repo)
- `HowToLens` (PyAutoBuild target: `howtolens` — pending registration in PyAutoBuild)

If any library repository (PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens, PyAutoConf) has uncommitted changes inside the worktree, **stop and tell the user** — those should be shipped with `/ship_library`, not this skill.

If the task has no `worktree:` field (legacy entry), fall back to inspecting the main checkouts as before — but warn the user that this task pre-dates the worktree flow and will not be parallel-safe.

For each affected workspace repository, run steps 2–3 below. Step 2 (draft) stays in the main Opus session; step 3 is delegated to a Sonnet subagent per `CLAUDE.md` → **Model Delegation**.

**Important:** Only edit files in `scripts/`. Never edit files in `notebooks/` directly — notebooks are auto-generated from scripts.

### 2. Draft the Commit Message and PR Body (Opus)

Before delegating execution, the main session drafts the artifacts the subagent will paste verbatim.

For each affected workspace repo:

- Inspect the diff (`git -C "$WT_ROOT/<workspace>" diff main --stat` then `git -C "$WT_ROOT/<workspace>" diff main`).
- Draft a **concise commit message**.
- Check `PyAutoPrompt/active.md` for the issue URL, then fetch issue comments and look for a "Library PR Created" comment. If one exists, capture the library PR URL for inclusion in the `## Upstream PR` section.
- Draft the **full PR body** following the PR body format below.

#### PR body format

The PR body **MUST** include a `## Scripts Changed` section listing each modified script with a brief description.

```markdown
## Summary
<concise description of what and why>

## Scripts Changed
- `scripts/overview/overview_1_the_basics.py` — updated `OldClass` usage to `new_function()`
- `scripts/imaging/start_here.py` — replaced deprecated `fit.log_likelihood` with `fit.figure_of_merit`

## Upstream PR
<library PR URL — include only if linked to an upstream library PR>

## Test Plan
- [ ] Smoke tests pass for all affected workspaces

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

If there is no upstream library PR, omit the `## Upstream PR` section entirely.

### 3. Execute: Commit, Push, Smoke Test, Open PR, Cross-Reference (Sonnet)

Delegate to a Sonnet subagent via the `Agent` tool with `model: "sonnet"`. Pass a self-contained prompt containing:

- `$WT_ROOT` (the full worktree path)
- The list of affected workspace repos
- The exact commit message drafted in step 2 (one per repo)
- The exact PR body drafted in step 2 (one per repo) — **paste verbatim** into `gh pr create --body`, never rewrite
- The target branch: `feature/<task-name>` (already exists)
- The PR label: `pending-release`
- The upstream library PR URL if one was detected in step 2 (for cross-referencing)

The subagent's contract (state explicitly in the prompt):

1. For each workspace repo, verify the branch is `feature/<task-name>`. If unexpectedly on `main`, **stop and report** — do not auto-switch.
2. `git -C "$WT_ROOT/<workspace>" add -A && git commit -m "<message>" && git push -u origin feature/<task-name>`.
3. `source "$WT_ROOT/activate.sh"`, then invoke the `/smoke_test` skill. If any smoke test fails, **stop immediately**. Return the failures verbatim. Do not create the PR. Do not attempt to fix the scripts.
4. If smoke tests pass: create the PR with `gh pr create --repo <owner/repo> --head feature/<task-name> --label "pending-release" --title "<title>" --body "<body>"`, pasting the drafted body via HEREDOC. Then **verify the label landed** with `gh pr view <number> --repo <owner/repo> --json labels --jq '[.labels[].name]'`. If `pending-release` is not in the output, **stop and report** — the label apply silently failed (usually because the label doesn't exist on the repo). Do NOT continue to step 5 until this is resolved. The fix is to run `bash admin_jammy/software/ensure_workspace_labels.sh` and then re-attach with `gh pr edit <number> --repo <owner/repo> --add-label pending-release`.
5. If a library PR URL was passed in: post a cross-reference comment on the library PR with `gh pr comment <library-PR-number> --repo <owner/repo> --body "Workspace PR: <workspace-PR-URL>"`.
6. Return a structured summary: one line per workspace with smoke-test pass/fail counts, commit SHA, workspace PR URL, and confirmation that the library-PR cross-reference comment was posted (if applicable).

Opus consumes the subagent's return value and proceeds to step 4 (merge). If the subagent reports a failure at any step, stop and report to the user — do not proceed.

### 4. Merge

Offer to merge (don't force). The user confirms before any merge happens.

**If linked to an upstream library PR — library-first merge gate:**

The library PR MUST be merged before the workspace PR is allowed to merge. Workspace scripts import the new API, so merging the workspace PR first would break `main` for anyone pulling the workspace before the library. Enforce this by checking the library PR's state:

```bash
gh pr view <library-PR-url> --json state --jq '.state'
```

- If the library PR state is **`MERGED`** → proceed with offering to merge the workspace PR.
- If the library PR state is **`OPEN`** (or any other non-merged state) → **refuse to merge** and display:

```
BLOCKED: cannot merge workspace PR until the library PR is merged.

  Library PR:   <URL> — state: <state>
  Workspace PR: <URL> — state: OPEN

  Merge the library PR first, then re-run this step (or the user
  can merge it manually in the GitHub UI). The workspace PR stays
  open until the library lands on main.
```

Do NOT use `--auto` on the workspace PR as a workaround — auto-merge can land the workspace change the instant its own CI goes green, even if the library PR is still in review. The gate is manual by design.

Once the library PR is merged, offer:

```
Library PR is merged. Merge workspace PR? <URL>
```

If confirmed:

```bash
gh pr merge <ws-PR-number> --repo <owner/repo> --merge
```

**If standalone (no upstream library PR):**

```
Merge workspace PR? <URL>
```

If confirmed:

```bash
gh pr merge <PR-number> --repo <owner/repo> --merge
```

If the user declines, skip merging — the PR stays open for manual review.

### 5. Update the GitHub Issue

#### 5a. Detect the associated issue

Check these sources in order:

1. **`PyAutoPrompt/active.md`** — look for a line containing a GitHub issue URL that matches this task.
2. **PR branch name** — if the branch name matches a known issue pattern, search for issues with matching keywords.
3. **Ask the user** — if neither source works, ask for the issue URL.

If no issue is found and the user confirms there isn't one, skip this step entirely.

#### 5b. Generate a session summary

Review the conversation and identify:

- **Key decisions** made during implementation that aren't captured in the code or PR
- **Alternatives considered** — approaches that were discussed but not taken
- **Gotchas discovered** — anything surprising or non-obvious encountered during the work
- **Future work ideas** — things the user mentioned wanting to do later

#### 5c. Post a completion comment

Post via `gh issue comment`:

```bash
gh issue comment <number> --repo <owner/repo> --body "$(cat <<'SHIP_EOF'
## Shipped — <YYYY-MM-DD>

### PRs
- <workspace PR URL(s), one per line>

### Upstream PR
<library PR URL — include only if linked. Omit this section otherwise.>

### Summary
<2-4 sentences: what was done and the outcome>

### Session Notes
<Key decisions, alternatives considered, gotchas, future work ideas from the CLI session
that didn't make it into the PR. Omit this section entirely if there's nothing to note.>

SHIP_EOF
)"
```

#### 5d. Move from `active.md` to `complete.md`

Move the entire task entry from `PyAutoPrompt/active.md` to `PyAutoPrompt/complete.md`.

In `complete.md`, append the completed entry:

```markdown
## <task-name>
- issue: <issue-url>
- completed: <YYYY-MM-DD>
- library-pr: <library PR URL — if linked>
- workspace-pr: <workspace PR URL(s)>
```

Remove the entire task block from `active.md`.

Push the PyAutoPrompt update so the completion is visible from any other machine:

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_push "prompt: ship <task-name> (#<issue>) → complete"
```

#### 5e. Handoff integration

If the task has a `location:` field in `active.md`:

- **If all work is complete** (both library and workspace PRs merged, or standalone workspace PR merged):
  Set `location: complete` before moving to `complete.md`. This is the normal flow — the task is done.

- **If work remains for the other environment** (e.g. workspace shipped from mobile but CLI needs to do post-merge cleanup):
  Instead of moving to `complete.md`, run the `/handoff park` logic:
  - Set `location:` to `ready-for-<other>` (cli if on mobile, mobile if on cli)
  - Ask for a summary of what's done and what remains
  - Commit and push `PyAutoPrompt`

Return the PR URL(s) to the user when done.

## Remote / mobile mode

**Environment detection:**
- If `~/Code/PyAutoLabs` exists and contains repo subdirectories → **laptop mode** (use all steps above)
- Otherwise → **mobile mode** (follow this section)

**Mobile behavior:**
1. Identify affected workspace repos from `active.md` as normal
2. Use clones in the current working directory instead of worktree paths:
   ```bash
   REPO_PATH="./<workspace>"
   ```
3. Set PYTHONPATH manually before running smoke tests:
   ```bash
   export PYTHONPATH="./PyAutoConf:./PyAutoFit:./PyAutoArray:./PyAutoGalaxy:./PyAutoLens:$PYTHONPATH"
   export NUMBA_CACHE_DIR=/tmp/numba_cache
   export MPLCONFIGDIR=/tmp/matplotlib
   ```
4. Commit, push, smoke test, and raise PRs using the same steps as laptop mode
5. Library-first merge gate applies identically on mobile
6. After shipping, use `/handoff park --complete` or `/handoff park` to set location appropriately
7. Skip worktree cleanup steps
