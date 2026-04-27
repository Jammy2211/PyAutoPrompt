# Ship Library: Test, Branch, Push, PR

Ship source-code library changes (PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens, PyAutoConf) by running tests, ensuring a feature branch, committing, pushing, and raising a PR — for every library repository touched by this task.

## Steps

### 1. Identify Affected Repositories

Read `PyAutoPrompt/active.md` to find the current task's `worktree:` field and its `repos:` list. These are the repos shipped in this session.

```bash
source admin_jammy/software/worktree.sh
WT_ROOT="$(grep -A1 '^## <task-name>' PyAutoPrompt/active.md | grep 'worktree:' | awk '{print $3}')"
# or simpler: WT_ROOT=~/Code/PyAutoLabs-wt/<task-name>
source "$WT_ROOT/activate.sh"
```

If the task has no `worktree:` field (legacy entry), fall back to inspecting the main checkouts as before — but warn the user that this task pre-dates the worktree flow and will not be parallel-safe.

For each affected repository under `$WT_ROOT/<repo>`, run steps 2–3 below. Step 2 (draft) stays in the main Opus session — it is judgment-heavy. Step 3 is mechanical and is delegated to a Sonnet subagent per `CLAUDE.md` → **Model Delegation**. All `git` and `pytest` commands must target the worktree path, not the main checkout.

### 2. Draft the Commit Message and PR Body (Opus)

Before delegating execution, the main session drafts the artifacts the subagent will use verbatim. This keeps the judgment-heavy writing in Opus and leaves only mechanical execution for Sonnet.

For each affected repo:

- Inspect the diff against `main` (`git -C "$WT_ROOT/<repo>" diff main --stat` then `git -C "$WT_ROOT/<repo>" diff main`) to understand what changed.
- Draft a **concise commit message** summarizing the change.
- Draft the **full PR body** following the format in the "Full PR Format" subsection below, including:
  - `## Summary`
  - `## API Changes` — high-level story, max 10 lines (see "Writing the API Changes Section" below)
  - `## Test Plan` — bullet list of how to verify
  - `<details>` block with the machine-readable full API changes breakdown

Writing the `## API Changes` section is the reason this step stays in Opus. Do not delegate it.

#### Writing the API Changes Section

Analyse **all** commits on the branch (not just the latest) and the full diff against the base branch. Identify every change that affects the public API:

- **Removed** classes, functions, methods, or arguments
- **Added** new public functions, classes, or arguments
- **Renamed** or **moved** public symbols
- **Changed signatures** (new required args, removed args, changed defaults)
- **Changed behaviour** of existing public functions (e.g. different return type)

The PR body has two parts: a short human-readable API Changes summary, and a full machine-readable details block.

#### Human-readable summary (max 10 lines)

Write a brief `## API Changes` section that captures the essence of what changed. Keep it **10 lines or fewer**. Focus on the high-level story (e.g. "replaced X classes with Y functions") rather than listing every symbol. End with "See full details below." pointing readers to the collapsible block.

If there are **no** API changes, write: `None — internal changes only.`

#### Machine-readable details block

After the Test Plan section, add a `<details>` block containing the full, structured API changes. Group by change type (Removed, Added, Renamed, Changed Signature, Changed Behaviour, Migration). Use code formatting for all symbol names. This block is collapsed by default on GitHub — clean for humans, parseable by automation.

#### Full PR Format

```markdown
## Summary
<concise description of what and why>

## API Changes
<high-level summary, max 10 lines — focus on the story not every symbol>
See full details below.

## Test Plan
- [ ] <how to verify the changes>

<details>
<summary>Full API Changes (for automation & release notes)</summary>

### Removed
- `module.OldClass` — replaced by `module.new_function()`

### Added
- `module.new_function(arg1, arg2=, **kwargs)` — does X

### Migration
- Before: `obj = module.OldClass(x); obj.method()`
- After: `module.new_function(x)`

</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

### 3. Execute: Test, Commit, Push, Open PR (Sonnet)

Delegate the mechanical execution to a Sonnet subagent via the `Agent` tool with `model: "sonnet"`. See `CLAUDE.md` → **Model Delegation** for the general contract. Opus must NOT run the tests or `gh pr create` itself — those are the subagent's job.

Pass the subagent a self-contained prompt containing:

- `$WT_ROOT` (the full worktree path)
- The list of affected repos and, for each, its test directory (e.g. `test_autofit/`, `test_autoarray/`, `test_autolens/`, or "no tests" if the repo has no test dir)
- The exact commit message drafted in step 2 (one per repo)
- The exact PR body drafted in step 2 (one per repo), including the `## API Changes` section and the `<details>` block — the subagent **must paste this verbatim** into `gh pr create --body`, never rewrite it
- The target branch: `feature/<task-name>` (already exists; created by `/start_library`)
- The PR label: `pending-release`

The subagent's contract (state explicitly in the prompt):

1. For each repo, `source "$WT_ROOT/activate.sh"` and run `python -m pytest <test_dir>/ -x` from inside `"$WT_ROOT/<repo>"`. Skip this step only if the repo has no test dir.
2. If any test fails, **stop immediately**. Return the failing test names and the tail of the traceback. Do not commit. Do not try to fix.
3. If the worktree is not on `feature/<task-name>`, **stop and report**. Do not auto-switch branches.
4. If tests pass: `git -C "$WT_ROOT/<repo>" add -A && git commit -m "<message>" && git push -u origin feature/<task-name>`.
5. Create the PR with `gh pr create --label "pending-release" --title "<title>" --body "<body>"`, pasting the drafted body verbatim (use a HEREDOC).
6. Return a structured summary: one line per repo with test pass/fail counts, the commit SHA, and the PR URL.

Opus consumes the subagent's return value and proceeds to step 5 (workspace impact analysis). If the subagent reports any failure, stop and report to the user — do not proceed to step 5.

### 4. Workspace Impact Analysis and Routing

After all PRs are created, analyse the workspace impact and present the user with options.

#### 4a. Analyse workspace impact

Read the `## API Changes` section from the PR(s) created in step 3.

For each changed public symbol, search workspace scripts:

```bash
grep -rn "<old_function_or_class>" \
  autofit_workspace/scripts/ \
  autogalaxy_workspace/scripts/ \
  autolens_workspace/scripts/ \
  autolens_workspace_test/scripts/ \
  euclid_strong_lens_modeling_pipeline/scripts/ \
  HowToLens/scripts/
```

Also check if any affected scripts are in `PyAutoBuild/autobuild/config/no_run.yaml`. Scripts listed there are **skipped during the release build** — meaning the integration test for that script is disabled. If a changed API symbol is used by a script in `no_run.yaml`, flag it as **hidden risk**: the script won't be tested in the release pipeline.

#### 4b. Present data-driven options

Display the analysis and a recommendation:

```
Workspace Impact Analysis
=========================

API Changes: <brief summary>

Affected scripts: <count> scripts across <count> workspaces
  autofit_workspace:
    scripts/overview/overview_1_the_basics.py:47 — uses OldClass
  autogalaxy_workspace:
    (none found)
  autolens_workspace:
    scripts/imaging/start_here.py:89 — uses OldClass
  autolens_workspace_test:
    (none found)

Recommendation: <see logic below>

Options:
  (i)   New scripts needed — write demos for new functionality
        → Run /start_workspace after this
  (ii)  API migration needed — update scripts that use changed symbols
        → Run /start_workspace after this
  (iii) No workspace impact — confirm with smoke tests
        → Run /smoke-test now
```

**Recommendation logic:**
- API Changes says "None — internal changes only" AND grep finds no matches → recommend **(iii)**
- API Changes has **Added** entries only (no removals/renames) → recommend **(i)** or **(iii)**, user decides if demos are warranted
- API Changes has **Removed/Renamed/Changed** entries AND grep finds matches → recommend **(ii)**, list affected scripts
- API Changes has Removed/Renamed entries but grep finds NO matches → recommend **(iii)**, note that API changed but no scripts appear affected

The user may also choose a combination of (i) and (ii) — new demos AND migration. Both route to `/start_workspace`.

#### 4c. Act on the user's choice

**If user picks (i) or (ii) or both:**

Post a **progress** comment on the issue (not "Shipped"):

```bash
gh issue comment <number> --repo <owner/repo> --body "$(cat <<'PROG_EOF'
## Library PR Created — <YYYY-MM-DD>

### PR
- <PR URL(s), one per line>

### Status
Workspace changes needed. Next: /start_workspace

### API Changes
<brief summary of what changed>

PROG_EOF
)"
```

Update `active.md` — change the task's status to `library-shipped, workspace-pending`. Add the library PR URL(s) to the entry:

```markdown
## <task-name>
- issue: <issue-url>
- session: claude --resume <session-id>
- status: library-shipped, workspace-pending
- library-pr: <PR URL>
- repos:
  - PyAutoFit: feature/<branch-name>
  - PyAutoArray: feature/<branch-name>
```

Do NOT move to `complete.md`. The work is not finished yet.

Push the PyAutoPrompt update so the new status is visible from any other machine:

```bash
source PyAutoPrompt/scripts/prompt_sync.sh
prompt_sync_push "prompt: <task-name> library shipped (#<library-pr>) — workspace-pending"
```

Tell the user: "Library PR created. Run `/start_workspace` to begin workspace updates."

**If user picks (iii):**

Run the `/smoke_test` skill to validate that workspace scripts still work. Smoke tests **must** be run with `$WT_ROOT/activate.sh` sourced — otherwise the scripts will `import autofit` from the main checkout and bypass the code we're about to ship. The `/smoke_test` skill picks up `PYTHONPATH`, `NUMBA_CACHE_DIR`, and `MPLCONFIGDIR` from the active environment, so sourcing the activate script is sufficient.

```bash
source "$WT_ROOT/activate.sh"
# then invoke /smoke_test
```

- **If all smoke tests pass:** Offer to merge the library PR:
  ```bash
  gh pr merge <PR-number> --repo <owner/repo> --merge --auto
  ```
  Post a "Shipped" comment on the issue:
  ```bash
  gh issue comment <number> --repo <owner/repo> --body "$(cat <<'SHIP_EOF'
  ## Shipped — <YYYY-MM-DD>

  ### PR
  - <PR URL(s), one per line>

  ### Summary
  <2-4 sentences: what was done and the outcome>

  ### Session Notes
  <Key decisions, alternatives considered, gotchas, future work ideas.
  Omit this section entirely if there's nothing to note.>

  SHIP_EOF
  )"
  ```
  Move the task entry from `active.md` to `complete.md`:

  In `complete.md`, append:
  ```markdown
  ## <task-name>
  - issue: <issue-url>
  - completed: <YYYY-MM-DD>
  - library-pr: <PR URL(s)>
  ```

  Remove the entire task block from `active.md`.

  Push the PyAutoPrompt update:

  ```bash
  source PyAutoPrompt/scripts/prompt_sync.sh
  prompt_sync_push "prompt: ship <task-name> (#<issue>) → complete"
  ```

- **If smoke tests fail:** Report the failures. Suggest: "Smoke tests failed — this looks like option (ii). Run `/start_workspace` to investigate and fix the affected scripts." Do NOT merge. Do NOT clean up `active.md`.

Return the PR URL(s) to the user when done.

## Remote / mobile mode

**Environment detection:**
- If `~/Code/PyAutoLabs` exists and contains repo subdirectories → **laptop mode** (use all steps above)
- Otherwise → **mobile mode** (follow this section)

**Mobile behavior:**
1. Identify affected repos from `active.md` as normal
2. For each repo, use the clone in the current working directory instead of a worktree path:
   ```bash
   REPO_PATH="./<repo>"
   ```
3. Set PYTHONPATH manually before running tests:
   ```bash
   export PYTHONPATH="./PyAutoConf:./PyAutoFit:./PyAutoArray:./PyAutoGalaxy:./PyAutoLens:$PYTHONPATH"
   export NUMBA_CACHE_DIR=/tmp/numba_cache
   export MPLCONFIGDIR=/tmp/matplotlib
   ```
4. Run tests, commit, push, and raise PRs using the same steps as laptop mode but with `REPO_PATH` instead of `$WT_ROOT/<repo>`
5. Workspace impact analysis works the same way (uses grep on local clones)
6. After shipping, use `/handoff park` to update `active.md` with `location: ready-for-cli` if workspace work follows
7. Skip worktree cleanup steps (no worktrees to remove on mobile)
