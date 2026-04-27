# "Never rewrite history" guard in CLAUDE.md / AGENTS.md

> ⚠️ **Caveat — drafted from a stale repo state.** This prompt was drafted on 2026-04-27 during a forensic sweep that found local checkouts up to 101 commits behind origin. The trigger looked like a structural workflow flaw, but later analysis showed the drift was largely driven by **stale local checkouts being edited without `git pull` first**, not by missing tooling. Now that PyAutoPrompt is the canonical source-of-truth and `skills/install.sh` auto-discovers across both repos, some of the recommendations below may be over-engineered for the day-to-day case. Re-evaluate whether each measure is still warranted — the cheap habits (pull before edit, never rewrite history) buy most of the win.

The single most damaging drift mechanism on 2026-04-27 was **independent `git init`
"fresh start" rewrites** on different machines. The three workspace repos ended
up with **no merge base at all** with origin — local and origin had each gone
through "Initial commit — fresh start for AI workflow" workflows, producing
identical content under entirely different SHAs. 41 commits had to be discarded.

The cause is a class of operations that humans and AI agents both find tempting:

- `rm -rf .git && git init` to "start clean"
- Squashing entire histories into a single "Initial commit"
- `git push --force` to main when local diverges from origin
- Cherry-picking onto a freshly-init'd local branch

Each of these is sometimes the right tool. None of them is the right tool when
the goal is "I want a clean working tree" — `git fetch && git reset --hard
origin/main && git clean -fd` does the same thing without breaking shared history.

## What to ship

Add a `## Never rewrite history` section to every PyAuto repo's `CLAUDE.md`
(and `AGENTS.md` if one exists), worded for both humans and AI agents:

```markdown
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
```

Also add to the `## General Rules` of any repo where one exists, a single line:

```markdown
- Before any `git init`, `git push --force`, or destructive history operation,
  stop and confirm with the user. These rules are non-negotiable.
```

## Acceptance

- The section is present in every PyAuto* and *_workspace* repo's CLAUDE.md.
- A casual `grep "Never rewrite history" -- '**/CLAUDE.md'` over the whole
  ecosystem returns hits in every repo.
- Cloud and local agent invocations from now on respect the rule by default
  (verify by reading agent transcripts post-deployment for any "fresh start"
  language).

## Out of scope

- A pre-commit hook that blocks "Initial commit"-style messages on remote-tracked
  branches. Useful but more brittle than CLAUDE.md discipline; can be a follow-up.
- Renaming existing fresh-start commits in history. They're already in.

## Files touched

One PR per repo, just `CLAUDE.md` (and `AGENTS.md` where present):

- PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens
- autofit_workspace, autogalaxy_workspace, autolens_workspace
- autofit_workspace_test, autolens_workspace_test
- autofit_workspace_developer, autolens_workspace_developer
- HowToLens (if it has CLAUDE.md)
- admin_jammy
- PyAutoPrompt
