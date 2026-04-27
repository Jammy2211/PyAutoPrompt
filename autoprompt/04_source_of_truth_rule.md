# "Pull before edit" / source-of-truth rule

> ⚠️ **Caveat — drafted from a stale repo state.** This prompt was drafted on 2026-04-27 during a forensic sweep that found local checkouts up to 101 commits behind origin. The trigger looked like a structural workflow flaw, but later analysis showed the drift was largely driven by **stale local checkouts being edited without `git pull` first**, not by missing tooling. Now that PyAutoPrompt is the canonical source-of-truth and `skills/install.sh` auto-discovers across both repos, some of the recommendations below may be over-engineered for the day-to-day case. Re-evaluate whether each measure is still warranted — the cheap habits (pull before edit, never rewrite history) buy most of the win.

Editing on a stale local checkout — without first pulling origin — is what
created the 41 redundant commits cleaned up on 2026-04-27. Two environments
(this laptop, plus cloud agents pushing PRs) made the *same* edits independently
because neither pulled the other's work first. The fix is a single rule, encoded
where every agent and every shell will read it.

## What to ship

In every PyAuto repo's `CLAUDE.md` (right after or alongside the "Never rewrite
history" rule from `03_history_rewrite_guard.md`), add:

```markdown
## Source of truth

The `github.com/PyAutoLabs/*` and `github.com/Jammy2211/*` and
`github.com/rhayes777/PyAutoFit` remotes are the **source of truth**. This local
checkout is downstream.

Before starting any work in a repo:

    git fetch origin
    git status

If `git status` reports `Your branch is behind 'origin/main' by N commits`, run
`git pull --ff-only` BEFORE editing anything. Editing on a stale checkout
produces duplicate-content commits when the upstream has already shipped the
same change. This is what created the 41 redundant local commits across three
workspace repos that had to be discarded on 2026-04-27.

If `git pull --ff-only` refuses (working tree dirty / commits ahead), STOP and
either commit/stash, or surface the divergence to the user — never paper over it.
```

Optional but recommended: a one-liner in shell rc that warns on `cd` into a
stale repo:

```bash
# in ~/.bashrc — runs on any cd
__pyauto_stale_warn() {
  [ -d .git ] || return
  local behind
  behind=$(git rev-list --count HEAD..@{u} 2>/dev/null) || return
  [ "$behind" = "0" ] && return
  printf "\033[33m⚠ %d commits behind upstream — git pull --ff-only before editing\033[0m\n" "$behind"
}
PROMPT_COMMAND="__pyauto_stale_warn; ${PROMPT_COMMAND:-}"
```

## Acceptance

- The "Source of truth" section is present in every PyAuto repo's CLAUDE.md.
- An agent (Claude Code, Codex, cloud Claude) reading CLAUDE.md before acting
  will see the rule before it considers any edit.
- The `__pyauto_stale_warn` shell hook fires when entering a stale repo (test
  manually).

## Companion: AGENTS.md

If/when a repo grows an `AGENTS.md` for non-Claude agents (e.g. Codex), the same
rule must appear there. Keep wording identical so cross-agent behavior matches.

## Out of scope

- Automated `git pull` on `cd`. Tempting but risky — can clobber dirty trees,
  and pulling at unexpected moments breaks reasoning about state. Warning is
  the right boundary.
- Forbidding parallel-agent work. The point is *not* to serialize, only to make
  the flow direction explicit (origin = hub, local = consumer).

## Files touched

Same set as `03_history_rewrite_guard.md`. These two prompts are tightly linked
and should ship in the same PR per repo.
