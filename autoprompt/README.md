# autoprompt/

Prompts about the PyAuto workflow infrastructure itself — i.e. the tooling that
makes prompts in this repo flow smoothly from idea to merged PR.

> ⚠️ **Caveat — drafted from a stale repo state.** These prompts were drafted on
> 2026-04-27 during a forensic sweep that found local checkouts up to 101 commits
> behind origin. The trigger looked like a structural workflow flaw, but later
> analysis showed the drift was largely driven by **stale local checkouts being
> edited without `git pull` first**, not by missing tooling. Now that PyAutoPrompt
> is the canonical source-of-truth and `skills/install.sh` auto-discovers across
> both repos, several of the recommendations here may be over-engineered for the
> day-to-day case. Re-evaluate whether each measure is still warranted before
> implementing — the cheap habits (pull before edit, never rewrite history) buy
> most of the win. Each prompt has its own caveat at the top.

These were drafted after a forensic sweep on 2026-04-27 found that local checkouts
across 12 repos had drifted up to 101 commits behind origin, with parallel
"fresh start" history rewrites and 41 redundant local commits. The root causes
were structural — multiple environments writing in parallel without flow rules,
no visible drift indicator, generated artifacts polluting `git status` so real
divergence was invisible, and skills/scripts that didn't enforce a "pull first"
discipline.

The seven prompts below address those root causes in increasing order of
investment / impact. Numbers indicate suggested implementation order; later
prompts assume earlier ones have landed.

| # | Prompt | Tier | Effort | Prevents |
|---|---|---|---|---|
| [01](01_status_dashboard.md) | `pyauto-status` shell function | 1 | 15 min | Drift being invisible |
| [02](02_gitignore_noise.md) | `.gitignore` workspace artifacts | 1 | 30 min | Generated files burying real `git status` signal |
| [03](03_history_rewrite_guard.md) | "Never rewrite history" rule in CLAUDE.md/AGENTS.md | 2 | 30 min | Independent `git init` fresh-starts on multiple machines |
| [04](04_source_of_truth_rule.md) | "Pull before edit" rule | 2 | 30 min | Duplicate-content commits from stale checkouts |
| [05](05_sync_slash_command.md) | `/sync` slash command | 3 | 2-3 h | Manual drift recovery being too expensive to do regularly |
| [06](06_repo_health_audit.md) | Monthly repo-health audit | 3 | 1-2 h | Untethered checkouts, dead branches, stale stashes |
| [07](07_worktree_only_edits.md) | Enforce worktree-only edits on canonical checkouts | 4 | 4-6 h | The whole class of drift, structurally |

If you do nothing else, do **01** and **03**. Those two cost ~45 minutes and
would have prevented the bulk of what was cleaned up on 2026-04-27.

The *biggest* single fix is **07** — once canonical checkouts of `PyAuto*` and
`*_workspace*` repos become read-only mirrors of origin/main with all editing
happening in task worktrees, the drift mechanism is structurally impossible.
The infrastructure already exists (`admin_jammy/software/worktree.sh`,
`/start_library`, `/start_workspace`, `/ship_*`); the missing piece is a rule
saying "no one ever `cd` into `~/Code/PyAutoLabs/<repo>/` to edit, ever."
