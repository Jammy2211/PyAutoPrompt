# Enforce worktree-only edits on canonical checkouts

This is the structural fix. The other six prompts add visibility and discipline;
this one removes the *opportunity* for drift entirely.

The PyAuto worktree infrastructure already exists:

- `admin_jammy/software/worktree.sh` — create/remove task worktrees, conflict-check
- `/start_library`, `/start_workspace` — set up worktrees and branches per task
- `/ship_library`, `/ship_workspace` — finish a task, remove the worktree
- Worktree root: `~/Code/PyAutoLabs-wt/<task-name>/`

But the rule "all editing happens in a worktree" was never enforced. On
2026-04-27, the workspace repos that drifted hadn't been edited in worktrees —
they'd been edited directly in the canonical checkout under
`~/Code/PyAutoLabs/<repo>/`. That's how local commits accumulated independently
of origin.

Once canonical checkouts become *read-only mirrors of origin/main*, the drift
mechanism is structurally impossible: any change has to go through a worktree,
which has to push to origin, which becomes the single source of truth by
construction.

## What to ship

Three layers, all required:

### 1. Make the rule explicit

Add to every PyAuto repo's `CLAUDE.md` (consolidate with the rules from
`03_history_rewrite_guard.md` and `04_source_of_truth_rule.md`):

```markdown
## No editing in canonical checkouts

The directory `~/Code/PyAutoLabs/<RepoName>/` is a **read-only mirror** of
`origin/main`. Never edit files in it directly. All work happens in
**task worktrees** under `~/Code/PyAutoLabs-wt/<task-name>/`, set up by
`/start_library` or `/start_workspace`.

If you find yourself wanting to `cd ~/Code/PyAutoLabs/<repo>/` to "just fix
something quickly" — stop. That's how parallel-history drift starts. Use
`/start_dev` to draft a prompt for it instead, even for one-line fixes.

The only operations allowed in the canonical checkout: `git fetch`, `git pull
--ff-only`, `git status`, `git log`. Everything else uses a worktree.
```

### 2. Soft enforcement — pre-edit hook

A `~/.bashrc` function or a `direnv` config that prints a loud warning when
`vim` / `nano` / `cursor` / `code` / etc. opens a file under
`~/Code/PyAutoLabs/<repo>/` (not `~/Code/PyAutoLabs-wt/`):

```bash
__pyauto_canonical_warn() {
  case "$PWD" in
    "$HOME/Code/PyAutoLabs/admin_jammy"|"$HOME/Code/PyAutoLabs/admin_jammy/"*) return ;;
    "$HOME/Code/PyAutoLabs/PyAutoPrompt"|"$HOME/Code/PyAutoLabs/PyAutoPrompt/"*) return ;;
    "$HOME/Code/PyAutoLabs/"*)
      printf "\033[31m⚠ Editing in canonical checkout. Did you mean to use a worktree?\033[0m\n" >&2
      printf "  /start_dev <prompt-file> → /start_library to create one.\n" >&2
      ;;
  esac
}
```

(Wired into `cd` or as a function `vw` / `vw 'just open my editor'`.) Whitelist
admin_jammy and PyAutoPrompt because those *are* edited directly — the rule is
specifically about library and workspace canonical checkouts.

### 3. Hard enforcement — `/sync` resets canonical checkouts unconditionally

After `05_sync_slash_command.md` ships, the `/sync` skill should treat canonical
checkouts of library/workspace repos as fully resettable to `origin/main` with
no confirmation needed for ff-pulls and only "is everything duplicated?" checks
for divergent state. Worktrees are where work lives; canonicals are throwaway.

The implication: any uncommitted local change in a canonical checkout should be
treated as junk by default. If you really need to keep it, move it to a
worktree first.

## Acceptance

- The CLAUDE.md rule lands in every library and workspace repo.
- The shell warning fires when opening a file in a canonical PyAuto repo, and
  doesn't fire for admin_jammy / PyAutoPrompt.
- `/sync` no longer asks "what about your dirty file in PyAutoArray/scratch.py?"
  — it just nukes it after a 2-second warning.
- A month after this lands, `pyauto-status` should show clean dirty:0 across
  all canonicals. Any non-zero is a sign someone broke the rule.

## Migration

The current canonicals already have stray dirty files (`PyAutoArray/test_autoarray/profile_tst.py`,
`PyAutoGalaxy/scrap.py` etc.). Move each to either a real prompt + worktree
flow, or delete it as scratch. Don't carry them across.

## Out of scope

- Symlinks / read-only filesystems for canonical checkouts. Possible but invasive
  (breaks `pip install -e`). Behavioural enforcement via warning + `/sync` reset
  is enough.
- Renaming the canonical-vs-worktree convention. Keep the existing path layout.

## Why this is the biggest single fix

Prompts 01-06 catch drift after it happens or warn about modes that lead to
drift. This one removes the path that produced the 2026-04-27 mess: making
local commits in a canonical checkout that origin doesn't know about.

If only one of these seven prompts ships, ship this one — the others become
nice-to-haves rather than load-bearing.

## Files touched

- CLAUDE.md across all library + workspace repos (one PR each, can batch)
- `~/.bashrc` (one-time, on this machine)
- `PyAutoPrompt/skills/sync/SKILL.md` (depends on `05_sync_slash_command.md`)
