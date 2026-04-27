# Cross-repo sync status dashboard

Drift across PyAuto repos is currently invisible — there's no single command
that shows whether each library and workspace is in sync with `origin/main`.
The 2026-04-27 audit found 12 repos with up to 101 commits of behind-origin
state, and we only noticed because of an unrelated investigation. Catching it
early needs a dashboard that's two keystrokes away.

## What to build

A shell function `pyauto-status` (loaded via `~/.bashrc` or `~/.local/bin/`)
that, for every git repo under `~/Code/PyAutoLabs/`, prints:

- repo name
- current branch
- upstream tracking ref (or `NONE` if missing — a real failure mode that hid
  `autolens_workspace_developer` for weeks)
- behind / ahead counts vs `@{u}`
- dirty file count
- a single-glyph flag column: `↓` for behind, `↑` for ahead, `*` for dirty

Implementation notes:

- Run `git fetch origin --quiet` per repo before counting (a stale dashboard is
  worse than none).
- Skip directories that aren't git repos cleanly (warn, don't crash).
- Use the upstream branch from `git rev-parse --abbrev-ref @{u}`, not a
  hardcoded `origin/main`. PyAutoFit recently moved from `main_build` → `main`
  and a hardcoded version would have hidden that.

A working draft is included in
`PyAutoPrompt/autoprompt/snippets/pyauto_status.sh` (see file). Test it; tune the
column widths; either inline it into `~/.bashrc` or symlink from `~/.local/bin/`.

## Acceptance

- Running `pyauto-status` from any directory shows all 12 repos in one screen.
- A repo on a non-default branch is visible (don't auto-checkout `main`).
- A repo with no upstream is flagged, not silently skipped.
- The whole sweep finishes in well under 10 seconds (parallel-friendly fetch).

## Out of scope

- Auto-pulling. This is a status command only — fixing drift is a separate
  prompt (`05_sync_slash_command.md`).
- Pretty colors / TUI. Plain text is faster to read and pipe.

## Files touched

- `~/.bashrc` (or `~/.local/bin/pyauto-status` + `chmod +x`)
- Optionally: `PyAutoPrompt/scripts/status.sh` already prints prompt-registry
  counts; consider adding a `--repos` flag that delegates to this function.
