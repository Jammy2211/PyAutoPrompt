#!/usr/bin/env bash
# pyauto_audit.sh — on-demand structural repo-health audit.
#
# Defines a shell function `pyauto-audit` that scans ~/Code/PyAutoLabs/ for
# state the dashboard (`pyauto-status`) doesn't surface:
#
#   1. Top-level directories with no .git (intentionally-not-a-repo or bug).
#      Skip prefixes "." (hidden) and "z_" (user's personal/staging convention).
#   2. Stashes older than $PYAUTO_AUDIT_STASH_DAYS (default 14) — drift-from-
#      stash is a real failure mode.
#   3. Local-only branches with no upstream and last commit older than
#      $PYAUTO_AUDIT_BRANCH_DAYS (default 30) — likely abandoned work.
#
# Run on demand. Always exits 0 — informational, the user reads + decides.
#
# Usage:
#   source ~/Code/PyAutoLabs/PyAutoPrompt/scripts/pyauto_audit.sh
#   pyauto-audit
#
# Override via env vars:
#   PYAUTO_AUDIT_ROOT         scan root (default $HOME/Code/PyAutoLabs)
#   PYAUTO_AUDIT_STASH_DAYS   stash age threshold in days (default 14)
#   PYAUTO_AUDIT_BRANCH_DAYS  branch age threshold in days (default 30)

PYAUTO_AUDIT_ROOT="${PYAUTO_AUDIT_ROOT:-$HOME/Code/PyAutoLabs}"
PYAUTO_AUDIT_STASH_DAYS="${PYAUTO_AUDIT_STASH_DAYS:-14}"
PYAUTO_AUDIT_BRANCH_DAYS="${PYAUTO_AUDIT_BRANCH_DAYS:-30}"

pyauto-audit() {
  local root="$PYAUTO_AUDIT_ROOT"
  if [[ ! -d "$root" ]]; then
    echo "pyauto-audit: $root does not exist" >&2
    return 1
  fi

  local now stash_thresh branch_thresh
  now=$(date +%s)
  stash_thresh=$(( PYAUTO_AUDIT_STASH_DAYS * 86400 ))
  branch_thresh=$(( PYAUTO_AUDIT_BRANCH_DAYS * 86400 ))

  # Section 1: non-git directories. Skip prefixes are hardcoded — if the
  # legitimate exceptions ever exceed 3-4 patterns, switch to a snooze file.
  local skip_prefixes=("." "z_")
  local non_git=() dir name skip prefix
  for dir in "$root"/*/; do
    [[ ! -d "$dir" ]] && continue
    name="$(basename "$dir")"
    skip=false
    for prefix in "${skip_prefixes[@]}"; do
      [[ "$name" == "$prefix"* ]] && skip=true && break
    done
    [[ "$skip" == "true" ]] && continue
    [[ -e "$dir.git" ]] && continue
    non_git+=("$name")
  done

  # Section 2: old stashes. `--format='%gd|%ad|%ct|%s'` gives the stash ref,
  # short date, commit timestamp, and subject in one line per entry.
  local stash_lines=() repo line ref short_date ts subj age
  for dir in "$root"/*/.git; do
    [[ -e "$dir" ]] || continue
    repo="${dir%/.git}"
    name="$(basename "$repo")"
    while IFS='|' read -r ref short_date ts subj; do
      [[ -z "$ts" ]] && continue
      age=$(( now - ts ))
      (( age < stash_thresh )) && continue
      stash_lines+=("$name: $ref ($short_date) $subj")
    done < <(git -C "$repo" stash list --date=short --format='%gd|%ad|%ct|%s' 2>/dev/null)
  done

  # Section 3: abandoned local-only branches. `for-each-ref` with empty
  # `%(upstream)` filters local-only branches, then we age-filter on commit
  # timestamp. Pipe delimiter (not tab) because bash treats consecutive
  # whitespace IFS chars as one separator, which would collapse the empty
  # upstream column into the timestamp.
  local branch_lines=() branch upstream branch_ts iso
  for dir in "$root"/*/.git; do
    [[ -e "$dir" ]] || continue
    repo="${dir%/.git}"
    name="$(basename "$repo")"
    while IFS='|' read -r branch upstream branch_ts; do
      [[ -z "$branch" ]] && continue
      [[ -n "$upstream" ]] && continue
      age=$(( now - branch_ts ))
      (( age < branch_thresh )) && continue
      iso=$(date -d "@$branch_ts" +%Y-%m-%d 2>/dev/null)
      branch_lines+=("$name: $branch (last: $iso)")
    done < <(git -C "$repo" for-each-ref --format='%(refname:short)|%(upstream)|%(committerdate:unix)' refs/heads/ 2>/dev/null)
  done

  # Output. Sections suppressed when empty; each prints its own header.
  local printed=false
  if (( ${#non_git[@]} > 0 )); then
    echo "Non-git directories under $root:"
    for name in "${non_git[@]}"; do echo "  $name/"; done
    printed=true
  fi
  if (( ${#stash_lines[@]} > 0 )); then
    [[ "$printed" == "true" ]] && echo ""
    echo "Old stashes (>$PYAUTO_AUDIT_STASH_DAYS days):"
    for line in "${stash_lines[@]}"; do echo "  $line"; done
    printed=true
  fi
  if (( ${#branch_lines[@]} > 0 )); then
    [[ "$printed" == "true" ]] && echo ""
    echo "Abandoned local-only branches (no upstream, last commit >$PYAUTO_AUDIT_BRANCH_DAYS days):"
    for line in "${branch_lines[@]}"; do echo "  $line"; done
    printed=true
  fi
  [[ "$printed" == "false" ]] && echo "pyauto-audit: clean (no findings under $root)"
  return 0
}
