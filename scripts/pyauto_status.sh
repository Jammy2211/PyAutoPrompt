#!/usr/bin/env bash
# pyauto_status.sh — cross-repo sync status dashboard.
#
# Defines a shell function `pyauto-status` that prints, for every git repo
# under ~/Code/PyAutoLabs/, the branch, upstream tracking ref, behind/ahead
# counts vs @{u}, dirty file count, and a flag column. Designed to run in
# under 10 seconds — fetches are parallelised one background job per repo.
#
# Usage:
#   source ~/Code/PyAutoLabs/PyAutoPrompt/scripts/pyauto_status.sh
#   pyauto-status
#
# Override the repo root (e.g. for testing) via PYAUTO_STATUS_ROOT.
#
# Note: this shell function shares its name with the /pyauto-status slash
# command (PyAutoPrompt/skills/pyauto-status/) but lives in a different
# namespace. The slash command shows workflow registry status (planned /
# active / complete tasks); this function shows git sync state.

PYAUTO_STATUS_ROOT="${PYAUTO_STATUS_ROOT:-$HOME/Code/PyAutoLabs}"

pyauto-status() {
  local root="$PYAUTO_STATUS_ROOT"
  if [[ ! -d "$root" ]]; then
    echo "pyauto-status: $root does not exist" >&2
    return 1
  fi

  # Discover repos. `.git` is a directory in normal checkouts and a file in
  # linked worktrees, so accept both. mindepth/maxdepth 2 limits us to the
  # immediate children of $root.
  local repos=()
  while IFS= read -r dir; do
    repos+=("$dir")
  done < <(
    find "$root" -mindepth 2 -maxdepth 2 \
      \( -name .git -type d -o -name .git -type f \) \
      -printf '%h\n' 2>/dev/null | sort
  )

  if [[ ${#repos[@]} -eq 0 ]]; then
    echo "pyauto-status: no git repos found under $root"
    return 0
  fi

  # Parallel fetch. One background job per repo; sentinel files mark fetch
  # failures so the dashboard can flag stale rows with `!` instead of
  # silently returning misleading counts.
  local fetch_status_dir
  fetch_status_dir="$(mktemp -d)"
  trap 'rm -rf "$fetch_status_dir"' RETURN

  local repo
  for repo in "${repos[@]}"; do
    (
      if ! git -C "$repo" fetch --quiet origin 2>/dev/null; then
        touch "$fetch_status_dir/$(basename "$repo")"
      fi
    ) &
  done
  wait

  # Header.
  local fmt='%-32s %-30s %-36s %6s %5s %4s %4s  %s\n'
  printf "$fmt" REPO BRANCH UPSTREAM BEHIND AHEAD MOD UNTR FLAGS
  printf "$fmt" "--------------------------------" \
    "------------------------------" \
    "------------------------------------" \
    "------" "-----" "----" "----" "-----"

  # Per-repo row. Porcelain is cached so the dirty-files listing below can
  # reuse it without a second `git status` per repo.
  declare -A repo_porcelain
  local name branch upstream behind ahead mod untr flags counts porcelain
  for repo in "${repos[@]}"; do
    name="$(basename "$repo")"

    branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    [[ "$branch" == "HEAD" ]] && branch="(detached)"
    [[ -z "$branch" ]] && branch="?"

    upstream="$(git -C "$repo" rev-parse --abbrev-ref '@{u}' 2>/dev/null || true)"
    flags=""

    if [[ -z "$upstream" ]]; then
      upstream="NONE"
      behind="?"
      ahead="?"
      flags+="!"
    else
      counts="$(git -C "$repo" rev-list --left-right --count "$upstream"...HEAD 2>/dev/null || true)"
      if [[ -n "$counts" ]]; then
        behind="${counts%%[[:space:]]*}"
        ahead="${counts##*[[:space:]]}"
      else
        behind="?"
        ahead="?"
      fi
      [[ -e "$fetch_status_dir/$name" ]] && flags+="!"
    fi

    porcelain="$(git -C "$repo" status --porcelain 2>/dev/null || true)"
    repo_porcelain["$name"]="$porcelain"

    if [[ -z "$porcelain" ]]; then
      mod=0
      untr=0
    else
      untr="$(printf '%s\n' "$porcelain" | grep -c '^??' || true)"
      mod="$(printf '%s\n' "$porcelain" | grep -cv '^??' || true)"
    fi

    [[ "$behind" =~ ^[0-9]+$ ]] && (( behind > 0 )) && flags+="↓"
    [[ "$ahead"  =~ ^[0-9]+$ ]] && (( ahead  > 0 )) && flags+="↑"
    (( mod + untr > 0 )) && flags+="*"

    printf "$fmt" "$name" "$branch" "$upstream" "$behind" "$ahead" "$mod" "$untr" "$flags"
  done

  # Per-repo dirty-file listing. Only repos with non-empty porcelain are
  # shown — keeps the output empty when everything is clean. The `??` and
  # ` M` etc. prefixes from porcelain are preserved so users can tell
  # untracked from modified at a glance.
  local printed_header=false
  for repo in "${repos[@]}"; do
    name="$(basename "$repo")"
    porcelain="${repo_porcelain[$name]}"
    [[ -z "$porcelain" ]] && continue
    if [[ "$printed_header" == "false" ]]; then
      echo ""
      echo "Dirty files:"
      printed_header=true
    fi
    echo "  $name:"
    printf '%s\n' "$porcelain" | sed 's/^/    /'
  done
}
