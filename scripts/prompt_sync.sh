#!/usr/bin/env bash
# prompt_sync.sh — keep PyAutoPrompt in sync with origin during task lifecycle.
#
# Sourced by skills that mutate PyAutoPrompt registry files (active.md,
# complete.md, planned.md, queue.md, ...) so we never accumulate local-only
# changes that drift from origin.
#
#   source PyAutoPrompt/scripts/prompt_sync.sh
#   prompt_sync_new_prompts                    # scan + commit + push new prompts
#   prompt_sync_push "prompt: <subject>"       # commit + push current state
#
# Both functions are no-ops when there is nothing to do, and safe to call
# multiple times in the same session.
#
# Replaces the previous admin_jammy/software/admin_sync.sh which operated on
# admin_jammy/prompt/. PyAutoPrompt is now the home of prompts and registry.

PROMPT_REPO="${PROMPT_REPO:-$HOME/Code/PyAutoLabs/PyAutoPrompt}"

# Commit and push any new untracked .md files at the repo root or under
# category dirs as one "sync new task ideas" commit. Each new file is listed
# individually in the commit body so the history shows which prompts arrived.
# Excludes issued/ (handled by prompt_sync_push when skills move files there)
# and tmp/ (scratch).
prompt_sync_new_prompts() {
  local untracked
  untracked=$(git -C "$PROMPT_REPO" ls-files --others --exclude-standard \
    -- '*.md' '*/*.md' \
    | grep -v '^issued/' \
    | grep -v '^tmp/' \
    || true)
  [ -z "$untracked" ] && return 0

  echo "Found new prompt files to sync:"
  echo "$untracked" | sed 's/^/  - /'

  local body
  body=$(echo "$untracked" | sed 's/^/- /')

  ( cd "$PROMPT_REPO" && git add -- $untracked && \
    git commit -m "$(printf 'prompt: sync new task ideas\n\n%s\n' "$body")" && \
    git push origin main )
}

# Stage any modifications under the repo, commit with the given subject,
# and push. Used by skills at task milestones (issue filed, repos
# registered, task shipped, etc.). No-op if nothing is staged.
prompt_sync_push() {
  local subject="${1:-prompt: sync PyAutoPrompt}"
  ( cd "$PROMPT_REPO" && \
    git add -A && \
    if git diff --cached --quiet; then return 0; fi && \
    git commit -m "$subject" && \
    git push origin main )
}
