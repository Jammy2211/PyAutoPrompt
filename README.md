# PyAutoPrompt

**The starting point of the PyAuto workflow.**

Every piece of work in the PyAuto ecosystem (PyAutoConf, PyAutoFit, PyAutoArray,
PyAutoGalaxy, PyAutoLens, and the `*_workspace*` repos) begins as a prompt in this
repo. A prompt is a markdown file describing a task in plain English, with enough
context that an AI agent (or a human) can pick it up and turn it into a tracked
GitHub issue, a feature branch, and a merged pull request.

This repo is the **single source of truth** for:

- pending prompts (work not yet started),
- in-flight tasks (`active.md`),
- completed work (`complete.md`),
- skills tightly coupled to the prompt workflow (under `skills/`).

It is the input side of the pipeline. The output side — code, tests, docs — lives
in the various PyAuto library and workspace repos.

---

## How a prompt flows through the workflow

```
  idea               ── you write it in ideas.md
    │
    ▼
  draft prompt       ── you write a markdown file under <category>/<name>.md
    │
    ▼
  /start_dev         ── reads the prompt, audits the code, drafts an issue,
    │                   creates the GitHub issue, registers the task in
    │                   active.md, moves the prompt to issued/
    ▼
  active.md entry    ── the task is now tracked across machines and sessions
    │
    ▼
  /start_library     ── creates a worktree, branch, opens dev environment
    or                  (or workspace variant — chosen automatically)
  /start_workspace
    │
    ▼
  development        ── code, tests, run smoke tests, commit
    │
    ▼
  /ship_library      ── runs tests, opens PR, waits for merge
    or
  /ship_workspace
    │
    ▼
  PR merged          ── post-merge cleanup deletes the worktree, removes the
    │                   active.md entry, appends a summary to complete.md
    ▼
  done
```

Two slash commands operate over the prompt registry without starting work:

- `/pyauto-status` — dashboard of `active.md`, `planned.md`, `complete.md`
- `/handoff` — park a task on this machine and resume on another (mobile, laptop, server)

---

## Repository layout

```
PyAutoPrompt/
├── README.md                ← this file
├── .gitignore
│
├── active.md                ← tasks currently in progress (one ## section per task)
├── complete.md              ← finished tasks (most recent first)
├── ideas.md                 ← raw incubating ideas, no structure required
├── planned.md               ← issued tasks blocked from starting (created on demand)
├── priority.md              ← hand-curated priority hints
├── queue.md                 ← processing queue for /register_and_iterate
│
├── autoarray/               ← prompts targeting PyAutoArray
├── autofit/                 ← prompts targeting PyAutoFit
├── autogalaxy/              ← prompts targeting PyAutoGalaxy
├── autolens/                ← prompts targeting PyAutoLens
├── autolens_workspace_developer/   ← prompts targeting the dev workspace
│
├── autobuild/                   ← prompts targeting build/release infrastructure (PyAutoBuild)
├── workspaces/              ← prompts targeting any *_workspace repo
│
├── cluster/                 ← cluster-lensing prompt series (numbered)
├── weak/                    ← weak-lensing prompt series (numbered)
│
├── issued/                  ← prompts that have been routed via /start_dev
│   └── autolens_workspace_developer/   ← per-target subdirs preserved
│
├── z_vault/                 ← deferred prompts (z_ prefix sorts last in listings)
│
├── autoprompt/              ← prompts about THIS repo's own infrastructure
│
├── scripts/
│   ├── status.sh            ← prompt inventory helper
│   └── prompt_sync.sh       ← commit/push helpers sourced by skills
│
└── skills/                  ← Claude Code skills tightly coupled to the prompt registry
    ├── start_dev/
    ├── start_library/
    ├── start_workspace/
    ├── ship_library/
    ├── ship_workspace/
    ├── pyauto-status/
    ├── register_and_iterate/
    ├── handoff/
    ├── worktree_status/
    ├── create_issue/
    └── plan_branches/
```

The `skills/` here hold **only the skills that read or write `active.md` / prompt
files**. General PyAuto tooling (release prep, dependency audits, smoke tests,
lint sweeps) lives in `admin_jammy/skills/`.

`scripts/prompt_sync.sh` is sourced by skills that mutate registry files
(`active.md`, `complete.md`, etc.) to commit and push back to origin. It
replaces the previous `admin_jammy/software/admin_sync.sh` which operated on
`admin_jammy/prompt/`.

---

## Conventions

### Naming

- Prompt filenames are lowercase `kebab_or_snake_case.md`.
- Numbered series use a leading number: `0_docs.md`, `1_simulator.md`. Skipping a
  number (e.g. `weak/2_*.md` not present) is fine — it usually means a step was
  consolidated or deferred.
- Category dirs match the target repo name (lowercased, no `Py` prefix):
  `autoarray/`, `autofit/`, `autogalaxy/`, `autolens/`. Workspace prompts go
  under `workspaces/` regardless of which workspace.

### Prompt file format

Free-form markdown. Strong conventions:

- Reference repos and files with `@RepoName/path/to/file.py` (e.g.
  `@PyAutoFit/autofit/non_linear/search.py`). `/start_dev` parses these to
  identify the primary target repo.
- One prompt = one task = one PR (ideally). If a prompt outlines several
  loosely-related changes, split before issuing.
- No frontmatter required. Title in the first line is helpful but optional.

### `active.md` schema

Each task is an H2 section:

```markdown
## <task-name-kebab-case>
- issue: https://github.com/<owner>/<repo>/issues/<n>
- session: claude --resume <session-id>           # optional
- status: <library-dev | workspace-dev | ready-to-ship | …>
- location: <cli-in-progress | ready-for-mobile | …>   # optional, used by /handoff
- worktree: ~/Code/PyAutoLabs-wt/<task-name>
- repos:
  - <RepoName>: feature/<branch-name>
- summary: |
    Free-form summary of progress and next steps.
```

### `complete.md` schema

```markdown
## <task-name>
- issue: https://github.com/<owner>/<repo>/issues/<n>
- completed: YYYY-MM-DD
- library-pr: <url> [, <url>]
- workspace-pr: <url> [, <url>]
- notes: |
    Long-form description of what landed, gotchas, follow-ups.
```

---

## Tracking and inspection

### Quick inventory

```bash
bash scripts/status.sh
```

Prints counts per category, lists the active and recently-completed tasks, and
flags anything in `z_vault/` that's been sitting for a while.

### From inside Claude Code

- `/pyauto-status` — dashboard of registry state (active, planned, recent complete)
- `/start_dev <category>/<name>.md` — read a prompt and route it
- `/handoff park` / `/handoff resume` — cross-machine task transitions
- `/worktree_status` — cross-references registry with task worktrees

---

## How this repo integrates with the rest

The PyAuto workflow has three repos with distinct roles:

| Repo | Purpose |
|------|---------|
| **PyAutoPrompt** (this repo) | Prompts, registry, prompt-coupled skills. The starting point. |
| **admin_jammy** | Personal admin notes (`euclid.md`, `papers.md`, `grants.md`, …) and general PyAuto tooling (`software/worktree.sh`, `software/admin_sync.sh`, generic skills like `audit_docs`, `dep_audit`, `repo_cleanup`). |
| **`PyAuto*` libraries and `*_workspace*` repos** | Where the actual code work happens. Each task gets a feature branch + worktree under `~/Code/PyAutoLabs-wt/<task-name>/`. |

Helper scripts that this repo's skills source:

- `admin_jammy/software/worktree.sh` — task worktree management (create, remove, conflict check).
- `admin_jammy/software/admin_sync.sh` — admin_jammy/PyAutoPrompt sync helpers.

These intentionally live in `admin_jammy/software/` because they're general
multi-repo tooling, not prompt-specific. The skills that need them source by
absolute path.

---

## Bootstrap on a new machine

```bash
cd ~/Code/PyAutoLabs
git clone git@github.com:PyAutoLabs/PyAutoPrompt.git
git clone git@github.com:Jammy2211/admin_jammy.git    # if not already present
bash admin_jammy/skills/install.sh                     # symlinks skills + commands
```

`install.sh` auto-discovers skills from both `admin_jammy/skills/` and
`PyAutoPrompt/skills/` and creates symlinks under `~/.claude/skills/` and
`~/.claude/commands/`. Re-run any time after pulling new skills from either repo.
