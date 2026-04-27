# PyAutoLens docs: update after HowToLens extraction

## Context

HowToLens has been extracted into its own standalone repository at `PyAutoLabs/HowToLens`. The bootstrap PR (sub-task 1) and the workspace cleanup (sub-task 2, autolens_workspace PR #80) are both merged. The notebooks that previously lived at `autolens_workspace/notebooks/howtolens/` no longer exist there, and the Colab URLs in the PyAutoLens readthedocs that point at them are now 404s. HowToLens has been tagged `2026.4.13.6` to match the autolens_workspace version at the time of extraction.

This is sub-task 3 of the HowToLens extraction project. All changes land in `@PyAutoLens` (the library repo). No workspace changes.

## What to do

### URL migration (broken links — must fix)

The 5 chapter RST pages at `@PyAutoLens/docs/howtolens/chapter_*.rst` are pure RST listings with Colab URLs that need rewriting.

- Old pattern: `https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.4.13.6/notebooks/howtolens/<chapter>/<tutorial>.ipynb`
- New pattern: `https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.4.13.6/notebooks/<chapter>/<tutorial>.ipynb`

Note the path prefix drops `/howtolens/` — the new repo has chapters directly under `notebooks/`.

Files:
- `docs/howtolens/chapter_1_introduction.rst` — 9 URLs
- `docs/howtolens/chapter_2_lens_modeling.rst`
- `docs/howtolens/chapter_3_search_chaining.rst`
- `docs/howtolens/chapter_4_pixelizations.rst`
- `docs/howtolens/chapter_optional.rst` — 2 URLs

### Prose updates (stale but not broken)

Reframe HowToLens from "shipped inside autolens_workspace" to "its own standalone repository":

- **`docs/howtolens/howtolens.rst`** — landing page opens "The best way to learn PyAutoLens is by going through the HowToLens lecture series on the autolens workspace." Rewrite to point at `PyAutoLabs/HowToLens`. Remove or rewrite the Jupyter Notebooks section's reference to `scripts` folder of the workspace (HowToLens ships both notebooks and scripts in its own repo now).
- **`docs/general/workspace.rst:57-65`** — "HowToLens" section describes the lectures as part of the workspace. Rewrite as a pointer to the standalone repo. Fix the broken `file:///Users/Jammy/.../docs/_build/tutorials/howtolens.html` link.
- **`docs/overview/overview_2_new_user_guide.rst:81`** — "checkout the HowToLens package now" link points at `Jammy2211/autolens_workspace/tree/release/notebooks/howtolens`. Replace with `PyAutoLabs/HowToLens`.
- **`README.rst:64`** — "The autolens_workspace GitHub repository: example scripts and the HowToLens Jupyter notebook lectures." Decouple — HowToLens is no longer part of the workspace. Add an explicit link to `PyAutoLabs/HowToLens` in the bulleted list.
- **`README.rst:76-84`** — HowToLens section's readthedocs link is fine. Consider adding a GitHub link to `PyAutoLabs/HowToLens` as a secondary entry point.
- **`paper/paper.md:88-89`** — "[autolens workspace] ... includes example scripts, lens datasets and the HowToLens lectures in Jupyter notebook format" — decouple the HowToLens mention; keep the workspace mention for the other items.
- **`paper/paper.md:177-185`** ("Workspace and HowToLens Tutorials" section) — reword to present the two as separate offerings; keep the Colab and readthedocs links.

### What NOT to change

- `docs/index.rst` toctree for howtolens — the local `.rst` files still exist, so the toctree stays valid.
- `docs/conf.py` — no howtolens-specific config.
- The `docs/howtolens/chapter_*.rst` files themselves — keep as readthedocs TOC pages; only swap URLs.
- Prose references to "HowToLens chapter N" in non-howtolens docs — those chapters still exist in the new repo and the references remain semantically correct.

### Branch / versioning decision

Colab URLs pin to `2026.4.13.6` (HowToLens was tagged to match the workspace version at the time of extraction). Future tags get created by `/pre_build` once HowToLens is registered in PyAutoBuild (follow-up already tracked in `complete.md`).

## Affected repos

- `@PyAutoLens` (primary)

## Deliverable

One PR on `Jammy2211/PyAutoLens` titled `docs: update HowToLens references after extraction to standalone repo`. After merge, all readthedocs HowToLens tutorial links resolve to the new repo, and prose consistently frames HowToLens as a standalone project rather than bundled with the workspace.
