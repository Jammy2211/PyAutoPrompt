Extract the HowToGalaxy Jupyter notebook lectures from `@autogalaxy_workspace` into their own standalone GitHub repo, following the exact same pattern we just used for HowToLens.

Starting state mirrors HowToLens pre-extraction:

- `autogalaxy_workspace/scripts/howtogalaxy/` and `autogalaxy_workspace/notebooks/howtogalaxy/` hold the content.
- Structure: `chapter_1_introduction`, `chapter_2_modeling`, `chapter_3_search_chaining`, `chapter_4_pixelizations`, `chapter_optional`, `simulators`.
- `Jammy2211/HowToGalaxy` exists but is empty (LICENSE only). It needs to be transferred to `PyAutoLabs/HowToGalaxy` (same "forgot to put it in the org" situation as HowToLens).
- `@PyAutoGalaxy` docs reference HowToGalaxy under Colab URLs pointing at `autogalaxy_workspace/.../howtogalaxy/...` — these need to migrate to the new repo.

Do this as three sub-tasks, with three separate GitHub issues:

1. **Bootstrap `PyAutoLabs/HowToGalaxy`** (current scope).
   - Transfer `Jammy2211/HowToGalaxy` → `PyAutoLabs/HowToGalaxy`.
   - Populate it from the existing `autogalaxy_workspace/howtogalaxy/` content, following the same folder layout as every other workspace (config / dataset / notebooks / output / scripts).
   - Copy CI workflows, pre-commit configs, and release automation from HowToLens as a template.
   - Tag `2026.4.13.6` — same tag as HowToLens, matches the current `autogalaxy_workspace` version, lets Colab URLs reference a stable snapshot.
   - Smoke-test the chapter-1 tutorials to confirm they still run after the move.

2. **Workspace cleanup** (autogalaxy_workspace).
   - Remove `scripts/howtogalaxy/` and `notebooks/howtogalaxy/`.
   - Rewrite internal references (READMEs, modeling.py prose mentioning `howtogalaxy`, build configs like `env_vars.yaml` and `no_run.yaml`).
   - Add a pointer to `PyAutoLabs/HowToGalaxy` wherever the old folder was referenced.

3. **PyAutoGalaxy docs update**.
   - Migrate every HowToGalaxy Colab URL from `autogalaxy_workspace/.../notebooks/howtogalaxy/...` to `PyAutoLabs/HowToGalaxy/blob/2026.4.13.6/notebooks/...`.
   - Rewrite README, `docs/index.rst`, `docs/general/workspace.rst`, `docs/overview/`, per-chapter `docs/howtogalaxy/` pages, and `paper/paper.md` prose to frame HowToGalaxy as a standalone repo alongside the workspace (not a sub-folder inside it).

Reference: the HowToLens trio is complete and shipped as:

- HowToLens bootstrap: https://github.com/PyAutoLabs/HowToLens/pull/1
- autolens_workspace cleanup: https://github.com/PyAutoLabs/autolens_workspace/pull/80
- PyAutoLens docs update: https://github.com/PyAutoLabs/PyAutoLens/pull/468

HowToFit is the third and final extraction, but it is out of scope for this task — handle it in its own follow-up.
