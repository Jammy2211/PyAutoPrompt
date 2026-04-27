# PyAutoFit library updates for HowToFit extraction (sub-3 of issue #38)

Sub-1 and sub-2 of the HowToFit extraction have shipped:
- Sub-1: PyAutoLabs/HowToFit standalone repo exists (PR #1 merged)
- Sub-2: `howtofit/` tree removed from autofit_workspace (PR #39 merged)

This sub-task updates **@PyAutoFit** — library docs and prose — so the tutorial references no longer
point into `autofit_workspace/howtofit/` or `pyautofit.readthedocs.io/howtofit/`, and instead point to
the new standalone `PyAutoLabs/HowToFit` GitHub repo.

## URL rewrite scheme

There is no `pyautofit.readthedocs.io/howtofit/...` anymore — those chapter pages will be deleted. For
every old readthedocs link and every `Jammy2211/autofit_workspace/.../howtofit/...` link, rewrite to
the HowToFit GitHub repo's notebooks:

- Old: `https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction.html`
  New: `https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/` (or landing page)
- Old: `https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html`
  New: `https://github.com/PyAutoLabs/HowToFit`
- Old: `https://github.com/Jammy2211/autofit_workspace/.../howtofit/chapter_graphical_models/tutorial_4_hierachical_models.ipynb`
  New: `https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models/tutorial_4_hierachical_models.ipynb`

## Scope

### 1. Delete the `docs/howtofit/` tree

Remove these four files:
- @PyAutoFit/docs/howtofit/howtofit.rst
- @PyAutoFit/docs/howtofit/chapter_1_introduction.rst
- @PyAutoFit/docs/howtofit/chapter_2_scientific_workflow.rst
- @PyAutoFit/docs/howtofit/chapter_3_graphical_models.rst

And the now-empty `docs/howtofit/` directory itself.

### 2. `docs/index.rst` toctree

Lines ~202-205 contain the Tutorials toctree entries for the HowToFit chapters. Remove those four
toctree lines. If the `:caption: Tutorials:` toctree becomes empty as a result, remove the whole
toctree block (including the caption header).

### 3. API doc cross-references

Seven API `.rst` files each contain a `- HowToFit: introduction chapter (detailed step-by-step
examples) <https://pyautofit.readthedocs.io/...>` line:

- @PyAutoFit/docs/api/model.rst
- @PyAutoFit/docs/api/plot.rst
- @PyAutoFit/docs/api/priors.rst
- @PyAutoFit/docs/api/analysis.rst
- @PyAutoFit/docs/api/database.rst
- @PyAutoFit/docs/api/searches.rst
- @PyAutoFit/docs/api/samples.rst

Rewrite each so the link points to `https://github.com/PyAutoLabs/HowToFit` (or the matching chapter
folder under `notebooks/` where it makes sense — e.g. the `priors` page can link to chapter 1 if that
covers prior composition; otherwise the repo root).

### 4. `docs/general/workspace.rst` — HowToFit section

The "HowToFit" section (lines ~16-24) still describes HowToFit as part of the workspace and links to
the readthedocs tutorials page. Rewrite it as a pointer-style section mirroring the sub-2 phrasing in
`autofit_workspace/README.rst`: HowToFit now lives in a standalone repo at
`https://github.com/PyAutoLabs/HowToFit`, clone/browse there for the chapters.

### 5. `docs/features/graphical.rst`

Contains two howtofit references (matching autofit_workspace/scripts/features/graphical_models.py that
was updated in sub-2). Rewrite to point at
`https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models/`.

### 6. `docs/overview/statistical_methods.rst:30`

Mirrors `autofit_workspace/scripts/overview/overview_3_statistical_methods.py` which was updated in
sub-2. Rewrite the URL from
`github.com/Jammy2211/autofit_workspace/blob/release/notebooks/howtofit/chapter_graphical_models/tutorial_4_hierachical_models.ipynb`
to
`github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models/tutorial_4_hierachical_models.ipynb`.

### 7. `docs/cookbooks/multiple_datasets.rst` and `docs/science_examples/astronomy.rst`

Each contains a `pyautofit.readthedocs.io/.../howtofit/...` URL. Rewrite to the new HowToFit repo
(chapter folder appropriate to the context).

### 8. `README.rst`

Three references:
- Top-of-README `HowToFit` header link (line ~37) → `https://github.com/PyAutoLabs/HowToFit`
- `autofit_workspace GitHub repository ... HowToFit Jupyter notebook lectures` sentence (line ~60) →
  drop the `/tree/main/notebooks/howtofit` sub-link, replace with a separate sentence pointing to the
  standalone HowToFit repo
- The `is provided on the HowToFit readthedocs page` link (line ~79) → point to the repo README instead

### 9. `paper/paper.md`

Three `HowToFit` references in the published JOSS paper prose. Rewrite URLs and prose to describe
HowToFit as a standalone tutorial repository. Keep the citation language (this is a peer-reviewed
paper — don't rewrite the science, only the URLs and the "part of the workspace" framing).

## Out of scope

- Anything under `build/lib/howtofit/` — that's generated Sphinx output and will self-heal on next
  docs build.
- Registering the `howtofit` build target in PyAutoBuild — tracked as a separate follow-up
  (`pyautobuild-register`) on issue #38.
- Anything in autofit_workspace (done in sub-2) or in the HowToFit repo itself (done in sub-1).

## Verification

- `grep -rn "howtofit" PyAutoFit --include="*.rst" --include="*.md" --exclude-dir=build` should show
  zero matches after the change, aside from the sub-3 commit message itself.
- Sphinx build (`cd docs && make html`) must succeed without broken-reference warnings for the removed
  `howtofit/` toctree entries.
