# Contents block renders as one paragraph in generated notebooks

## The problem

Every workspace tutorial script that has a `__Contents__` block at the top of
its module docstring uses the same pattern:

```
__Contents__

**Model:** Compose the lens model fitted to the data.
**Plotters:** Overview of plotting tools used for visualization.
**Dataset Paths:** The `dataset_type` describes the type of data ...
**Grid:** Define the 2d grid of (y,x) coordinates ...
```

When PyAutoBuild's `generate.py` converts the script to a notebook, this
block becomes the contents of the first markdown cell. Markdown collapses
consecutive non-blank lines into a single paragraph, so on GitHub
(and in JupyterLab) every contents entry runs together as one continuous
sentence — exactly the opposite of what the index is for.

Confirmed in `ic50_workspace`'s `scripts/simulator.py` and verified by
inspecting `autolens_workspace/notebooks/imaging/simulator.ipynb`'s top
cell — the same one-big-paragraph rendering appears.

## The fix

Convert each `**Section:** description.` line to a Markdown bullet:

```
__Contents__

- **Model:** Compose the lens model fitted to the data.
- **Plotters:** Overview of plotting tools used for visualization.
- **Dataset Paths:** The `dataset_type` describes the type of data ...
- **Grid:** Define the 2d grid of (y,x) coordinates ...
```

Wrapped lines need a 2-space indent so they continue the same list item:

```
- **Real Data Preview:** Show one example real GDSC2 curve to anchor what
  the simulator is trying to reproduce.
```

The change is purely Markdown — no Python semantics change, no .py-script
behaviour change, no notebook-execution change. Diff is text-only inside
triple-quoted module docstrings.

A worked example shipped with `ic50_workspace` (commit
`4cde480` on `main`):
<https://github.com/Jammy2211/ic50_workspace/commit/4cde480>

## Scope

Apply this fix to every script in every workspace that has a
`__Contents__` (or equivalent inline index) block in its module docstring.
Confirmed-affected workspaces:

- `autolens_workspace`
- `autogalaxy_workspace`
- `autofit_workspace`
- `autolens_workspace_test`
- `autogalaxy_workspace_test`
- `autofit_workspace_test`
- `HowToLens`
- `HowToGalaxy`
- `HowToFit`
- `euclid_strong_lens_modeling_pipeline`
- `BSc_Galaxies_Project`

For each repo:

1. `grep -rln "__Contents__" scripts/` to find the affected files.
2. For each file, inside the top-level `"""..."""` module docstring,
   rewrite the `__Contents__` block so each `**Section:**` entry becomes
   a `- **Section:**` bullet, with continuation lines indented two
   spaces.
3. Spot-check by regenerating the notebook for one or two scripts
   (`PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py <project>`)
   and visually confirming the cell now renders as a list, not a
   paragraph.
4. Commit per repo with a single tidy commit (e.g.
   `docs: render __Contents__ blocks as Markdown lists`) and push;
   notebook regeneration is normally handled by the next `pre_build`
   run, so don't dirty up unrelated notebooks unless the workspace's
   own CI requires it.

## Other Markdown paragraph-collapse risks worth a quick scan

Same root cause may bite anywhere a docstring uses adjacent
non-blank lines that the author intended as separate items:

- `__Model__` blocks in workspace simulators that list bullet-like items
  with a single leading space (` - foo` vs `- foo`) — these usually do
  render as lists because Markdown allows up to 3 leading spaces, but
  worth eyeballing.
- Any `Steps`, `Notes`, `Outputs` block where each line starts with a
  bold label.

Don't go beyond `__Contents__` unless you find an additional concrete
broken example — this prompt's scope is the contents-block fix.
