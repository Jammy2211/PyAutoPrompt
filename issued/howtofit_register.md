Register `howtofit` as a first-class build target in @PyAutoBuild, mirroring the pattern
that was established for `howtolens` and `howtogalaxy` (PyAutoBuild PR #53 —
`register-howto-repos`).

This is sub-4 of the HowToFit extraction umbrella:
https://github.com/PyAutoLabs/autofit_workspace/issues/38

Context:
- HowToFit now lives as a standalone repository at
  https://github.com/PyAutoLabs/HowToFit
- Sub-1 (scaffold), sub-2 (remove `howtofit/` from autofit_workspace), and sub-3
  (update PyAutoFit library URLs + `docs/howtofit/`) all merged on 2026-04-22.
- The final follow-up is to make PyAutoBuild build, release, and publish
  HowToFit the same way it already does for HowToLens and HowToGalaxy.

## Files to modify in @PyAutoBuild

Mirror the exact pattern added for HowToLens/HowToGalaxy in each file:

1. **`pre_build.sh`** — add a `run_workspace "HowToFit" "howtofit"` line after
   the existing HowToLens line (around line 65).

2. **`.github/workflows/release.yml`** — five stages, roughly eight edit
   locations:
   - Checkout block for HowToFit (`repository: PyAutoLabs/HowToFit`,
     `path: howtofit`)
   - `script_matrix.py` argument list — add `howtofit`
   - Test matrix entry (`- name: howtofit`,
     `workspace_repository: PyAutoLabs/HowToFit`)
   - Workspace-repository resolver branches (two of them) — add
     `elif [ "$workspace" = "howtofit" ]` branch each
   - Release matrix entry (`- repository: PyAutoLabs/HowToFit`, `name: howtofit`)

3. **`autobuild/bump_colab_urls.sh`** — extend the regex at line 30 and the
   comment at line 12 to include `HowToFit`.

4. **`autobuild/config/copy_files.yaml`** — add an empty `howtofit:` entry after
   `howtolens:` (around line 11).

5. **`autobuild/config/no_run.yaml`** — add `howtofit: []` after
   `howtogalaxy: []` (around line 52).

## Smoke check

After making the edits, run:

```
python autobuild/script_matrix.py autofit autogalaxy autolens autofit_test autolens_test howtogalaxy howtolens howtofit
```

and confirm it exits zero and emits the expected per-workspace matrix rows.

## Acceptance

- PyAutoBuild PR opened with the five files updated.
- `script_matrix.py` sanity check passes locally with `howtofit` in the list.
- Umbrella issue autofit_workspace#38 can be closed once this PR merges.
