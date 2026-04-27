Three workspace repos currently have `release` set as their GitHub default branch. `release` is supposed to be a downstream branch updated only by PyAutoBuild when it merges `main → release` during a release cut. Having `release` as the default means `gh pr create` without `--base main`, and the GitHub UI "Compare & pull request" button, both silently target `release`. PRs that land on `release` are orphaned from `main`'s history and get overwritten by the next `main → release` sync.

This already happened: autolens_workspace PRs #54, #55, #58, #59, #61 and autogalaxy_workspace PR #28 were merged to `release` over 2026-04-13 → 2026-04-14 without anyone noticing. All have since been replayed onto `main` via cherry-pick PRs (autolens_workspace #62, #63 and autogalaxy_workspace #29), but the underlying misconfiguration is still there and will keep causing drift until fixed.

## Reason

Currnetly, release is what links to the pypi release, such that if a user runs pip install autolens, the
release branch should be in sync.

What we should do instead is have workspaces use main, but have it so the two following things happen:

1) Workspaces have tagged or version github branches, so that when a PyAuto version is released users can pair it to a specific workspace.
2) All PyAutoFit / PyAutoGalaxy / PyAutoLens docs tell the user (e.g. during installation) to use a version number for the workspace clone or download, paired to their version.
3) If a user has an out of sync workspace and source code, they get an error they have to manually disable.

This will mean a user can run the main branches of the source repos and workspace repos without issue. 

## Action

On each of the three repos, change the default branch from `release` back to `main`:

- https://github.com/PyAutoLabs/autolens_workspace/settings/branches — set default to `main`
- https://github.com/PyAutoLabs/autogalaxy_workspace/settings/branches — set default to `main`
- https://github.com/PyAutoLabs/autofit_workspace/settings/branches — set default to `main`

(autolens_workspace_test is already correct — default is `main`.)

Then add solution to points 1), 2) and 3) above.

## Verification

After each change, confirm with:

```bash
gh repo view PyAutoLabs/<repo> --json defaultBranchRef --jq '.defaultBranchRef.name'
```

Expect `main` for all four workspace repos.

## Why not purely rely on the `/ship_workspace` skill fix?

The skill has already been patched to pass `--base main` explicitly on every `gh pr create` call (admin_jammy commit `8152d05`). That covers Claude-driven shipping. It does NOT cover:

- PRs opened manually in the GitHub UI (the "Compare & pull request" banner defaults to the repo's default branch).
- PRs from contributors who don't use the shipping skill.
- `git push -u origin <branch>`'s suggested PR URL (pre-fills the default base).

So changing the repo default is the root-cause fix; the skill patch is defence-in-depth.

So, also remove this fix once we are on main everywhere.