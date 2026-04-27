  ---
  I need you to investigate and fix a bug in `mge_model_from` in PyAutoGalaxy. The
  helper advertises a `gaussian_per_basis` parameter but does not produce the
  behaviour users expect when `gaussian_per_basis > 1`.

  ## Location

  File: `/home/jammy/Code/PyAutoLabs/PyAutoGalaxy/autogalaxy/analysis/model_util.py`
  Function: `mge_model_from(...)` (defined near line 7)

  Current signature:

      def mge_model_from(
          mask_radius: float,
          total_gaussians: int = 30,
          gaussian_per_basis: int = 1,
          centre_prior_is_uniform: bool = True,
          centre: Tuple[float, float] = (0.0, 0.0),
          centre_fixed: Optional[Tuple[float, float]] = None,
          centre_sigma: float = 0.3,
          ell_comps_prior_is_uniform: bool = False,
          ell_comps_uniform_width: float = 0.2,
          ell_comps_sigma: float = 0.3,
          use_spherical: bool = False,
      ) -> af.Collection:

  Re-exported in PyAutoLens as `al.model_util.mge_model_from`.

  ## The bug

  Inside the function, `centre_0`/`centre_1` and `ell_comps_0`/`ell_comps_1` are
  constructed ONCE, outside the `for j in range(gaussian_per_basis)` loop. Then
  every Gaussian across every basis is assigned the same prior *instance*, which
  in PyAutoFit ties the parameters across all bases.

  Net effect: calling `mge_model_from(mask_radius=1.0, total_gaussians=30,
  gaussian_per_basis=2)` produces a Basis with 60 Gaussians but only 4 free
  parameters (2 centre + 2 ell_comps), regardless of how many bases are
  requested. Adding bases does nothing observable — the extra Gaussians just add
  redundant, perfectly-tied components. This is almost certainly a refactor
  regression; `gaussian_per_basis` was originally meant to add *model flexibility*
  (e.g. to represent twisting isophotes with ellipticity varying by scale).

  First thing you should do: empirically verify the bug. Call the helper with
  `gaussian_per_basis=1` and `gaussian_per_basis=2` and compare `.prior_count`
  and the actual prior structure. Confirm 4 free params in both cases (if it's
  really broken) before doing anything else. Do not fix based only on my reading
  of the code.

  ## Intended behaviour (from the user)

  - Each basis must ALWAYS have INDEPENDENT ell_comps. This is not optional —
    if you want a single-orientation MGE you just use `gaussian_per_basis=1`.

  - Centres should be SHARED across bases by default (the common case is a
    galaxy with one luminosity centre but complex isophotal shape).

  - Add a boolean flag, default False, to optionally give each basis its own
    centre. Suggested name: `centre_per_basis: bool = False`. When True, each
    basis gets independently-drawn centre priors; the shared `centre_fixed`
    override still applies if set.

  Expected free-parameter counts after the fix, for an elliptical model
  (use_spherical=False):

      gaussian_per_basis=1                         → 2 centre + 2 ell_comps = 4
      gaussian_per_basis=2 (default shared centre) → 2 centre + 4 ell_comps = 6
      gaussian_per_basis=2, centre_per_basis=True  → 4 centre + 4 ell_comps = 8
      gaussian_per_basis=K (shared centre)         → 2 + 2K
      gaussian_per_basis=K, centre_per_basis=True  → 2K + 2K = 4K

  Spherical case (use_spherical=True): no ell_comps, only centres.
  Shared centre: 2. Per-basis centre: 2K.

  ## Implementation guidance

  The fix is to move prior construction INSIDE the `for j in range(gaussian_per_basis)`
  loop for anything that should be per-basis. Move ell_comps prior construction
  inside the loop unconditionally. Move centre prior construction inside the loop
  only when `centre_per_basis=True`. Leave `centre_fixed` handling alone — if
  centres are fixed they are fixed for everyone.

  Do NOT change:
  - The `PYAUTO_WORKSPACE_SMALL_DATASETS` branch that caps params at 2 & 1.
  - The `centre_fixed` semantics.
  - The default of `gaussian_per_basis=1`.
  - The existing kwarg names for centre/ell_comps priors.

  Keep the change backward compatible: any existing call with
  `gaussian_per_basis=1` must produce identical output. Any existing call with
  `gaussian_per_basis>1` currently produces 4 free params; after the fix it will
  produce more. That is the intended behaviour change — flag it clearly in the
  docstring and in any commit message.

  Update the docstring to explain the per-basis coupling semantics and document
  `centre_per_basis`. The current docstring is misleading on this point.

  ## Verification

  1. Unit test or ad-hoc script that constructs the helper with:
     - `gaussian_per_basis=1` (elliptical + spherical)
     - `gaussian_per_basis=2` default (should be 6 free params, 8 spherical=4)
     - `gaussian_per_basis=2, centre_per_basis=True` (should be 8 free params)
     - `gaussian_per_basis=3` to confirm the formula generalises
     - `centre_fixed=(0.0, 0.0)` — centres should be fixed, not free
     Assert `.prior_count` matches the expected values above.

  2. Grep for callers of `mge_model_from` in these repos and ensure none of them
     rely on the old (buggy) behaviour of tying ell_comps across bases:

         /home/jammy/Code/PyAutoLabs/PyAutoGalaxy
         /home/jammy/Code/PyAutoLabs/PyAutoLens
         /home/jammy/Code/PyAutoLabs/autolens_workspace
         /home/jammy/Code/PyAutoLabs/autogalaxy_workspace
         /home/jammy/Code/PyAutoLabs/autolens_workspace_test
         /home/jammy/Code/PyAutoLabs/autolens_workspace_developer

     There are ~30+ call sites across workspace scripts and notebooks (e.g.
     `autolens_workspace/scripts/multi/features/slam/{independent,simultaneous}.py`,
     `imaging/start_here.py`, SLaM pipelines). Most call with `gaussian_per_basis=1`
     or `gaussian_per_basis=2`. The ones using `=2` will now actually get the extra
     flexibility they were asking for; confirm none assumed 4-param coupling.

  3. Run the PyAutoGalaxy unit tests that touch `model_util`:

         cd /home/jammy/Code/PyAutoLabs/PyAutoGalaxy
         python -m pytest test_autogalaxy/analysis/test_model_util.py -x

     If no tests exist for this function, add a small one covering all four
     cases from step 1.

  ## Follow-up (DO report, DO NOT fix in this session)

  After the library fix is verified working, note in your final summary that the
  following workspace SLaM scripts currently call `mge_model_from(total_gaussians=20,
  gaussian_per_basis=2)` for the light pipeline and may now want to be updated to
  `total_gaussians=30, gaussian_per_basis=2` (a user preference, not a bug):

  - `autolens_workspace/scripts/multi/features/slam/independent.py` (second `lens_bulge` before `light_lp`)
  - `autolens_workspace/scripts/multi/features/slam/simultaneous.py` (second `lens_bulge` before `light_lp`)

  These updates should be a separate task.

  ## Planning + branch survey

  The repo is `PyAutoGalaxy` and the change modifies library source, so follow
  the project's CLAUDE.md planning requirement: produce a plan, get approval,
  run `/plan_branches`, then implement. Report the git repo root and current
  branch before editing.

  ---
  Let me know if you want me to tighten it, add anything, or broaden the investigation scope.