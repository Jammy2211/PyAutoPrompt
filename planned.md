## fix-interferometer-sparse-operator-irregular-meshes
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/314
- planned: 2026-05-15
- classification: library (PyAutoArray) — math rewrite, expert-only
- suggested-branch: feature/fix-interferometer-sparse-curvature
- summary: |
    The `InterferometerSparseOperator.curvature_matrix_via_sparse_operator_from`
    path in `autoarray/inversion/inversion/interferometer/inversion_interferometer_util.py`
    only handles Pmax=1 (one source pixel per image pixel, weight 1) — i.e. the
    Rectangular case. For Delaunay (Pmax=3 via barycentric interpolation, weights
    summing to 1) the returned curvature matrix disagrees with the mapping path
    by ~34% Frobenius norm and the regularized matrix loses positive-definiteness
    (smallest eigenvalue -4.82 vs +0.66 for mapping), triggering a Cholesky
    LinAlgError downstream. PR #315 (just shipped) added a defensive
    NotImplementedError guard for Delaunay; this task replaces the guard with a
    real fix.

    Scope:
      - Rewrite `curvature_matrix_via_sparse_operator_from` to handle Pmax > 1
        correctly. Likely mirror the imaging sparse path
        (`inversion_imaging_util.curvature_matrix_diag_from`) which already
        handles Delaunay+barycentric correctly via the flat COO triplet format
        from `mapper.sparse_triplets_curvature`.
      - Audit RectangularAdaptDensity (and any other mesh that emits Pmax > 1
        weights). The canonical script
        `autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/rectangular_sparse.py`
        notes a ~0.4 % discrepancy between sparse-vs-mapping which is currently
        accepted as "numerical reformulation". If that discrepancy stems from the
        same Pmax handling issue, the fix should converge it to <1e-4.
      - Add the missing regression test
        `test__curvature_matrix__interferometer_sparse_operator__delaunay__identical_to_mapping`
        mirroring the existing imaging sparse interpolation test
        (`test_abstract.py:160`) so sparse-vs-mapping parity is asserted for
        every Pmax > 1 mesh going forward.
      - Once the math fix lands and the parity test passes, **remove the guard**
        at `InversionInterferometerSparse.curvature_matrix_diag` and convert
        `test__apply_sparse_operator__delaunay_mapper__raises_not_implemented`
        into the parity assertion.

    Diagnostic numbers (from Phase 1 reproducer, see issue #314 for the script):
      - ||sparse - mapping||_F = 20.77 on Delaunay(pixels=578); ||mapping||_F = 60.32
      - Smallest eigenvalue of curvature_reg_matrix_reduced: -4.82 sparse vs +0.66 mapping
      - Top-10 disagreeing rows: [477, 500, 133, 486, 503, 134, 322, 512, 348, 381]
        — all on active vertices, not in the zeroed-edge block
      - Disagreement is mostly off-diagonal (max diag 4.34, max off-diag 1.80,
        row L1 totals reach 60) — source-pixel cross-coupling is structurally
        wrong, not a numerical-precision issue.
      - Independent of `zeroed_pixels` and independent of `use_jax`.

    Why deferred: the math investigation requires intimate knowledge of the
    W-tilde formalism and the sparse-operator FFT-based curvature
    accumulation. Aris or another PyAutoArray inversion maintainer is the
    appropriate owner.

    Downstream effect when fixed: the dev-workspace Delaunay profilers at
    `autolens_workspace_developer/jax_profiling/jit/{interferometer,datacube}/delaunay.py`
    can opt into `dataset.apply_sparse_operator(use_jax=True, show_progress=True)`
    for a substantial speedup (the precision-matrix precompute amortises the
    inversion's per-fit recomputation). Hannah Stacey's ALMA cube fit is the
    primary beneficiary.

## weak-visualization
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/496
- planned: 2026-05-06
- classification: library (then workspace follow-up)
- suggested-branch: feature/weak-visualization
- blocked-by: use-pathlib (using PyAutoLens via ~/Code/PyAutoLabs-wt/use-pathlib)
- affected-repos:
  - PyAutoLens
  - autolens_workspace
