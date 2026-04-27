The MGE gradient testing script (@autolens_workspace_developer/jax_profiling/imaging/mge_gradients.py)
shows that `jax.value_and_grad` returns NaN gradients starting at the NNLS reconstruction step
(`jaxnnls.solve_nnls_primal`). All upstream steps (ray-trace, mapping matrix, blurred mapping matrix,
data vector, curvature matrix) produce valid finite gradients. The NNLS step poisons everything
downstream.

The jaxnnls library (v1.0.1, installed at site-packages/jaxnnls/) already has a `@jax.custom_vjp`
on `solve_nnls_primal` that uses relaxed KKT implicit differentiation. The forward pass solves the
exact NNLS, then re-solves with relaxed complementarity (`s * z >= target_kappa` instead of
`s * z = 0`) to regularise the backward pass. The backward pass factorises the relaxed KKT system
and solves for gradients via `diff_nnls()` in `diff_qp.py`.

The NaN gradients likely come from ill-conditioning in the KKT factorisation during the backward
pass. Possible causes:

1. The curvature matrix Q passed from @PyAutoArray is poorly conditioned (40x40 for 40 MGE Gaussians)
2. The relaxation parameter `target_kappa=1e-3` (default) is too small, leaving the relaxed solution
   too close to the boundary where `P_inv_vec = z / s` explodes
3. The relaxed solver (`solve_relaxed_nnls` in `pdip_relaxed.py`) does not converge within its
   50-iteration limit, leaving the relaxed variables in a bad state
4. Cholesky factorisation of `Q + diag(P_inv_vec)` fails silently (JAX doesn't raise on NaN)

The call chain is:
  @PyAutoArray/autoarray/inversion/inversion/inversion_util.py:reconstruction_positive_only_from
    -> jaxnnls.solve_nnls_primal(curvature_reg_matrix, data_vector)
    -> diff_qp.py: solve_nnls_primal (custom_vjp)
    -> pdip.py: solve_nnls (forward)
    -> pdip_relaxed.py: solve_relaxed_nnls (relaxation for backward)
    -> diff_qp.py: diff_nnls (backward, KKT factorisation)

Investigate and fix the NaN gradients. Start by:

1. Adding diagnostics to the gradient script to log the condition number of Q, whether the relaxed
   solver converged, and the magnitude of `P_inv_vec` at the backward pass
2. Testing whether increasing `target_kappa` (e.g. 1e-1, 1e-2) fixes the NaN issue
3. If the fix is in jaxnnls itself, the source is at site-packages/jaxnnls/ — editable in-place
   for testing, then we can fork/PR upstream or vendor the fix into @PyAutoArray
4. If the fix requires changes to how @PyAutoArray calls jaxnnls (e.g. preconditioning Q,
   passing a different target_kappa), make those changes in inversion_util.py
