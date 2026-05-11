Make installing `af.NSS` a single safe command for a science user
installing PyAutoFit fresh — eliminate the multi-hour multi-step
install saga that the profiling project survived in session-of-2026-05-11.

This is **Phase 4** of `z_features/nss_first_class_sampler.md`.
**Depends on Phase 1** (so we know what API surface to vendor or
pin against), and informs Phase 5 (workspace tutorials can only
recommend `af.NSS` to users if installation isn't a 3-hour adventure).

__Why this matters — the install pain from session-of-2026-05-11__

`z_projects/profiling/FINDINGS_v3.md` "Methodology saga" records the
following failure modes during a clean install of `nss` into a fresh
HPC venv:

1. `pip install git+https://github.com/yallup/nss.git` pulled
   **mainline** `blackjax==1.3` from PyPI. nss imports
   `blackjax.ns.adaptive.init`, which does NOT exist in mainline.
   The fork at `handley-lab/blackjax` does — but pip doesn't know to
   prefer it.

2. `pip install --force-reinstall --no-deps git+https://github.com/handley-lab/blackjax.git`
   force-replaced the mainline blackjax. But the fork has its own
   runtime imports (`fastprogress`, `optax`, `absl-py`, `chex`,
   `jaxopt<=0.8.3`, `ott-jax`, `lineax`, `equinox`, `jaxtyping`,
   `wadler-lindig`), each of which had to be added one at a time
   with `--no-deps` after walking the chain of `ModuleNotFoundError`s.

3. `pip install --no-deps fastprogress optax` triggered installation
   of `fastprogress 1.1.5` which transitively pulls
   `python-fasthtml` and all of `starlette`, `uvicorn`, `httpx`,
   `websockets`, ... none of which are needed at runtime.
   Workaround: `pip install --no-deps "fastprogress<1.1"` — older
   versions don't have the `python-fasthtml` dep.

4. Default `pip install pocomc` (a separate sampler, but illustrative
   of the same class of pain) pulled `torch` with CUDA-13 nvidia-*
   shared libs (~3 GB), which clobbered JAX's CUDA-12 `libcudnn.so`
   files at the OS dynamic-linker level. Every subsequent JAX GPU
   call failed with `DNN library initialization failed`.
   Workaround: pre-install CPU-only torch from the PyTorch CPU index
   before `pip install pocomc`.

5. `pip install numpyro` default-pulled `numpyro 0.21.0` which uses
   `jax.api_util.debug_info` — a symbol that JAX 0.4.38 doesn't
   export. Workaround: `pip install --no-deps "numpyro<0.16"`.

None of these are problems we can ask a science user installing
PyAutoFit to navigate. **One safe install command is the deliverable
of this prompt.**

__What to investigate__

There are four candidate approaches. The Phase 4 deliverable evaluates
each and ships one (the recommendation below is option A; if option C
becomes feasible during the work, switch to that and document).

### Option A — Vendor a minimal nss into PyAutoFit (recommended)

Copy the parts of `yallup/nss` we actually use into
`@PyAutoFit/autofit/non_linear/search/nest/nss/_vendor/`:

- `nss/ns.py` (run_nested_sampling main loop) — ~200 lines.
- `nss/utils.py` (Results dataclass) — drop the `ott` import; nss's
  MMD/W2 metrics are never populated in practice (FINDINGS_v3
  confirms `Results.mmd` etc. are default-0). The dataclass survives
  without ott.

Plus the parts of `handley-lab/blackjax` we use:

- `blackjax.ns.adaptive.init` and the slice-MCMC inner kernel.
- Strip the rest of mainline blackjax (we don't use SMC / HMC /
  adjusted-mclmc from blackjax itself in `af.NSS`).

Result: a vendored subdirectory with maybe ~500–800 lines of code.
**No external `nss` or `blackjax` install needed.** The vendored
copy lives under PyAutoFit's licence (verify yallup's is compatible
— BSD-3-Clause as of latest yallup/nss check).

Pro:
- **Single safe install**: `pip install autofit` just works.
- No git+ URLs in install instructions.
- We own the version we test against — no surprise upstream breaks.

Con:
- We carry the maintenance burden when yallup/nss evolves. Worth it
  only if `nss_jit` becomes a recommended production sampler.
- Need to track upstream for bug fixes / performance improvements.

Mitigation: pin a specific `nss` commit hash in a docstring next to
the vendored code, document the `--upgrade-nss` workflow (a script
that re-vendors from upstream HEAD), and tag the vendored version
in PyAutoFit's CHANGELOG when bumped.

### Option B — Coordinate with yallup to publish to PyPI

Push for yallup/nss to publish a self-contained PyPI release that
bundles or pins the right `blackjax` fork explicitly. Then PyAutoFit
just lists `nss>=X.Y` as an optional dep.

Pro:
- The upstream community benefits.
- Standard `pip install autofit[nss]` extra works.
- No maintenance burden in PyAutoFit.

Con:
- Requires upstream coordination — unbounded timeline.
- The `handley-lab/blackjax` fork is itself a fork-of-a-fork; yallup
  doesn't own it. Multi-party coordination.

Recommendation: pursue **in parallel** with option A, but ship A first.
When B lands upstream, the vendored copy under option A can be
deleted and replaced with the `pip install` dep.

### Option C — `pip install autofit[nss]` extra (defer)

PyAutoFit's `pyproject.toml` declares an `[options.extras_require]
nss = [...]` entry that pins the specific git+ URLs and
`--no-deps`-equivalent install order.

Pro:
- Most "standard pip" pattern.

Con:
- `setuptools` pip extras do not support `--no-deps` per-dep. You
  can specify `git+https://...` URLs but not the install ordering.
- The fork's transitive deps would still pull mainline blackjax →
  the cure for which is `--force-reinstall --no-deps` on the
  handley-lab fork **after** the rest are installed. pip extras
  can't sequence that.

Verdict: not feasible with current pip semantics. Move on.

### Option D — `python -m autofit.install_nss` helper (fallback)

A standalone Python script that does the install dance:

```python
# autofit/install_nss.py
import subprocess
def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "--index-url", "https://download.pytorch.org/whl/cpu", "torch"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "git+https://github.com/yallup/nss.git", "--no-deps"])
    # ...etc, in order
```

User runs `python -m autofit.install_nss` after `pip install autofit`.

Pro:
- Works today with no upstream coordination.
- Documented + scripted, not "ten manual pip commands".

Con:
- Adds a second-step ceremony users have to know about.
- Surface area for things to break if pip / PyPI changes.
- Doesn't solve the fundamental fragility — `--force-reinstall
  --no-deps` is still happening; just hidden in the script.

Use this only if option A's vendoring turns out infeasible
(legally or technically).

### Decision matrix

| Option | Effort | Robustness | User experience |
|---|---|---|---|
| A — vendor | medium | high | `pip install autofit` works |
| B — upstream coordinate | low (PyAutoFit) + unbounded (yallup) | high | `pip install autofit[nss]` works once upstream lands |
| C — pip extra | low | low — pip can't sequence | broken |
| D — install script | low | medium | two-step install |

**Recommendation: ship A, pursue B in parallel.** D is the fallback
if A turns out to have licence or technical blockers.

__What to build (option A)__

1. **Vendor the code.** Create
   `@PyAutoFit/autofit/non_linear/search/nest/nss/_vendor/` with:
   - `__init__.py` — exports the entrypoints `run_nested_sampling`
     (from `ns.py`) and `Results` (from `utils.py`).
   - `ns.py` — copy from yallup/nss with the imports rewired to
     point at the vendored blackjax-fork pieces.
   - `utils.py` — copy with the `ott` import + MMD/W2 fields removed
     (or made optional).
   - `_blackjax_fork.py` — extract the minimum `init` and slice
     kernel from `handley-lab/blackjax`.
   - `LICENSE-NSS` and `LICENSE-BLACKJAX-FORK` — preserve upstream
     attribution. Verify both are BSD-3 or MIT before committing.

2. **Wire the `af.NSS` Phase 1 wrapper** (or update it if Phase 1 is
   already merged) to `from autofit.non_linear.search.nest.nss._vendor
   import run_nested_sampling` instead of `from nss.ns import ...`.

3. **Smoke test.** A fresh `python -m venv tmp_venv && source
   tmp_venv/bin/activate && pip install autofit && python -c
   "import autofit; search = autofit.NSS(); print('OK')"` must
   succeed without any `pip install` of nss, blackjax, optax, chex,
   etc. The vendored code has to be self-contained except for `jax`
   itself, which PyAutoFit already declares as an extra.

4. **CI workflow.** Add a job to PyAutoFit's `.github/workflows/`
   that does a clean venv install and runs the smoke from step 3.
   Triggers on every PR — catches install regressions before users
   see them.

5. **Document the vendoring source.** Top-of-file docstring on each
   vendored module:

   ```python
   """
   Vendored from yallup/nss @ <commit-sha>, BSD-3-Clause.
   Source: https://github.com/yallup/nss/tree/<commit-sha>/src/nss/ns.py

   See PyAutoPrompt/autofit/nss_install_simplification.md for the
   vendoring rationale. To re-vendor from upstream HEAD, run
   scripts/revendor_nss.py.
   """
   ```

__What to verify__

1. **Fresh-venv install smoke** — see step 3 above. Must be the
   primary CI gate.

2. **All Phase 1 unit tests still pass** with the vendored backend.
   Run the same `test_autofit/non_linear/search/nest/nss/` suite the
   Phase 1 prompt defines; the only difference is the import path.

3. **`autolens_workspace_developer/searches_minimal/nss_jit.py`
   continues to work** — it imports `from nss.ns import
   run_nested_sampling` (the public yallup PyPI path, not the
   vendored one). Vendoring doesn't remove that — users who want
   to use yallup/nss directly still can, via their own install. This
   gate confirms our vendoring doesn't accidentally break the
   external-nss path.

4. **`autoconf.jax_wrapper`-style guards.** `import autofit` must
   not import the vendored nss eagerly (defer to first use of
   `af.NSS(...)`). Otherwise installing PyAutoFit imposes a JAX
   cost on every Python startup.

5. **Licence audit.** Confirm both yallup/nss (BSD-3-Clause, check
   `pyproject.toml`) and handley-lab/blackjax (BSD-3-Clause, fork
   of mainline-MIT) are compatible with PyAutoFit's licence. If a
   conflict surfaces, **stop and fall back to option D**.

__Out of scope__

- `nss_grad` / HMC variant — that uses different blackjax primitives.
  Phase 4 only vendors what `af.NSS` (slice-sampling) needs. When
  `nss_grad` lands (post-NaN-gradient-fix from session-1's probe), a
  follow-up prompt extends the vendored bundle.
- Removing `pocomc`, `numpyro`, `blackjax-smc` from the install
  conversation — those are profiling-only samplers in
  `z_projects/profiling/`, not first-class PyAutoFit samplers (yet).
- Generating Conda / mamba install instructions — pip-first; conda
  comes later if the demand exists.

__Risks / open questions__

1. **Licence compatibility.** Both upstream packages are BSD-3
   today; verify at vendoring time. If either changes upstream
   licence in a way that conflicts with PyAutoFit's, the vendoring
   option dies and we fall back to D.

2. **Maintenance burden.** Every yallup/nss bug fix or perf
   improvement requires a re-vendor. Mitigate with a
   `scripts/revendor_nss.py` helper that automates the copy + the
   import rewriting. Run quarterly or when users report a bug
   upstream has already fixed.

3. **Coordinating with the `autolens_workspace_developer/searches_minimal`
   experimental scripts.** Those scripts directly import `from
   nss.ns import ...` because that's the natural way to run the
   yallup/nss sampler externally. The vendoring under PyAutoFit
   doesn't remove that — but if PyAutoFit's vendored copy diverges
   from upstream, users running both may hit version skew. Document
   the relationship in `autolens_workspace_developer/CLAUDE.md`.

4. **JAX version pin.** The Phase 0 priors + Phase 1 wrapper pin
   against JAX 0.4.38 (the version the profiling project confirmed
   on the HPC). The vendored nss must work with that JAX. If
   yallup's HEAD requires a newer JAX, vendor an older commit
   compatible with PyAutoFit's pinned JAX, not HEAD.

__Reference__

- `@z_projects/profiling/FINDINGS_v3.md` — full install saga
- `@z_projects/profiling/hpc/install_nss.sh` — current install
  script (the "do this manually" version we're trying to replace)
- `@PyAutoFit/pyproject.toml` — PyAutoFit's existing optional
  dependency declarations (look at how `[nautilus]` is structured)
- `@PyAutoPrompt/z_features/nss_first_class_sampler.md` — Phase 4
  in the sequenced roadmap
- `@PyAutoPrompt/autofit/nss_search_wrapper.md` — Phase 1 (the
  consumer of the vendored backend)
