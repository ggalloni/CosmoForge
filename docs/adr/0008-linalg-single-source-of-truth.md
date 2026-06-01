# ADR-0008: All dense linear algebra goes through `cosmocore/basics/linalg.py`

## Status

Accepted — established by precedent (`matrix_mult`, `matrix_inverse_symm`, `cholesky_decomposition`, `matrix_slogdet_symm`); made explicit by the 2026-04-30 Cholesky refactor (added `cholesky_factor`, `cholesky_solve`).

## Context

CosmoForge's hot path is dominated by dense linear algebra on n_pix × n_pix matrices that reach ~62 GB at ECLIPSE-class resolutions. The library that provides those operations today is SciPy / LAPACK via thin wrappers in `cosmocore/basics/linalg.py`. Some operations (e.g. `np.matmul`) are already trivially wrapped (`matrix_mult` is `np.matmul`) for no obvious reason at the call site — but for a forward-looking reason: the wrapper is the integration point.

Future work the project is on the hook for:

- **GPU / accelerator backends** (CuPy, JAX, PyTorch) — a serious option once n_pix exceeds shared-memory limits.
- **Packed symmetric storage** (`dpptrf`, halves a 62 GB cov to 31 GB) — a planned follow-up.
- **MPI shared-memory or distributed factorisations** — also tracked.
- **Backend-specific numerical guards** (regularisation floors, conditioning checks) that should apply uniformly, not site-by-site.

If linear-algebra calls are scattered across `harmonic.py`, `pixel.py`, `core.py`, `picslike.py`, `fisher.py`, etc. as direct SciPy imports, every one of those changes becomes a multi-file grep-and-replace with high risk of missing a site or applying inconsistent defaults.

## Decision

All dense linear-algebra primitives consumed by CosmoForge code live in `cosmocore/basics/linalg.py` (and the closely-related `cosmocore/basics/smw.py` for SMW-specific patterns). Call sites import from `cosmocore.basics`, not from `numpy.linalg`, `scipy.linalg`, or `scipy.linalg.lapack` directly.

Concretely:

- Matrix multiplication → `matrix_mult` (not `np.matmul` / `@` for the production hot path)
- Symmetric inverse → `matrix_inverse_symm`
- Symmetric log-determinant → `matrix_slogdet_symm`
- Cholesky decomposition (returns `L`) → `cholesky_decomposition`
- Cholesky factor (returns `(L, lower=True)` tuple, scipy-`cho_factor` style) → `cholesky_factor`
- Cholesky solve → `cholesky_solve`
- Trace of a matrix product → `matrix_trace`
- Diagonal accumulation → `add_diagonal`

New primitives are added here as needed; existing primitives are not bypassed for "convenience" inline.

The wrappers stay **thin** — they delegate to the chosen backend with project-standard defaults. Behaviour is added (regularisation, conditioning checks, backend dispatch) only when warranted, and once added applies uniformly to every call site.

## Consequences

- **Backend swap is local**: replacing SciPy/LAPACK with a GPU backend touches `basics/linalg.py` and (potentially) `basics/smw.py`. Call-site code is unchanged unless the data lives on a different device — a separate migration tracked in its own plan.
- **Defaults are encapsulated**: a single canonical choice (e.g. always `lower=True` for Cholesky, F-order for symmetric routines) instead of each call site picking its own.
- **Future numerical guards are uniform**: when conditioning monitoring or regularisation is added, it applies everywhere.
- **A future reader sees a five-line wrapper** around `scipy.linalg.cho_factor` (or any other primitive) and asks "why?" — the answer is here.
- **Trade-off**: very thin wrappers have no behavioural value today, only future-proofing value. Reviewers may push back. The principle exists precisely so that "use the wrapper" is a project standard, not a per-PR debate.
- **Test code may import directly from SciPy** for ad-hoc reference computations (e.g., asserting that `cholesky_solve(N_chol, X) ≈ scipy.linalg.solve(N, X)`). Production code in `cosmocore`, `qube`, and `picslike` does not.

## References

- Existing precedent: `cosmocore/basics/linalg.py` (already wraps `np.matmul`, `lapack.dpotrf`, `lapack.dpotri`, `lapack.dpotrs`).
- Motivating refactor: `.claude/plans/2026-04-30-cholesky-refactor.md` (added `cholesky_factor`, `cholesky_solve`).
- Deferred follow-ups that benefit from the principle: packed symmetric storage; MPI shared-memory `basis_manager`; GPU backend exploration.

## Update (2026-05-02)

The wrappers exist because the project carries factors, not inverses. Made explicit here so future readers do not mistake the asymmetric handling of `N_inv` in the codebase for a bug:

1. **QML hot paths consume `N` only via `cholesky_solve(N_chol, X)`.** The dense `N_inv` is never materialised. At ECLIPSE-class resolutions this is the difference between fitting and OOMing — `N_inv` is roughly the same size as the covariance it inverts (~62 GB at the QU eclipse cell), and materialising it once doubles peak RSS. The Fisher hot path, the `q` quadratic, and every SMW step (ADR-0001) consume `N` through the factor.

2. **The compression-basis sites in `pixel.py` are the deliberate exception.** They keep dense `N_inv` and reconstruct symmetric `N` lazily on demand. Rewriting them onto the factor path requires primitives that do not exist in `basics/linalg.py` yet — `dtrmm`-based one-sided sandwiches, the `sandwich-cho_solve` pattern for `A^T N^{-1} A`, and a per-field factor variant that keeps the cost subadditive in the spin-2 doubling. Catalogued in `project_compression_basis_n_inv_deferred.md`; out of scope until the compression follow-up paper, when the algebraic rewrites have a forcing function.

3. **New code defaults to the factor path.** Reintroducing `N_inv` materialisation in a hot path requires a new ADR Update that explains why the algebra cannot be expressed through `cholesky_solve`. "Convenience" is not a sufficient reason; the precedent set by PR #16 (drop dense `N_inv`, in-place Cholesky over the noise covariance) freed the memory that landing the eclipse-QU cluster runs depended on.

4. **`cholesky_factor` returns a `(L, lower=True)` tuple in scipy's `cho_factor` style.** Call sites pass the tuple to `cholesky_solve` directly; they do not unpack and pass `L` alone, because the lower-flag travels with the factor and rewriting code that drops it on the floor is a class of bug the wrapper exists to prevent (see `feedback_chol_factor_dirty_upper.md`).

The aggregate effect of these four points is that the project's noise-covariance representation in production code is the Cholesky factor; `N_inv` is a vestige in the compression-basis branch only and a candidate for removal once the algebraic rewrites land.
