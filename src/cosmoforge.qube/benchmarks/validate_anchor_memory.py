"""Check the fiducial-anchor memory predictions against tracemalloc, step by step.

Usage, from the repo root:

    uv run python src/cosmoforge.qube/benchmarks/validate_anchor_memory.py 3000 38

Arguments are n_pix and the inference lmax (lmax_signal = 2 * lmax), spin-0 T,
correlated noise, switch optimisation active.

A. Today's setup path (noise anchor and the committed fiducial anchor), with a
   production-like S_fixed passed in, as Core does.
B. The restructured "direct" anchored setup the 113 -> ~49 GiB claim was about,
   built here as a throwaway: S_full into the C buffer, N streamed from disk,
   in-place trsm over V^T, syrk for A, streamed N for T.

Reading the output: tracemalloc only sees allocations made after a run starts.
Add back the inputs allocated before it: N and S_fixed (n^2 each) in A, V
(m*n) in B. B also symmetrises A out of place for the comparison (+2 m^2), which
production would do in place. Measured at n=3000, lmax=38 (m=1517), these give
278 MiB (A, noise), 295 MiB (A, fiducial), 121 MiB intrinsic (B) against
predictions of 276.7 and 120.9 MiB.
"""

import gc
import sys
import tempfile
import time
import tracemalloc

import numpy as np
from scipy.linalg import blas, lapack

from cosmocore.basis import HarmonicBasis
from cosmocore.signal_kernels import compute_00_contribution
from qube.memory_budget import BudgetConfig, predict_qube_budget

n, lmax = int(sys.argv[1]), int(sys.argv[2])
lsig = 2 * lmax
SCR = tempfile.mkdtemp()
MiB = 1024**2
rng = np.random.default_rng(0)
theta = np.arccos(rng.uniform(-1, 1, n))
phi = rng.uniform(0, 2 * np.pi, n)
ells = np.arange(lsig + 1.0)
fid = np.where(ells >= 2, 1.0 / np.maximum(ells, 1) ** 2, 0.0)
win = fid.copy()
win[lmax + 1 :] = 0.0
high = fid - win
pv = np.column_stack(
    [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
)
N = 1e-3 * np.eye(n) + 1e-5 * np.ones((n, n))
np.save(f"{SCR}/N.npy", N)


def kernel(cl, out):
    """Spin-0 signal into ``out``'s lower triangle (what Core's kernel does)."""
    compute_00_contribution(cl, out, pv, pv, np.empty(lsig + 1), 0)


S_fixed = np.zeros((n, n))
kernel(high, S_fixed)  # warm the JIT (C order)
kernel(high, np.zeros((n, n), order="F"))  # and F order: numba specialises on layout
_t = time.perf_counter()
kernel(high, S_fixed)
print(f"S_fixed kernel (excluded from A's timing): {time.perf_counter() - _t:.2f}s")
S_fixed = S_fixed + np.tril(S_fixed, -1).T  # symmetric, as Core hands it over


class Steps:
    def __init__(self):
        tracemalloc.start()
        self.rows = []
        self.t0 = self.t = time.perf_counter()

    def mark(self, name):
        cur, peak = tracemalloc.get_traced_memory()
        now = time.perf_counter()
        self.rows.append((name, cur / MiB, peak / MiB, now - self.t))
        self.t = now
        tracemalloc.reset_peak()

    def report(self, title):
        print(f"\n{title}")
        for name, cur, peak, dt in self.rows:
            print(f"  {name:38s} peak {peak:7.1f}  after {cur:7.1f} MiB  {dt:6.2f}s")
        print(f"  total time {time.perf_counter() - self.t0:.2f}s")
        tracemalloc.stop()


def run_current(anchor):
    hb = HarmonicBasis(
        N.copy(),
        theta,
        phi,
        lmax_signal=lsig,
        lmin=2,
        lmax=lmax,
        S_fixed=S_fixed.copy(),
        anchor_C_ell=anchor,
    )
    gc.collect()
    st = Steps()
    st.mark("inputs (N, S_fixed)")
    hb._build_basis()
    hb._harmonic_basis._V_blocks = None
    st.mark("build V")
    hb._compute_effective_noise()
    st.mark("N_eff = N + S_fixed")
    if anchor is not None:
        hb._anchor_at_fiducial()
        st.mark("anchor: += V^T L0 V")
    hb._factorise_noise()
    st.mark("cholesky")
    hb._compute_smw_components()
    st.mark("V_N_inv, M, T")
    hb.release_pixel_projector()
    gc.collect()
    st.mark("release V")
    st.report(f"A. today's setup, anchor={'fiducial' if anchor is not None else 'noise'}")
    return hb


ref = run_current(None)
anc = run_current(win)
del ref
gc.collect()

# ---- B. direct anchored setup ----------------------------------------------
hb = HarmonicBasis(np.eye(n), theta, phi, lmax_signal=lmax)  # only for V
hb._build_basis()
Vc = np.ascontiguousarray(hb._V)  # production would build V C-ordered
m = Vc.shape[0]
del hb
Nd = np.load(f"{SCR}/N.npy", mmap_mode="r")
gc.collect()
st = Steps()
st.mark("inputs (V)")
C = np.empty((n, n), order="F")
kernel(fid, C)  # full fiducial signal, every multipole, into C
st.mark("S_full into C")
blk = 256
for j in range(0, n, blk):  # C += N, streamed from disk, column blocks
    C[:, j : j + blk] += Nd[:, j : j + blk]
st.mark("C += N (streamed)")
L, info = lapack.dpotrf(C, lower=1, overwrite_a=1, clean=0)
assert info == 0 and L is C
st.mark("cholesky in place")
X = blas.dtrsm(1.0, L, Vc.T, lower=1, overwrite_b=1)  # V^T is F-ordered n x m
assert np.shares_memory(X, Vc)
st.mark("X = L^-1 V^T in place")
A = blas.dsyrk(1.0, X, trans=1, lower=1)
A = A + np.tril(A, -1).T  # symmetric, for the comparison only
st.mark("A = X^T X")
Z = blas.dtrsm(1.0, L, X, lower=1, trans_a=1, overwrite_b=1)
del L, C
gc.collect()
st.mark("Z = L^-T X in place; free L")
T = np.zeros((m, m))
for i in range(0, n, blk):  # T = Z^T N Z, N streamed by row block
    T += Z[i : i + blk].T @ (np.asarray(Nd[i : i + blk]) @ Z)
st.mark("T = Z^T N Z (N streamed)")
st.report("B. direct anchored setup (throwaway)")

# ---- predictions --------------------------------------------------------------
sq, mn, m2 = n * n * 8 / MiB, m * n * 8 / MiB, m * m * 8 / MiB
b = predict_qube_budget(BudgetConfig(n_pix=n, n_modes=m, lmax_signal=lsig, lmax=lmax))
print(f"\nn={n} m={m}: n^2={sq:.1f} mn={mn:.1f} m^2={m2:.1f} MiB")
print(
    f"memory_budget basis_setup peak: {b.stage('basis_setup').peak_bytes / MiB:.1f} MiB, "
    f"kept {b.stage('basis_setup').persistent_bytes / MiB:.1f} MiB"
)
print(f"direct prediction, peak n^2+mn+m^2 = {sq + mn + m2:.1f} MiB")
print(
    "A == anchored M :",
    np.allclose(A, anc._V_Ninv_VT, rtol=1e-8, atol=1e-10 * np.abs(A).max()),
)
print(
    "T == anchored T :",
    np.allclose(T, anc._noise_cov_T, rtol=1e-8, atol=1e-10 * np.abs(T).max()),
)
