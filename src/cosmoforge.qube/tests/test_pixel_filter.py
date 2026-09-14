"""Acceptance tests for ``pixel_filter=`` (ADR-0020).

Restriction through the library's own seams must reproduce the
ridge-regularised dense conjugation of the **truncated** operator
``F_trunc = U_k Σ_k W_kᵀ``. An oracle built on the full ``F`` is a different
estimator and disagrees at ``range_epsilon²``, so the oracle here always
conjugates the truncation the filter actually kept.

The invertible-filter equality (a filter of full rank leaves the estimator
alone) is vacuous under restriction and is *not* the oracle test; it appears
separately as the hits-weighting no-op control.
"""

from __future__ import annotations

import os

import healpy as hp
import numpy as np
import pytest
import yaml

from cosmocore.filters import Filter, harmonic_deprojection, hits_weighting
from cosmocore.signal_kernels import signal_matrix
from qube import Fisher, Spectra

#: Deprojected multipoles. Two whole multipoles at nside 4 remove enough rank
#: to make the filtered answer differ from the unfiltered one by O(1).
NULL_ELLS = [2, 3]

#: Projector tolerance against the oracle. Measured 1.4e-14 on the build.
PROJECTOR_TOL = 1e-10

#: Graded tolerance. The oracle agreement floors at ``cond * eps_machine`` and
#: ``cond ~ range_epsilon^-2``, so this tracks the epsilon below, not machine
#: precision.
GRADED_TOL = 1e-8
GRADED_EPSILON = 1e-3

#: The noise bias is quadratic in ``C^-1``, so it carries the *oracle's* ridge
#: rather than the filter's error. Sweeping the ridge over 10 / 1e3 / 1e6 times
#: the median noise diagonal moved it over 2.1e-9 / 3.0e-11 / 8.3e-10 while the
#: Fisher stayed at 1e-14 throughout: the residual is the regularisation, not
#: the restriction.
NOISE_BIAS_TOL = 1e-8

#: T+QU tolerance. The oracle inverts a ridge-regularised covariance whose T and
#: P blocks sit some sixteen orders of magnitude apart, and the largest Fisher
#: element there is 4.7e10; the residual is that dynamic range, not the seams.
#: ``test_the_oracle_harness_is_transparent`` pins the harness itself at zero.
MULTIFIELD_TOL = 1e-8


# ---------------------------------------------------------------- helpers


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    return float(np.abs(a - b).max() / den)


def _worst_diagonal_change(a, b):
    """Largest per-element relative change on the diagonal.

    ``_rel`` normalises by the global maximum, and a Fisher diagonal spans nine
    orders of magnitude here, so it cannot see a deprojected multipole being
    annihilated.
    """
    a, b = np.abs(np.diag(np.asarray(a, float))), np.abs(np.diag(np.asarray(b, float)))
    return float(np.max(np.abs(a - b) / np.maximum(a, 1e-300)))


def _mask(local_path, name, ncomponents):
    """The shipped nside-4 mask, widened to one column per component."""
    path = os.path.join(local_path, "tests", "data", "nside4", name, "mask.fits")
    m = hp.read_map(path)
    return np.repeat(np.asarray(m, float)[:, None], ncomponents, axis=1)


def _rewritten(config, tmp_path, **keys):
    """The same config with a few keys overridden, in a temporary file."""
    with open(config) as f:
        body = yaml.safe_load(f)
    body.update(keys)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(body))
    return str(path)


def _graded_from(projector, mask, epsilon=GRADED_EPSILON):
    """A graded operator suppressing, rather than deleting, a projector's null.

    Built from the projector's own removed directions so the two filters act on
    the same subspace and the only difference under test is Σ.
    """
    Q = projector.Q_removed
    gains = np.logspace(-1, -6, Q.shape[1])
    F = np.eye(Q.shape[0]) - Q @ np.diag(1.0 - gains) @ Q.T
    return Filter.from_operator(F, range_epsilon=epsilon, mask=mask)


def _truncated(f):
    """The operator the filter actually kept, ``U_k Σ_k W_kᵀ``, and its null."""
    F_trunc = (f.U * f.sigma) @ f.W.T
    return F_trunc, np.eye(f.n_pixels) - f.U @ f.U.T


class _OracleMixin:
    """Ridge-regularised dense conjugation, at the full pixel dimension.

    The ridge is what makes the conjugated covariance invertible on the null
    space; it never touches the range, so the estimator it defines is the
    filtered one.
    """

    F = P_null = None
    ridge_factor = 10.0

    def _conjugate_noise(self, N):
        # The ridge follows the noise diagonal rather than one scalar: T and P
        # sit orders of magnitude apart, and a single scalar regularises one of
        # them far too weakly.
        ridge = self.P_null @ np.diag(self.ridge_factor * np.diag(N)) @ self.P_null
        return np.asfortranarray(self.F @ N @ self.F.T + ridge)

    def setup_covariance_matrices(self):
        super().setup_covariance_matrices()
        self.noise_cov1 = self._conjugate_noise(self.noise_cov1)
        if self.noise_cov2 is not None:
            self.noise_cov2 = self._conjugate_noise(self.noise_cov2)
        return self.noise_cov1, self.noise_cov2

    def setup_signal_matrix(self):
        S = signal_matrix(self.collection, self.lmax_signal)
        self.signal_matrix = np.asfortranarray(self.F @ S @ self.F.T)
        return self.signal_matrix

    def _build_derivative_matrix(self, ell, spectrum_idx=0):
        dC = super()._build_derivative_matrix(ell, spectrum_idx)
        return np.asfortranarray(self.F @ dC @ self.F.T)

    def setup_maps(self):
        super().setup_maps()
        for name in ("maps1", "maps2"):
            maps = getattr(self, name, None)
            if maps is not None:
                setattr(self, name, np.ascontiguousarray(self.F @ maps))


_OracleFisher = type("_OracleFisher", (_OracleMixin, Fisher), {})
_OracleSpectra = type("_OracleSpectra", (_OracleMixin, Spectra), {})


def _run(config, mask, *, pixel_filter=None, basis=False, **kwargs):
    fisher = Fisher(config, basis=basis, mask=mask, pixel_filter=pixel_filter, **kwargs)
    fisher.run()
    spectra = Spectra(config, fisher=fisher)
    spectra.run()
    return fisher, spectra


def _run_oracle(config, mask, pixel_filter):
    F_trunc, P_null = _truncated(pixel_filter)
    fisher = _OracleFisher(config, basis=False, mask=mask)
    fisher.F, fisher.P_null = F_trunc, P_null
    fisher.run()
    spectra = _OracleSpectra(config, fisher=fisher)
    spectra.F, spectra.P_null = F_trunc, P_null
    spectra.run()
    return fisher, spectra


# ---------------------------------------------------------------- fixtures

#: ``(config directory, spins, ncomponents, deprojected slot)``.
GEOMETRIES = {
    "T": ("T", [0], 1, "T"),
    "QU": ("QU", [2], 2, "E"),
    "TQU": ("TQU", [0, 2], 3, "E"),
}


@pytest.fixture
def geometry(request, local_path, config_resolver):
    """Config path, mask and a projector deprojecting two whole multipoles."""
    name, spins, ncomponents, slot = GEOMETRIES[request.param]
    config = config_resolver(f"tests/data/nside4/{name}/config.yaml")
    mask = _mask(local_path, name, ncomponents)
    projector = harmonic_deprojection(mask=mask, spins=spins, ells=NULL_ELLS, slot=slot)
    yield config, mask, projector
    os.unlink(config)


# ---------------------------------------------------------------- the oracle


@pytest.mark.parametrize("geometry", ["T", "QU"], indirect=True)
class TestRidgeOracle:
    """The destination's acceptance sentence: T-only and QU, both filter kinds."""

    def test_projector_matches_the_oracle(self, geometry):
        config, mask, projector = geometry
        native_f, native_s = _run(config, mask, pixel_filter=projector)
        oracle_f, oracle_s = _run_oracle(config, mask, projector)

        assert _rel(native_f.fisher, oracle_f.fisher) < PROJECTOR_TOL
        assert _rel(native_s.qml_results, oracle_s.qml_results) < PROJECTOR_TOL
        assert _rel(native_s.qml_noise_bias, oracle_s.qml_noise_bias) < NOISE_BIAS_TOL

    def test_the_filter_is_not_a_no_op(self, geometry):
        """Guards the oracle test: two identities also agree to 1e-14."""
        config, mask, projector = geometry
        assert 0 < projector.rank < projector.n_pixels
        plain_f, _ = _run(config, mask)
        native_f, _ = _run(config, mask, pixel_filter=projector)
        # The deprojected multipoles lose essentially all of their Fisher
        # information, so the worst diagonal element moves by ~100 per cent.
        assert _worst_diagonal_change(plain_f.fisher, native_f.fisher) > 0.5

    def test_graded_matches_the_truncated_operator_oracle(self, geometry):
        config, mask, projector = geometry
        graded = _graded_from(projector, mask)
        assert not graded.is_projector

        native_f, native_s = _run(config, mask, pixel_filter=graded)
        oracle_f, oracle_s = _run_oracle(config, mask, graded)

        assert _rel(native_f.fisher, oracle_f.fisher) < GRADED_TOL
        assert _rel(native_s.qml_results, oracle_s.qml_results) < GRADED_TOL


@pytest.mark.parametrize("geometry", ["QU"], indirect=True)
class TestOtherPaths:
    """Each path against the traditional one, filtered and unfiltered.

    The unfiltered run is the control: the paths already disagree by something
    in this configuration, and the filter is only suspect if the filtered gap
    is worse than that.
    """

    #: The destination's four paths, less the traditional one they are all
    #: measured against. Compression is lossy, which is exactly why the
    #: unfiltered control matters more here than anywhere else.
    PATHS = {
        "harmonic": {"method": "harmonic"},
        "pixel-direct": {"method": "pixel"},
        "pixel-compressed": {"method": "pixel", "mode_fraction": 0.9},
    }

    @pytest.mark.parametrize("kind", ["projector", "graded"])
    @pytest.mark.parametrize("path", list(PATHS))
    def test_path_agreement_is_not_degraded(self, geometry, path, kind):
        config, mask, projector = geometry
        basis = self.PATHS[path]
        # The graded case is the only thing that exercises the Σ scaling in
        # ComputationBasis._restrict_V; a projector leaves V' = V W.
        pixel_filter = projector if kind == "projector" else _graded_from(projector, mask)

        plain_ref_f, plain_ref_s = _run(config, mask)
        plain_f, plain_s = _run(config, mask, basis=basis)
        control_fisher = _rel(plain_f.fisher, plain_ref_f.fisher)
        control_qb = _rel(plain_s.qml_results, plain_ref_s.qml_results)

        filtered_ref_f, filtered_ref_s = _run(config, mask, pixel_filter=pixel_filter)
        filtered_f, filtered_s = _run(
            config, mask, pixel_filter=pixel_filter, basis=basis
        )

        assert _rel(filtered_f.fisher, filtered_ref_f.fisher) <= max(
            10 * control_fisher, 1e-11
        )
        assert _rel(filtered_s.qml_results, filtered_ref_s.qml_results) <= max(
            10 * control_qb, 1e-11
        )

    def test_a_narrowed_signal_band_still_agrees(self, geometry, tmp_path):
        """``lmax < lmax_signal`` is the configuration that can build S_fixed.

        Nothing else in the suite narrows the band, and the narrowing changes
        what the basis represents, so the filter is re-checked against the
        traditional path there. Whether this fixture actually populates
        ``S_fixed`` was not established: ``HarmonicBasis``'s own
        ``_compute_s_fixed_from_fiducial`` stays unexercised either way, and
        the acceptance question it guards is recorded on the map, not asserted
        here.
        """
        config, mask, projector = geometry
        windowed = _rewritten(config, tmp_path, lmax_signal=12)

        plain_ref_f, plain_ref_s = _run(windowed, mask)
        plain_f, plain_s = _run(windowed, mask, basis={"method": "harmonic"})
        control_fisher = _rel(plain_f.fisher, plain_ref_f.fisher)
        control_qb = _rel(plain_s.qml_results, plain_ref_s.qml_results)

        filtered_ref_f, filtered_ref_s = _run(windowed, mask, pixel_filter=projector)
        filtered_f, filtered_s = _run(
            windowed, mask, pixel_filter=projector, basis={"method": "harmonic"}
        )

        assert _rel(filtered_f.fisher, filtered_ref_f.fisher) <= max(
            10 * control_fisher, 1e-11
        )
        assert _rel(filtered_s.qml_results, filtered_ref_s.qml_results) <= max(
            10 * control_qb, 1e-11
        )


@pytest.mark.parametrize("geometry", ["TQU"], indirect=True)
class TestFieldBlocks:
    """T+QU, where the per-field bookkeeping the filter switches off would fire.

    Both filters go through the branch that the component-structure decision
    changed; the mixing one also crosses the T-P block, which is the only
    crossing a filter can make (Q-U mixing never leaves its own block).
    """

    def test_the_oracle_harness_is_transparent(self, geometry):
        """An identity conjugation through the mixin must change nothing.

        Without this the multi-field tolerance below could be hiding a defect in
        the test harness rather than the conditioning it claims.
        """
        config, mask, projector = geometry
        native, _ = _run(config, mask)
        oracle = _OracleFisher(config, basis=False, mask=mask)
        n = projector.n_pixels
        oracle.F, oracle.P_null = np.eye(n), np.zeros((n, n))
        oracle.run()
        assert _rel(native.fisher, oracle.fisher) == 0.0

    def test_single_slot_projector_matches_the_oracle(self, geometry):
        config, mask, projector = geometry
        native_f, native_s = _run(config, mask, pixel_filter=projector)
        oracle_f, oracle_s = _run_oracle(config, mask, projector)
        assert _rel(native_f.fisher, oracle_f.fisher) < MULTIFIELD_TOL
        assert _rel(native_s.qml_results, oracle_s.qml_results) < MULTIFIELD_TOL

    def test_component_mixing_filter_matches_the_oracle(self, geometry):
        config, mask, projector = geometry
        n = projector.n_pixels
        rng = np.random.default_rng(20260909)
        W = np.linalg.qr(rng.standard_normal((n, n - 40)))[0]
        mixing = Filter.from_subspace(W, mask=mask)

        native_f, native_s = _run(config, mask, pixel_filter=mixing)
        oracle_f, oracle_s = _run_oracle(config, mask, mixing)
        assert _rel(native_f.fisher, oracle_f.fisher) < MULTIFIELD_TOL
        assert _rel(native_s.qml_results, oracle_s.qml_results) < MULTIFIELD_TOL


class TestCrossSpectra:
    """``do_cross`` filters through the same seams, on the traditional path."""

    def test_cross_matches_the_oracle(self, local_path, config_resolver):
        config = config_resolver("tests/data/nside4/QU/cross_config.yaml")
        mask = _mask(local_path, "QU", 2)
        projector = harmonic_deprojection(mask=mask, spins=[2], ells=NULL_ELLS, slot="E")
        native_f, native_s = _run(config, mask, pixel_filter=projector)
        oracle_f, oracle_s = _run_oracle(config, mask, projector)
        os.unlink(config)

        assert _rel(native_f.fisher, oracle_f.fisher) < PROJECTOR_TOL
        assert _rel(native_s.qml_results, oracle_s.qml_results) < PROJECTOR_TOL


# ---------------------------------------------------------------- controls


@pytest.mark.parametrize("geometry", ["QU"], indirect=True)
class TestInvertibleControl:
    """Hits weighting is invertible, so restriction must change nothing."""

    def test_hits_weighting_is_a_no_op(self, geometry):
        config, mask, _ = geometry
        rng = np.random.default_rng(11)
        hits = rng.uniform(0.5, 2.0, mask.shape[0])
        with pytest.warns(UserWarning, match="kept every singular value"):
            weighting = hits_weighting(hits, mask=mask, range_epsilon=1e-7)
        assert weighting.rank == weighting.n_pixels

        plain_f, plain_s = _run(config, mask)
        weighted_f, weighted_s = _run(config, mask, pixel_filter=weighting)
        assert _rel(plain_f.fisher, weighted_f.fisher) < GRADED_TOL
        assert _rel(plain_s.qml_results, weighted_s.qml_results) < GRADED_TOL


@pytest.mark.parametrize("geometry", ["QU"], indirect=True)
class TestPrefilteredFlags:
    def test_prefiltered_maps_agree_with_raw_maps(self, geometry):
        """For a projector, ``Uᵀ (F d)`` and ``Σ Wᵀ d`` are the same vector."""
        config, mask, projector = geometry
        reference_f, reference_s = _run(config, mask, pixel_filter=projector)

        # W @ (Wᵀ d) is F d for a projector, so this is the same data handed in
        # pre-filtered rather than raw.
        prefiltered = np.ascontiguousarray(projector.W @ reference_s.maps1)
        spectra = Spectra(
            config, fisher=reference_f, maps1=prefiltered, maps_prefiltered=True
        )
        spectra.run()
        assert _rel(spectra.qml_results, reference_s.qml_results) < PROJECTOR_TOL

    def test_prefiltered_maps_warn_when_power_leaves_the_range(self, geometry):
        config, mask, projector = geometry
        fisher = Fisher(config, basis=False, mask=mask, pixel_filter=projector)
        fisher.run()
        rng = np.random.default_rng(3)
        maps = rng.standard_normal((projector.n_pixels, fisher.params.nsims))
        spectra = Spectra(config, fisher=fisher, maps1=maps, maps_prefiltered=True)
        with pytest.warns(UserWarning, match="outside the filter's range"):
            spectra.run()

    def test_noise_prefiltered_is_free_for_a_projector(self, geometry):
        config, mask, projector = geometry
        plain = Fisher(config, basis=False, mask=mask, pixel_filter=projector)
        plain.run()
        flagged = Fisher(
            config,
            basis=False,
            mask=mask,
            pixel_filter=projector,
            noise_prefiltered=True,
        )
        flagged.run()
        assert _rel(plain.fisher, flagged.fisher) < PROJECTOR_TOL

    def test_noise_prefiltered_bites_for_a_graded_filter(self, geometry):
        """``Σ Wᵀ N W Σ`` and ``Uᵀ N U`` are different noise models."""
        config, mask, projector = geometry
        graded = _graded_from(projector, mask)
        plain = Fisher(config, basis=False, mask=mask, pixel_filter=graded)
        plain.run()
        flagged = Fisher(
            config,
            basis=False,
            mask=mask,
            pixel_filter=graded,
            noise_prefiltered=True,
        )
        flagged.run()
        assert _worst_diagonal_change(plain.fisher, flagged.fisher) > 0.1


# ---------------------------------------------------------------- guards


@pytest.mark.parametrize("geometry", ["QU"], indirect=True)
class TestGuards:
    def test_wrong_pixel_count_raises(self, geometry):
        config, mask, projector = geometry
        short = Filter.from_subspace(np.linalg.qr(projector.W[:-2])[0])
        with pytest.raises(ValueError, match="was built for"):
            Fisher(config, basis=False, mask=mask, pixel_filter=short).run()

    def test_wrong_ordering_raises(self, geometry):
        config, mask, projector = geometry
        # The nside-4 fixtures are full sky, so a permutation of the mask is
        # the same mask; punching a hole is what changes the active ordering.
        holed = mask.copy()
        holed[0] = 0.0
        misfingerprinted = Filter.from_subspace(projector.W, mask=holed)
        with pytest.raises(ValueError, match="different active-pixel ordering"):
            Fisher(config, basis=False, mask=mask, pixel_filter=misfingerprinted).run()

    def test_out_file_handoff_refuses_a_filter(self, geometry):
        """The ``out*`` files hold ``r x r`` under a filter; the loader reads ``n x n``.

        Reproduces the worker-rank state that used to reach this adapter by
        accident: before ``Fisher.run()`` shared ``reduced_noise_cov1``, every
        rank but 0 saw it as ``None`` and fell past the in-memory adapter to
        disk, where the reshape raised an opaque ``ValueError``.
        """
        config, mask, projector = geometry
        fisher = Fisher(config, basis=False, mask=mask, pixel_filter=projector)
        fisher.run()
        assert fisher.reduced_noise_cov1 is not None
        fisher.reduced_noise_cov1 = None
        with pytest.raises(NotImplementedError, match="not supported with a"):
            Spectra(config, fisher=fisher)

    def test_spectra_refuses_a_conflicting_filter(self, geometry):
        config, mask, projector = geometry
        fisher = Fisher(config, basis=False, mask=mask)
        fisher.run()
        with pytest.raises(ValueError, match="conflicts with the supplied fisher"):
            Spectra(config, fisher=fisher, pixel_filter=projector)

    def test_m_block_compression_refuses_a_filter(self, geometry):
        """Reached through the basis directly: ``compress`` is not a Fisher key."""
        from cosmocore.basis import HarmonicBasis

        _, mask, projector = geometry
        n = 60
        theta = np.linspace(0.3, 0.9, n)
        phi = np.linspace(0.0, 1.2, n)
        templates = np.column_stack([np.ones(n), theta, theta**2])
        blocked = Filter.from_deprojection(templates)
        with pytest.raises(NotImplementedError):
            HarmonicBasis(
                np.eye(blocked.rank),
                theta,
                phi,
                6,
                compress=True,
                pixel_filter=blocked,
            )
        # The same construction without a filter is allowed, so the refusal is
        # the filter's doing and not a pre-existing one.
        HarmonicBasis(np.eye(n), theta, phi, 6, compress=True)
        assert projector.rank > 0
