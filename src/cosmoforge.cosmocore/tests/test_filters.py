"""Operator constructors in ``cosmocore.filters``: their own self-checks."""

import healpy as hp
import numpy as np
import pytest

from cosmocore.filters import (
    MIN_RANGE_EPSILON,
    Filter,
    _centred_phi,
    _field_columns,
    _scan_frame,
    harmonic_deprojection,
    hits_weighting,
    scan_polynomial,
)
from cosmocore.geometry import active_pixel_index, active_pixels
from cosmocore.spectrum_key import Slot

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)

# The scan axis, and a patch 50 degrees from it. A constant-elevation scan
# observes at some elevation; it does not encircle its own axis, and a patch
# centred on the pole makes every stripe a full ring with no cut-free azimuth.
POLE = (np.radians(15.0), np.radians(70.0))
PATCH = (np.radians(65.0), np.radians(70.0))


@pytest.fixture
def cap():
    """An 18-degree cap, shared by three components (T, Q, U)."""
    disc = hp.query_disc(NSIDE, hp.ang2vec(*PATCH), np.radians(18.0))
    mask = np.zeros((NPIX, 3))
    mask[disc, :] = 1.0
    return mask


def _survives(f, x):
    """Fraction of a vector's norm left after restriction to ``range(F)``."""
    return np.linalg.norm(f.W.T @ x) / np.linalg.norm(x)


def _harmonic(mask, spins, ell, m, slot):
    """One real cut-sky harmonic of one slot, in the analysis ordering.

    Built with healpy's forward synthesis, which the constructor no longer uses:
    it takes its templates from the estimator's own V. So these vectors are an
    *independent* construction of the same harmonics, and the annihilation
    tests below cross-validate V against healpy rather than checking the
    constructor against itself.
    """
    columns_of = _field_columns(spins, mask.shape[1])
    lmax = max(ell, 2)
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    alm[hp.Alm.getidx(lmax, ell, m)] = 1.0
    stack = np.zeros((mask.shape[1], NPIX))
    if slot is Slot.S:
        stack[columns_of[0][0]] = hp.alm2map(alm, NSIDE, lmax=lmax)
    else:
        pair = [alm, np.zeros_like(alm)]
        if slot is not Slot.G:
            pair = pair[::-1]
        stack[list(columns_of[1])] = hp.alm2map_spin(pair, NSIDE, 2, lmax)
    return stack.ravel()[active_pixel_index(mask)]


class TestFieldColumns:
    def test_spin_widths(self):
        assert _field_columns([0, 2], 3) == [(0,), (1, 2)]
        assert _field_columns([2], 2) == [(0, 1)]
        assert _field_columns([0, 0], 2) == [(0,), (1,)]

    def test_component_count_mismatch_names_the_distinction(self):
        with pytest.raises(ValueError, match="per field, mask columns are per component"):
            _field_columns([0, 2], 2)


class TestHitsWeighting:
    def test_invertible_hits_are_full_rank_and_say_so(self, cap):
        rng = np.random.default_rng(1)
        hits = np.zeros_like(cap)
        active = cap[:, 0] > 0.5
        hits[active, :] = rng.uniform(50.0, 500.0, size=(active.sum(), 3))
        n = active_pixel_index(cap).size
        with pytest.warns(UserWarning, match="kept every singular value"):
            f = hits_weighting(hits, mask=cap, range_epsilon=1e-6)
        assert f.rank == n
        assert not f.is_projector
        # sigma is sqrt(hits), so sigma^2 must reproduce the hit counts.
        np.testing.assert_allclose(
            np.sort(f.sigma**2), np.sort(hits[active, :].T.ravel())
        )

    def test_unobserved_pixel_inside_the_footprint_lowers_the_rank(self, cap):
        hits = np.where(cap > 0.5, 100.0, 0.0)
        active = np.flatnonzero(cap[:, 0] > 0.5)
        hits[active[0], :] = 0.0
        n = active_pixel_index(cap).size
        f = hits_weighting(hits, mask=cap, range_epsilon=1e-6)
        assert f.rank == n - 3

    def test_rank_threshold_is_required(self, cap):
        with pytest.raises(ValueError, match="rank threshold is required"):
            hits_weighting(np.ones_like(cap), mask=cap)

    def test_negative_hits_rejected(self, cap):
        hits = np.ones_like(cap)
        hits[0, 0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            hits_weighting(hits, mask=cap, range_epsilon=1e-6)


class TestScanPolynomial:
    def test_kills_per_stripe_ramps(self, cap):
        f = scan_polynomial(mask=cap, pole=POLE, n_stripes=4, degree=1)
        assert f.is_projector

        per_component = active_pixels(cap)
        offsets = np.cumsum([0, *(len(a) for a in per_component)])
        theta, phi = _scan_frame(POLE, NPIX)
        observed = np.unique(np.concatenate(per_component))
        edges = np.linspace(theta[observed].min(), theta[observed].max(), 5)
        edges[-1] = np.nextafter(edges[-1], np.inf)

        rng = np.random.default_rng(0)
        ramp = np.zeros(offsets[-1])
        for component, pixels in enumerate(per_component):
            stripe_of = np.searchsorted(edges, theta[pixels], side="right") - 1
            for stripe in range(4):
                within = np.flatnonzero(stripe_of == stripe)
                x = _centred_phi(phi[pixels[within]])
                ramp[offsets[component] + within] = rng.normal() + rng.normal() * x
        assert _survives(f, ramp) < 1e-12

    def test_spares_rapid_along_scan_oscillation(self, cap):
        f = scan_polynomial(mask=cap, pole=POLE, n_stripes=4, degree=1)
        per_component = active_pixels(cap)
        offsets = np.cumsum([0, *(len(a) for a in per_component)])
        _, phi = _scan_frame(POLE, NPIX)

        def oscillation(cycles):
            out = np.zeros(offsets[-1])
            for component, pixels in enumerate(per_component):
                out[offsets[component] : offsets[component + 1]] = np.cos(
                    cycles * phi[pixels]
                )
            return out

        slow, fast = _survives(f, oscillation(8)), _survives(f, oscillation(20))
        assert fast > 0.9
        # A per-stripe degree-1 polynomial can partly fit a slow wave and not a
        # fast one, so the ordering is the check, not the absolute level.
        assert slow < fast

    def test_pole_changes_the_subspace_but_not_the_rank(self, cap):
        kwargs = dict(mask=cap, n_stripes=4, degree=1)
        here = scan_polynomial(pole=POLE, **kwargs)
        moved = scan_polynomial(pole=(POLE[0] + np.radians(30.0), POLE[1]), **kwargs)
        # The whole point of requiring n_stripes: a pole sweep compares
        # estimators of equal dimension, so a difference is geometry and not a
        # rank that shifted underfoot.
        assert here.rank == moved.rank
        overlap = np.linalg.svd(here.W.T @ moved.W, compute_uv=False)
        assert overlap.min() < 0.9

    def test_stripe_encircling_the_pole_warns(self):
        """A cap centred on the scan axis has no cut-free azimuth."""
        disc = hp.query_disc(NSIDE, hp.ang2vec(*POLE), np.radians(25.0))
        mask = np.zeros((NPIX, 1))
        mask[disc, 0] = 1.0
        with pytest.warns(UserWarning, match="not a constant-elevation"):
            scan_polynomial(mask=mask, pole=POLE, n_stripes=4, degree=1)

    def test_thin_stripe_warns_about_unreachable_degree(self, cap):
        with pytest.warns(UserWarning, match="fewer than the"):
            scan_polynomial(mask=cap, pole=POLE, n_stripes=40, degree=3)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"n_stripes": 0}, "at least 1"), ({"degree": -1}, "non-negative")],
    )
    def test_rejects_degenerate_knobs(self, cap, kwargs, match):
        full = {"mask": cap, "pole": POLE, "n_stripes": 4, "degree": 1, **kwargs}
        with pytest.raises(ValueError, match=match):
            scan_polynomial(**full)


class TestHarmonicDeprojection:
    SPINS = [0, 2]

    def test_kills_its_templates_to_template_epsilon(self, cap):
        """In-band survival tracks template_epsilon, not machine precision.

        At the 1e-3 default, template directions weaker than the cut are
        declared dependent and left in place, so an in-band harmonic survives
        at that level. Tightening the cut removes all 16 and drops survival to
        machine precision.
        """
        worst = {}
        for epsilon in (1e-3, 1e-10):
            f = harmonic_deprojection(
                mask=cap,
                spins=self.SPINS,
                ells=range(4),
                slot=Slot.S,
                template_epsilon=epsilon,
            )
            worst[epsilon] = max(
                _survives(f, _harmonic(cap, self.SPINS, ell, m, Slot.S))
                for ell, m in ((0, 0), (2, 1), (3, 3))
            )
        assert worst[1e-3] < 1e-2
        assert worst[1e-10] < 1e-12

    def test_sharpness_in_ell_is_set_by_the_patch_size(self, cap):
        """Deprojecting multipoles on a small patch is not sharp in ell.

        Out-of-band survival rises monotonically and reaches unity only well
        above the patch's resolution scale, ell ~ 180 / radius: the patch
        cannot tell a slowly varying out-of-band harmonic apart from the band
        it nulled. Widening the patch moves the transition down in ell, which
        is what pins the mechanism rather than just the trend.
        """
        wide = np.zeros_like(cap)
        wide[hp.query_disc(NSIDE, hp.ang2vec(*PATCH), np.radians(45.0)), :] = 1.0

        curves = {}
        for name, mask in (("narrow", cap), ("wide", wide)):
            f = harmonic_deprojection(
                mask=mask, spins=self.SPINS, ells=range(4), slot="T"
            )
            curves[name] = [
                _survives(f, _harmonic(mask, self.SPINS, ell, 1, Slot.S))
                for ell in (4, 6, 9, 15, 25)
            ]
            assert np.all(np.diff(curves[name]) > 0.0), name
            assert curves[name][0] < 0.5, name
            assert curves[name][-1] > 0.9, name

        # 18 deg resolves ell ~ 10, 45 deg resolves ell ~ 4, so the wider patch
        # is transparent sooner at every multipole tested.
        assert np.all(np.array(curves["wide"]) > np.array(curves["narrow"]))

    def test_scalar_slot_leaves_polarization_alone(self, cap):
        f = harmonic_deprojection(mask=cap, spins=self.SPINS, ells=range(4), slot="T")
        survival = _survives(f, _harmonic(cap, self.SPINS, 2, 1, Slot.G))
        np.testing.assert_allclose(survival, 1.0, atol=1e-12)

    def test_nulling_e_is_not_purification(self, cap):
        """Removing cut-sky E harmonics removes B power too, and vice versa.

        A cut-sky pure-B harmonic is not orthogonal to the E harmonics, so an
        E-slot deprojection bites into B. This is the numerical statement of the
        docstring's warning that the constructor is not purification.
        """
        f = harmonic_deprojection(mask=cap, spins=self.SPINS, ells=range(2, 5), slot="E")
        for ell, m in ((2, 0), (3, 2)):
            assert _survives(f, _harmonic(cap, self.SPINS, ell, m, Slot.C)) < 0.999

    def test_cmb_and_internal_slot_labels_agree(self, cap):
        kwargs = dict(mask=cap, spins=self.SPINS, ells=range(2, 5))
        by_alias = harmonic_deprojection(slot="B", **kwargs)
        by_letter = harmonic_deprojection(slot=Slot.C, **kwargs)
        np.testing.assert_allclose(by_alias.W, by_letter.W)

    def test_rejects_slot_with_no_matching_field(self, cap):
        mask = cap[:, :1]
        with pytest.raises(ValueError, match="needs a spin-2 field"):
            harmonic_deprojection(mask=mask, spins=[0], ells=range(2, 4), slot="E")

    def test_rejects_multipole_below_the_slot_spin(self, cap):
        with pytest.raises(ValueError, match="does not exist for it"):
            harmonic_deprojection(mask=cap, spins=self.SPINS, ells=[1, 2], slot="E")

    def test_rejects_unknown_slot(self, cap):
        with pytest.raises(ValueError, match="unknown slot"):
            harmonic_deprojection(mask=cap, spins=self.SPINS, ells=range(4), slot="Q")


def test_constructors_agree_on_the_fingerprint(cap):
    """Every constructor records the same active-pixel ordering."""
    scan = scan_polynomial(mask=cap, pole=POLE, n_stripes=4, degree=1)
    band = harmonic_deprojection(mask=cap, spins=[0, 2], ells=range(4), slot="T")
    assert scan.fingerprint == band.fingerprint
    # And they intersect, since both carry Q_removed.
    both = scan & band
    assert isinstance(both, Filter)
    assert both.rank <= min(scan.rank, band.rank)


class TestRangeRank:
    """The second of the two threshold knobs, which no constructor reaches.

    Every shipped constructor supplies ``range_epsilon`` or builds its record
    directly, so ``from_operator(range_rank=...)`` is only ever exercised by a
    caller writing their own operator. It is public and documented, so it is
    tested here rather than left to them.
    """

    def _graded(self, n=40):
        rng = np.random.default_rng(2)
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        return (Q * np.logspace(0, -8, n)) @ Q.T

    def test_range_rank_keeps_exactly_that_many_directions(self):
        # A smooth spectrum has no gap to cut at, so the cut warns and says so.
        with pytest.warns(UserWarning, match="without a spectral gap"):
            f = Filter.from_operator(self._graded(), range_rank=12)
        assert f.rank == 12
        assert f.sigma.size == 12
        assert f.range_rank == 12
        assert f.range_epsilon is None
        assert not f.is_projector

    def test_range_rank_outside_the_spectrum_is_refused(self):
        with pytest.raises(ValueError, match="outside 1"):
            Filter.from_operator(self._graded(n=40), range_rank=41)

    def test_the_two_knobs_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Filter.from_operator(self._graded(), range_epsilon=1e-3, range_rank=5)


class TestFromOperatorRecognisesItsInput:
    """Two behaviours of ``from_operator`` no shipped constructor reaches."""

    def test_a_projection_matrix_comes_back_as_a_projector(self):
        """Unit singular values mean ``U`` aliases ``W``, not a graded record."""
        rng = np.random.default_rng(4)
        Q = np.linalg.qr(rng.standard_normal((30, 8)))[0]
        f = Filter.from_operator(Q @ Q.T, range_epsilon=1e-6)
        assert f.is_projector
        assert f.U is f.W
        assert f.rank == 8
        np.testing.assert_allclose(f.sigma, 1.0)
        # Q_removed is what lets two projectors intersect by complement-of-union.
        assert f.Q_removed is not None and f.Q_removed.shape == (30, 22)

    def test_an_epsilon_below_the_floor_is_clipped_loudly(self):
        """The floor is sqrt(eps) because the restricted covariance carries Σ²."""
        rng = np.random.default_rng(6)
        Q = np.linalg.qr(rng.standard_normal((20, 20)))[0]
        F = (Q * np.logspace(0, -12, 20)) @ Q.T
        with pytest.warns(UserWarning, match="below MIN_RANGE_EPSILON"):
            f = Filter.from_operator(F, range_epsilon=1e-13)
        assert f.range_epsilon == pytest.approx(MIN_RANGE_EPSILON)
