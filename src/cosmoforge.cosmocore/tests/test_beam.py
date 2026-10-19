"""Tests for the beam seam: ``coswinbeam`` conventions + ``beam=`` injection (ADR-0017).

No dedicated beam test file existed before A5 (review Part II gap); this covers the
two cosine-window conventions and the new array-injection path together.

``beam`` shadows ``params.beam_file`` (the ``smoothing_type="file"`` adapter). The
injected object is exactly what ``hp.read_cl(beam_file)`` returns: a 2D float array with
at least 3 rows (T, E, B window functions), pixwin already folded in. An injected beam
wins over ``smoothing_type`` (explicit injection wins, as for every input seam).
"""

import tempfile

import healpy as hp
import numpy as np
import pytest

from cosmocore.beam import coswinbeam

from .test_core import ConcreteCore


def _beam_params(beam_file, *, nside=4, lmax=8):
    from cosmocore.settings import InputParams

    params = InputParams()
    params.nside = nside
    params.lmax = lmax
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"
    params.smoothing_type = "file"
    params.beam_file = beam_file

    npix = 12 * nside**2
    mask = np.ones(npix, dtype=np.float64)
    mask_f = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    hp.write_map(mask_f.name, mask, overwrite=True)
    params.maskfile = mask_f.name
    return params


def _write_beam(path, nell, seed=0):
    """Write a (3, nell) T/E/B beam window file (as hp.read_cl reads back)."""
    rng = np.random.default_rng(seed)
    bls = np.abs(rng.standard_normal((3, nell))).astype(np.float64)
    hp.write_cl(path, [bls[0], bls[1], bls[2]], overwrite=True)


def test_injected_beam_matches_file_path():
    """compute_beams(injected_beam=A) equals compute_beams reading A from a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        beam_file = f"{tmpdir}/beam.fits"
        _write_beam(beam_file, nell=32)
        params = _beam_params(beam_file)

        core = ConcreteCore(params)
        core.setup_fields()
        bm = core.collection.beam_manager

        ref = bm.compute_beams(
            lmax=8, nside=4, smoothtype="file", fwhmarcmin=0.0, beam_file=beam_file
        )
        injected = hp.read_cl(beam_file).astype(np.float64)
        inj = bm.compute_beams(
            lmax=8,
            nside=4,
            smoothtype="file",
            fwhmarcmin=0.0,
            beam_file="/nonexistent/beam.fits",
            injected_beam=injected,
        )
        assert ref.keys() == inj.keys()
        for k in ref:
            np.testing.assert_array_equal(inj[k], ref[k])


def test_injected_beam_wins_over_smoothtype():
    """An injected beam is used even when ``smoothtype`` is not ``"file"``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        beam_file = f"{tmpdir}/beam.fits"
        _write_beam(beam_file, nell=32)
        params = _beam_params(beam_file)
        core = ConcreteCore(params)
        core.setup_fields()
        bm = core.collection.beam_manager

        injected = hp.read_cl(beam_file).astype(np.float64)
        out = bm.compute_beams(
            lmax=8,
            nside=4,
            smoothtype="gaussian",  # would compute an analytic beam without injection
            fwhmarcmin=60.0,
            beam_file="",
            injected_beam=injected,
        )
        np.testing.assert_array_equal(out["T"], injected[0][:9])


def test_injected_beam_too_few_rows_raises():
    """A beam with fewer than 3 rows fails loudly at the convergence point."""
    with tempfile.TemporaryDirectory() as tmpdir:
        beam_file = f"{tmpdir}/beam.fits"
        _write_beam(beam_file, nell=32)
        params = _beam_params(beam_file)
        core = ConcreteCore(params)
        core.setup_fields()
        bm = core.collection.beam_manager

        bad = np.ones((2, 32), dtype=np.float64)
        import pytest

        with pytest.raises(ValueError, match="at least 3 rows"):
            bm.compute_beams(
                lmax=8,
                nside=4,
                smoothtype="file",
                fwhmarcmin=0.0,
                beam_file="",
                injected_beam=bad,
            )


def test_beam_is_named_kwarg():
    """``beam`` is an explicit signature parameter on Core."""
    import inspect

    from cosmocore.core import Core

    assert "beam" in inspect.signature(Core.__init__).parameters


def test_coswinbeam_conventions_differ():
    """Legacy (ell1=nside) and NPIPE (ell1=1) cosine windows are distinct.

    Characterisation test closing the pre-A5 beam test gap: both are monotone-ish
    windows in [0, 1] that start at 1 and taper to 0 by ell2=3*nside, and the
    NPIPE variant (which starts tapering from ell=1) is never above legacy.
    """
    nside = 8
    legacy = coswinbeam(nside, ell1=nside, ell2=3 * nside)
    npipe = coswinbeam(nside, ell1=1, ell2=3 * nside)
    assert legacy.shape == npipe.shape
    for b in (legacy, npipe):
        assert b[0] == 1.0
        assert 0.0 <= b[3 * nside] <= 1e-12 or b[3 * nside] <= b[nside]
        assert np.all(b >= -1e-12) and np.all(b <= 1.0 + 1e-12)
    # NPIPE tapers earlier, so at the legacy plateau edge it is already lower.
    assert npipe[nside] <= legacy[nside] + 1e-12
    assert not np.array_equal(legacy, npipe)


def _analytic_beams(smoothtype, apply_pixwin, *, nside=4, lmax=12):
    from cosmocore.beam import BeamManager

    return BeamManager([]).compute_beams(
        lmax=lmax,
        nside=nside,
        smoothtype=smoothtype,
        fwhmarcmin=60.0,
        beam_file="",
        apply_pixwin=apply_pixwin,
    )


@pytest.mark.parametrize(
    "smoothtype", ["none", "gaussian", "cosine_legacy", "cosine_npipe"]
)
def test_apply_pixwin_multiplies_analytic_beam_by_hp_pixwin(smoothtype):
    """apply_pixwin turns b_ℓ into b_ℓ·p_ℓ: T by the T window, E and B by the pol one."""
    off = _analytic_beams(smoothtype, apply_pixwin=False)
    on = _analytic_beams(smoothtype, apply_pixwin=True)
    pw_t, pw_p = hp.pixwin(4, pol=True, lmax=12)

    np.testing.assert_array_equal(on["T"], off["T"] * pw_t)
    np.testing.assert_array_equal(on["E"], off["E"] * pw_p)
    np.testing.assert_array_equal(on["B"], off["B"] * pw_p)
    assert not np.array_equal(on["T"], off["T"])


def test_apply_pixwin_is_read_from_params():
    """The InputParams flag reaches the field beams (it was inert before 1.4.0)."""
    from cosmocore.settings import InputParams

    beams = {}
    for flag in (False, True):
        params = InputParams()
        params.nside = 4
        params.lmax = 8
        params.smoothing_type = "gaussian"
        params.fwhmarcmin = 60.0
        params.apply_pixwin = flag
        mask_f = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        mask = np.ones(12 * 4**2)
        hp.write_map(mask_f.name, [mask] * params.nfields, overwrite=True)
        params.maskfile = mask_f.name

        core = ConcreteCore(params)
        core.setup_fields()
        bm = core.collection.beam_manager
        bm.set_beams_from_params(params)
        beams[flag] = bm.get_beam_dict()

    pw_t, pw_p = hp.pixwin(4, pol=True, lmax=8)
    np.testing.assert_array_equal(beams[True]["T"], beams[False]["T"] * pw_t)
    np.testing.assert_array_equal(beams[True]["E"], beams[False]["E"] * pw_p)
    np.testing.assert_array_equal(beams[True]["B"], beams[False]["B"] * pw_p)


def test_apply_pixwin_ignored_for_file_and_injected_beam():
    """A beam from file or injected is used as given; apply_pixwin only warns."""
    from cosmocore.beam import BeamManager

    with tempfile.TemporaryDirectory() as tmpdir:
        beam_file = f"{tmpdir}/beam.fits"
        _write_beam(beam_file, nell=32)
        bls = hp.read_cl(beam_file).astype(np.float64)
        bm = BeamManager([])

        for kwargs in (
            {"smoothtype": "file", "beam_file": beam_file},
            {"smoothtype": "gaussian", "beam_file": "", "injected_beam": bls},
        ):
            with pytest.warns(UserWarning, match="apply_pixwin is ignored"):
                out = bm.compute_beams(
                    lmax=8, nside=4, fwhmarcmin=60.0, apply_pixwin=True, **kwargs
                )
            np.testing.assert_array_equal(out["T"], bls[0][:9])
            np.testing.assert_array_equal(out["E"], bls[1][:9])


def test_apply_pixwin_beyond_pixwin_table_raises():
    """healpy tabulates the pixel window only to 4*nside; past it, fail clearly."""
    with pytest.raises(ValueError, match="4\\*nside=16"):
        _analytic_beams("gaussian", apply_pixwin=True, nside=4, lmax=20)
