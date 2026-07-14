"""Tests for the ``cls_data=``/``fiducial_cls=`` injection kwargs (ADR-0017, A4).

``cls_data`` shadows ``params.inputclfile`` and forwards into
``FieldCollection.set_cls`` (which already dispatches ``None`` → file read);
``fiducial_cls`` shadows ``params.fiducialfile`` for the S_fixed fiducial re-read.
Injected objects are "exactly what ``readcl`` returns": a ``{label: C_ℓ}`` dict of
physical C_ℓ (no ``input_convention`` conversion — that is the file adapter's job).
"""

import tempfile

import numpy as np

from cosmocore.in_out import readcl

from .test_core import ConcreteCore
from .test_core_pipeline import _make_params


def _cls_dict(core):
    return core.collection.spectra_manager._cls_dict


def test_cls_data_is_named_kwarg():
    """``cls_data``/``fiducial_cls`` are explicit signature parameters on Core."""
    import inspect

    from cosmocore.core import Core

    core_sig = inspect.signature(Core.__init__).parameters
    assert "cls_data" in core_sig
    assert "fiducial_cls" in core_sig


def test_injected_cls_data_matches_file_path():
    """Injecting ``cls_data`` yields the same spectra as reading from the cls file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _make_params(tmpdir, nside=4, lmax=8)

        ref = ConcreteCore(params)
        ref.setup_fields()
        ref.setup_cls(lmax=8)

        injected = readcl(params.inputclfile, params, lmax=8)
        params.inputclfile = "/nonexistent/cls.txt"
        inj = ConcreteCore(params, cls_data=injected)
        inj.setup_fields()
        inj.setup_cls(lmax=8)

        ref_cls, inj_cls = _cls_dict(ref), _cls_dict(inj)
        assert ref_cls.keys() == inj_cls.keys()
        for label in ref_cls:
            np.testing.assert_array_equal(inj_cls[label], ref_cls[label])


def test_beam_smoothing_does_not_mutate_the_caller_spectra():
    """``set_cls`` takes ownership of its input: the caller's arrays stay put.

    Beam smoothing multiplies the stored spectra in place. A shallow copy of the
    input dict would let that reach back into the caller's arrays — re-smoothing
    them on every subsequent ``set_cls`` with the same source.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _make_params(tmpdir, nside=4, lmax=8)

        core = ConcreteCore(params)
        core.setup_fields()

        caller_cls = readcl(params.inputclfile, params, lmax=8)
        pristine = {label: arr.copy() for label, arr in caller_cls.items()}

        core.collection.set_cls(caller_cls, lmax=8)
        core.setup_beams(lmax=8)
        core.collection.beam_manager.apply_smoothing(
            core.collection.spectra_manager, lmax=8
        )

        for label in pristine:
            np.testing.assert_array_equal(caller_cls[label], pristine[label])
