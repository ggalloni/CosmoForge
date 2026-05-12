"""Tests for the label-keyed user-facing accessors (A')."""

import os

import numpy as np
import pytest

from qube import Fisher


@pytest.fixture
def fisher_tqu_ready(config_resolver):
    """Fisher run end-to-end on T+QU so spectra_list and fisher are populated."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file)
    fisher.run()
    os.unlink(config_file)
    return fisher


def test_spectra_list_populated_after_run(fisher_tqu_ready):
    """spectra_list is publicly accessible after run() and matches nspectra."""
    spectra_list = fisher_tqu_ready.spectra_list
    assert spectra_list is not None
    assert len(spectra_list) == fisher_tqu_ready.params.nspectra


def test_get_bandpower_slices_tqu(fisher_tqu_ready):
    """For TQU under SYMMETRIC, the slice map covers TT, EE, BB, EB, TE, TB."""
    slices = fisher_tqu_ready.get_bandpower_slices()
    assert slices is not None
    expected_labels = {"TT", "EE", "BB", "EB", "TE", "TB"}
    assert set(slices.keys()) == expected_labels

    nbins = fisher_tqu_ready.bins.nbins
    # Every block has width nbins and the blocks tile the full axis.
    starts = sorted(s.start for s in slices.values())
    stops = sorted(s.stop for s in slices.values())
    assert starts == [i * nbins for i in range(len(expected_labels))]
    assert stops == [(i + 1) * nbins for i in range(len(expected_labels))]


def test_get_error_bars_as_dict_matches_flat(fisher_tqu_ready):
    """as_dict=True returns the same numbers as the flat array, just sliced."""
    flat = fisher_tqu_ready.get_error_bars()
    by_label = fisher_tqu_ready.get_error_bars(as_dict=True)
    slices = fisher_tqu_ready.get_bandpower_slices()
    for label, slc in slices.items():
        np.testing.assert_array_equal(by_label[label], flat[slc])


def test_get_fisher_block_matches_manual_slicing(fisher_tqu_ready):
    """get_fisher_block(label_i, label_j) is identical to slice indexing."""
    F = fisher_tqu_ready.get_fisher_matrix()
    slices = fisher_tqu_ready.get_bandpower_slices()

    tt_auto = fisher_tqu_ready.get_fisher_block("TT")
    np.testing.assert_array_equal(tt_auto, F[slices["TT"], slices["TT"]])

    te_x_ee = fisher_tqu_ready.get_fisher_block("TE", "EE")
    np.testing.assert_array_equal(te_x_ee, F[slices["TE"], slices["EE"]])

    nbins = fisher_tqu_ready.bins.nbins
    assert tt_auto.shape == (nbins, nbins)
    assert te_x_ee.shape == (nbins, nbins)


def test_get_fisher_block_unknown_label_raises(fisher_tqu_ready):
    with pytest.raises(KeyError, match="Unknown spectrum label"):
        fisher_tqu_ready.get_fisher_block("XX")
    with pytest.raises(KeyError, match="Unknown spectrum label"):
        fisher_tqu_ready.get_fisher_block("TT", "ZZ")
