"""
Tests for PICSLike class.

This module contains unit and integration tests for the PICSLike class
functionality including initialization, signal matrix computation,
likelihood computation, and result management.

Note: Many PICSLike operations require MPI. These tests are designed to
run in single-process mode (mpirun -n 1) or with MPI mocked where necessary.
"""

import os

import numpy as np
import pytest

from picslike import LikelihoodResult, PICSLike


class TestPICSLikeInit:
    """Test suite for PICSLike initialization."""

    def test_initialization_from_config(self, fast_config_path):
        """Test PICSLike initialization from config file."""
        picslike = PICSLike(params_file=fast_config_path)

        assert picslike is not None
        assert picslike.rank == 0  # Single process
        assert picslike.size == 1  # Single process
        assert picslike.maps1 is None  # Not loaded yet
        assert picslike.parameter_grid is None  # Not set up yet

    def test_initialization_attributes(self, fast_config_path):
        """Test that initialization sets expected attributes."""
        picslike = PICSLike(params_file=fast_config_path)

        assert hasattr(picslike, "comm")
        assert hasattr(picslike, "rank")
        assert hasattr(picslike, "size")
        assert hasattr(picslike, "maps1")
        assert hasattr(picslike, "maps2")
        assert hasattr(picslike, "parameter_grid")
        assert hasattr(picslike, "likelihood_result")
        assert hasattr(picslike, "simulation_index")

    def test_simulation_index_default(self, fast_config_path):
        """Test default simulation index is 0."""
        picslike = PICSLike(params_file=fast_config_path)
        assert picslike.simulation_index == 0


class TestSetSimulationIndex:
    """Test suite for set_simulation_index method."""

    def test_set_valid_simulation_index(self, fast_config_path):
        """Test setting a valid simulation index."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_maps()

        picslike.set_simulation_index(5)
        assert picslike.simulation_index == 5

    def test_set_simulation_index_out_of_range(self, fast_config_path):
        """Test error handling for out-of-range simulation index."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_maps()

        with pytest.raises(ValueError, match="out of range"):
            picslike.set_simulation_index(9999)

    def test_set_negative_simulation_index(self, fast_config_path):
        """Test error handling for negative simulation index."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_maps()

        with pytest.raises(ValueError, match="out of range"):
            picslike.set_simulation_index(-1)


class TestSetupParameterGrid:
    """Test suite for parameter grid setup."""

    def test_setup_parameter_grid(self, fast_config_path):
        """Test parameter grid setup from config."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_parameter_grid()

        assert picslike.parameter_grid is not None
        assert picslike.parameter_grid.get_total_points() > 0

    def test_parameter_names_set(self, fast_config_path):
        """Test that parameter names are extracted from config."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_parameter_grid()

        assert hasattr(picslike, "parameter_names")

    def test_parameter_ranges_set(self, fast_config_path):
        """Test that parameter ranges are set from config."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_parameter_grid()

        assert hasattr(picslike, "parameter_ranges")
        assert len(picslike.parameter_ranges) > 0


class TestSetupMaps:
    """Test suite for map setup."""

    def test_setup_maps_loads_data(self, fast_config_path):
        """Test that setup_maps loads map data."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_maps()

        assert picslike.maps1 is not None
        assert picslike.maps1.shape[1] == picslike.params.nsims

    def test_setup_maps_before_geometry_raises(self, fast_config_path):
        """Test that setup_maps fails if geometry not set up."""
        picslike = PICSLike(params_file=fast_config_path)

        with pytest.raises(ValueError, match="Pixel information not available"):
            picslike.setup_maps()


class TestComputeSignalMatrix:
    """Test suite for signal matrix computation (requires full pipeline)."""

    def test_compute_signal_matrix_requires_covariance(self, fast_config_path):
        """Test that compute_signal_matrix requires covariance setup."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_parameter_grid()

        param_point = list(picslike.parameter_grid.grid_points)[0]
        with pytest.raises((ValueError, AttributeError)):
            picslike.compute_signal_matrix(param_point)

    def test_compute_signal_matrix_shape(self, fast_config_path):
        """Test that signal matrix has correct shape."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_parameter_grid()

        param_point = list(picslike.parameter_grid.grid_points)[0]
        signal_matrix = picslike.compute_signal_matrix(param_point)

        assert signal_matrix.shape[0] == signal_matrix.shape[1]
        assert signal_matrix.shape == picslike.noise_cov1.shape


class TestPrepareCovariance:
    """Test suite for covariance preparation (requires full pipeline)."""

    def test_prepare_covariance_creates_inverse(self, fast_config_path):
        """Test that prepare_covariance creates inverse covariance."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_parameter_grid()

        param_point = list(picslike.parameter_grid.grid_points)[0]
        picslike.compute_signal_matrix(param_point)
        picslike.prepare_covariance_matrix()

        assert hasattr(picslike, "inv_cov")
        assert picslike.inv_cov is not None


class TestGetters:
    """Test suite for getter methods before computation."""

    def test_get_chi_squared_before_compute_raises(self, fast_config_path):
        """Test that get_chi_squared raises error if not computed."""
        picslike = PICSLike(params_file=fast_config_path)

        with pytest.raises(RuntimeError, match="Likelihood not computed"):
            picslike.get_chi_squared()

    def test_get_log_likelihood_before_compute_raises(self, fast_config_path):
        """Test that get_log_likelihood raises error if not computed."""
        picslike = PICSLike(params_file=fast_config_path)

        with pytest.raises(RuntimeError, match="Likelihood not computed"):
            picslike.get_log_likelihood()

    def test_get_best_fit_before_compute_raises(self, fast_config_path):
        """Test that get_best_fit raises error if not computed."""
        picslike = PICSLike(params_file=fast_config_path)

        with pytest.raises(RuntimeError, match="Likelihood not computed"):
            picslike.get_best_fit()

    def test_get_simulation_results_before_compute_raises(self, fast_config_path):
        """Test that get_simulation_results raises error if not computed."""
        picslike = PICSLike(params_file=fast_config_path)

        with pytest.raises(RuntimeError, match="Simulation results not available"):
            picslike.get_simulation_results()

    def test_get_mean_likelihood_result_before_compute_raises(self, fast_config_path):
        """Test that get_mean_likelihood_result raises error if not computed."""
        picslike = PICSLike(params_file=fast_config_path)

        with pytest.raises(RuntimeError, match="Likelihood not computed"):
            picslike.get_mean_likelihood_result()


class TestSaveResults:
    """Test suite for save_results method."""

    def test_save_results_before_compute_raises(self, fast_config_path, temp_output_dir):
        """Test that save_results raises error if not computed."""
        picslike = PICSLike(params_file=fast_config_path)

        with pytest.raises(RuntimeError, match="No results to save"):
            picslike.save_results(str(temp_output_dir / "results.pkl"))


class TestComputeMeanLikelihoodResult:
    """Test suite for mean likelihood computation."""

    def test_compute_mean_empty_list_raises(self, fast_config_path):
        """Test that empty results list raises error."""
        picslike = PICSLike(params_file=fast_config_path)

        with pytest.raises(ValueError, match="No simulation results provided"):
            picslike._compute_mean_likelihood_result([])

    def test_compute_mean_single_result(
        self,
        fast_config_path,
        sample_likelihood_result,
    ):
        """Test mean computation with single result."""
        picslike = PICSLike(params_file=fast_config_path)

        mean_result = picslike._compute_mean_likelihood_result([sample_likelihood_result])

        np.testing.assert_array_equal(
            mean_result.chi_squared_values,
            sample_likelihood_result.chi_squared_values,
        )

    def test_compute_mean_multiple_results(
        self,
        fast_config_path,
        sample_parameter_grid,
    ):
        """Test mean computation with multiple results."""
        picslike = PICSLike(params_file=fast_config_path)

        n_points = sample_parameter_grid.get_total_points()

        chi2_1 = np.ones(n_points) * 100
        chi2_2 = np.ones(n_points) * 200

        result1 = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=chi2_1,
            log_likelihood_values=-0.5 * chi2_1,
        )
        result2 = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=chi2_2,
            log_likelihood_values=-0.5 * chi2_2,
        )

        mean_result = picslike._compute_mean_likelihood_result([result1, result2])

        expected_mean = np.ones(n_points) * 150
        np.testing.assert_array_almost_equal(
            mean_result.chi_squared_values,
            expected_mean,
        )


class TestIntegration:
    """Integration tests for full pipeline.

    These tests use fast_config with a 2x2 grid and 10 simulations for speed.
    Tests are ordered to build on each other:
    - test_full_pipeline covers setup + single-point + full compute + getters + save
    - test_run_method tests the all-in-one run() method
    """

    def test_full_pipeline(self, fast_config_path, temp_output_dir):
        """Test full pipeline: setup, single-point, compute, getters, save."""
        picslike = PICSLike(params_file=fast_config_path)

        # Setup pipeline
        picslike.setup_parameter_grid()
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_maps()

        # Verify setup completed
        assert picslike.parameter_grid is not None
        assert picslike.maps1 is not None
        assert picslike.noise_cov1 is not None
        assert picslike.collection is not None

        # Single-point likelihood
        param_point = list(picslike.parameter_grid.grid_points)[0]
        chi2, log_like = picslike._compute_likelihood_point(param_point)

        assert len(chi2) == picslike.params.nsims
        assert len(log_like) == picslike.params.nsims
        assert np.all(chi2 > 0)
        assert np.all(np.isfinite(log_like))

        # Verify log-likelihood formula consistency across simulations
        if len(chi2) > 1:
            delta_chi2 = chi2[1:] - chi2[:-1]
            delta_log_like = log_like[1:] - log_like[:-1]
            np.testing.assert_array_almost_equal(delta_log_like, -0.5 * delta_chi2)

        # Full compute
        picslike.compute()

        assert picslike.likelihood_result is not None

        # Getters after computation
        chi2_all = picslike.get_chi_squared()
        assert chi2_all is not None
        assert len(chi2_all) == picslike.parameter_grid.get_total_points()

        log_like_all = picslike.get_log_likelihood()
        assert log_like_all is not None
        assert len(log_like_all) == picslike.parameter_grid.get_total_points()

        best_fit = picslike.get_best_fit()
        assert isinstance(best_fit, dict)
        assert len(best_fit) > 0

        sim_results = picslike.get_simulation_results()
        assert sim_results is not None
        assert len(sim_results) == picslike.params.nsims

        mean_result = picslike.get_mean_likelihood_result()
        assert mean_result is not None
        assert mean_result == picslike.likelihood_result

        # Save results
        output_path = temp_output_dir / "test_results.npz"
        picslike.save_results(str(output_path))

        assert output_path.exists()
        for i in range(picslike.params.nsims):
            sim_path = temp_output_dir / f"test_results_sim_{i:02d}.npz"
            assert sim_path.exists()

    def test_run_method(self, fast_config_path):
        """Test the full run() pipeline method."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.run()

        assert picslike.likelihood_result is not None
        assert picslike.simulation_results is not None
        assert picslike.parameter_grid is not None


class TestMPIDistribution:
    """Test suite for MPI point distribution."""

    def test_points_distribution_single_process(self, fast_config_path):
        """Test that single process gets all points."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_parameter_grid()

        points = picslike.parameter_grid.get_points_for_process(0, 1)
        assert len(points) == picslike.parameter_grid.get_total_points()

    def test_points_distribution_multiple_processes_simulation(self, fast_config_path):
        """Test simulated multi-process distribution."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_parameter_grid()

        size = 4
        all_points = []

        for rank in range(size):
            points = picslike.parameter_grid.get_points_for_process(rank, size)
            all_points.extend(points)

        total_points = picslike.parameter_grid.get_total_points()
        assert len(all_points) == total_points
        assert len(set(all_points)) == total_points


class TestCompressedLikelihood:
    """Test suite for compressed likelihood computation."""

    def test_compressed_likelihood_consistency_tqu(self, fast_config_path):
        """Test compressed vs traditional for TQU (multi-field) analysis.

        Validates the spin-2 compressed path by comparing against
        the full pixel-space computation. Also verifies that compressed
        pipeline runs without errors for multi-field configs.
        """
        # Run without compression (traditional pixel-space)
        picslike_standard = PICSLike(params_file=fast_config_path)
        picslike_standard.setup_parameter_grid()
        picslike_standard.setup_fields()
        picslike_standard.setup_geometry()
        picslike_standard.setup_covariance_matrices()
        picslike_standard.setup_cls(lmax=picslike_standard.lmax_signal)
        picslike_standard.setup_beams(lmax=picslike_standard.lmax_signal)
        picslike_standard.setup_maps()

        # Run with compression (spin-2 aware SMW path)
        picslike_compressed = PICSLike(
            params_file=fast_config_path,
            compression={"method": "harmonic"},
        )
        picslike_compressed.setup_parameter_grid()
        picslike_compressed.setup_fields()
        picslike_compressed.setup_geometry()
        picslike_compressed.setup_covariance_matrices()
        picslike_compressed.setup_cls(lmax=picslike_compressed.lmax_signal)
        picslike_compressed.setup_beams(lmax=picslike_compressed.lmax_signal)
        picslike_compressed.setup_computation_basis(method="harmonic")
        picslike_compressed.setup_maps()

        # Verify compression was set up
        assert picslike_compressed.basis_manager is not None

        # Compute likelihood at a single point for comparison
        param_point = list(picslike_standard.parameter_grid.grid_points)[0]

        chi2_standard, log_like_standard = picslike_standard._compute_likelihood_point(
            param_point
        )
        chi2_compressed, log_like_compressed = (
            picslike_compressed._compute_likelihood_point(param_point)
        )

        # Extract first simulation's values
        chi2_std = (
            chi2_standard[0] if hasattr(chi2_standard, "__len__") else chi2_standard
        )
        chi2_comp = (
            chi2_compressed[0] if hasattr(chi2_compressed, "__len__") else chi2_compressed
        )
        log_std = (
            log_like_standard[0]
            if hasattr(log_like_standard, "__len__")
            else log_like_standard
        )
        log_comp = (
            log_like_compressed[0]
            if hasattr(log_like_compressed, "__len__")
            else log_like_compressed
        )

        # Chi-squared and log-likelihood should match
        rel_diff_chi2 = abs(chi2_comp - chi2_std) / abs(chi2_std)
        assert rel_diff_chi2 < 1e-7, (
            f"TQU chi-squared relative difference too large: {rel_diff_chi2:.2e}. "
            f"Standard={chi2_std:.6f}, Compressed={chi2_comp:.6f}"
        )

        rel_diff_log = abs(log_comp - log_std) / abs(log_std)
        assert rel_diff_log < 1e-7, (
            f"TQU log-likelihood relative difference too large: {rel_diff_log:.2e}. "
            f"Standard={log_std:.6f}, Compressed={log_comp:.6f}"
        )

    def test_compressed_likelihood_consistency_tqu_do_cross(self, fast_config_path):
        """Compressed vs traditional pixel-space for ``do_cross=True``.

        Regression for the SMW cross-quadratic identity
        ``d1^T C^{-1} d2 = d1^T N^{-1} d2 - y1^T K^{-1} y2`` (with
        ``y_i = V N^{-1} d_i``) in the harmonic-basis fast path. We feed
        distinct ``maps2`` (``maps1`` plus a fixed-seed perturbation) so
        the cross-quadratic does NOT reduce to the auto case — a bug
        that ignored ``maps2`` or reused ``projected1`` for both sides
        would diverge from the traditional pixel-space cross. The
        compressed SMW path and the pixel-space cross must agree to
        numerical precision.
        """
        rng = np.random.default_rng(20260514)

        def _make_picslike(compression=None):
            pls = PICSLike(params_file=fast_config_path, compression=compression)
            pls.setup_parameter_grid()
            pls.setup_fields()
            pls.setup_geometry()
            pls.setup_covariance_matrices()
            pls.setup_cls(lmax=pls.lmax_signal)
            pls.setup_beams(lmax=pls.lmax_signal)
            if compression is not None:
                pls.setup_computation_basis(method=compression["method"])
            pls.setup_maps()
            return pls

        picslike_standard = _make_picslike(compression=None)
        picslike_compressed = _make_picslike(compression={"method": "harmonic"})

        # Build maps2 = maps1 + small perturbation, identical across the
        # two PICSLike instances so the cross-quadratic has the same
        # ground truth on both paths.
        perturbation = 0.01 * rng.standard_normal(picslike_standard.maps1.shape)
        for pls in (picslike_standard, picslike_compressed):
            pls.params.do_cross = True
            pls.maps2 = pls.maps1 + perturbation

        assert picslike_compressed.basis_manager is not None

        param_point = list(picslike_standard.parameter_grid.grid_points)[0]
        chi2_std, log_std = picslike_standard._compute_likelihood_point(param_point)
        chi2_comp, log_comp = picslike_compressed._compute_likelihood_point(param_point)

        chi2_std0 = chi2_std[0] if hasattr(chi2_std, "__len__") else chi2_std
        chi2_comp0 = chi2_comp[0] if hasattr(chi2_comp, "__len__") else chi2_comp
        log_std0 = log_std[0] if hasattr(log_std, "__len__") else log_std
        log_comp0 = log_comp[0] if hasattr(log_comp, "__len__") else log_comp

        rel_diff_chi2 = abs(chi2_comp0 - chi2_std0) / abs(chi2_std0)
        assert rel_diff_chi2 < 1e-7, (
            f"do_cross chi-squared relative difference too large: "
            f"{rel_diff_chi2:.2e}. "
            f"Standard={chi2_std0:.6f}, Compressed={chi2_comp0:.6f}"
        )

        rel_diff_log = abs(log_comp0 - log_std0) / abs(log_std0)
        assert rel_diff_log < 1e-7, (
            f"do_cross log-likelihood relative difference too large: "
            f"{rel_diff_log:.2e}. "
            f"Standard={log_std0:.6f}, Compressed={log_comp0:.6f}"
        )

    def test_compressed_likelihood_pixel_basis_smoke(self, fast_config_path):
        """Smoke test for the polymorphic pixel-basis use_basis path.

        Verifies that ``_compute_likelihood_point`` runs end-to-end with
        ``compression={"method": "pixel"}`` — exercising the ABC
        polymorphic branch (``bm.get_inverse`` / ``bm.to_basis`` /
        ``bm.get_logdet``) added to picslike alongside the harmonic SMW
        path. Pre-fix this branch would have ``AttributeError``'d on
        ``bm._V_N_inv``. Numerical consistency vs the traditional
        pixel-space path is checked at a relaxed tolerance because the
        eigenmode truncation is lossy in general.
        """
        picslike_standard = PICSLike(params_file=fast_config_path)
        picslike_standard.setup_parameter_grid()
        picslike_standard.setup_fields()
        picslike_standard.setup_geometry()
        picslike_standard.setup_covariance_matrices()
        picslike_standard.setup_cls(lmax=picslike_standard.lmax_signal)
        picslike_standard.setup_beams(lmax=picslike_standard.lmax_signal)
        picslike_standard.setup_maps()

        picslike_pixel = PICSLike(
            params_file=fast_config_path,
            compression={"method": "pixel", "epsilon": 1e-12},
        )
        picslike_pixel.setup_parameter_grid()
        picslike_pixel.setup_fields()
        picslike_pixel.setup_geometry()
        picslike_pixel.setup_covariance_matrices()
        picslike_pixel.setup_cls(lmax=picslike_pixel.lmax_signal)
        picslike_pixel.setup_beams(lmax=picslike_pixel.lmax_signal)
        picslike_pixel.setup_computation_basis(method="pixel", epsilon=1e-12)
        picslike_pixel.setup_maps()

        assert picslike_pixel.basis_manager is not None
        assert picslike_pixel.basis_manager.method == "pixel"

        param_point = list(picslike_standard.parameter_grid.grid_points)[0]
        chi2_std, log_std = picslike_standard._compute_likelihood_point(param_point)
        chi2_pix, log_pix = picslike_pixel._compute_likelihood_point(param_point)

        chi2_std0 = chi2_std[0] if hasattr(chi2_std, "__len__") else chi2_std
        chi2_pix0 = chi2_pix[0] if hasattr(chi2_pix, "__len__") else chi2_pix
        log_pix0 = log_pix[0] if hasattr(log_pix, "__len__") else log_pix

        # Smoke: no AttributeError, finite values.
        assert np.isfinite(chi2_pix0)
        assert np.isfinite(log_pix0)
        # Numerical agreement with traditional pixel-space (eigenvalue
        # truncation at 1e-12 keeps essentially all modes for nside=4).
        rel_diff_chi2 = abs(chi2_pix0 - chi2_std0) / abs(chi2_std0)
        assert rel_diff_chi2 < 1e-3, (
            f"pixel-basis chi-squared relative difference too large: "
            f"{rel_diff_chi2:.2e}. "
            f"Standard={chi2_std0:.6f}, Pixel={chi2_pix0:.6f}"
        )

    @pytest.mark.slow
    def test_compressed_likelihood_consistency_single_field(self, local_path):
        """Test compressed vs traditional for B-only (single-field, nside=8).

        Exercises the inference-window optimisation: S_fixed absorbs
        multipoles outside ``[lmin, lmax]``; SMW handles the varying band.
        """
        import tempfile

        import yaml

        config_path = os.path.join(
            local_path, "tests/data/nside8/B/fortran_reference/config.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            if isinstance(value, str) and value.startswith("../tests/"):
                config[key] = os.path.join(local_path, value.replace("../", ""))

        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, temp_config, default_flow_style=False)
        temp_config.close()
        b_config = temp_config.name

        # Run without compression
        picslike_standard = PICSLike(params_file=b_config)
        picslike_standard.setup_parameter_grid()
        picslike_standard.setup_fields()
        picslike_standard.setup_geometry()
        picslike_standard.setup_covariance_matrices()
        picslike_standard.setup_cls(lmax=picslike_standard.lmax_signal)
        picslike_standard.setup_beams(lmax=picslike_standard.lmax_signal)
        picslike_standard.setup_maps()

        # Run with compression
        picslike_compressed = PICSLike(
            params_file=b_config,
            compression={"method": "harmonic"},
        )
        picslike_compressed.setup_parameter_grid()
        picslike_compressed.setup_fields()
        picslike_compressed.setup_geometry()
        picslike_compressed.setup_covariance_matrices()
        picslike_compressed.setup_cls(lmax=picslike_compressed.lmax_signal)
        picslike_compressed.setup_beams(lmax=picslike_compressed.lmax_signal)
        picslike_compressed.setup_computation_basis(method="harmonic")
        picslike_compressed.setup_maps()

        assert picslike_compressed.basis_manager is not None
        assert picslike_compressed.basis_manager._use_switch_optimization

        param_point = list(picslike_standard.parameter_grid.grid_points)[1]

        chi2_standard, log_like_standard = picslike_standard._compute_likelihood_point(
            param_point
        )
        chi2_compressed, log_like_compressed = (
            picslike_compressed._compute_likelihood_point(param_point)
        )

        chi2_std = (
            chi2_standard[0] if hasattr(chi2_standard, "__len__") else chi2_standard
        )
        chi2_comp = (
            chi2_compressed[0] if hasattr(chi2_compressed, "__len__") else chi2_compressed
        )
        log_std = (
            log_like_standard[0]
            if hasattr(log_like_standard, "__len__")
            else log_like_standard
        )
        log_comp = (
            log_like_compressed[0]
            if hasattr(log_like_compressed, "__len__")
            else log_like_compressed
        )

        rel_diff_chi2 = abs(chi2_comp - chi2_std) / abs(chi2_std)
        assert rel_diff_chi2 < 1e-10, (
            f"Chi-squared relative difference too large: {rel_diff_chi2:.2e}. "
            f"Standard={chi2_std:.6f}, Compressed={chi2_comp:.6f}"
        )

        rel_diff_log = abs(log_comp - log_std) / abs(log_std)
        assert rel_diff_log < 1e-10, (
            f"Log-likelihood relative difference too large: {rel_diff_log:.2e}. "
            f"Standard={log_std:.6f}, Compressed={log_comp:.6f}"
        )
