LikelihoodResult Class
======================

.. currentmodule:: picslike.likelihood_result

The LikelihoodResult class stores and analyzes the results of likelihood computations,
providing methods for best-fit extraction, confidence intervals, and marginalization.

Overview
--------

LikelihoodResult provides:

* Storage of chi-squared and log-likelihood values across parameter grid
* Best-fit parameter extraction from likelihood surface
* Confidence interval computation via profile likelihood
* Parameter marginalization for 1D posteriors
* Result persistence and loading

Class Documentation
-------------------

.. autoclass:: LikelihoodResult
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

Initialization
^^^^^^^^^^^^^^

.. automethod:: LikelihoodResult.__init__
   :no-index:

Best-Fit Analysis
^^^^^^^^^^^^^^^^^

.. automethod:: LikelihoodResult.get_best_fit
   :no-index:
.. automethod:: LikelihoodResult.get_chi_squared_minimum
   :no-index:
.. automethod:: LikelihoodResult.get_maximum_likelihood
   :no-index:

Statistical Analysis
^^^^^^^^^^^^^^^^^^^^

.. automethod:: LikelihoodResult.get_confidence_intervals
   :no-index:
.. automethod:: LikelihoodResult.get_marginalized_likelihood
   :no-index:
.. automethod:: LikelihoodResult.get_summary_statistics
   :no-index:

Persistence
^^^^^^^^^^^

.. automethod:: LikelihoodResult.save
   :no-index:
.. automethod:: LikelihoodResult.load
   :no-index:

Usage Examples
--------------

Creating a LikelihoodResult
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from picslike import LikelihoodResult, ParameterGrid

   # Assuming you have a parameter grid and computed chi-squared values
   chi2_values = np.array([...])  # Chi-squared at each grid point
   log_like_values = -0.5 * chi2_values

   result = LikelihoodResult(
       parameter_grid=grid,
       chi_squared_values=chi2_values,
       log_likelihood_values=log_like_values,
   )

Extracting Best-Fit Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get best-fit parameter values
   best_fit = result.get_best_fit()
   print(f"Best-fit omega_b: {best_fit['omega_b']:.5f}")
   print(f"Best-fit omega_c: {best_fit['omega_c']:.4f}")

   # Get minimum chi-squared
   chi2_min = result.get_chi_squared_minimum()
   print(f"Minimum chi-squared: {chi2_min:.2f}")

Computing Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 68% confidence intervals (1-sigma)
   intervals_68 = result.get_confidence_intervals(0.68)
   for param, (lower, upper) in intervals_68.items():
       print(f"{param}: [{lower:.5f}, {upper:.5f}]")

   # 95% confidence intervals (2-sigma)
   intervals_95 = result.get_confidence_intervals(0.95)

Marginalizing Over Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get 1D marginalized likelihood for omega_b
   marg_omega_b = result.get_marginalized_likelihood("omega_b")

   # Plot marginalized likelihood
   import matplotlib.pyplot as plt

   omega_b_values = grid.parameter_ranges["omega_b"]
   plt.plot(omega_b_values, marg_omega_b)
   plt.xlabel(r"$\Omega_b$")
   plt.ylabel("Marginalized Likelihood")
   plt.show()

Summary Statistics
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get comprehensive summary
   summary = result.get_summary_statistics()

   print(f"Number of parameters: {summary['n_parameters']}")
   print(f"Total grid points: {summary['n_grid_points']}")
   print(f"Chi-squared minimum: {summary['chi2_min']:.2f}")
   print(f"Best-fit values: {summary['best_fit']}")

Saving and Loading Results
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Save results to file
   result.save("output/likelihood_results.npz")

   # Load results from file
   loaded_result = LikelihoodResult.load("output/likelihood_results.npz")

   # Verify loaded correctly
   assert loaded_result.get_chi_squared_minimum() == result.get_chi_squared_minimum()

Result Format
-------------

The LikelihoodResult stores:

* **parameter_grid**: The ParameterGrid used for computation
* **chi_squared_values**: 1D array of chi-squared values indexed by grid point
* **log_likelihood_values**: 1D array of log-likelihood values

File Format
^^^^^^^^^^^

Results are saved in NumPy's compressed archive format (``.npz``):

.. code-block:: python

   # Contents of saved file
   {
       "chi_squared_values": np.ndarray,
       "log_likelihood_values": np.ndarray,
       "parameter_names": list,
       "parameter_ranges": dict,
       # ... additional metadata
   }

Statistical Notes
-----------------

Confidence Intervals
^^^^^^^^^^^^^^^^^^^^

Confidence intervals are computed using the profile likelihood method:

.. math::

   \Delta \chi^2 = \chi^2(\theta) - \chi^2_{min}

For a single parameter, the confidence levels correspond to:

* 68% (1σ): :math:`\Delta \chi^2 < 1.0`
* 95% (2σ): :math:`\Delta \chi^2 < 4.0`
* 99% (3σ): :math:`\Delta \chi^2 < 9.0`

Marginalization
^^^^^^^^^^^^^^^

Marginalized likelihoods are computed by integrating over nuisance parameters:

.. math::

   \mathcal{L}_{marg}(\theta_i) = \int \mathcal{L}(\theta_i, \theta_j) d\theta_j

For grid-based evaluation, this is approximated as a sum over grid points.

See Also
--------

* :class:`picslike.picslike.PICSLike` : Main likelihood computation class
* :class:`picslike.parameter_grid.ParameterGrid` : Parameter space management
