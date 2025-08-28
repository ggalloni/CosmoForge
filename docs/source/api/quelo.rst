CosmoForge.Quelo Package
========================

Quelo is the analysis engine of CosmoForge, implementing Fisher matrix analysis and 
Quadratic Maximum Likelihood (QML) power spectrum estimation for cosmological parameter 
inference from CMB data.

Overview
--------

Quelo provides two main analysis methods:

* **Fisher Matrix Analysis**: Fast parameter forecasting and covariance estimation
* **QML Power Spectrum Estimation**: Optimal power spectrum recovery from noisy data

Both methods support MPI parallelization for large-scale analyses and can handle 
temperature and polarization data with realistic instrumental effects.

Key Features
------------

Fisher Matrix Analysis
^^^^^^^^^^^^^^^^^^^^^^^

* Fast parameter forecasting
* Covariance matrix computation  
* Support for multiple field types (T, E, B)
* Cross-correlation analysis
* Instrumental noise modeling

QML Power Spectrum Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Optimal power spectrum estimation
* Noise bias correction
* Cross-correlation support
* Fisher matrix renormalization
* Mock data generation capabilities

Package Structure
-----------------

.. autosummary::
   :toctree: generated/
   :recursive:

   cosmoforge.quelo

Main Modules
------------

.. note::
   Full API documentation for Quelo package modules will be added as the package 
   documentation is completed.
