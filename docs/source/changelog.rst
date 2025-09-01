Changelog
=========

All notable changes to CosmoForge will be documented here.

Version 0.1.0 (In Development)
-------------------------------

**Features:**

* Initial release of CosmoForge framework
* Complete cosmocore package with mathematical utilities
* HEALPix integration for spherical harmonic analysis
* Numba-optimized computational functions
* YAML-based configuration system
* **Automatic field label expansion**: Support for concatenated field labels
  (e.g., "QU" → ["Q", "U"]) and underscore-separated multi-character fields
  (e.g., "T1_T2" → ["T1", "T2"])
* Comprehensive documentation with Sphinx

**API:**

* ``cosmocore.basics``: Mathematical utilities and Legendre polynomials
* ``cosmocore.core``: Abstract base classes for analysis workflows
* ``cosmocore.fields``: Cosmological field representations
* ``cosmocore.harmonic``: Spherical harmonic transforms and power spectra
* ``cosmocore.in_out``: I/O operations for cosmological data
* ``cosmocore.pixel``: Pixel-space operations and signal matrices
* ``cosmocore.settings``: Parameter management and configuration

**Documentation:**

* Complete API reference with NumPy-style docstrings
* Installation and quick start guides
* Tutorial series for common use cases
* Read the Docs integration
