Contributing
============

We welcome contributions to CosmoForge! This document explains how to contribute.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a development environment
4. Install in development mode

.. code-block:: bash

   git clone https://github.com/yourusername/CosmoForge.git
   cd CosmoForge
   pip install -e ".[dev]"

Code Style
----------

We use the following tools for code quality:

* **ruff** for linting and formatting
* **pytest** for testing
* **pre-commit** hooks for automated checks

Run quality checks:

.. code-block:: bash

   # Linting
   ruff check src/
   
   # Formatting
   ruff format src/
   
   # Tests
   pytest tests/

Documentation
-------------

Documentation is built with Sphinx. To build locally:

.. code-block:: bash

   cd docs
   make html

The built documentation will be in ``docs/build/html/``.

Pull Requests
-------------

1. Create a feature branch from main
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Submit a pull request

Please ensure:

* All tests pass
* Code follows style guidelines  
* Documentation is updated
* Commit messages are descriptive
