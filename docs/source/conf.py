# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the source code path to Python path for autodoc
sys.path.insert(0, os.path.abspath("../../src"))
# Add specific package paths
sys.path.insert(0, os.path.abspath("../../src/cosmoforge.cosmocore"))
sys.path.insert(0, os.path.abspath("../../src/cosmoforge.quelo"))
sys.path.insert(0, os.path.abspath("../../src/cosmoforge.picslike"))
sys.path.insert(0, os.path.abspath("../../src/cosmoforge.meta"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CosmoForge"
copyright = "2025, Giacomo Galloni"
author = "Giacomo Galloni"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # For NumPy/Google style docstrings
    "sphinx.ext.autosummary",  # For automatic summary generation
    "sphinx.ext.intersphinx",  # For cross-references to other projects
    "sphinx.ext.mathjax",  # For mathematical expressions
    "sphinx_autodoc_typehints",  # For type hints support
    "myst_parser",  # For Markdown support
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = False  # Disable autosummary to avoid import issues
autosummary_imported_members = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Mock imports for modules that might not be available during documentation build
autodoc_mock_imports = [
    "scipy",
    "healpy",
    "numba",
    "mpi4py",
    "yaml",
    "astropy",
    "matplotlib",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "healpy": ("https://healpy.readthedocs.io/en/latest/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # Use Read the Docs theme
html_static_path = ["_static"]

html_logo = "_static/logo_cosmoforge_white.png"

# Copy .nojekyll file to output directory for GitHub Pages
html_extra_path = [".nojekyll"]

# HTML theme options
html_theme_options = {
    "analytics_id": "",  # Provided by you if you have Google Analytics
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# -- Source file suffixes ----------------------------------------------------
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}
