# Configuration file for the Sphinx documentation builder

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# General configuration
project = 'QWIM Dashboard'
copyright = '2025, QWIM Team'
author = 'QWIM Team'
release = '1.0.0'

# Add extensions needed for your documentation
extensions = [
    'sphinx.ext.autodoc',     # Auto-generate API documentation
    'sphinx.ext.napoleon',    # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',    # Include source code in documentation
    'sphinx.ext.autosummary', # Generate summaries of functions/classes
    'sphinx.ext.intersphinx', # Link to other project's documentation
    'sphinx_autodoc_typehints', # Use Python 3 annotations for documentation
]

# Napoleon configuration (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

# Theme configuration
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme

# Intersphinx mapping for linking to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'polars': ('https://pola-rs.github.io/polars/py-polars/html/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'shiny': ('https://shiny.posit.co/py/api/', None),
}

# Additional options
autodoc_member_order = 'bysource'  # Document members in source code order
autodoc_typehints = 'description'  # Put typehints in descriptions
add_module_names = False           # Don't prefix members with module names