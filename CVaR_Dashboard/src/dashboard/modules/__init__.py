"""
QWIM Dashboard Modules
=====================

This package contains the modules used by the QWIM Dashboard application:

- inputs_module: Handles time series selection and data filtering
- analysis_module: Provides data visualization and statistical analysis
- results_module: Displays results and insights from the analysis
"""

# Re-export core components for easier imports
from .inputs_module import inputs_ui, inputs_server
from .analysis_module import analysis_ui, analysis_server
from .results_module import results_ui, results_server

# Define available modules
__all__ = [
    "inputs_module",
    "analysis_module",
    "results_module"
]