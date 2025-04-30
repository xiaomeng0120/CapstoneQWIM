"""
QWIM Dashboard Package
======================

A comprehensive dashboard application for time series analysis and visualization.

The dashboard is organized into modules:
- inputs_module: For data selection and filtering
- analysis_module: For exploratory data analysis and visualization
- results_module: For displaying insights and comparative analysis
"""

__version__ = "1.0.0"
__build_date__ = "2025-03-15"

# Re-export main App for easier imports
from .main_App import app, get_data, get_series_names

# Export version information
from . import modules