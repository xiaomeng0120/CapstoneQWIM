"""
Portfolio Module for QWIM Dashboard
===================================

This module provides comprehensive portfolio analysis capabilities for the QWIM Dashboard.

The module offers visualization and analysis of portfolio performance through three subtabs:

* **Comparison**: Compare portfolio to benchmark with interactive charts
* **Analysis**: Calculate performance metrics and analyze return distributions
* **Weights**: Visualize and analyze portfolio weight evolution over time

Features
--------
* Portfolio vs benchmark performance comparison with multiple visualization options
* Statistical analysis including returns distribution, drawdowns, and rolling metrics
* Portfolio weight analysis and component weight distribution
* Full quantitative metrics via QuantStats integration
* Interactive plots with zoom, pan, and export capabilities
* Date range filtering for all analyses
* Data export functionality for further analysis

Components
----------
This module consists of two main components:

* ``portfolios_ui``: The user interface definition
* ``portfolios_server``: The server logic for handling inputs and generating outputs

File Structure
-------------
This module expects the following data files:

* ``sample_portfolio_values.csv`` in ``data/processed/``
* ``benchmark_portfolio_values.csv`` in ``data/processed/``
* ``sample_portfolio_weights_ETFs.csv`` in ``data/raw/``

Dependencies
-----------
* pandas, numpy: Data manipulation and calculations
* plotly: Interactive visualizations
* polars: High-performance data processing
* great_tables: HTML table generation
* quantstats: Portfolio analytics and metrics
* shiny, shinywidgets: UI framework

Examples
--------
To use this module in a Shiny application:

.. code-block:: python

    from dashboard.modules.portfolios_module import portfolios_ui, portfolios_server
    
    # In your Shiny app layout
    app_ui = ui.page_fluid(
        portfolios_ui()
    )
    
    # In your server function
    def server(input, output, session):
        portfolios_server(input, output, session)

Notes
-----
This module is designed to be part of a larger dashboard application 
but can also be run as a standalone Shiny application.
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from scipy import stats
from great_tables import GT
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget
import sys
import quantstats as qs  # Add this import for quantstats

# Define project directory structure using pathlib
# This ensures paths are correctly constructed regardless of operating system
PROJECT_ROOT = Path(__file__).resolve().parents[3]
"""Path to the project root directory."""

DATA_DIR = PROJECT_ROOT / "data"
"""Path to the data directory."""

RAW_DATA_DIR = DATA_DIR / "raw"
"""Path to the raw data directory."""

PROCESSED_DATA_DIR = DATA_DIR / "processed"
"""Path to the processed data directory."""

# Define specific file paths for all resources used in this module
PORTFOLIO_VALUES_FILE = PROCESSED_DATA_DIR / "sample_portfolio_values.csv"
"""Path to the portfolio values CSV file."""

BENCHMARK_VALUES_FILE = PROCESSED_DATA_DIR / "benchmark_portfolio_values.csv"
"""Path to the benchmark values CSV file."""

PORTFOLIO_WEIGHTS_FILE = RAW_DATA_DIR / "sample_portfolio_weights_ETFs.csv"
"""Path to the portfolio weights CSV file."""

# Create any missing directories
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# Log the file paths for debugging purposes
print(f"Project Root: {PROJECT_ROOT}")
print(f"Portfolio Values File: {PORTFOLIO_VALUES_FILE}")
print(f"Benchmark Values File: {BENCHMARK_VALUES_FILE}")
print(f"Portfolio Weights File: {PORTFOLIO_WEIGHTS_FILE}")

@module.ui
def portfolios_ui():
    """Create the UI for the Portfolios tab.
    
    This function generates a complete user interface layout for the Portfolios tab,
    including comparison, analysis, and weights subtabs with their respective
    inputs and visualization containers.
    
    Returns
    -------
    shiny.ui.page_fluid
        A fluid page layout containing all UI elements for the Portfolio tab,
        organized into three subtabs.
    
    Notes
    -----
    The UI includes various interactive elements:
    
    * Date range pickers for filtering data
    * Visualization type selectors
    * Download buttons for plots and data
    * Option toggles for customizing visualizations
    
    The UI is structured as a tabset with three main panels:
    
    1. **Comparison**: Portfolio vs benchmark performance comparison
    2. **Analysis**: Portfolio performance metrics and return distribution analysis
    3. **Weights**: Portfolio component weight analysis and visualization
    
    See Also
    --------
    portfolios_server : Server-side logic for this UI
    """
    return ui.page_fluid(
        ui.h2("Portfolio Analysis"),
        ui.navset_card_tab(
            ui.nav_panel(
                "Comparison",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Date Range"),
                        ui.input_date_range(
                            "ID_comparison_date_range",
                            "Select Date Range:",
                            start=datetime(2018, 1, 1),
                            end=datetime.now(),
                            min=datetime(2010, 1, 1),
                            max=datetime.now(),
                            format="yyyy-mm-dd",
                            separator=" to ",
                        ),
                        ui.h4("Visualization Options"),
                        ui.input_radio_buttons(
                            "ID_comparison_viz_type",
                            "Visualization Type:",
                            {
                                "value": "Portfolio Value",
                                "normalized": "Normalized Value (Base=100)",
                                "pct_change": "Percent Change",
                                "cum_return": "Cumulative Return"
                            },
                            selected="value",
                        ),
                        ui.input_checkbox(
                            "ID_comparison_show_diff", "Show Difference", False
                        ),
                        ui.h4("Download Options"),
                        ui.download_button(
                            "output_ID_download_comparison_plot", "Download Plot"
                        ),
                        ui.download_button(
                            "output_ID_download_comparison_data", "Download Data"
                        ),
                    ),
                    ui.h3("Portfolio vs Benchmark Comparison"),
                    output_widget("output_ID_comparison_plot"),
                    ui.h4("Value Statistics"),
                    ui.output_ui("output_ID_comparison_stats"),
                ),
            ),
            ui.nav_panel(
                "Analysis",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Date Range"),
                        ui.input_date_range(
                            "ID_analysis_date_range",
                            "Select Date Range:",
                            start=datetime(2018, 1, 1),
                            end=datetime.now(),
                            min=datetime(2010, 1, 1),
                            max=datetime.now(),
                            format="yyyy-mm-dd",
                            separator=" to ",
                        ),
                        ui.h4("Analysis Options"),
                        ui.input_radio_buttons(
                            "ID_analysis_type",
                            "Analysis Type:",
                            {
                                "returns": "Returns Distribution",
                                "drawdowns": "Drawdowns Analysis",
                                "rolling": "Rolling Statistics"
                            },
                            selected="returns",
                        ),
                        ui.panel_conditional(
                            "input.ID_analysis_type === 'rolling'",
                            ui.input_slider(
                                "ID_rolling_window",
                                "Rolling Window (Days):",
                                min=10,
                                max=252,
                                value=60,
                                step=10
                            ),
                        ),
                        ui.h4("Download Options"),
                        ui.download_button(
                            "output_ID_download_analysis_plot", "Download Plot"
                        ),
                        ui.download_button(
                            "output_ID_download_analysis_data", "Download Data"
                        ),
                    ),
                    ui.h3("Portfolio Performance Analysis"),
                    output_widget("output_ID_portfolio_analysis_plot"),
                    ui.h4("Performance Metrics"),
                    ui.output_ui("output_ID_portfolio_analysis_stats"),
                    ui.h4("QuantStats Metrics"),
                    ui.output_ui("output_ID_portfolio_quantstats_metrics"),
                ),
            ),
            ui.nav_panel(
                "Weights",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Date Range"),
                        ui.input_date_range(
                            "ID_weights_date_range",
                            "Select Date Range:",
                            start=datetime(2018, 1, 1),
                            end=datetime.now(),
                            min=datetime(2010, 1, 1),
                            max=datetime.now(),
                            format="yyyy-mm-dd",
                            separator=" to ",
                        ),
                        ui.h4("Visualization Options"),
                        ui.input_radio_buttons(
                            "ID_weights_viz_type",
                            "Visualization Type:",
                            {
                                "area": "Stacked Area Plot",
                                "bar": "Stacked Bar Chart",
                                "line": "Line Chart",
                                "heatmap": "Heatmap"
                            },
                            selected="area",
                        ),
                        ui.input_checkbox(
                            "ID_weights_show_pct", "Show as Percentages", True
                        ),
                        ui.input_checkbox(
                            "ID_weights_sort_components", "Sort Components by Weight", False
                        ),
                        ui.h4("Download Options"),
                        ui.download_button(
                            "output_ID_download_weights_plot", "Download Plot"
                        ),
                        ui.download_button(
                            "output_ID_download_weights_data", "Download Data"
                        ),
                    ),
                    ui.h3("Portfolio Weights Evolution"),
                    output_widget("output_ID_weights_plot"),
                    ui.h4("Weight Distribution"),
                    output_widget("output_ID_weight_distribution_plot"),
                    ui.h4("Component Summary"),
                    ui.output_ui("output_ID_weight_summary_table"),
                ),
            ),
        ),
    )

@module.server
def portfolios_server(input, output, session, inputs_data=None, data_r=None):
    """Server logic for the Portfolios tab.
    
    This function handles all server-side processing for the Portfolios tab,
    including data loading, metric calculations, and plot generation.
    
    Parameters
    ----------
    input : shiny.module_input.ModuleInput
        Input object containing all user interface inputs.
    output : shiny.module_output.ModuleOutput
        Output object for rendering results back to the UI.
    session : shiny.module_session.Session
        Shiny session object.
    inputs_data : dict, optional
        Dictionary containing shared data from inputs module.
    data_r : reactive.Value or dict, optional
        Reactive value or dictionary containing shared data.
        
    Returns
    -------
    None
        This function defines reactive elements and outputs but does not return a value.
    
    Notes
    -----
    This function defines many nested functions using the reactive programming
    model of Shiny. Each output element has a corresponding function that
    generates its content.
    
    The server logic handles error states gracefully, providing user-friendly
    messages when data is unavailable or calculations fail.
    
    See Also
    --------
    portfolios_ui : UI definition for this module
    """
    # Use a flag to track if data is loaded
    data_loaded = reactive.Value(False)
    
    # Define error handling for the module
    def safe_load(func, default_value=None):
        """Safely execute a function with error handling.
        
        This is a utility function that wraps function calls with try/except
        to ensure that errors don't crash the application.
        
        Parameters
        ----------
        func : callable
            Function to execute safely.
        default_value : any, optional
            Value to return if the function raises an exception.
            
        Returns
        -------
        any
            Result of the function call if successful, or the default_value
            if an exception occurs.
            
        Examples
        --------
        >>> result = safe_load(lambda: 1/0, default_value="Error")
        >>> print(result)
        'Error'
        
        >>> result = safe_load(lambda: 42, default_value=0)
        >>> print(result)
        42
        """
        try:
            return func()
        except Exception as e:
            print(f"Error in portfolios module: {str(e)}")
            return default_value
    
    @reactive.calc
    def load_portfolio_data():
        """Load portfolio data from CSV file.
        
        This function reads the portfolio values data from the predefined file path,
        handles different column name formats, and ensures proper date formatting.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing portfolio data with 'Date' and 'Value' columns.
            Returns an empty DataFrame with these columns if loading fails.
        
        Notes
        -----
        The function attempts to use polars for faster loading when available,
        and falls back to pandas if necessary. It also handles column name
        case variations and ensures the Date column is in datetime format.
        
        See Also
        --------
        load_benchmark_data : Similar function for benchmark data
        """
        try:
            # Use the predefined file path constants
            portfolio_path = PORTFOLIO_VALUES_FILE  # Use the global constant
            
            # Try polars first for better performance
            try:
                df = pl.read_csv(portfolio_path)
                if "Date" in df.columns and "Value" in df.columns:
                    # Convert to pandas for compatibility
                    return df.to_pandas()
                else:
                    # Try different column names
                    df = df.rename({"date": "Date", "value": "Value"})
                    return df.to_pandas()
            except:
                # Fallback to pandas
                df = pd.read_csv(portfolio_path)
                if "date" in df.columns and "Date" not in df.columns:
                    df = df.rename(columns={"date": "Date"})
                if "value" in df.columns and "Value" not in df.columns:
                    df = df.rename(columns={"value": "Value"})
                
                # Convert Date column to datetime with utc=True
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
                
                return df
        except Exception as e:
            print(f"Error loading portfolio data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame({"Date": [], "Value": []})
            
    @reactive.calc
    def load_benchmark_data():
        """Load benchmark data from CSV file.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing benchmark data with date and value columns
        """
        try:
            # Use the predefined file path constants
            benchmark_path = BENCHMARK_VALUES_FILE  # Use the global constant
            
            # Try polars first for better performance
            try:
                df = pl.read_csv(benchmark_path)
                if "Date" in df.columns and "Value" in df.columns:
                    # Convert to pandas for compatibility
                    return df.to_pandas()
                else:
                    # Try different column names
                    df = df.rename({"date": "Date", "value": "Value"})
                    return df.to_pandas()
            except:
                # Fallback to pandas
                df = pd.read_csv(benchmark_path)
                if "date" in df.columns and "Date" not in df.columns:
                    df = df.rename(columns={"date": "Date"})
                if "value" in df.columns and "Value" not in df.columns:
                    df = df.rename(columns={"value": "Value"})
                return df
        except Exception as e:
            print(f"Error loading benchmark data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame({"Date": [], "Value": []})
    
    @reactive.calc
    def load_weights_data():
        """Load portfolio weights data from CSV file.
        
        Returns
        ------- 
        pd.DataFrame
            DataFrame containing portfolio weights
        """
        try:
            # Use the predefined global constant for the weights file path
            file_path = PORTFOLIO_WEIGHTS_FILE
            
            if not file_path.exists():
                print(f"Portfolio weights file not found at: {file_path}")
                # Return a minimal valid DataFrame with a Date column
                return pd.DataFrame({"Date": [pd.Timestamp.now()]})
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Check if Date column exists, if not try to find a suitable column or create one
            if "Date" not in df.columns:
                # Look for alternative date columns with different case
                date_cols = [col for col in df.columns if col.lower() == "date"]
                if date_cols:
                    # Rename the first matching column to "Date"
                    df = df.rename(columns={date_cols[0]: "Date"})
                else:
                    # Look for columns that might contain dates
                    potential_date_cols = [col for col in df.columns if 
                                          any(time_str in col.lower() for time_str in 
                                             ["date", "time", "period", "day", "month", "year"])]
                    
                    if potential_date_cols:
                        # Try to convert the first potential date column
                        try:
                            df = df.rename(columns={potential_date_cols[0]: "Date"})
                            # Test conversion
                            pd.to_datetime(df["Date"], errors='coerce')
                        except:
                            # If conversion fails, create a date column
                            print("Failed to find a valid date column, creating one")
                            df["Date"] = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=len(df)-1), 
                                                     periods=len(df), freq='D')
                    else:
                        # If no potential date columns, create a date column
                        print("No date column found, creating one")
                        df["Date"] = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=len(df)-1), 
                                                 periods=len(df), freq='D')
            
            # Ensure Date column is datetime
            try:
                df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            except Exception as e:
                print(f"Error converting dates to datetime: {e}")
                # If date conversion fails, replace with a generated date range
                df["Date"] = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=len(df)-1), 
                                         periods=len(df), freq='D')
            
            # Clean up columns - handle potential numeric column names
            numeric_columns = [col for col in df.columns 
                              if col != "Date" and (str(col).isdigit() or (str(col).startswith('-') and str(col)[1:].isdigit()))]
            
            if numeric_columns:
                print(f"Dropping likely index columns: {numeric_columns}")
                df = df.drop(columns=numeric_columns)
            
            # Ensure we have at least one component column other than Date
            if len(df.columns) <= 1:
                print("No component columns found, adding placeholder")
                df["Component_1"] = 1.0  # Add a placeholder component with weight 1.0
                
            return df
        except Exception as e:
            print(f"Error loading weights data: {e}")
            # Return a minimal valid DataFrame with required columns
            return pd.DataFrame({
                "Date": [pd.Timestamp.now()],
                "Component_1": [1.0]  # Add a placeholder component
            })
    
    @reactive.calc
    def get_weights_data():
        """Filter portfolio weights data based on date range.
        
        Returns
        -------
        pd.DataFrame
            Filtered portfolio weights data
        """
        weights_df = load_weights_data()
        
        # Return empty DataFrame if no data
        if weights_df.empty:
            return pd.DataFrame()
            
        # Ensure Date column is datetime (without timezone information)
        if "Date" in weights_df.columns:
            try:
                weights_df["Date"] = pd.to_datetime(weights_df["Date"], utc=True).dt.tz_localize(None)
            except:
                print("Error converting dates to datetime format in weights data")
            
        # Filter by date range
        date_range = input.ID_weights_date_range()
        if date_range[0] and date_range[1]:
            # Convert the input dates to timezone-naive datetime objects
            start_date = pd.Timestamp(date_range[0]).tz_localize(None)
            end_date = pd.Timestamp(date_range[1]).tz_localize(None)
            
            filtered = weights_df[
                (weights_df["Date"] >= start_date) & 
                (weights_df["Date"] <= end_date)
            ]
        else:
            filtered = weights_df
            
        return filtered
    
    @reactive.calc
    def process_weights_data():
        """Process portfolio weights data for visualization.
        
        Returns
        -------
        pd.DataFrame
            Processed weights data ready for visualization
        """
        weights_df = get_weights_data()
        
        # Return empty DataFrame if no data
        if weights_df.empty:
            return pd.DataFrame()
        
        # Get components (all columns except Date)
        components = [col for col in weights_df.columns if col != "Date"]
        
        # Check if we need to convert to percentages
        show_pct = input.ID_weights_show_pct()
        
        if show_pct:
            # Calculate row sums
            weights_df["RowSum"] = weights_df[components].sum(axis=1)
            
            # Convert to percentages
            for comp in components:
                weights_df[comp] = weights_df[comp] / weights_df["RowSum"] * 100
                
            # Drop the RowSum column
            weights_df = weights_df.drop(columns=["RowSum"])
        
        # Sort components by weight if requested
        if input.ID_weights_sort_components():
            # Calculate average weight for each component
            avg_weights = weights_df[components].mean()
            # Sort components by average weight
            sorted_components = avg_weights.sort_values(ascending=False).index.tolist()
            
            # Reorder columns
            weights_df = weights_df[["Date"] + sorted_components]
        
        return weights_df
    
    @reactive.calc
    def calculate_weight_statistics():
        """Calculate statistics for portfolio weights."""
        weights_df = get_weights_data()
        
        # Return empty DataFrame if no data
        if weights_df.empty:
            return pd.DataFrame()
            
        # Get components (all columns except Date)
        components = [col for col in weights_df.columns if col != "Date"]
        
        # Check if we need to convert to percentages
        show_pct = input.ID_weights_show_pct()
        
        stats = []
        
        for comp in components:
            comp_weights = weights_df[comp].copy()
            
            if show_pct:
                # Convert to percentage of row sum
                row_sums = weights_df[components].sum(axis=1)
                comp_weights = comp_weights / row_sums * 100
            
            # Calculate statistics - ensure all values are numeric to avoid string mixing
            try:
                min_weight = float(comp_weights.min())
                max_weight = float(comp_weights.max())
                mean_weight = float(comp_weights.mean())
                std_dev = float(comp_weights.std())
                current_weight = float(comp_weights.iloc[-1]) if not comp_weights.empty else 0.0
            except (ValueError, TypeError):
                # Handle any conversion errors
                min_weight = 0.0
                max_weight = 0.0
                mean_weight = 0.0
                std_dev = 0.0
                current_weight = 0.0
                
            # Add as a proper dictionary with numeric values
            stats.append({
                "Component": str(comp),  # Ensure component is a string
                "Min Weight": min_weight,
                "Max Weight": max_weight,
                "Mean Weight": mean_weight,
                "Std Dev": std_dev,
                "Current Weight": current_weight
            })
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Sort by mean weight if we have data
        if not stats_df.empty:
            stats_df = stats_df.sort_values("Mean Weight", ascending=False)
        
        return stats_df
    
    @output
    @render_widget
    def output_ID_weights_plot():
        """Generate portfolio weights evolution plot.
        
        Creates an interactive visualization showing how portfolio component
        weights have changed over time using different chart types.
        
        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure showing the weights evolution using the selected visualization type.
        
        Notes
        -----
        This function supports multiple visualization types:
        
        * **area**: Stacked area chart
        * **bar**: Stacked bar chart
        * **line**: Line chart
        * **heatmap**: Heatmap visualization
        
        Weights can be displayed as absolute values or as percentages.
        
        The plot includes interactive features like date range selection,
        zoom controls, and hover information.
        
        See Also
        --------
        get_weights_data : Provides the data for this plot
        process_weights_data : Processes the weights data for visualization
        """
        try:
            weights_df = process_weights_data()
            viz_type = input.ID_weights_viz_type()
            show_pct = input.ID_weights_show_pct()
            
            # Check if we have valid data
            if weights_df.empty or "Date" not in weights_df.columns:
                # Return empty plot with helpful message
                fig = px.scatter(title="No portfolio weights data available")
                fig.add_annotation(
                    text="Please check that your weights file exists and contains a 'Date' column",
                    showarrow=False,
                    font=dict(size=14)
                )
                return fig
            
            # Get components (all columns except Date)
            components = [col for col in weights_df.columns if col != "Date"]
            
            if not components:
                # No components found
                fig = px.scatter(title="No component columns found")
                fig.add_annotation(
                    text="Your weights file must contain columns other than 'Date'",
                    showarrow=False,
                    font=dict(size=14)
                )
                return fig
            
            # Define y-axis title based on percentage option - ensure it's a string
            y_title = "Weight (%)" if show_pct else "Weight"
            
            # Create plot based on visualization type
            if viz_type == "area":
                # Create stacked area chart
                fig = px.area(
                    weights_df,
                    x="Date",
                    y=components,
                    title="Portfolio Weight Evolution Over Time",
                    labels={"Date": "Date", "value": str(y_title), "variable": "Component"},
                    template="plotly_white"
                )
                
            elif viz_type == "bar":
                # Create stacked bar chart
                fig = px.bar(
                    weights_df,
                    x="Date",
                    y=components,
                    title="Portfolio Weight Evolution Over Time",
                    labels={"Date": "Date", "value": str(y_title), "variable": "Component"},
                    template="plotly_white"
                )
                
            elif viz_type == "line":
                # Create line chart
                fig = px.line(
                    weights_df,
                    x="Date",
                    y=components,
                    title="Portfolio Weight Evolution Over Time",
                    labels={"Date": "Date", "value": str(y_title), "variable": "Component"},
                    template="plotly_white"
                )
                
            elif viz_type == "heatmap":
                # Pivot the data for heatmap visualization
                # For heatmap, we need the data in long format
                weights_long = weights_df.melt(
                    id_vars=["Date"], 
                    value_vars=components,
                    var_name="Component", 
                    value_name="Weight"
                )
                
                # Create heatmap
                fig = px.density_heatmap(
                    weights_long,
                    x="Date",
                    y="Component",
                    z="Weight",
                    title="Portfolio Weight Heatmap",
                    labels={"Date": "Date", "Component": "Component", "Weight": y_title},
                    template="plotly_white",
                    color_continuous_scale="YlGnBu"
                )
                
                # Add component labels if needed
                if viz_type == "heatmap":
                    for i, comp in enumerate(components):
                        fig.add_annotation(
                            x=-0.05,
                            y=i,
                            xref="paper",
                            yref="y",
                            text=comp,
                            showarrow=False,
                            font=dict(size=10),
                        )
            
            # Update layout with explicit string conversion and increased space for legend
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=str(y_title),  # Convert to string explicitly
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.12,  # Increased from 1.02 to 1.12 to add more space
                    xanchor="right",
                    x=1
                ),
                margin=dict(
                    l=40, 
                    r=40, 
                    t=100,  # Increased from default to add more space at the top
                    b=40
                ),
                hovermode="x unified",
                title=dict(
                    y=0.95,  # Move title position slightly higher
                    yanchor="top"
                )
            )
            
            # If we're showing percentages, update y-axis to show 0-100
            if show_pct:
                fig.update_yaxes(range=[0, 100])
                
                # For stacked plots, make sure they add up to 100%
                if viz_type in ["area", "bar"]:
                    fig.update_layout(yaxis_range=[0, 100])
            
            # For bar chart, add pattern bar padding
            if viz_type == "bar":
                fig.update_layout(barmode="stack", bargap=0.1)
            
            # Add range selector for date axis
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Return the figure
            return fig
        except Exception as e:
            print(f"Error generating weights plot: {str(e)}")
            fig = px.scatter(title=f"Error: {str(e)}")
            return fig
    
    @output
    @render_widget
    def output_ID_weight_distribution_plot():
        """Generate weight distribution plot for the current portfolio."""
        try:
            weights_df = get_weights_data()
            show_pct = input.ID_weights_show_pct()
            
            # Validate DataFrame and Date column
            if weights_df.empty:
                # Return empty plot with helpful message
                fig = px.scatter(title="No portfolio weights data available")
                fig.add_annotation(
                    text="Please check that your weights file exists in the data/raw directory",
                    showarrow=False,
                    font=dict(size=14)
                )
                return fig
            
            # Check if Date column exists
            if 'Date' not in weights_df.columns:
                # Log the issue and add a Date column with today's date
                print("Date column missing, adding one")
                weights_df["Date"] = pd.Timestamp.now()
                
            # Get components (all columns except Date)
            components = [col for col in weights_df.columns if col != "Date" and col != "RowSum"]
            
            if not components:
                # Add a placeholder component if none exist
                print("No component columns found, adding placeholder")
                weights_df["Component_1"] = 1.0
                components = ["Component_1"]
            
            # Ensure we have data for all components
            for comp in components:
                if weights_df[comp].isna().all():
                    # Replace all NaN values with zeros for the component
                    weights_df[comp] = 0
                
            # Create working copy of the dataframe
            plot_df = weights_df.copy()
            
            # Convert to percentages if needed
            if show_pct:
                try:
                    # Calculate row sums (only considering the component columns)
                    row_sums = plot_df[components].sum(axis=1)
                    
                    # Convert component values to percentages
                    for comp in components:
                        plot_df[comp] = plot_df[comp] / row_sums * 100
                except Exception as e:
                    print(f"Error converting to percentages: {str(e)}")
                    # Continue with original weights if percentage calculation fails
            
            # Convert to long format for box plot - using the validated components only
            try:
                weights_long = pd.melt(
                    plot_df,
                    value_vars=components,  # Use only valid component columns
                    var_name="Component",
                    value_name="Weight"
                )
            except Exception as e:
                # Handle any error during the melt operation
                print(f"Error in weight distribution plot melt: {str(e)}")
                fig = px.scatter(title=f"Error processing weights data: {str(e)}")
                fig.add_annotation(
                    text="Please check your weights data format",
                    showarrow=False,
                    font=dict(size=14)
                )
                return fig
            
            # Define y-axis title
            y_title = "Weight (%)" if show_pct else "Weight"
            
            # Create box plot
            try:
                fig = px.box(
                    weights_long,
                    x="Component",
                    y="Weight",
                    title="Weight Distribution by Component",
                    labels={"Component": "Component", "Weight": y_title},
                    color="Component",
                    template="plotly_white",
                    # Increase the height to improve vertical scaling
                    height=600  # Increased from default height
                )
            except Exception as e:
                print(f"Error creating box plot: {str(e)}")
                fig = px.scatter(title=f"Error creating weight distribution plot: {str(e)}")
                fig.add_annotation(
                    text="Please check your weights data format and values",
                    showarrow=False,
                    font=dict(size=14)
                )
                return fig
            
            # Add current weights as scatter points
            try:
                last_date = plot_df["Date"].max()
                current_weights = plot_df[plot_df["Date"] == last_date]
                
                if not current_weights.empty:
                    # Convert to long format - only using actual components
                    current_long = pd.melt(
                        current_weights,
                        id_vars=["Date"],
                        value_vars=components,  # Use only valid component columns
                        var_name="Component",
                        value_name="Weight"
                    )
                    
                    # Add scatter points
                    fig.add_trace(
                        go.Scatter(
                            x=current_long["Component"],
                            y=current_long["Weight"],
                            mode="markers",
                            marker=dict(
                                color="black",
                                size=10,  # Increased marker size
                                symbol="diamond"
                            ),
                            name="Current Weight"
                        )
                    )
            except Exception as e:
                print(f"Error adding current weights: {str(e)}")
                # Continue without current weights if there's an error
            
            # Calculate the y-axis range more intelligently
            if not weights_long.empty:
                weight_min = weights_long["Weight"].min()
                weight_max = weights_long["Weight"].max()
                
                # Create better padding
                y_padding = (weight_max - weight_min) * 0.15
                
                # Set reasonable minimum (don't go below zero for weights)
                y_min = max(0, weight_min - y_padding)
                
                # Set maximum with adequate padding
                y_max = weight_max + y_padding
                
                # For percentage weights, ensure we have sensible bounds
                if show_pct:
                    # Never go below 0 for percentages
                    y_min = max(0, y_min)
                    
                    # Ensure upper bound is sensible for percentages
                    if y_max < 100:
                        # If all weights are under 100%, add reasonable padding
                        y_max = min(100, y_max * 1.1)
                    elif y_max < 110:
                        # If some weights are near 100%, cap at slightly above 100%
                        y_max = 110
                    else:
                        # For unusually large percentage values, add modest padding
                        y_max = y_max * 1.05
            else:
                # Default range if no data
                y_min = 0
                y_max = 100 if show_pct else 1
            
            # Update layout with better spacing and margins
            try:
                y_title_str = str(y_title) if y_title is not None else "Weight"
                fig.update_layout(
                    xaxis_title="Component",
                    yaxis_title=y_title_str,
                    showlegend=False,
                    # Increased top margin for better title spacing
                    margin=dict(l=50, r=50, t=100, b=50),
                    # Better spacing between boxes
                    boxgap=0.2,  
                    boxgroupgap=0.3,
                    # Set specific y-axis range for better vertical scaling
                    yaxis=dict(
                        range=[y_min, y_max]
                    ),
                    # Improve overall appearance
                    plot_bgcolor='white',
                    title=dict(
                        y=0.95,  # Position title higher
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    )
                )
                
                # Rotate x-axis labels if there are many components
                if len(components) > 4:
                    fig.update_layout(
                        xaxis=dict(
                            tickangle=45,
                            tickmode="array",
                            tickvals=list(range(len(components))),
                            ticktext=components
                        )
                    )
                
            except Exception as e:
                print(f"Error updating layout: {str(e)}")
                # The plot should still display even if layout customization fails
                
            return fig
        except Exception as e:
            print(f"Error generating weight distribution plot: {str(e)}")
            fig = px.scatter(title=f"Error: {str(e)}")
            return fig
    
    @output
    @render.ui
    def output_ID_weight_summary_table():
        """Generate weight summary table using great_tables."""
        from great_tables import GT
        
        try:
            stats_df = calculate_weight_statistics()
            show_pct = input.ID_weights_show_pct()
            
            if stats_df.empty:
                return ui.p("No portfolio weights data available")
            
            # Determine if we're showing percentages for formatting
            value_suffix = "%" if show_pct else ""
            
            # Create great_tables object
            gt_table = (
                GT(stats_df)
                .tab_header(
                    title="Portfolio Component Statistics",
                    subtitle="Summary of component weights over selected time period"
                )
                .fmt_number(
                    columns=["Min Weight", "Max Weight", "Mean Weight", "Std Dev", "Current Weight"],
                    decimals=2
                )
                .tab_source_note(
                    source_note="Data calculated from portfolio weights file"
                )
            )
            
            # Try different HTML rendering methods
            try:
                # Try direct HTML rendering first
                html_content = gt_table.as_raw_html()
                return ui.HTML(html_content)
            except Exception as e1:
                print(f"First rendering attempt failed: {str(e1)}")
                try:
                    # Try the render() method next with the required context parameter
                    html_content = gt_table.render(context="html")  # Add context parameter
                    return ui.HTML(html_content)
                except Exception as e2:
                    print(f"Second rendering attempt failed: {str(e2)}")
                    try:
                        # Fallback to a custom HTML table when GT rendering fails
                        html = "<table border='1' style='width:100%; border-collapse:collapse;'>"
                        html += "<thead><tr style='background-color:#f2f2f2;'>"
                        html += "<th style='padding:8px; text-align:left;'>Component</th>"
                        html += "<th style='padding:8px; text-align:right;'>Min Weight</th>"
                        html += "<th style='padding:8px; text-align:right;'>Max Weight</th>"
                        html += "<th style='padding:8px; text-align:right;'>Mean Weight</th>"
                        html += "<th style='padding:8px; text-align:right;'>Std Dev</th>"
                        html += "<th style='padding:8px; text-align:right;'>Current Weight</th>"
                        html += "</tr></thead><tbody>"
                        
                        # Add rows
                        for _, row in stats_df.iterrows():
                            html += "<tr>"
                            html += f"<td style='padding:8px; border-top:1px solid #ddd;'>{row['Component']}</td>"
                            
                            # Format numeric columns with 2 decimal places
                            numeric_cols = ["Min Weight", "Max Weight", "Mean Weight", "Std Dev", "Current Weight"]
                            for col in numeric_cols:
                                val = row[col]
                                if pd.notna(val) and isinstance(val, (int, float)):
                                    formatted_val = f"{val:.2f}{value_suffix}"
                                else:
                                    formatted_val = "N/A"
                                html += f"<td style='padding:8px; border-top:1px solid #ddd; text-align:right;'>{formatted_val}</td>"
                            
                            html += "</tr>"
                        
                        html += "</tbody></table>"
                        html += "<div style='font-size:0.8em; margin-top:10px;'>Data calculated from portfolio weights file</div>"
                        
                        return ui.HTML(html)
                    except Exception as e3:
                        print(f"All rendering attempts failed: {str(e3)}")
                        
                        # Last resort: simple text representation
                        text_output = ui.div(
                            ui.h4("Portfolio Component Statistics"),
                            ui.tags.pre(stats_df.to_string(index=False))
                        )
                        return text_output
        except Exception as e:
            print(f"Error generating weight summary table: {str(e)}")
            return ui.div(
                ui.tags.b("Error generating component statistics"),
                ui.p(str(e))
            )
    
    @output
    @render.download(filename="portfolio_weights.csv")
    def output_ID_download_weights_data():
        """Download handler for weights data in CSV format.
        
        Returns
        -------
        str
            CSV formatted string of the weights data
        """
        weights_df = process_weights_data()
        
        if weights_df.empty:
            return "No data available"
        
        # Return as CSV
        return weights_df.to_csv(index=False)
    
    @render.download(filename="portfolio_weights_plot.png")
    def output_ID_download_weights_plot():
        """Generate a downloadable version of the weights plot.
        
        Returns
        ------- 
        bytes
            PNG image of the weights plot
        """
        from io import BytesIO
        
        # Get the same figure as in the plot
        fig = output_ID_weights_plot()
        
        try:
            # Write to PNG
            img_bytes = BytesIO()
            fig.write_image(img_bytes, format="png", width=1200, height=800, scale=2)
            img_bytes.seek(0)
            
            # Return bytes
            return img_bytes.read()
        except Exception as e:
            print(f"Error generating plot image: {str(e)}")
            return f"Error: {str(e)}"
    
    @render.download(filename="portfolio_comparison.csv")
    def output_ID_download_comparison_data():
        """Download handler for comparison data in CSV format.
        
        Returns
        -------
        str
            CSV formatted string of the comparison data
        """
        portfolio_df, benchmark_df = get_comparison_data()
        
        if portfolio_df.empty or benchmark_df.empty:
            return "No data available"
        
        # Merge the dataframes for comparison
        merged_df = pd.merge(
            portfolio_df, 
            benchmark_df,
            on="Date",
            suffixes=("_Portfolio", "_Benchmark")
        )
        
        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            "Value_Portfolio": "My-Portfolio", 
            "Value_Benchmark": "My-Benchmark"
        })
        
        # Return as CSV
        return merged_df.to_csv(index=False)
    
    @render.download(filename="portfolio_comparison_plot.png")
    def output_ID_download_comparison_plot():
        """Generate a downloadable version of the comparison plot.
        
        Returns
        ------- 
        bytes
            PNG image of the comparison plot
        """
        from io import BytesIO
        
        # Get the same figure as in the plot
        fig = output_ID_comparison_plot()
        
        try:
            # Write to PNG
            img_bytes = BytesIO()
            fig.write_image(img_bytes, format="png", width=1200, height=800, scale=2)
            img_bytes.seek(0)
            
            # Return bytes
            return img_bytes.read()
        except Exception as e:
            print(f"Error generating comparison plot image: {str(e)}")
            return f"Error: {str(e)}"
    
    @render.download(filename="portfolio_analysis.csv")
    def output_ID_download_analysis_data():
        """Download handler for analysis data in CSV format.
        
        Returns
        -------
        str
            CSV formatted string of the analysis data
        """
        portfolio_df, benchmark_df = get_comparison_data()
        
        if portfolio_df.empty or benchmark_df.empty:
            return "No data available"
        
        # Calculate daily returns
        portfolio_df["Portfolio_Return"] = portfolio_df["Value"].pct_change()
        benchmark_df["Benchmark_Return"] = benchmark_df["Value"].pct_change()
        
        # Merge the dataframes for analysis
        merged_df = pd.merge(
            portfolio_df[["Date", "Value", "Portfolio_Return"]], 
            benchmark_df[["Date", "Value", "Benchmark_Return"]],
            on="Date",
            suffixes=("_Portfolio", "_Benchmark")
        )
        
        # Rename value columns
        merged_df = merged_df.rename(columns={
            "Value_Portfolio": "Portfolio_Value", 
            "Value_Benchmark": "Benchmark_Value"
        })
        
        # Return as CSV
        return merged_df.to_csv(index=False)
    
    @render.download(filename="portfolio_analysis_plot.png")
    def output_ID_download_analysis_plot():
        """Generate a downloadable version of the analysis plot.
        
        Returns
        ------- 
        bytes
            PNG image of the analysis plot
        """
        from io import BytesIO
        
        # Get the same figure as in the plot
        fig = output_ID_portfolio_analysis_plot()
        
        try:
            # Write to PNG
            img_bytes = BytesIO()
            fig.write_image(img_bytes, format="png", width=1200, height=800, scale=2)
            img_bytes.seek(0)
            
            # Return bytes
            return img_bytes.read()
        except Exception as e:
            print(f"Error generating analysis plot image: {str(e)}")
            return f"Error: {str(e)}"
       
    @reactive.calc
    def get_comparison_data():
        """Filter portfolio and benchmark data based on user-selected date range.
        
        This function loads portfolio and benchmark data from CSV files,
        processes the dates to ensure proper formatting, and filters the
        data based on the date range selected by the user in the UI.
        
        Returns
        -------
        tuple of pandas.DataFrame
            A tuple containing (portfolio_df, benchmark_df), each DataFrame
            having 'Date' and 'Value' columns. Both DataFrames are filtered
            to the selected date range and sorted by date.
        
        Notes
        -----
        If the date filtering results in empty DataFrames, the function
        falls back to the original unfiltered data to ensure visualizations
        don't break.
        
        The function handles various date formats and timezone issues to ensure
        consistent comparison between portfolio and benchmark data.
        
        Examples
        --------
        >>> portfolio_df, benchmark_df = get_comparison_data()
        >>> portfolio_df.head()
           Date        Value
        0  2020-01-01  100.0
        1  2020-01-02  101.2
        ...
        """
        portfolio_df = load_portfolio_data()
        benchmark_df = load_benchmark_data()
        
        # Process portfolio data
        if not portfolio_df.empty and "Date" in portfolio_df.columns:
            try:
                # Handle various date formats safely
                if portfolio_df["Date"].dtype == 'object' or portfolio_df["Date"].dtype.name == 'category':
                    # Convert strings to datetime
                    portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"], errors='coerce', utc=True)
                # Remove timezone info if present
                if hasattr(portfolio_df["Date"].dtype, 'tz'):
                    portfolio_df["Date"] = portfolio_df["Date"].dt.tz_localize(None)
                # Drop NaT values that might have resulted from failed conversions
                portfolio_df = portfolio_df.dropna(subset=["Date"])
            except Exception as e:
                print(f"Error processing portfolio dates: {str(e)}")
                # Create a minimal valid dataframe if conversion fails
                portfolio_df = pd.DataFrame({"Date": [pd.Timestamp.now()], "Value": [0]})
        
        # Process benchmark data
        if not benchmark_df.empty and "Date" in benchmark_df.columns:
            try:
                # Handle various date formats safely
                if benchmark_df["Date"].dtype == 'object' or benchmark_df["Date"].dtype.name == 'category':
                    # Convert strings to datetime
                    benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"], errors='coerce', utc=True)
                # Remove timezone info if present
                if hasattr(benchmark_df["Date"].dtype, 'tz'):
                    benchmark_df["Date"] = benchmark_df["Date"].dt.tz_localize(None)
                # Drop NaT values that might have resulted from failed conversions
                benchmark_df = benchmark_df.dropna(subset=["Date"])
            except Exception as e:
                print(f"Error processing benchmark dates: {str(e)}")
                # Create a minimal valid dataframe if conversion fails
                benchmark_df = pd.DataFrame({"Date": [pd.Timestamp.now()], "Value": [0]})
        
        # Filter by date range
        date_range = input.ID_comparison_date_range()
        if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
            try:
                # Convert the input dates to timezone-naive datetime objects
                start_date = pd.Timestamp(date_range[0]).tz_localize(None)
                end_date = pd.Timestamp(date_range[1]).tz_localize(None)
                
                # Filter portfolio data
                portfolio_filtered = portfolio_df[
                    (portfolio_df["Date"] >= start_date) & 
                    (portfolio_df["Date"] <= end_date)
                ]
                
                # Filter benchmark data
                benchmark_filtered = benchmark_df[
                    (benchmark_df["Date"] >= start_date) & 
                    (benchmark_df["Date"] <= end_date)
                ]
                
                # Ensure we have data after filtering
                if portfolio_filtered.empty:
                    print("No portfolio data matches the selected date range")
                    portfolio_filtered = portfolio_df
                    
                if benchmark_filtered.empty:
                    print("No benchmark data matches the selected date range")
                    benchmark_filtered = benchmark_df
            except Exception as e:
                print(f"Error filtering data by date range: {str(e)}")
                portfolio_filtered = portfolio_df
                benchmark_filtered = benchmark_df
        else:
            portfolio_filtered = portfolio_df
            benchmark_filtered = benchmark_df
        
        # Sort by date
        if not portfolio_filtered.empty and "Date" in portfolio_filtered.columns:
            portfolio_filtered = portfolio_filtered.sort_values("Date")
            
        if not benchmark_filtered.empty and "Date" in benchmark_filtered.columns:
            benchmark_filtered = benchmark_filtered.sort_values("Date")
            
        return portfolio_filtered, benchmark_filtered
    
    @output
    @render_widget
    def output_ID_comparison_plot():
        """Generate the portfolio comparison plot.
        
        Creates an interactive plot comparing portfolio and benchmark performance.
        The visualization type can be changed between absolute values, normalized 
        values, percentage change, or cumulative returns based on user selection.
        
        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure object containing the comparison visualization.
            
        Notes
        -----
        The function supports multiple visualization types:
        
        * value: Shows absolute values over time
        * normalized: Shows values normalized to 100 at the start
        * pct_change: Shows daily percentage changes
        * cum_return: Shows cumulative returns over time
        
        If no data is available, returns an empty plot with an explanatory message.
        
        The plot includes interactive features like date range selection,
        hover information, and the option to show the difference between
        portfolio and benchmark as a secondary y-axis.
        
        See Also
        --------
        get_comparison_data : Provides the data for this plot
        output_ID_comparison_stats : Generates statistics related to this plot
        """
        try:
            portfolio_df, benchmark_df = get_comparison_data()
            show_diff = input.ID_comparison_show_diff()
            viz_type = input.ID_comparison_viz_type()
            
            # Check if we have data
            if portfolio_df.empty or benchmark_df.empty:
                # Return an empty plot if no data
                fig = px.scatter(title="No data available")
                fig.add_annotation(
                    text="Please check that portfolio and benchmark data files exist",
                    showarrow=False,
                    font=dict(size=14),
                )
                return fig
            
            # Make sure dates are datetime without timezone information
            try:
                portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"]).dt.tz_localize(None)
                benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"]).dt.tz_localize(None)
            except:
                print("Error converting dates to timezone-naive format")
            
            # Merge data on Date for comparison
            merged_df = pd.merge(
                portfolio_df, 
                benchmark_df,
                on="Date",
                suffixes=("_Portfolio", "_Benchmark")
            )
            
            # Rename columns for clarity
            merged_df = merged_df.rename(columns={
                "Value_Portfolio": "My-Portfolio", 
                "Value_Benchmark": "My-Benchmark"
            })
            
            # Calculate additional metrics based on visualization type
            if viz_type == "normalized":
                # Normalize to 100 at start date
                first_portfolio = merged_df["My-Portfolio"].iloc[0]
                first_benchmark = merged_df["My-Benchmark"].iloc[0]
                
                if first_portfolio == 0 or first_benchmark == 0:
                    # Handle zero division
                    merged_df["Portfolio_Normalized"] = 100
                    merged_df["Benchmark_Normalized"] = 100
                else:
                    merged_df["Portfolio_Normalized"] = merged_df["My-Portfolio"] / first_portfolio * 100
                    merged_df["Benchmark_Normalized"] = merged_df["My-Benchmark"] / first_benchmark * 100
                    
                y_column_portfolio = "Portfolio_Normalized"
                y_column_benchmark = "Benchmark_Normalized"
                y_axis_title = "Value (Base = 100)"
                plot_title = "Normalized Portfolio vs Benchmark Performance"
                
            elif viz_type == "pct_change":
                # Calculate daily percentage change
                merged_df["Portfolio_PctChange"] = merged_df["My-Portfolio"].pct_change() * 100
                merged_df["Benchmark_PctChange"] = merged_df["My-Benchmark"].pct_change() * 100
                
                # Replace NaN in first row
                merged_df.iloc[0, merged_df.columns.get_indexer(["Portfolio_PctChange", "Benchmark_PctChange"])] = 0
                
                y_column_portfolio = "Portfolio_PctChange"
                y_column_benchmark = "Benchmark_PctChange"
                y_axis_title = "Daily Change (%)"
                plot_title = "Daily Percentage Change: Portfolio vs Benchmark"
                
            elif viz_type == "cum_return":
                # Calculate cumulative return
                first_portfolio = merged_df["My-Portfolio"].iloc[0]
                first_benchmark = merged_df["My-Benchmark"].iloc[0]
                
                if first_portfolio == 0 or first_benchmark == 0:
                    # Handle zero division
                    merged_df["Portfolio_CumReturn"] = 0
                    merged_df["Benchmark_CumReturn"] = 0
                else:
                    merged_df["Portfolio_CumReturn"] = (merged_df["My-Portfolio"] / first_portfolio - 1) * 100
                    merged_df["Benchmark_CumReturn"] = (merged_df["My-Benchmark"] / first_benchmark - 1) * 100
                    
                y_column_portfolio = "Portfolio_CumReturn"
                y_column_benchmark = "Benchmark_CumReturn"
                y_axis_title = "Cumulative Return (%)"
                plot_title = "Cumulative Return: Portfolio vs Benchmark"
                
            else:  # Default to "value"
                y_column_portfolio = "My-Portfolio"
                y_column_benchmark = "My-Benchmark"
                y_axis_title = "Value"
                plot_title = "Portfolio vs Benchmark: Absolute Value"
            
            # Create the figure
            fig = make_subplots(specs=[[{"secondary_y": show_diff}]])
            
            # Add portfolio trace
            fig.add_trace(
                go.Scatter(
                    x=merged_df["Date"],
                    y=merged_df[y_column_portfolio],
                    name="My-Portfolio",
                    line=dict(color="blue", width=2)
                ),
                secondary_y=False
            )
            
            # Add benchmark trace
            fig.add_trace(
                go.Scatter(
                    x=merged_df["Date"],
                    y=merged_df[y_column_benchmark],
                    name="My-Benchmark",
                    line=dict(color="red", width=2)
                ),
                secondary_y=False
            )
            
            # Add difference trace if requested
            if show_diff:
                # Calculate difference
                if viz_type == "value":
                    merged_df["Difference"] = merged_df["My-Portfolio"] - merged_df["My-Benchmark"]
                    diff_title = "Absolute Difference"
                else:
                    merged_df["Difference"] = merged_df[y_column_portfolio] - merged_df[y_column_benchmark]
                    diff_title = "Difference"
                    
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=merged_df["Date"],
                        y=merged_df["Difference"],
                        name="Difference",
                        line=dict(color="green", dash="dash", width=1.5)
                    ),
                    secondary_y=True
                )
                
                # Update secondary y-axis
                fig.update_yaxes(
                    title_text=diff_title,
                    secondary_y=True,
                    showgrid=False
                )
            
            # Update layout
            fig.update_layout(
                title=plot_title,
                xaxis_title="Date",
                yaxis_title=y_axis_title,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                template="plotly_white",
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Add range selector for date axis
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            return fig
        except Exception as e:
            # Handle any errors gracefully
            print(f"Error in comparison plot: {str(e)}")
            fig = px.scatter(title=f"Error: {str(e)}")
            return fig

    @reactive.calc
    def calculate_portfolio_metrics():
        """Calculate key portfolio performance metrics.
        
        This function computes essential performance metrics for both the portfolio
        and benchmark, including returns, volatility, Sharpe ratio, and maximum drawdown.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Metric', 'My-Portfolio', 'My-Benchmark', and 'Difference',
            containing calculated metrics for both the portfolio and benchmark, as well as
            the difference between them.
        
        Notes
        -----
        The metrics calculated include:
        
        * Cumulative Return: Total return over the entire period
        * Annualized Return: Return normalized to a yearly basis
        * Annualized Volatility: Standard deviation of returns on a yearly basis
        * Sharpe Ratio: Risk-adjusted return (assuming 0% risk-free rate)
        * Maximum Drawdown: Largest percentage drop from peak to trough
        
        Requires at least 5 data points for both portfolio and benchmark to calculate metrics.
        
        See Also
        --------
        calculate_quantstats_metrics : More comprehensive metrics calculation
        """
        try:
            portfolio_df, benchmark_df = get_comparison_data()
            
            # Check if we have sufficient data
            if len(portfolio_df) < 5 or len(benchmark_df) < 5:
                # Return placeholder DataFrame with error message
                return pd.DataFrame({
                    "Metric": ["Insufficient data for calculations"],
                    "My-Portfolio": ["N/A"],
                    "My-Benchmark": ["N/A"],
                    "Difference": ["N/A"]
                })
            
            # Ensure datetime format
            portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"], utc=True)
            benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"], utc=True)
            
            # Remove timezone info after standardizing to UTC
            portfolio_df["Date"] = portfolio_df["Date"].dt.tz_localize(None)
            benchmark_df["Date"] = benchmark_df["Date"].dt.tz_localize(None)
            
            # Sort by date
            portfolio_df = portfolio_df.sort_values("Date")
            benchmark_df = benchmark_df.sort_values("Date")
            
            # Calculate daily returns
            portfolio_returns = portfolio_df["Value"].pct_change().dropna()
            benchmark_returns = benchmark_df["Value"].pct_change().dropna()
            
            # Calculate metrics
            total_days = len(portfolio_returns)
            years = total_days / 252  # Assuming 252 trading days per year
            
            # Performance metrics
            metrics = []
            
            # 1. Calculate cumulative return
            portfolio_cum_return = (portfolio_df["Value"].iloc[-1] / portfolio_df["Value"].iloc[0] - 1)
            benchmark_cum_return = (benchmark_df["Value"].iloc[-1] / benchmark_df["Value"].iloc[0] - 1)
            difference = portfolio_cum_return - benchmark_cum_return
            
            metrics.append({
                "Metric": "Cumulative Return",
                "My-Portfolio": portfolio_cum_return,
                "My-Benchmark": benchmark_cum_return,
                "Difference": difference
            })
            
            # 2. Calculate annualized return
            if years > 0:
                portfolio_ann_return = (1 + portfolio_cum_return) ** (1 / years) - 1
                benchmark_ann_return = (1 + benchmark_cum_return) ** (1 / years) - 1
                difference = portfolio_ann_return - benchmark_ann_return
                
                metrics.append({
                    "Metric": "Annualized Return",
                    "My-Portfolio": portfolio_ann_return,
                    "My-Benchmark": benchmark_ann_return,
                    "Difference": difference
                })
            
            # 3. Calculate volatility
            if len(portfolio_returns) > 1:
                portfolio_vol = portfolio_returns.std() * np.sqrt(252)
                benchmark_vol = benchmark_returns.std() * np.sqrt(252)
                difference = portfolio_vol - benchmark_vol
                
                metrics.append({
                    "Metric": "Annualized Volatility",
                    "My-Portfolio": portfolio_vol,
                    "My-Benchmark": benchmark_vol,
                    "Difference": difference
                })
            
            # 4. Calculate Sharpe Ratio (assuming 0% risk-free rate)
            if len(portfolio_returns) > 1:
                portfolio_sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
                benchmark_sharpe = (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252))
                difference = portfolio_sharpe - benchmark_sharpe
                
                metrics.append({
                    "Metric": "Sharpe Ratio",
                    "My-Portfolio": portfolio_sharpe,
                    "My-Benchmark": benchmark_sharpe,
                    "Difference": difference
                })
            
            # 5. Calculate max drawdown
            portfolio_cum_returns = (1 + portfolio_returns).cumprod()
            benchmark_cum_returns = (1 + benchmark_returns).cumprod()
            
            portfolio_max_drawdown = 1 - portfolio_cum_returns / portfolio_cum_returns.cummax()
            benchmark_max_drawdown = 1 - benchmark_cum_returns / benchmark_cum_returns.cummax()
            
            portfolio_max_dd = portfolio_max_drawdown.max()
            benchmark_max_dd = benchmark_max_drawdown.max()
            difference = portfolio_max_dd - benchmark_max_dd
            
            metrics.append({
                "Metric": "Maximum Drawdown",
                "My-Portfolio": portfolio_max_dd,
                "My-Benchmark": benchmark_max_dd,
                "Difference": difference
            })
            
            # Create DataFrame
            metrics_df = pd.DataFrame(metrics)
            
            return metrics_df
        except Exception as e:
            print(f"Error calculating portfolio metrics: {str(e)}")
            # Return placeholder DataFrame with error message
            return pd.DataFrame({
                "Metric": ["Error calculating metrics"],
                "My-Portfolio": [str(e)],
                "My-Benchmark": ["N/A"],
                "Difference": ["N/A"]
            })

    @reactive.calc
    def calculate_quantstats_metrics():
        """Calculate comprehensive portfolio metrics using quantstats.
        
        This function provides extensive portfolio performance analytics using the
        quantstats package, including risk metrics, risk-adjusted return metrics,
        drawdown statistics, and various ratios.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all metrics from quantstats.reports.metrics(mode='full')
            for both the portfolio and benchmark, with an additional column showing
            the difference between them.
        
        Notes
        -----
        Requires at least 5 data points for both portfolio and benchmark.
        
        The metrics include but are not limited to:
        
        * Various return metrics (daily, monthly, yearly, CAGR)
        * Risk metrics (volatility, Calmar ratio, Sortino ratio)
        * Value-at-Risk and Conditional Value-at-Risk
        * Alpha, Beta, and other risk-adjusted return measures
        * Maximum drawdown and recovery statistics
        
        The function formats metric names for readability and calculates
        the difference between portfolio and benchmark metrics.
        
        See Also
        --------
        calculate_portfolio_metrics : Basic performance metrics calculation
        output_ID_portfolio_quantstats_metrics : Rendering of these metrics
        """
        try:
            portfolio_df, benchmark_df = get_comparison_data()
            
            # Check if we have sufficient data
            if len(portfolio_df) < 5 or len(benchmark_df) < 5:
                # Return placeholder DataFrame with error message
                return pd.DataFrame({
                    "Metric": ["Insufficient data for calculations"],
                    "My-Portfolio": ["Need at least 5 data points for QuantStats metrics"],
                    "Benchmark-Portfolio": ["Need at least 5 data points for QuantStats metrics"],
                    "Difference": ["N/A"]  # Add Difference column to match expected structure
                })
            
            # Ensure datetime format and sort
            portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"], utc=True).dt.tz_localize(None)
            benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"], utc=True).dt.tz_localize(None)
            portfolio_df = portfolio_df.sort_values("Date")
            benchmark_df = benchmark_df.sort_values("Date")
            
            # Create returns series (needed for quantstats)
            portfolio_df["Return"] = portfolio_df["Value"].pct_change()
            benchmark_df["Return"] = benchmark_df["Value"].pct_change()
            
            # Drop first row with NaN returns
            portfolio_returns = portfolio_df.dropna()
            benchmark_returns = benchmark_df.dropna()
            
            # Set index to Date for quantstats compatibility
            portfolio_returns = portfolio_returns.set_index("Date")["Return"]
            benchmark_returns = benchmark_returns.set_index("Date")["Return"]
            
            # Calculate quantstats metrics for portfolio with benchmark as reference
            try:
                portfolio_metrics = qs.reports.metrics(
                    portfolio_returns, 
                    benchmark_returns, 
                    mode='full',
                    display=False  # Don't display, just return the DataFrame
                )
            except Exception as e:
                print(f"Error calculating portfolio metrics: {str(e)}")
                portfolio_metrics = pd.Series(dtype='float64')  # Empty series
            
            # Calculate quantstats metrics for benchmark with portfolio as reference
            try:
                benchmark_metrics = qs.reports.metrics(
                    benchmark_returns, 
                    portfolio_returns, 
                    mode='full',
                    display=False  # Don't display, just return the DataFrame
                )
            except Exception as e:
                print(f"Error calculating benchmark metrics: {str(e)}")
                benchmark_metrics = pd.Series(dtype='float64')  # Empty series
            
            # Create a new dataframe from scratch to avoid axis length mismatch
            result_data = []
            
            # Process metrics and build a unified list of dictionaries
            all_metrics = set()
            
            # Get all unique metric names
            if isinstance(portfolio_metrics, pd.Series):
                all_metrics.update(portfolio_metrics.index)
            elif isinstance(portfolio_metrics, pd.DataFrame):
                all_metrics.update(portfolio_metrics.index.values)
            
            if isinstance(benchmark_metrics, pd.Series):
                all_metrics.update(benchmark_metrics.index)
            elif isinstance(benchmark_metrics, pd.DataFrame):
                all_metrics.update(benchmark_metrics.index.values)
            
            # Create a unified DataFrame with all metrics
            for metric in all_metrics:
                row = {"Metric": metric.replace('_', ' ').title()}
                
                # Add portfolio value
                if isinstance(portfolio_metrics, pd.Series) and metric in portfolio_metrics.index:
                    row["My-Portfolio"] = portfolio_metrics[metric]
                elif isinstance(portfolio_metrics, pd.DataFrame) and metric in portfolio_metrics.index:
                    row["My-Portfolio"] = portfolio_metrics.loc[metric].iloc[0] if len(portfolio_metrics.columns) > 0 else None
                else:
                    row["My-Portfolio"] = None
                
                # Add benchmark value
                if isinstance(benchmark_metrics, pd.Series) and metric in benchmark_metrics.index:
                    row["Benchmark-Portfolio"] = benchmark_metrics[metric]
                elif isinstance(benchmark_metrics, pd.DataFrame) and metric in benchmark_metrics.index:
                    row["Benchmark-Portfolio"] = benchmark_metrics.loc[metric].iloc[0] if len(benchmark_metrics.columns) > 0 else None
                else:
                    row["Benchmark-Portfolio"] = None
                
                # Calculate difference if both values are numeric
                if (pd.notna(row.get("My-Portfolio")) and 
                    pd.notna(row.get("Benchmark-Portfolio")) and
                    isinstance(row.get("My-Portfolio"), (int, float)) and 
                    isinstance(row.get("Benchmark-Portfolio"), (int, float))):
                    row["Difference"] = row["My-Portfolio"] - row["Benchmark-Portfolio"]
                else:
                    row["Difference"] = None
                    
                result_data.append(row)
            
            # Create the final DataFrame
            metrics_df = pd.DataFrame(result_data)
            
            # Return empty DataFrame with placeholders if no data was processed
            if len(metrics_df) == 0:
                return pd.DataFrame({
                    "Metric": ["No metrics calculated"],
                    "My-Portfolio": ["N/A"],
                    "Benchmark-Portfolio": ["N/A"],
                    "Difference": ["N/A"]
                })
            
            return metrics_df
            
        except Exception as e:
            print(f"Error calculating quantstats metrics: {str(e)}")
            # Return placeholder DataFrame with error message and all required columns
            return pd.DataFrame({
                "Metric": ["Error calculating QuantStats metrics"],
                "My-Portfolio": [str(e)],
                "Benchmark-Portfolio": [str(e)],
                "Difference": ["N/A"]
            })

    @output
    @render_widget
    def output_ID_portfolio_analysis_plot():
        """Generate portfolio analysis plot.
        
        Creates interactive visualizations for analyzing portfolio performance
        through different lenses based on the selected analysis type.
        
        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure containing the selected analysis visualization.
        
        Notes
        -----
        This function creates different types of analysis plots based on user selection:
        
        * **returns**: Two-panel plot showing returns distribution histogram and time series
        * **drawdowns**: Plot showing drawdowns over time for portfolio and benchmark
        * **rolling**: Three-panel plot showing rolling annualized return, volatility, and Sharpe ratio
        
        For rolling statistics, the window size is configurable through the UI.
        
        Plots include appropriate spacing between elements, clear labels, and 
        interactive features for exploration.
        
        See Also
        --------
        get_comparison_data : Provides the data for these plots
        calculate_portfolio_metrics : Calculation of key metrics
        """
        try:
            portfolio_df, benchmark_df = get_comparison_data()
            analysis_type = input.ID_analysis_type()
            
            # Check if we have data
            if portfolio_df.empty or benchmark_df.empty:
                fig = px.scatter(title="No data available")
                fig.add_annotation(
                    text="Please check that portfolio and benchmark data files exist",
                    showarrow=False,
                    font=dict(size=14),
                )
                return fig
            
            # Ensure datetime format without timezone information
            try:
                portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"]).dt.tz_localize(None)
                benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"]).dt.tz_localize(None)
            except:
                print("Error converting dates to timezone-naive format in analysis plot")
            
            # Sort by date
            portfolio_df = portfolio_df.sort_values("Date")
            benchmark_df = benchmark_df.sort_values("Date")
            
            # Calculate daily returns
            portfolio_df["Return"] = portfolio_df["Value"].pct_change()
            benchmark_df["Return"] = benchmark_df["Value"].pct_change()
            
            # Drop first row with NaN returns
            portfolio_returns = portfolio_df.dropna()
            benchmark_returns = benchmark_df.dropna()
            
            # Create appropriate analysis plot based on selected type
            if analysis_type == "returns":
                # Create returns distribution plot
                data = pd.DataFrame({
                    "Portfolio Returns": portfolio_returns["Return"],
                    "Benchmark Returns": benchmark_returns["Return"]
                })
                
                # Increased vertical spacing from 0.1 to 0.25 for more space between subplots
                fig = make_subplots(
                    rows=2, 
                    cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.25,  # Increased from 0.1 to 0.25
                    subplot_titles=["Daily Returns Distribution", "Returns Over Time"]
                )
                
                # Add histogram
                for column, color in zip(data.columns, ["blue", "red"]):
                    fig.add_trace(
                        go.Histogram(
                            x=data[column],
                            name=column,
                            marker_color=color,
                            opacity=0.7,
                            nbinsx=30
                        ),
                        row=1, col=1
                    )
                
                # Add time series
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_returns["Date"],
                        y=portfolio_returns["Return"],
                        name="My-Portfolio",
                        line=dict(color="blue")
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_returns["Date"],
                        y=benchmark_returns["Return"],
                        name="My-Benchmark",
                        line=dict(color="red")
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title="Returns Analysis",
                    barmode="overlay",
                    # Increased the y position of the legend to add more space
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.15,  # Increased from 1.02 to 1.15
                        xanchor="right",
                        x=1
                    ),
                    template="plotly_white",
                    # Increase the height to provide more spacing
                    height=700,  # Increased from default
                    # Increase top margin for more space at the top
                    margin=dict(t=120, b=50, l=50, r=50)  # Increased top margin
                )
                
                # Update y-axis titles
                fig.update_yaxes(title_text="Frequency", row=1, col=1)
                fig.update_yaxes(title_text="Daily Return", row=2, col=1)
                fig.update_xaxes(title_text="Daily Return", row=1, col=1)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                
                # Add more space between subplot title and plot
                # Update the position of the first subplot title
                if fig.layout.annotations:
                    # First annotation is the title of the top subplot
                    fig.layout.annotations[0].update(y=fig.layout.annotations[0].y + 0.03)
                
            elif analysis_type == "drawdowns":
                # Calculate drawdowns
                portfolio_cum_returns = (1 + portfolio_returns["Return"]).cumprod()
                benchmark_cum_returns = (1 + benchmark_returns["Return"]).cumprod()
                
                portfolio_drawdowns = 1 - portfolio_cum_returns / portfolio_cum_returns.cummax()
                benchmark_drawdowns = 1 - benchmark_cum_returns / benchmark_cum_returns.cummax()
                
                # Create figure
                fig = go.Figure()
                
                # Add traces
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_returns["Date"],
                        y=portfolio_drawdowns,
                        name="Portfolio Drawdowns",
                        line=dict(color="blue"),
                        fill="tozeroy"
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_returns["Date"],
                        y=benchmark_drawdowns,
                        name="Benchmark Drawdowns",
                        line=dict(color="red", dash="dash")
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title="Drawdown Analysis",
                    xaxis_title="Date",
                    yaxis_title="Drawdown",
                    yaxis=dict(
                        tickformat=".1%",
                        range=[0, max(portfolio_drawdowns.max(), benchmark_drawdowns.max()) * 1.1]
                    ),
                    # Increase legend spacing
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.15,  # Increased from 1.02 to 1.15
                        xanchor="right",
                        x=1
                    ),
                    template="plotly_white",
                    # Increase margins
                    margin=dict(t=120, b=50, l=50, r=50)  # Increased top margin
                )
                
            elif analysis_type == "rolling":
                # Get rolling window size
                window = input.ID_rolling_window()
                
                # Calculate rolling statistics
                portfolio_rolling_return = portfolio_returns["Return"].rolling(window=window).mean() * 252
                benchmark_rolling_return = benchmark_returns["Return"].rolling(window=window).mean() * 252
                
                portfolio_rolling_vol = portfolio_returns["Return"].rolling(window=window).std() * np.sqrt(252)
                benchmark_rolling_vol = benchmark_returns["Return"].rolling(window=window).std() * np.sqrt(252)
                
                portfolio_rolling_sharpe = portfolio_rolling_return / portfolio_rolling_vol
                benchmark_rolling_sharpe = benchmark_rolling_return / benchmark_rolling_vol
                
                # Create DataFrame for plotting
                rolling_df = pd.DataFrame({
                    "Date": portfolio_returns["Date"].iloc[window-1:],
                    "Portfolio Return": portfolio_rolling_return,
                    "Benchmark Return": benchmark_rolling_return,
                    "Portfolio Volatility": portfolio_rolling_vol,
                    "Benchmark Volatility": benchmark_rolling_vol,
                    "Portfolio Sharpe": portfolio_rolling_sharpe,
                    "Benchmark Sharpe": benchmark_rolling_sharpe
                })
                
                # Create figure with three subplots - increase vertical spacing
                fig = make_subplots(
                    rows=3, 
                    cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.15,  # Increased from 0.1 to 0.15
                    subplot_titles=[
                        f"Rolling {window}-day Annualized Return", 
                        f"Rolling {window}-day Annualized Volatility",
                        f"Rolling {window}-day Sharpe Ratio"
                    ]
                )
                
                # Add return traces
                fig.add_trace(
                    go.Scatter(
                        x=rolling_df["Date"],
                        y=rolling_df["Portfolio Return"],
                        name="Portfolio Return",
                        line=dict(color="blue")
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_df["Date"],
                        y=rolling_df["Benchmark Return"],
                        name="Benchmark Return",
                        line=dict(color="red")
                    ),
                    row=1, col=1
                )
                
                # Add volatility traces
                fig.add_trace(
                    go.Scatter(
                        x=rolling_df["Date"],
                        y=rolling_df["Portfolio Volatility"],
                        name="Portfolio Volatility",
                        line=dict(color="blue", dash="dash")
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_df["Date"],
                        y=rolling_df["Benchmark Volatility"],
                        name="Benchmark Volatility",
                        line=dict(color="red", dash="dash")
                    ),
                    row=2, col=1
                )
                
                # Add Sharpe ratio traces
                fig.add_trace(
                    go.Scatter(
                        x=rolling_df["Date"],
                        y=rolling_df["Portfolio Sharpe"],
                        name="Portfolio Sharpe",
                        line=dict(color="blue", dash="dot")
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_df["Date"],
                        y=rolling_df["Benchmark Sharpe"],
                        name="Benchmark Sharpe",
                        line=dict(color="red", dash="dot")
                    ),
                    row=3, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Rolling {window}-day Statistics",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.15,  # Increased from 1.02 to 1.15
                        xanchor="right",
                        x=1
                    ),
                    template="plotly_white",
                    height=800,  # Increased from 700 to 800
                    margin=dict(t=120, b=50, l=50, r=50)  # Increased top margin
                )
                
                # Update y-axis titles
                fig.update_yaxes(title_text="Annualized Return", row=1, col=1)
                fig.update_yaxes(title_text="Annualized Volatility", row=2, col=1)
                fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
                fig.update_xaxes(title_text="Date", row=3, col=1)
                
                # Adjust subplot title positions
                if fig.layout.annotations:
                    for i in range(3):
                        if i < len(fig.layout.annotations):
                            fig.layout.annotations[i].update(y=fig.layout.annotations[i].y + 0.02)
                
            return fig
        except Exception as e:
            # Handle any errors gracefully
            print(f"Error in portfolio analysis plot: {str(e)}")
            fig = px.scatter(title=f"Error: {str(e)}")
            return fig

    @output
    @render.ui
    def output_ID_portfolio_analysis_stats():
        """Render portfolio analysis metrics tables with tabs for basic and advanced metrics."""
        # Create a tabset containing both basic metrics and quantstats metrics
        return ui.navset_tab(
            ui.nav_panel("Basic Metrics", 
                   ui.output_ui("output_ID_portfolio_basic_metrics"))
         )

    # Add a new function to render the basic metrics (previously part of output_ID_portfolio_analysis_stats)
    @output
    @render.ui
    def output_ID_portfolio_basic_metrics():
        """Render portfolio basic analysis metrics table."""
        from great_tables import GT
        
        try:
            metrics_df = calculate_portfolio_metrics()
            
            # Check if we have data
            if "Insufficient data for calculations" in metrics_df["Metric"].values or "Error calculating metrics" in metrics_df["Metric"].values:
                return ui.div(
                    ui.tags.b(metrics_df["Metric"].iloc[0]),
                    ui.p(metrics_df["My-Portfolio"].iloc[0])
                )
            
            # Get date range values and handle None values
            date_range = input.ID_analysis_date_range()
            start_date = date_range[0] if date_range and date_range[0] else "earliest"
            end_date = date_range[1] if date_range and date_range[1] else "latest"
            
            # Format dates with safe handling of None values
            if isinstance(start_date, str):
                start_date_str = start_date
            else:
                start_date_str = start_date.strftime('%Y-%m-%d')
                
            if isinstance(end_date, str):
                end_date_str = end_date
            else:
                end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Create great_tables object
            gt_table = (
                GT(metrics_df)
                .tab_header(
                    title="Portfolio Performance Metrics",
                    subtitle=f"Date Range: {start_date_str} to {end_date_str}"
                )
                .fmt_percent(
                    columns=["My-Portfolio", "My-Benchmark", "Difference"],
                    rows=lambda x: x["Metric"].isin(["Cumulative Return", "Annualized Return", "Maximum Drawdown", "Annualized Volatility"]),
                    decimals=2
                )
                .fmt_number(
                    columns=["My-Portfolio", "My-Benchmark", "Difference"],
                    rows=lambda x: x["Metric"].isin(["Sharpe Ratio"]),
                    decimals=2
                )
            )
            
            # Add source note which should work in most versions
            gt_table = gt_table.tab_source_note(
                source_note="Data calculated from portfolio and benchmark values"
            )
            
            # Try different HTML rendering methods that might be available
            try:
                # Try the as_raw_html() method first
                html_content = gt_table.as_raw_html()
                return ui.HTML(html_content)
            except:
                try:
                    # Try the render() method next with the required context parameter
                    html_content = gt_table.render(context="html")  # Add context parameter
                    return ui.HTML(html_content)
                except:
                    try:
                        # Try the as_raw_html() method next
                        html_content = gt_table.as_raw_html()
                        return ui.HTML(html_content)
                    except Exception as e:
                        print(f"Error rendering analysis table HTML: {str(e)}")
                        return ui.p("Error rendering table. Please check great_tables version and documentation.")
                
        except Exception as e:
            print(f"Error rendering portfolio analysis stats: {str(e)}")
            return ui.div(
                ui.tags.b("Error generating statistics table"),
                ui.p(str(e))
            )

    @output
    @render.ui
    def output_ID_comparison_stats():
        """Generate summary statistics for the portfolio comparison."""
        from great_tables import GT
        
        try:
            portfolio_df, benchmark_df = get_comparison_data()
            viz_type = input.ID_comparison_viz_type()
            
            # Check if we have data
            if portfolio_df.empty or benchmark_df.empty:
                return ui.p("No data available for statistics")
            
            # Calculate statistics
            stats = []
            
            # Start and end dates
            if not portfolio_df.empty and "Date" in portfolio_df.columns:
                try:
                    # Ensure dates are datetime objects before formatting
                    start_date = pd.to_datetime(portfolio_df["Date"].min(), utc=True).tz_localize(None)
                    end_date = pd.to_datetime(portfolio_df["Date"].max(), utc=True).tz_localize(None)
                    date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                except Exception as e:
                    print(f"Date formatting error: {str(e)}")
                    # Fallback to string representation if formatting fails
                    date_range = f"{str(portfolio_df['Date'].min())} to {str(portfolio_df['Date'].max())}"
            else:
                date_range = "N/A"
            
            stats.append({
                "Statistic": "Date Range",
                "Value": date_range
            })
            
            # Number of data points
            if not portfolio_df.empty:
                num_points = len(portfolio_df)
                stats.append({
                    "Statistic": "Data Points",
                    "Value": num_points
                })
            
            # Portfolio Statistics
            if not portfolio_df.empty and "Value" in portfolio_df.columns:
                # Start and end values
                start_value = portfolio_df["Value"].iloc[0]
                end_value = portfolio_df["Value"].iloc[-1]
                
                stats.append({
                    "Statistic": "Portfolio Start Value",
                    "Value": start_value
                })
                
                stats.append({
                    "Statistic": "Portfolio End Value",
                    "Value": end_value
                })
                
                # Total return
                total_return = (end_value / start_value - 1)
                stats.append({
                    "Statistic": "Portfolio Total Return",
                    "Value": total_return
                })
            
            # Benchmark Statistics
            if not benchmark_df.empty and "Value" in benchmark_df.columns:
                # Start and end values
                start_value = benchmark_df["Value"].iloc[0]
                end_value = benchmark_df["Value"].iloc[-1]
                
                stats.append({
                    "Statistic": "Benchmark Start Value",
                    "Value": start_value
                })
                
                stats.append({
                    "Statistic": "Benchmark End Value",
                    "Value": end_value
                })
                
                # Total return
                total_return = (end_value / start_value - 1)
                stats.append({
                    "Statistic": "Benchmark Total Return",
                    "Value": total_return
                })
            
            # Create DataFrame
            stats_df = pd.DataFrame(stats)
            
            # Create great_tables object with minimal styling to avoid errors
            gt_table = (
                GT(stats_df)
                .tab_header(
                    title="Value Statistics",
                    subtitle="Summary statistics for portfolio and benchmark"
                )
                .fmt_number(
                    columns=["Value"],
                    rows=lambda x: x["Statistic"].str.contains("Value"),
                    decimals=2
                )
                .fmt_percent(
                    columns=["Value"],
                    rows=lambda x: x["Statistic"].str.contains("Return"),
                    decimals=2
                )
                .fmt_number(
                    columns=["Value"],
                    rows=lambda x: x["Statistic"] == "Data Points",
                    decimals=0
                )
            )
            
            # Try different HTML rendering methods that might be available
            try:
                # Try the as_raw_html() method first
                html_content = gt_table.as_raw_html()
                return ui.HTML(html_content)
            except:
                try:
                    # Try the render() method next with the required context parameter
                    html_content = gt_table.render(context="html")  # Add context parameter
                    return ui.HTML(html_content)
                except:
                    try:
                        # Try the as_raw_html() method next
                        html_content = gt_table.as_raw_html()
                        return ui.HTML(html_content)
                    except Exception as e:
                        print(f"Error rendering comparison table HTML: {str(e)}")
                        return ui.p("Error rendering table. Please check great_tables version and documentation.")
            
        except Exception as e:
            print(f"Error generating comparison stats: {str(e)}")
            return ui.div(
                ui.tags.b("Error generating statistics"),
                ui.p(str(e))
            )

    @output
    @render.ui
    def output_ID_portfolio_quantstats_metrics():
        """Render comprehensive portfolio metrics table using quantstats and great_tables.
        
        Creates an HTML table displaying extensive portfolio performance metrics
        calculated using the quantstats package, with side-by-side comparison of
        portfolio and benchmark metrics.
        
        Returns
        -------
        shiny.ui.tags.div or shiny.ui.HTML
            HTML content containing the formatted metrics table.
        
        Notes
        -----
        This function attempts multiple rendering methods for compatibility:
        
        1. First tries great_tables' as_raw_html() method
        2. Then tries great_tables' render(context="html") method
        3. Falls back to custom HTML table generation if both methods fail
        
        The table includes:
        
        * Comprehensive metrics for both portfolio and benchmark
        * Difference column showing the spread between portfolio and benchmark
        * Appropriate formatting for percentage and numeric values
        * Rounded values for the Difference column (to 2 decimal places)
        
        See Also
        --------
        calculate_quantstats_metrics : Calculation of the metrics displayed in this table
        """
        from great_tables import GT
        
        try:
            metrics_df = calculate_quantstats_metrics()
            
            # Check if we have data or if there was an error
            error_keywords = ["Insufficient data", "Error calculating"]
            has_error = False
            
            if not metrics_df.empty:
                for keyword in error_keywords:
                    if any(metrics_df["Metric"].astype(str).str.contains(keyword)):
                        has_error = True
                        break
            
            if has_error:
                return ui.div(
                    ui.tags.b(metrics_df["Metric"].iloc[0]),
                    ui.p(metrics_df["My-Portfolio"].iloc[0])
                )
            
            # Get date range values for the subtitle
            date_range = input.ID_analysis_date_range()
            start_date = date_range[0] if date_range and date_range[0] else "earliest"
            end_date = date_range[1] if date_range and date_range[1] else "latest"
            
            # Format dates with safe handling
            if isinstance(start_date, str):
                start_date_str = start_date
            else:
                start_date_str = start_date.strftime('%Y-%m-%d')
                
            if isinstance(end_date, str):
                end_date_str = end_date
            else:
                end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Create a copy of the DataFrame to avoid modifying the original
            display_df = metrics_df.copy()
            
            # Pre-format the Difference column to round to 2 decimal places
            display_df["Difference"] = display_df["Difference"].apply(
                lambda x: round(float(x), 2) if pd.notna(x) and isinstance(x, (int, float)) else x
            )
            
            # Identify rows for percentage formatting
            percent_patterns = ["return", "ratio", "alpha", "beta", "sharpe", "sortino", 
                              "volatility", "drawdown", "var", "cvar", "calmar", "common"]
            
            # Create a mask for percentage rows - using string methods safely
            percentage_mask = display_df["Metric"].astype(str).str.lower().apply(
                lambda x: any(pattern in x for pattern in percent_patterns)
            )
            
            # Create a list of row indices for percentage formatting
            percentage_row_indices = percentage_mask[percentage_mask].index.tolist()
            
            # Create a list of non-percentage row indices
            non_percentage_row_indices = percentage_mask[~percentage_mask].index.tolist()
            
            # Create great_tables object with appropriate formatting
            gt_table = GT(display_df).tab_header(
                title="QuantStats Portfolio Metrics Comparison (Full)",
                subtitle=f"Period: {start_date_str} to {end_date_str}"
            )
            
            # Apply percentage formatting if we have percentage rows
            if percentage_row_indices:
                gt_table = gt_table.fmt_percent(
                    columns=["My-Portfolio", "Benchmark-Portfolio", "Difference"],
                    rows=percentage_row_indices,
                    decimals=2
                )
            
            # Apply number formatting if we have non-percentage rows
            if non_percentage_row_indices:
                gt_table = gt_table.fmt_number(
                    columns=["My-Portfolio", "Benchmark-Portfolio", "Difference"],
                    rows=non_percentage_row_indices,
                    decimals=2
                )
            
            # Add source note
            gt_table = gt_table.tab_source_note(
                source_note="Metrics calculated using QuantStats Python package"
            )
            
            # Try different HTML rendering methods
            try:
                # Try the as_raw_html() method first
                html_content = gt_table.as_raw_html()
                return ui.HTML(html_content)
            except Exception as e1:
                print(f"First rendering attempt failed: {str(e1)}")
                try:
                    # Try the render() method next with the required context parameter
                    html_content = gt_table.render(context="html")  # Add context parameter
                    return ui.HTML(html_content)
                except Exception as e2:
                    print(f"Second rendering attempt failed: {str(e2)}")
                    try:
                        # Fallback to custom HTML table with rounded values
                        html = "<table border='1' style='width:100%; border-collapse:collapse;'>"
                        html += "<thead><tr style='background-color:#f2f2f2;'>"
                        html += "<th style='padding:8px; text-align:left;'>Metric</th>"
                        html += "<th style='padding:8px; text-align:right;'>My-Portfolio</th>"
                        html += "<th style='padding:8px; text-align:right;'>Benchmark-Portfolio</th>"
                        html += "<th style='padding:8px; text-align:right;'>Difference</th>"
                        html += "</tr></thead><tbody>"
                        
                        for _, row in display_df.iterrows():
                            metric = row['Metric']
                            
                            # Format My-Portfolio value
                            port_val = row['My-Portfolio']
                            if pd.notna(port_val) and isinstance(port_val, (int, float)):
                                # Check if it's a percentage metric
                                if any(pattern in str(metric).lower() for pattern in percent_patterns):
                                    port_val = f"{port_val:.2%}"
                                else:
                                    port_val = f"{port_val:.2f}"
                            else:
                                port_val = str(port_val) if pd.notna(port_val) else "N/A"
                            
                            # Format Benchmark value
                            bench_val = row['Benchmark-Portfolio']
                            if pd.notna(bench_val) and isinstance(bench_val, (int, float)):
                                # Check if it's a percentage metric
                                if any(pattern in str(metric).lower() for pattern in percent_patterns):
                                    bench_val = f"{bench_val:.2%}"
                                else:
                                    bench_val = f"{bench_val:.2f}"
                            else:
                                bench_val = str(bench_val) if pd.notna(bench_val) else "N/A"
                            
                            # Format Difference value - always to 2 decimal places
                            diff_val = row['Difference']
                            if pd.notna(diff_val) and isinstance(diff_val, (int, float)):
                                # Check if it's a percentage metric
                                if any(pattern in str(metric).lower() for pattern in percent_patterns):
                                    diff_val = f"{diff_val:.2%}"
                                else:
                                    diff_val = f"{diff_val:.2f}"
                            else:
                                diff_val = str(diff_val) if pd.notna(diff_val) else "N/A"
                            
                            # Add row to table
                            html += f"<tr><td style='padding:8px; border-top:1px solid #ddd;'>{metric}</td>"
                            html += f"<td style='padding:8px; border-top:1px solid #ddd; text-align:right;'>{port_val}</td>"
                            html += f"<td style='padding:8px; border-top:1px solid #ddd; text-align:right;'>{bench_val}</td>"
                            html += f"<td style='padding:8px; border-top:1px solid #ddd; text-align:right;'>{diff_val}</td></tr>"
                        
                        html += "</tbody></table>"
                        html += "<div style='font-size:0.8em; margin-top:10px;'>Metrics calculated using QuantStats Python package</div>"
                        
                        return ui.HTML(html)
                        
                    except Exception as e3:
                        print(f"All rendering attempts failed: {str(e3)}")
                        return ui.p("Error rendering metrics table. See console for details.")
            
        except Exception as e:
            print(f"Error rendering QuantStats metrics table: {str(e)}")
            return ui.div(
                ui.tags.b("Error generating QuantStats metrics table"),
                ui.p(str(e))
            )