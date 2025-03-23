"""
Inputs Module for QWIM Dashboard
================================

This module provides the UI and server components for the Inputs tab
of the QWIM Dashboard.

It enables users to select time series data, filter by date ranges,
and visualize selected data before proceeding to analysis.

Features
--------
* Time series selection through checkboxes
* Date range filtering with presets
* Interactive data preview with summary statistics
* Optimized visualizations for selected time series

Functions
---------
.. autosummary::
   :toctree: generated/
   
   inputs_ui
   inputs_server

See Also
--------
analysis_module : Provides analysis capabilities for selected data
results_module : Presents analysis results in various formats
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from shiny import module, reactive, render, ui
from shinywidgets import render_plotly, output_widget, render_widget


@module.ui
def inputs_ui():
    """Create the UI for the Data Selection tab.
    
    This function generates the complete user interface for the Inputs tab of
    the QWIM Dashboard. It provides controls for users to select time series data,
    filter by date ranges, and preview the selected data.
    
    The UI contains:
    
    - Time period selection with both presets and custom date ranges
    - Series selection through checkboxes with bulk selection options
    - Data preview with statistical summaries and visualization
    - Quick selection presets for common time periods and series groupings
    
    Returns
    -------
    shiny.ui.page_fluid
        A fluid page layout containing all UI elements for the Inputs tab
    
    Notes
    -----
    This UI layout uses a sidebar design pattern, with controls on the left
    and previews on the right. The time period presets provide quick access to
    common time frames, while custom date selection offers more flexibility.
    
    See Also
    --------
    inputs_server : Server-side logic for this UI
    apply_filters : Function that processes the filter selections
    """
    return ui.page_fluid(
        ui.h2("Data Selection and Filtering"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Time Period"),
                ui.input_select(
                    "ID_preset_time_periods",
                    "Preset Time Periods:",
                    {
                        "custom": "Custom",
                        "1y": "Last 1 Year",
                        "5y": "Last 5 Years",
                        "10y": "Last 10 Years",
                        "ytd": "Year to Date",
                    },
                    selected="5y",
                ),
                ui.input_date_range(
                    "ID_custom_date_range",
                    "Custom Date Range:",
                    start="2018-01-01",
                    end="2023-01-01",
                ),
                ui.h3("Series Selection"),
                ui.input_checkbox_group(
                    "ID_selected_series",
                    "Select Time Series:",
                    [],
                    selected=[],
                ),
                ui.h3("Date Range"),
                ui.input_date_range(
                    "ID_date_range",
                    "Select Date Range:",
                    start="2002-01-01",
                    end="2025-03-31",
                    format="yyyy-mm-dd",
                    separator=" to ",
                ),
                ui.input_action_button(
                    "ID_apply_filters", 
                    "Apply Filters", 
                    class_="btn-primary",
                ),
                ui.hr(),
                ui.h3("Quick Selections"),
                ui.input_action_button(
                    "ID_select_all", 
                    "Select All Series", 
                    class_="btn-outline-secondary",
                ),
                ui.input_action_button(
                    "ID_clear_selection", 
                    "Clear Selection", 
                    class_="btn-outline-secondary",
                ),
                ui.hr(),
                ui.h4("Presets"),
                ui.input_select(
                    "ID_time_period_preset",
                    "Time Period Preset:",
                    {
                        "all": "All Data",
                        "1y": "Last Year",
                        "5y": "Last 5 Years",
                        "10y": "Last 10 Years",
                        "ytd": "Year to Date",
                    },
                    selected="all",
                ),
                ui.input_select(
                    "ID_series_preset",
                    "Series Preset:",
                    {
                        "none": "None",
                        "first3": "First 3 Series",
                        "trending": "Trending Up Series",
                        "volatile": "Most Volatile Series",
                    },
                    selected="none",
                ),
                ui.input_action_button(
                    "ID_apply_presets", 
                    "Apply Presets", 
                    class_="btn-outline-info",
                ),
            ),
            ui.h4("Selected Series Overview"),
            output_widget("output_ID_overview_plot"),
            ui.h4("Series Statistics"),
            ui.output_table("output_ID_series_stats"),
            ui.h4("Selected Data"),
            ui.output_data_frame("output_ID_selected_data_preview"),
        ),
    )


@module.server
def inputs_server(input, output, session, data_r, series_names_r):
    """Server logic for the Inputs tab.
    
    This function implements all server logic for the Data Selection tab,
    handling reactive events, data filtering, and visualization generation.
    It manages user selections for time series data and date ranges, and
    produces optimized visualizations of the selected data.
    
    Parameters
    ----------
    input : shiny.module_input.ModuleInput
        Input object containing all user interface inputs
    output : shiny.module_output.ModuleOutput
        Output object for rendering results back to the UI
    session : shiny.Session
        Shiny session object for managing client state
    data_r : reactive.Value
        Reactive value containing the complete dataset
    series_names_r : reactive.Value
        Reactive value containing all available time series names
        
    Returns
    -------
    reactive.calc
        Reactive calculation function that returns filtered data and selected series
        
    Notes
    -----
    This server function contains several types of reactive components:
    
    - reactive.effect: Handle UI updates and button clicks
    - reactive.calc: Process data filtering operations
    - output renderers: Generate visualizations and tables
    
    The apply_filters function is both used internally and returned for
    use by other modules that need access to the selected data.
    
    See Also
    --------
    inputs_ui : UI component for the Inputs tab
    apply_filters : Main filtering function that processes user selections
    """
    # Initialize reactive values
    filtered_data = reactive.Value(None)
    selected_series = reactive.Value([])
    
    @reactive.effect
    def _update_series_options():
        """Update the series selection options when series names change.
        
        This effect synchronizes the UI checkbox group with the available
        series names, preserving any existing selections when possible.
        """
        series_names = series_names_r()
        ui.update_checkbox_group(
            "ID_selected_series",
            choices=series_names,
            selected=selected_series(),
        )
    
    @reactive.effect
    @reactive.event(input.ID_select_all)
    def _handle_select_all():
        """Handle the 'Select All' button click.
        
        Selects all available series when the user clicks the 'Select All' button.
        """
        series_names = series_names_r()
        ui.update_checkbox_group(
            "ID_selected_series",
            selected=series_names,
        )
    
    @reactive.effect
    @reactive.event(input.ID_clear_selection)
    def _handle_clear_selection():
        """Handle the 'Clear Selection' button click.
        
        Clears all selected series when the user clicks the 'Clear Selection' button.
        """
        ui.update_checkbox_group(
            "ID_selected_series",
            selected=[],
        )
    
    @reactive.effect
    @reactive.event(input.ID_apply_presets)
    def _handle_apply_presets():
        """Handle the 'Apply Presets' button click.
        
        Applies the selected time period and series presets, updating
        both the date range and series selection accordingly.
        
        The function supports several preset types:
        - Time presets: 1y, 5y, 10y, ytd, all
        - Series presets: first3, trending, volatile
        """
        data = data_r()
        time_preset = input.ID_time_period_preset()
        series_preset = input.ID_series_preset()
        
        # Parse the date column
        dates = pd.to_datetime(data.select("date").to_pandas()["date"])
        
        # Apply time preset
        if time_preset == "1y":
            max_date = dates.max()
            min_date = max_date - pd.DateOffset(years=1)
        elif time_preset == "5y":
            max_date = dates.max()
            min_date = max_date - pd.DateOffset(years=5)
        elif time_preset == "10y":
            max_date = dates.max()
            min_date = max_date - pd.DateOffset(years=10)
        elif time_preset == "ytd":
            max_date = dates.max()
            min_date = pd.Timestamp(max_date.year, 1, 1)
        else:  # all data
            min_date = dates.min()
            max_date = dates.max()
        
        # Update date range input
        ui.update_date_range(
            "ID_date_range",
            start=min_date.strftime("%Y-%m-%d"),
            end=max_date.strftime("%Y-%m-%d"),
        )
        
        # Apply series preset
        selected = []
        if series_preset == "first3":
            selected = series_names_r()[:3] if len(series_names_r()) >= 3 else series_names_r()
        elif series_preset == "trending":
            # Identify trending series (those with positive slope)
            slopes = {}
            for series in series_names_r():
                y = data.select(series).to_numpy().flatten()
                x = np.arange(len(y))
                slope, _ = np.polyfit(x, y, 1)
                slopes[series] = slope
            
            # Sort by descending slope and select top 3
            trending_series = sorted(slopes.items(), key=lambda x: x[1], reverse=True)
            selected = [s[0] for s in trending_series[:3]]
        elif series_preset == "volatile":
            # Identify volatile series (those with highest coefficient of variation)
            volatility = {}
            for series in series_names_r():
                values = data.select(series).to_numpy().flatten()
                # Handle cases with zero mean
                mean_value = np.mean(values)
                volatility[series] = np.std(values) / mean_value if mean_value != 0 else np.inf
            
            # Sort by descending volatility and select top 3
            volatile_series = sorted(volatility.items(), key=lambda x: x[1], reverse=True)
            selected = [s[0] for s in volatile_series[:3] if not np.isinf(s[1])]
            
            # If no valid series found (all had zero means), just take first 3
            if not selected and len(series_names_r()) > 0:
                selected = series_names_r()[:3] if len(series_names_r()) >= 3 else series_names_r()
        else:  # Keep current selection
            selected = input.ID_selected_series()
        
        ui.update_checkbox_group(
            "ID_selected_series",
            selected=selected,
        )
        
        # Apply the filters after updating selections
        apply_filters()
    
    @reactive.calc
    @reactive.event(input.ID_apply_filters)
    def apply_filters():
        """Apply data filters based on user selections.
        
        This function filters the data based on the selected date range and series,
        and updates the reactive values for filtered data and selected series.
        
        Returns
        -------
        dict
            Dictionary containing:
            - filtered_data: polars.DataFrame with filtered data
            - selected_series: list of selected series names
        """
        data = data_r()
        date_range = input.ID_date_range()
        selected = input.ID_selected_series()
        
        # Filter by date range
        if date_range[0] and date_range[1]:
            filtered = data.filter(
                (pl.col("date") >= date_range[0]) & 
                (pl.col("date") <= date_range[1]),
            )
        else:
            filtered = data
        
        # Update reactive values
        filtered_data.set(filtered)
        selected_series.set(selected)
        
        return {
            "filtered_data": filtered,
            "selected_series": selected,
        }
    
    @output
    @render_widget
    def output_ID_overview_plot():
        """Generate overview plot of selected time series.
        
        Creates an optimized visualization of the selected time series,
        with automatic downsampling for large datasets and performance
        optimizations to ensure smooth rendering in the dashboard.
        
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure for rendering with render_widget
        """
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Get current filters
        filters_result = apply_filters()
        filtered = filters_result["filtered_data"]
        selected = filters_result["selected_series"]
        
        if not selected or filtered is None or filtered.is_empty():
            # Return a simple empty plot with minimal overhead
            fig = go.Figure()
            fig.update_layout(
                title="No data selected",
                xaxis_title="Date",
                yaxis_title="Value",
                showlegend=False,
                template="plotly_white",
                height=400,
                margin=dict(l=50, r=50, t=70, b=50),
            )
            fig.add_annotation(
                text="Select one or more time series and apply filters to view data",
                showarrow=False,
                font=dict(size=14),
            )
            return fig
        
        # Convert to pandas for easier plotting
        # Ensure selected is a list before concatenation
        columns_to_select = ["date"] + (selected if isinstance(selected, list) else list(selected))
        pd_data = filtered.select(columns_to_select).to_pandas()
        pd_data["date"] = pd.to_datetime(pd_data["date"])
        
        # Downsample data if there are too many points
        data_points = len(pd_data)
        if data_points > 500:
            # Calculate frequency based on number of points
            freq = f"{max(1, data_points // 500)}D"
            
            # Group by date frequency, preserving important statistics
            downsampled = pd_data.set_index("date").resample(freq).agg({
                col: ['mean', 'min', 'max', 'first', 'last'] for col in pd_data.columns if col != "date"
            }).reset_index()
            
            # Convert multi-level columns to single level
            downsampled.columns = ['date'] + [f"{col[0]}_{col[1]}" for col in downsampled.columns[1:]]
            
            # Create mapping to original columns for plotting
            col_map = {}
            for series in selected:
                col_map[series] = f"{series}_mean"
            
            # Use downsampled data for plotting
            plot_data = downsampled
        else:
            # Use original data if small enough
            plot_data = pd_data
            col_map = {series: series for series in selected}
        
        # Limit number of subplots if there are many series
        max_subplots = min(len(selected), 5)  # Limit to 5 subplots max
        if len(selected) > max_subplots:
            selected_limited = selected[:max_subplots]
            fig = make_subplots(
                rows=max_subplots, 
                cols=1,
                shared_xaxes=True,
                subplot_titles=selected_limited + [f"+ {len(selected) - max_subplots} more series (not shown)"],
                vertical_spacing=0.05,
            )
        else:
            selected_limited = selected
            fig = make_subplots(
                rows=len(selected), 
                cols=1,
                shared_xaxes=True,
                subplot_titles=selected,
                vertical_spacing=0.05,
            )
        
        # Add traces for each series with optimized approach
        for i, series in enumerate(selected_limited):
            # Calculate y-axis range efficiently
            if data_points > 500:  # Use min/max columns from downsampled data
                series_min = plot_data[f"{series}_min"].min()
                series_max = plot_data[f"{series}_max"].max()
            else:
                series_min = plot_data[series].min()
                series_max = plot_data[series].max()
                
            y_margin = (series_max - series_min) * 0.1 if series_max > series_min else 0.1
            
            # Ensure we have valid y-range
            if pd.isna(series_min) or pd.isna(series_max) or series_min == series_max:
                y_min, y_max = -1, 1  # Default range
            else:
                y_min = series_min - y_margin
                y_max = series_max + y_margin
            
            # Add line trace
            plot_col = col_map.get(series, series)
            fig.add_trace(
                go.Scatter(
                    x=plot_data["date"], 
                    y=plot_data[plot_col],
                    mode="lines",
                    name=series,
                    line=dict(width=2),
                ),
                row=i+1, 
                col=1,
            )
            
            # Only add min/max markers for smaller datasets
            if data_points <= 500:
                non_na_data = pd_data.dropna(subset=[series])
                
                if len(non_na_data) > 0:
                    try:
                        # Use nlargest/nsmallest for better performance
                        max_row = non_na_data.nlargest(1, series).iloc[0]
                        fig.add_trace(
                            go.Scatter(
                                x=[max_row["date"]],
                                y=[max_row[series]],
                                mode="markers+text",
                                text=["Max"],
                                textposition="top center",
                                marker=dict(size=8, color="red"),
                                showlegend=False,
                            ),
                            row=i+1,
                            col=1,
                        )
                        
                        min_row = non_na_data.nsmallest(1, series).iloc[0]
                        fig.add_trace(
                            go.Scatter(
                                x=[min_row["date"]],
                                y=[min_row[series]],
                                mode="markers+text",
                                text=["Min"],
                                textposition="bottom center",
                                marker=dict(size=8, color="blue"),
                                showlegend=False,
                            ),
                            row=i+1,
                            col=1,
                        )
                    except (KeyError, ValueError, IndexError):
                        pass
            
            # Set y-axis range
            fig.update_yaxes(
                range=[y_min, y_max],
                row=i+1,
                col=1,
            )
        
        # Simplified layout for better performance
        fig.update_layout(
            title="Selected Time Series Overview",
            showlegend=False,
            height=max(400, 200 + 200 * len(selected_limited)),
            template="plotly_white",
            margin=dict(l=50, r=50, t=70, b=50),
        )
        
        # Add range selector only to the bottom subplot
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]),
            ),
            row=len(selected_limited),
            col=1,
        )
        
        # Performance-optimized configuration
        config = {
            'displayModeBar': False,  # Hide the mode bar for cleaner UI
            'doubleClick': 'reset',   # Reset on double click instead of complex calculations
        }
        
        return fig
    
    @output
    @render.table
    def output_ID_series_stats():
        """Generate summary statistics table for selected time series.
        
        Creates a table showing key statistical measures for each selected
        series, including mean, standard deviation, quantiles, and trend.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing summary statistics for each selected series
        """
        # Get current filters
        filters_result = apply_filters()
        filtered = filters_result["filtered_data"]
        selected = filters_result["selected_series"]
        
        if not selected or filtered is None or filtered.is_empty():
            return pd.DataFrame({"Message": ["No data selected"]})
        
        # Calculate statistics for each selected series
        stats = []
        for series in selected:
            series_data = filtered.select(series).to_numpy().flatten()
            
            # Calculate linear trend
            x = np.arange(len(series_data))
            if len(series_data) > 1:  # Need at least 2 points for regression
                slope, intercept = np.polyfit(x, series_data, 1)
                trend = "↑" if slope > 0 else "↓"
                trend_strength = abs(slope) / np.mean(series_data) if np.mean(series_data) != 0 else 0
                trend_desc = f"{trend} {trend_strength:.2%}"
            else:
                trend_desc = "N/A"
            
            stats.append({
                "Series": series,
                "Mean": round(np.mean(series_data), 2),
                "Std Dev": round(np.std(series_data), 2),
                "Min": round(np.min(series_data), 2),
                "25%": round(np.percentile(series_data, 25), 2),
                "Median": round(np.median(series_data), 2),
                "75%": round(np.percentile(series_data, 75), 2),
                "Max": round(np.max(series_data), 2),
                "Trend": trend_desc,
            })
        
        return pd.DataFrame(stats)
    
    @output
    @render.data_frame
    def output_ID_selected_data_preview():
        """Generate a preview of the selected data.
        
        Creates a data grid showing the first 10 rows of the selected
        data, including the date and all selected time series.
        
        Returns
        -------
        render.DataGrid
            DataGrid object showing a preview of the filtered data
        """
        # Get current filters
        filters_result = apply_filters()
        filtered = filters_result["filtered_data"]
        selected = filters_result["selected_series"]
        
        if not selected or filtered is None or filtered.is_empty():
            return pd.DataFrame({"Message": ["No data selected"]})
        
        # Select only date and selected series
        # Ensure selected is a list before concatenation
        columns_to_select = ["date"] + (selected if isinstance(selected, list) else list(selected))
        preview_data = filtered.select(columns_to_select)
        
        # Convert to pandas and format date
        pd_data = preview_data.to_pandas()
        pd_data["date"] = pd.to_datetime(pd_data["date"])
        
        # Configure the data frame display options
        return render.DataGrid(
            pd_data.head(10),
            width="100%",
            height="300px",
            summary=f"Showing first 10 of {len(pd_data)} rows",
        )
    
    @reactive.effect
    def _initialize_defaults():
        """Initialize default selections when the module loads.
        
        This effect runs once when the module initializes, selecting
        the first 3 series by default and applying the initial filters.
        """
        # Wait for data to be available
        if data_r() is not None and len(series_names_r()) > 0:
            # Select first 3 series by default
            default_selected = series_names_r()[:3] if len(series_names_r()) >= 3 else series_names_r()
            ui.update_checkbox_group(
                "ID_selected_series",
                selected=default_selected,
            )
            # Apply default filters
            apply_filters()
    
    # Return reactive value for use in other modules
    return apply_filters