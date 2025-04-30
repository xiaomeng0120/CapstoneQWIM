"""
Results Module for QWIM Dashboard
=================================

This module provides the UI and server components for the Results tab
of the QWIM Dashboard, offering summary visualizations and comparative analysis.

Features
--------
* Interactive time series visualizations with multiple plot types
* Data normalization options (Z-score, Min-Max, etc.)
* Statistical summary tables for quantitative analysis
* Automated insights generation for quick interpretation
* Comparative analysis between multiple selected series

Functions
---------
.. autosummary::
   :toctree: generated/

   results_ui
   results_server
   normalize_data
   transform_data

See Also
--------
inputs_module : Handles data selection and initial filtering
analysis_module : Provides detailed analytical capabilities
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.io import write_image
from plotnine import (aes, element_text, facet_wrap, geom_line, geom_point,
                     ggplot, labs, scale_color_brewer, theme, theme_minimal)
from shiny import module, reactive, render, ui
from shinywidgets import render_plotly, output_widget, render_widget


@module.ui
def results_ui():
    """Create the UI for the Results tab.
    
    This function generates the complete user interface for the Results tab of
    the QWIM Dashboard, providing visualizations, statistics, and insights
    about the selected time series data. The interface is organized in a sidebar layout
    with controls on the left and visualizations on the right.
    
    The UI contains:
    
    - Visualization Options: Controls for plot type, normalization method, and baseline
    - Interactive Plot: Main visualization area for selected time series
    - Summary Statistics: Table displaying key metrics for selected series
    - Automated Insights: Dynamically generated interpretations and observations
    - Download Options: Buttons to export plots and data
    
    Returns
    -------
    shiny.ui.page_fluid
        A fluid page layout containing all UI elements for the Results tab
    
    Notes
    -----
    This UI layout uses a sidebar design pattern, with controls on the left and
    visualizations on the right. The visualization options adapt based on the
    selected plot type, with certain options only becoming relevant for specific
    plot types (e.g., baseline date for index plots).
    
    See Also
    --------
    results_server : Server-side logic for this UI
    normalize_data : Function that handles data normalization
    transform_data : Function that transforms data based on plot type
    """
    return ui.page_fluid(
        ui.h2("Results and Insights"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Visualization Options"),
                ui.input_radio_buttons(
                    "ID_plot_type",
                    "Plot Type:",
                    {
                        "time_series": "Time Series",
                        "percent_change": "Percent Change",
                        "cumulative": "Cumulative Change",
                        "comparative": "Comparative Analysis",
                    },
                    selected="time_series",
                ),
                ui.hr(),
                ui.input_radio_buttons(
                    "ID_normalize",
                    "Normalization:",
                    {
                        "none": "None",
                        "min_max": "Min-Max",
                        "z_score": "Z-Score",
                    },
                    selected="none",
                ),
                ui.hr(),
                ui.input_select(
                    "ID_baseline_date",
                    "Baseline Date for Comparison:",
                    choices=[],  # Will be populated in server
                ),
                ui.h4("Download Options"),
                ui.download_button("ID_download_plot", "Download Plot"),
                ui.download_button("ID_download_data", "Download Data"),
            ),
            ui.h3("Visualization"),
           # ui.output_ui("output_ID_results_plot", height="600px"),
            output_widget("output_ID_results_plot", height="600px"),
            ui.h3("Summary Statistics"),
            ui.output_table("output_ID_results_table"),
            ui.h3("Insights"),
            ui.output_ui("output_ID_results_insights"),
        ),
    )


@module.server
def results_server(input, output, session, inputs_data, data_r, series_names_r):
    """Server logic for the Results tab.
    
    This function implements all server logic for the Results tab of the QWIM Dashboard,
    handling data transformation, visualization, statistical analysis, and insight generation.
    It processes user inputs to create interactive visualizations and summary statistics
    for selected time series data.
    
    Parameters
    ----------
    input : shiny.module_input.ModuleInput
        Input object containing all user interface inputs
    output : shiny.module_output.ModuleOutput
        Output object for rendering results back to the UI
    session : shiny.Session
        Shiny session object for managing client state
    inputs_data : reactive.Value
        Reactive value containing user selections from the Inputs tab
    data_r : reactive.Value
        Reactive value containing the complete dataset
    series_names_r : reactive.Value
        Reactive value containing all available series names
        
    Returns
    -------
    None
        This function does not return a value but creates reactive outputs
    
    Notes
    -----
    This server function contains several types of components:
    
    - Helper functions for data processing and transformation
    - Reactive calculations for dynamic content generation
    - Output renderers for visualizations, tables, and insights
    - Download handlers for exporting data and plots
    
    See Also
    --------
    results_ui : UI component for the Results tab
    normalize_data : Function for data normalization
    transform_data : Function for plot-specific data transformation
    """
    # Update baseline date options based on filtered data
    @reactive.effect
    def _update_baseline_date_options():
        """Update the baseline date dropdown when filtered data changes.
        
        This effect refreshes the available baseline date options whenever 
        the filtered data changes, ensuring that users can only select dates
        that are present in the current dataset.
        """
        selected_data = inputs_data()
        filtered_data = selected_data["filtered_data"]
        
        if filtered_data is not None and not filtered_data.is_empty():
            # Get unique dates from filtered data
            dates = filtered_data.select("date").to_pandas()["date"].dt.strftime('%Y-%m-%d').tolist()
            
            # Update select input
            ui.update_select(
                "ID_baseline_date", 
                choices=dates,
                selected=dates[0] if dates else None,
            )
    
    def normalize_data(data, method, selected_series):
        """Normalize data based on the selected method.
        
        This function applies various normalization techniques to the selected 
        time series data, making it easier to compare series with different scales.
        
        Parameters
        ----------
        data : polars.DataFrame
            The filtered data containing time series to normalize
        method : str
            Normalization method to apply:
            - 'none': No normalization
            - 'zscore': Z-score normalization (mean=0, std=1)
            - 'minmax': Min-Max normalization (range [0,1])
            - 'percent': Percent change from first value
        selected_series : list or tuple
            Names of selected time series to normalize
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the normalized data with original date column
            and transformed series values
            
        Notes
        -----
        The function handles edge cases like division by zero and
        ensures date formatting is preserved during transformation.
        """
        # Convert to pandas for easier manipulation
        # Ensure selected_series is a list before concatenation
        cols_to_select = ["date"] + (
            selected_series if isinstance(selected_series, list) else list(selected_series)
        )
        pd_data = data.select(cols_to_select).to_pandas()
        pd_data["date"] = pd.to_datetime(pd_data["date"])
        
        # No normalization
        if method == "none":
            return pd_data
        
        # Create a copy to avoid modifying the original
        normalized = pd_data.copy()
        
        # Convert selected_series to list if it's not already
        series_list = selected_series if isinstance(selected_series, list) else list(selected_series)
        
        # Apply normalization to each series
        for series in series_list:
            series_data = pd_data[series].values
            
            if method == "zscore":
                # Z-score normalization
                mean = np.mean(series_data)
                std = np.std(series_data)
                if std > 0:  # Avoid division by zero
                    normalized[series] = (series_data - mean) / std
                
            elif method == "minmax":
                # Min-max normalization
                min_val = np.min(series_data)
                max_val = np.max(series_data)
                if max_val > min_val:  # Avoid division by zero
                    normalized[series] = (series_data - min_val) / (max_val - min_val)
                
            elif method == "percent":
                # Percent change from first value
                first_val = series_data[0]
                if first_val != 0:  # Avoid division by zero
                    normalized[series] = (series_data / first_val - 1) * 100
        
        return normalized

    def transform_data(data, plot_type, selected_series, baseline_date=None):
        """Transform data based on the selected plot type.
        
        This function applies plot-specific transformations to the dataset,
        such as indexing values to a baseline date or calculating cumulative
        changes.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The normalized data to transform
        plot_type : str
            Plot type determining the transformation:
            - 'time_series': No transformation
            - 'percent_change': Calculate percent change between points
            - 'cumulative': Calculate cumulative change
            - 'comparative': Prepare for comparison visualization
        selected_series : list or tuple
            Names of selected time series to transform
        baseline_date : str, optional
            Reference date for index plots (default is None)
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the transformed data ready for visualization
            
        Notes
        -----
        For index plots, all series are rebased to 100 at the baseline date,
        making it easier to compare relative performance over time.
        """
        # Create a copy to avoid modifying the original
        transformed = data.copy()
        
        # Convert selected_series to list if it's not already
        series_list = selected_series if isinstance(selected_series, list) else list(selected_series)
        
        # For index plot, rebase to 100 at baseline date
        if plot_type == "index" and baseline_date:
            baseline_date = pd.to_datetime(baseline_date)
            
            # Find the closest date if exact match not found
            if baseline_date not in transformed["date"].values:
                closest_idx = (transformed["date"] - baseline_date).abs().idxmin()
                baseline_date = transformed.loc[closest_idx, "date"]
            
            # Get baseline values
            baseline_row = transformed[transformed["date"] == baseline_date]
            
            if not baseline_row.empty:
                for series in series_list:
                    baseline_value = baseline_row[series].values[0]
                    if baseline_value != 0:  # Avoid division by zero
                        transformed[series] = (transformed[series] / baseline_value) * 100
        
        return transformed

    @reactive.calc
    def get_plot_title():
        """Generate appropriate plot title based on user selections.
        
        Creates a dynamic title that reflects the current visualization type,
        normalization method, and baseline date (if applicable).
        
        Returns
        -------
        str
            Formatted plot title describing the current visualization
        """
        plot_type = input.ID_plot_type()
        normalize = input.ID_normalize()
        
        if plot_type == "index":
            baseline_date = input.ID_baseline_date()
            return f"Index Plot (Baseline: {baseline_date})"
        else:
            title_components = []
            
            if normalize == "zscore":
                title_components.append("Z-Score Normalized")
            elif normalize == "minmax":
                title_components.append("Min-Max Normalized")
            elif normalize == "percent":
                title_components.append("Percent Change")
            
            if plot_type == "area":
                title_components.append("Area Plot")
            else:
                title_components.append("Line Plot")
            
            return " ".join(title_components)
    
    @reactive.calc
    def get_y_axis_label():
        """Generate appropriate y-axis label based on user selections.
        
        Creates a descriptive y-axis label that matches the current
        normalization method to help users interpret the visualization.
        
        Returns
        -------
        str
            Y-axis label text appropriate for the current normalization
        """
        normalize = input.ID_normalize()
        
        if normalize == "zscore":
            return "Standard Deviations"
        elif normalize == "minmax":
            return "Normalized Value (0-1)"
        elif normalize == "percent":
            return "Percent Change (%)"
        else:
            return "Value"
    
    # Generate the main results plot
    @output
    @render_widget
    def output_ID_results_plot():
        """
        Generate the main results visualization plot using plotly express.
        
        Creates an interactive visualization of the selected time series with 
        appropriate transformations based on user selections. The plot includes
        statistical markers, trend indicators, and interactive elements.
        
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure for rendering with render_widget
        """
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np
        
        selected_data = inputs_data()
        filtered_data = selected_data["filtered_data"]
        selected_series = selected_data["selected_series"]
        
        # Check if we have valid data to visualize
        if filtered_data is None or filtered_data.is_empty() or not selected_series:
            fig = go.Figure()
            fig.update_layout(
                title="No data available for visualization",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
                height=600
            )
            fig.add_annotation(
                text="Select one or more series and apply filters to view data",
                showarrow=False,
                font=dict(size=14),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )
            return fig
        
        # Normalize data
        normalized_data = normalize_data(
            data=filtered_data,
            method=input.ID_normalize(),
            selected_series=selected_series
        )
        
        # Transform data based on plot type
        transformed_data = transform_data(
            data=normalized_data,
            plot_type=input.ID_plot_type(),
            selected_series=selected_series,
            baseline_date=input.ID_baseline_date()
        )
        
        # Convert selected_series to list if it's a tuple
        series_list = selected_series if isinstance(selected_series, list) else list(selected_series)
        
        # Create base plotly figure
        plot_type = input.ID_plot_type()
        
        # Choose appropriate plot function based on plot type
        if plot_type == "comparative":
            # For comparative analysis, use a dedicated subplot layout
            from plotly.subplots import make_subplots
            
            # Create a figure with subplots - one row for each series
            fig = make_subplots(
                rows=len(series_list), 
                cols=1,
                shared_xaxes=True, 
                vertical_spacing=0.05,
                subplot_titles=series_list
            )
            
            # Add each series to its own row
            for i, series in enumerate(series_list):
                # Add main line trace
                series_data = transformed_data[["date", series]].dropna()
                fig.add_trace(
                    go.Scatter(
                        x=series_data["date"], 
                        y=series_data[series],
                        mode="lines",
                        name=series,
                        line=dict(width=2),
                    ),
                    row=i+1, 
                    col=1,
                )
                
                # Calculate y-axis range for better visualization
                y_values = series_data[series].dropna()
                if len(y_values) > 0:
                    y_min, y_max = y_values.min(), y_values.max()
                    y_range = y_max - y_min
                    fig.update_yaxes(
                        range=[y_min - 0.1*y_range, y_max + 0.1*y_range],
                        row=i+1, 
                        col=1,
                        title=series if i == 0 else None
                    )
        
        elif plot_type == "percent_change":
            # For percent change, calculate period-over-period changes
            fig = go.Figure()
            
            for series in series_list:
                # Calculate percent change
                series_data = transformed_data[["date", series]].copy()
                series_data["pct_change"] = series_data[series].pct_change() * 100
                series_data = series_data.dropna()
                
                # Add line for percent change
                fig.add_trace(
                    go.Scatter(
                        x=series_data["date"],
                        y=series_data["pct_change"],
                        mode="lines",
                        name=f"{series} (% change)",
                    )
                )
                
                # Add zero line for reference
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        elif plot_type == "cumulative":
            # For cumulative change, show cumulative sum of changes
            fig = go.Figure()
            
            for series in series_list:
                # Calculate cumulative sum of changes
                series_data = transformed_data[["date", series]].copy()
                
                # Compute cumulative change from first value
                first_value = series_data[series].iloc[0]
                if first_value != 0:  # Avoid division by zero
                    series_data["cumulative"] = ((series_data[series] / first_value) - 1) * 100
                    
                    # Add line for cumulative change
                    fig.add_trace(
                        go.Scatter(
                            x=series_data["date"],
                            y=series_data["cumulative"],
                            mode="lines",
                            name=f"{series} (cumulative %)",
                        )
                    )
            
            # Add zero line for reference
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        else:  # Default time series plot
            # For regular time series, use px.line with all series
            fig = px.line(
                transformed_data,
                x="date",
                y=series_list,
                title=get_plot_title(),
                labels={"value": get_y_axis_label(), "date": "Date"},
                template="plotly_white"
            )
        
        # Add markers for important points (for all plot types except comparative)
        if plot_type != "comparative":
            for series in series_list:
                try:
                    # Drop NaN values before finding min/max
                    clean_data = transformed_data.dropna(subset=[series])
                    
                    if len(clean_data) > 0:
                        # Find max and min values
                        max_idx = clean_data[series].idxmax()
                        min_idx = clean_data[series].idxmin()
                        
                        # Find last value
                        last_idx = clean_data.index[-1]
                        
                        # Add markers for max, min and last points
                        for idx, marker_name, position, color in [
                            (max_idx, "Max", "top center", "red"), 
                            (min_idx, "Min", "bottom center", "blue"), 
                            (last_idx, "Latest", "top right", "green")
                        ]:
                            if idx is not None:
                                fig.add_trace(
                                    go.Scatter(
                                        x=[clean_data.loc[idx, "date"]],
                                        y=[clean_data.loc[idx, series]],
                                        mode="markers+text",
                                        text=[marker_name],
                                        textposition=position,
                                        marker=dict(size=10, color=color),
                                        name=f"{series} - {marker_name}",
                                        showlegend=False
                                    )
                                )
                except (KeyError, ValueError, IndexError) as e:
                    # Skip this series if there's an error
                    print(f"Error adding markers for {series}: {e}")
                    continue
        
        # Improve layout
        fig.update_layout(
            title=get_plot_title(),
            xaxis_title="Date",
            yaxis_title=get_y_axis_label(),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            template="plotly_white",
            height=600
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider=dict(visible=True),
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
        
        # Add configuration options for better interactivity
        config = {
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'results_plot',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        
        # Return the figure directly for render_widget
        return fig

    # Generate summary statistics table
    @output
    @render.table
    def output_ID_results_table():
        """
        Generate a summary statistics table for selected series.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing summary statistics
        """
        selected_data = inputs_data()
        filtered_data = selected_data["filtered_data"]
        selected_series = selected_data["selected_series"]
        
        # Fix: Check filtered_data properly and if selected_series is empty
        if filtered_data is None or filtered_data.is_empty() or not selected_series:
            return pd.DataFrame({"Message": ["No data available for statistics"]})
        
        # Fix: Convert selected_series to list if it's a tuple
        series_list = selected_series if isinstance(selected_series, list) else list(selected_series)
        
        # Convert to pandas for easier stat calculation
        columns_to_select = ["date"] + series_list
        pd_data = filtered_data.select(columns_to_select).to_pandas()
        
        # Calculate statistics
        stats = []
        for series in series_list:
            # Skip if series doesn't exist in the data
            if series not in pd_data.columns:
                continue
                
            try:
                latest_value = pd_data[series].iloc[-1]
                mean = pd_data[series].mean()
                std_dev = pd_data[series].std()
                min_val = pd_data[series].min()
                max_val = pd_data[series].max()
                
                # Calculate changes
                first_value = pd_data[series].iloc[0]
                pct_change_total = ((latest_value / first_value) - 1) * 100 if first_value != 0 else np.nan
                
                # For 1-month change
                if len(pd_data) >= 2:
                    prev_value = pd_data[series].iloc[-2]
                    pct_change_1m = ((latest_value / prev_value) - 1) * 100 if prev_value != 0 else np.nan
                else:
                    pct_change_1m = np.nan
                
                stats.append({
                    "Series": series,
                    "Latest Value": round(latest_value, 2),
                    "Average": round(mean, 2),
                    "Std Dev": round(std_dev, 2),
                    "Min": round(min_val, 2),
                    "Max": round(max_val, 2),
                    "Total Change %": round(pct_change_total, 2),
                    "1-Month Change %": round(pct_change_1m, 2)
                })
            except Exception as e:
                # Skip this series if there's an error
                print(f"Error calculating stats for {series}: {e}")
                continue
        
        return pd.DataFrame(stats)

    # Generate insights based on data
    @output
    @render.ui
    def output_ID_results_insights():
        """Generate insights and observations based on the selected data.
        
        This function performs an automated analysis of the selected time series data
        and generates human-readable insights about trends, volatility, correlations,
        and performance comparisons. The insights are formatted with Markdown syntax
        for improved readability.
        
        The analysis includes:
        
        - Overall trend detection (upward, downward, stable)
        - Percentage changes over the selected period
        - Volatility assessment
        - Correlation analysis between multiple series
        - Comparative performance rankings
        
        Returns
        -------
        shiny.ui.Tag
            UI elements containing formatted insights as HTML
            
        Notes
        -----
        The function handles potential errors gracefully, skipping problematic
        series or analysis types rather than failing completely. Empty or invalid
        data results in an appropriate message rather than an error.
        """
        selected_data = inputs_data()
        filtered_data = selected_data["filtered_data"]
        selected_series = selected_data["selected_series"]
        
        # Check if we have valid data to analyze
        if filtered_data is None or filtered_data.is_empty() or not selected_series:
            return ui.p("No data available for insights.")
        
        # Convert selected_series to list if it's a tuple
        series_list = selected_series if isinstance(selected_series, list) else list(selected_series)
        
        # Convert to pandas for analysis
        columns_to_select = ["date"] + series_list
        pd_data = filtered_data.select(columns_to_select).to_pandas()
        pd_data["date"] = pd.to_datetime(pd_data["date"])
        
        # List to store insight texts
        insights = []
        
        # Overall trend analysis for each series
        for series in series_list:
            # Skip if series doesn't exist in the data
            if series not in pd_data.columns:
                continue
                
            try:
                # Drop NaN values before analysis
                clean_data = pd_data[series].dropna()
                
                if len(clean_data) > 1:  # Need at least 2 points for analysis
                    # Simple linear regression to determine trend
                    y = clean_data.values
                    x = np.arange(len(y))
                    slope, _ = np.polyfit(x, y, 1)
                    
                    # Calculate basic statistics
                    latest_value = clean_data.iloc[-1]
                    first_value = clean_data.iloc[0]
                    max_value = clean_data.max()
                    min_value = clean_data.min()
                    
                    # Format values to 2 decimal places
                    latest_value = round(latest_value, 2)
                    first_value = round(first_value, 2)
                    max_value = round(max_value, 2)
                    min_value = round(min_value, 2)
                    
                    # Calculate percentage change if first value isn't zero
                    if first_value != 0:
                        percent_change = round(((latest_value / first_value) - 1) * 100, 2)
                        change_text = f"changed by **{percent_change}%**"
                    else:
                        change_text = "changed by an undefined percentage (starting from zero)"
                    
                    # Generate trend text based on slope direction
                    if slope > 0:
                        trend_text = (
                            f"**{series}** shows an overall **upward trend**, "
                            f"{change_text} over the selected period."
                        )
                    elif slope < 0:
                        trend_text = (
                            f"**{series}** shows an overall **downward trend**, "
                            f"{change_text} over the selected period."
                        )
                    else:
                        trend_text = (
                            f"**{series}** shows a **stable pattern**, "
                            f"{change_text} over the selected period."
                        )
                    
                    insights.append(trend_text)
                    
                    # Add value range information
                    insights.append(f"* Starting value: {first_value}, Latest value: {latest_value}")
                    insights.append(f"* Range: Min {min_value} to Max {max_value}")
                    
                    # Identify volatility using coefficient of variation
                    std_dev = clean_data.std()
                    mean = clean_data.mean()
                    cv = (std_dev / mean) * 100 if mean != 0 else float("inf")
                    
                    if cv > 25:
                        insights.append(f"* **High volatility** observed (CV: {round(cv, 2)}%)")
                    elif cv > 10:
                        insights.append(f"* **Moderate volatility** observed (CV: {round(cv, 2)}%)")
                    else:
                        insights.append(f"* **Low volatility** observed (CV: {round(cv, 2)}%)")
                    
                    insights.append("<br>")
            except Exception as e:
                # Skip this series if there's an error
                print(f"Error analyzing trend for {series}: {e}")
                continue
        
        # Comparative analysis if multiple series are selected
        if len(series_list) > 1:
            insights.append("### Comparative Analysis")
            
            try:
                # Calculate correlations between series
                corr_matrix = pd_data[series_list].corr()
                
                # Find highest correlation pair (comparing each unique pair)
                corr_values = []
                for i in range(len(series_list)):
                    for j in range(i+1, len(series_list)):
                        corr_values.append((
                            series_list[i],
                            series_list[j],
                            corr_matrix.iloc[i, j],
                        ))
                
                # Sort by absolute correlation value (highest first)
                corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Report top correlation with interpretation
                if corr_values:
                    s1, s2, corr = corr_values[0]
                    corr_rounded = round(corr, 2)
                    
                    if corr > 0.7:
                        insights.append(
                            f"* **{s1}** and **{s2}** show a **strong positive correlation** "
                            f"({corr_rounded})."
                        )
                    elif corr > 0.3:
                        insights.append(
                            f"* **{s1}** and **{s2}** show a **moderate positive correlation** "
                            f"({corr_rounded})."
                        )
                    elif corr > -0.3:
                        insights.append(
                            f"* **{s1}** and **{s2}** show **little to no correlation** "
                            f"({corr_rounded})."
                        )
                    elif corr > -0.7:
                        insights.append(
                            f"* **{s1}** and **{s2}** show a **moderate negative correlation** "
                            f"({corr_rounded})."
                        )
                    else:
                        insights.append(
                            f"* **{s1}** and **{s2}** show a **strong negative correlation** "
                            f"({corr_rounded})."
                        )
                
                # Calculate and rank performance metrics
                performances = []
                for series in series_list:
                    clean_data = pd_data[series].dropna()
                    if len(clean_data) > 1:
                        first = clean_data.iloc[0]
                        last = clean_data.iloc[-1]
                        if first != 0:  # Avoid division by zero
                            perf = ((last / first) - 1) * 100
                            performances.append((series, perf))
                
                # Report best and worst performers
                if performances:
                    performances.sort(key=lambda x: x[1], reverse=True)
                    best_series, best_perf = performances[0]
                    worst_series, worst_perf = performances[-1]
                    
                    insights.append(
                        f"* Best performing: **{best_series}** ({round(best_perf, 2)}% change)"
                    )
                    insights.append(
                        f"* Worst performing: **{worst_series}** ({round(worst_perf, 2)}% change)"
                    )
            except Exception as e:
                print(f"Error in comparative analysis: {e}")
                insights.append("* Comparative analysis could not be completed due to data issues.")
        
        # Return insights as HTML for display
        if insights:
            return ui.HTML("<br>".join(insights))
        else:
            return ui.p("No insights could be generated from the selected data.")

    # Download handlers for plot and data
    @render.download(filename="results_plot.png")
    def ID_download_plot():
        """Generate a downloadable PNG of the current plot.
        
        This function creates a high-quality PNG image of the current visualization,
        with enhanced formatting for download purposes. The downloaded plot includes
        all selected series with statistical markers (min, max, latest) and proper
        formatting for titles, legends, and axes.
        
        Returns
        -------
        bytes
            PNG image data in binary format
            
        Notes
        -----
        The function uses the kaleido engine for rendering Plotly figures to static
        images, which requires the kaleido package to be installed. The output
        resolution is set to a higher quality than the on-screen version.
        
        See Also
        --------
        ID_download_data : Function for downloading data in CSV format
        normalize_data : Function that prepares data for visualization
        transform_data : Function that transforms data based on plot type
        """
        selected_data = inputs_data()
        filtered_data = selected_data["filtered_data"]
        selected_series = selected_data["selected_series"]
        
        # Return empty bytes if no valid data
        if filtered_data is None or filtered_data.is_empty() or not selected_series:
            return b""
        
        # Prepare data with the same transformations as the displayed plot
        normalized_data = normalize_data(
            data=filtered_data,
            method=input.ID_normalize(),
            selected_series=selected_series,
        )
        
        transformed_data = transform_data(
            data=normalized_data,
            plot_type=input.ID_plot_type(),
            selected_series=selected_series,
            baseline_date=input.ID_baseline_date(),
        )
        
        # Create plotly figure with enhanced formatting for download
        fig = px.line(
            transformed_data,
            x="date",
            y=selected_series,
            title=get_plot_title(),
            labels={"value": get_y_axis_label(), "date": "Date"},
            template="plotly_white",
        )
        
        # Add markers for important points with consistent handling of list/tuple
        series_list = selected_series if isinstance(selected_series, list) else list(selected_series)
        for series in series_list:
            try:
                # Drop NaN values before analysis
                clean_data = transformed_data.dropna(subset=[series])
                
                if len(clean_data) > 0:
                    # Find max point
                    max_idx = clean_data[series].idxmax()
                    fig.add_trace(
                        go.Scatter(
                            x=[clean_data.loc[max_idx, "date"]],
                            y=[clean_data.loc[max_idx, series]],
                            mode="markers+text",
                            text=["Max"],
                            textposition="top center",
                            marker=dict(size=10, color="red"),
                            name=f"{series} - Max",
                            showlegend=False,
                        )
                    )
                    
                    # Find min point
                    min_idx = clean_data[series].idxmin()
                    fig.add_trace(
                        go.Scatter(
                            x=[clean_data.loc[min_idx, "date"]],
                            y=[clean_data.loc[min_idx, series]],
                            mode="markers+text",
                            text=["Min"],
                            textposition="bottom center",
                            marker=dict(size=10, color="blue"),
                            name=f"{series} - Min",
                            showlegend=False,
                        )
                    )
                    
                    # Add latest point
                    last_idx = len(clean_data) - 1
                    fig.add_trace(
                        go.Scatter(
                            x=[clean_data.iloc[last_idx]["date"]],
                            y=[clean_data.iloc[last_idx][series]],
                            mode="markers+text",
                            text=["Latest"],
                            textposition="top right",
                            marker=dict(size=10, color="green"),
                            name=f"{series} - Latest",
                            showlegend=False,
                        )
                    )
            except (KeyError, ValueError, IndexError) as e:
                # Skip this series if there's an error
                print(f"Error adding markers for {series}: {e}")
                continue
        
        # Improve layout for better image output
        fig.update_layout(
            height=800,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        # Write to bytes buffer
        import io
        
        buf = io.BytesIO()
        write_image(fig, buf, format="png", engine="kaleido")
        buf.seek(0)
        return buf.read()

    @render.download(filename="results_data.csv")
    def ID_download_data():
        """Generate a downloadable CSV of the current data.
        
        This function creates a CSV file containing the transformed data currently
        being visualized in the dashboard. The CSV includes all selected series
        after applying any normalization and transformation methods specified
        in the UI.
        
        Returns
        -------
        bytes
            CSV data as UTF-8 encoded bytes
            
        Notes
        -----
        The CSV includes a header row with column names and uses standard CSV
        formatting (comma separators, quoted strings when necessary). The data
        is encoded in UTF-8 format for maximum compatibility.
        
        Examples
        --------
        The downloaded CSV will have the following format:
        
        .. code-block:: text
        
            date,series1,series2,...
            2022-01-01,10.5,20.3,...
            2022-02-01,11.2,19.8,...
        
        See Also
        --------
        ID_download_plot : Function for downloading plot as PNG image
        normalize_data : Function that prepares data for visualization
        transform_data : Function that transforms data based on plot type
        """
        selected_data = inputs_data()
        filtered_data = selected_data["filtered_data"]
        selected_series = selected_data["selected_series"]
        
        # Return empty bytes if no valid data
        if filtered_data is None or filtered_data.is_empty() or not selected_series:
            return b""
        
        # Apply the same transformations as the displayed data
        normalized_data = normalize_data(
            data=filtered_data,
            method=input.ID_normalize(),
            selected_series=selected_series,
        )
        
        transformed_data = transform_data(
            data=normalized_data,
            plot_type=input.ID_plot_type(),
            selected_series=selected_series,
            baseline_date=input.ID_baseline_date(),
        )
        
        # Export to CSV using a string buffer
        import io
        
        buf = io.StringIO()
        transformed_data.to_csv(buf, index=False)
        buf.seek(0)
        return buf.getvalue().encode("utf-8")  # Explicitly specify UTF-8 encoding