"""
Data Analysis Module for QWIM Dashboard
=======================================

This module provides the UI and server components for the Data Analysis tab
of the QWIM Dashboard.

The module offers several key features:
    * Time series evolution analysis with trend detection
    * Statistical analysis with descriptive statistics
    * Distribution visualization with histogram and density plots
    * Correlation and distance heatmaps for multiple time series

Classes and Functions:
    * analysis_ui - Creates the UI components for the Data Analysis tab
    * analysis_server - Handles the server-side logic for analysis functionality
"""

from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO

import d3blocks
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from great_tables import GT
from plotly.subplots import make_subplots
from plotnine import (aes, element_text, facet_wrap, geom_line, geom_point,
                     ggplot, labs, scale_color_brewer, theme, theme_minimal)
from scipy import stats as scipy_stats  # Rename to avoid naming conflicts
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import gaussian_kde
from shiny import module, reactive, render, ui
from shinywidgets import render_plotly, output_widget, render_widget

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@module.ui
def analysis_ui():
    """Create the UI for the Data Analysis tab.
    
    This function generates the complete user interface for the Data Analysis tab,
    organizing content into multiple sub-tabs with various interactive controls.
    
    The UI includes:
    
    - Time Evolution: Time series visualization with trend analysis
    - Data Statistics: Statistical analysis with descriptive statistics
    - Heatmap: Correlation/distance matrix visualizations
    
    Each tab includes appropriate inputs, filters, and visualization options.
    
    Returns
    -------
    shiny.ui.page_fluid
        A fluid page layout containing all UI elements for the Data Analysis tab
    
    See Also
    --------
    analysis_server : Server-side logic for this UI
    """
    return ui.page_fluid(
        ui.h2("Time Series Analysis"),
        ui.navset_tab(
            ui.nav_panel(
                "Time Evolution",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Series Selection"),
                        ui.input_select("ID_time_evo_series", "Select Series:", []),
                        ui.h4("Date Range"),
                        ui.input_date_range(
                            "ID_time_evo_date_range",
                            "Select Date Range:",
                            start=datetime(2008, 7, 2),
                            end=datetime(2025, 3, 2),
                            min=datetime(2008, 7, 2),
                            max=datetime(2025, 3, 2),
                            format="yyyy-mm-dd",
                            separator=" to ",
                        ),
                        ui.input_checkbox("ID_show_trendline", "Show Trendline", True),
                        ui.h4("Visualization Options"),
                        ui.input_radio_buttons(
                            "ID_time_evo_viz_type",
                            "Visualization Type:",
                            {
                                "raw": "Raw Values",
                                "normalized": "Normalized Values",
                                "pct_change": "Percent Change",
                            },
                            selected="raw",
                        ),
                        ui.input_checkbox(
                            "ID_show_stats_markers", "Show Statistical Markers", True
                        ),
                    ),
                    ui.h3("Time Series Evolution"),
                    # ui.output_ui("output_ID_time_evolution_plot", height="600px"),
                    output_widget("output_ID_time_evolution_plot", height="600px"),
                    ui.h4("Series Statistics"),
                    ui.output_table("output_ID_time_evolution_stats"),
                ),
            ),
            ui.nav_panel(
                "Data Statistics",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Series Selection"),
                        ui.input_select("ID_stats_series", "Select Series:", []),
                        ui.h4("Date Range"),
                        ui.input_date_range(
                            "ID_stats_date_range",
                            "Select Date Range:",
                            start=datetime(2002, 1, 1),
                            end=datetime(2025, 3, 31),
                            min=datetime(2002, 1, 1),
                            max=datetime(2025, 3, 31),
                            format="yyyy-mm-dd",
                            separator=" to ",
                        ),
                        ui.h4("Statistics Options"),
                        ui.input_checkbox_group(
                            "ID_selected_stats",
                            "Select Statistics to Display:",
                            {
                                "basic": "Basic Statistics",
                                "moments": "Statistical Moments",
                                "quantiles": "Quantiles",
                                "tests": "Statistical Tests",
                            },
                            selected=["basic", "moments"],
                        ),
                    ),
                    ui.h3("Statistical Analysis"),
                    ui.output_ui("output_ID_stats_tables"),
                    ui.h4("Distribution Visualization"),
                    # ui.output_ui("output_ID_stats_dist_plot", height="400px"),
                    output_widget("output_ID_stats_dist_plot", height="400px")
                ),
            ),
            ui.nav_panel(
                "Heatmap",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Heatmap Options"),
                        ui.input_select(
                            "ID_heatmap_ordering",
                            "Order Series By:",
                            {
                                "clustering": "Clustering",
                                "name": "Name",
                                "frequency": "Frequency",
                            },
                            selected="clustering",
                        ),
                        ui.h4("Date Range"),
                        ui.input_date_range(
                            "ID_heatmap_date_range",
                            "Select Date Range:",
                            start=datetime(2008, 7, 2),
                            end=datetime(2025, 3, 2),
                            min=datetime(2008, 7, 2),
                            max=datetime(2025, 3, 2),
                            format="yyyy-mm-dd",
                            separator=" to ",
                        ),
                        ui.h4("Visualization Options"),
                        ui.input_radio_buttons(
                            "ID_heatmap_color_scale",
                            "Color Scale:",
                            {
                                "viridis": "Viridis",
                                "plasma": "Plasma",
                                "inferno": "Inferno",
                                "magma": "Magma",
                                "cividis": "Cividis",
                            },
                            selected="viridis",
                        ),
                        ui.input_radio_buttons(
                            "ID_heatmap_metric",
                            "Metric to Display:",
                            {
                                "values": "Raw Values",
                                "correlation": "Correlation",
                                "distance": "Distance",
                            },
                            selected="correlation",
                        ),
                    ),
                    ui.h3("Time Series Heatmap"),
                    # ui.output_ui("output_ID_d3_heatmap"),
                    output_widget("output_ID_d3_heatmap"),
                    ui.tags.div(
                        {"id": "heatmap-container", "style": "height: 800px; width: 100%;"}
                    ),
                    ui.h4("Interpretation"),
                    ui.output_ui("output_ID_heatmap_interpretation"),
                ),
            ),
        ),
    )


@module.server
def analysis_server(input, output, session, inputs_data, data_r, series_names_r):
    """Server logic for the Data Analysis tab.
    
    This function implements all reactive server logic for the Data Analysis tab,
    processing user inputs, performing statistical analyses, and generating
    visualizations. It handles three main panels:
    
    1. Time Evolution - Time series visualization with trend analysis
    2. Data Statistics - Statistical analysis with descriptive statistics
    3. Heatmap - Correlation and distance matrix visualization
    
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
    
    Notes
    -----
    This function contains multiple nested reactive functions:
    
    - Reactive effects: Update UI elements based on data changes
    - Reactive calculations: Filter and transform data based on user inputs
    - Output renderers: Generate visualizations and tables for display
    - Download handlers: Create downloadable content for users
    
    The function uses plotly.express for creating interactive visualizations that are
    displayed through shinywidgets, providing a more integrated and responsive user
    experience compared to standard Shiny outputs.
    
    See Also
    --------
    analysis_ui : UI component for the Data Analysis tab
    get_time_evo_data : Reactive calc function for time evolution data
    get_stats_data : Reactive calc function for statistics data
    get_heatmap_data : Reactive calc function for heatmap data
    """
    # New data section
    def _load_new_data():
        print("In loading Data")
        """Load the new dataset from the raw/etf_data directory."""
        data_path = Path("data/raw/etf_data.csv")
        logger.info(f"Loading data from {data_path}")
        
        if not data_path.exists():
            logger.error(f"Data file not found at {data_path}")
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        # Load data using polars
        df = pl.read_csv(data_path)
        
        # Ensure date column is properly formatted
        df = df.with_columns(pl.col("Date").str.to_date())
        
        logger.info(f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns")
        print("############################")
        return df
    
    def _get_series_names(data):
        print("Right in new func")
        series_names = [col for col in data.columns if col != "Date"]
        logger.info(f"Found {len(series_names)} time series: {', '.join(series_names)}")
        return series_names

    # Update series selection dropdowns when series names change
    @reactive.effect
    def _update_series_selections():
        """Update series selection dropdowns with available series names.
        
        This effect updates the dropdown menus whenever the available
        series names change, ensuring UI elements stay synchronized with
        available data.
        """
        print("findout")
        new_data = _load_new_data()  # 调用加载新数据的函数
        print(new_data)
            # 如果新数据为 None，跳过后续操作
        if new_data is None:
            print("Error: No data loaded.")
            return
        
        series_names = _get_series_names(new_data)
        print(series_names)
        
        ui.update_select(
            "ID_time_evo_series",
            choices=series_names,
            selected=series_names[0] if series_names else None,
        )
        
        ui.update_select(
            "ID_stats_series",
            choices=series_names,
            selected=series_names[0] if series_names else None,
        )
    
    # Get filtered data for Time Evolution tab
    @reactive.calc
    def get_time_evo_data():
        """Filter data for the Time Evolution tab based on user selections.
        
        Processes the raw dataset to extract the selected time series within
        the specified date range. This filtered dataset is used for time
        evolution visualizations and statistics.
        
        Returns
        -------
        pl.DataFrame or None
            Filtered dataframe containing date and selected series columns,
            or None if no series is selected
        """
        print("here")
        # Load the new data using _load_new_data function
        new_data = _load_new_data()  # Ensure you are calling your modified data loading

        date_range = input.ID_time_evo_date_range()
        print(date_range)
        series = input.ID_time_evo_series()
        print(series)

        if not series:
            return None
        
        # Filter by date range
        if date_range[0] and date_range[1]:
            filtered = new_data.filter(
                (pl.col("Date") >= date_range[0]) & 
                (pl.col("Date") <= date_range[1]),
            )
        else:
            filtered = new_data
        
        # Select only relevant columns
        filtered = filtered.select(["Date", series])
        print(" Here Printing:")
        print(filtered)
        
        return filtered
    
    # Generate time evolution plot
    @output
    @render_widget
    def output_ID_time_evolution_plot():
        print("Entered?")
        """Generate the time evolution plot for the selected series.
        
        Creates an interactive Plotly Express visualization showing the time evolution
        of the selected series with optional trend line and statistical markers.
        The plot adapts based on user-selected visualization type (raw values,
        normalized, or percent change) and uses shinywidgets for enhanced interactivity.
        
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure for rendering with render_widget
        """
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        filtered_data = get_time_evo_data()
        series = input.ID_time_evo_series()
        viz_type = input.ID_time_evo_viz_type()
        show_trendline = input.ID_show_trendline()
        show_stats_markers = input.ID_show_stats_markers()
        
        if filtered_data is None or filtered_data.is_empty():
            # Return an empty plot if no data
            fig = px.scatter(title="No data available")
            fig.add_annotation(
                text="Select a series and date range to view data",
                showarrow=False,
                font=dict(size=14),
            )
            return fig
        
        # Convert to pandas for easier plotting
        pd_data = filtered_data.select(["Date", series]).to_pandas()
        pd_data["Date"] = pd.to_datetime(pd_data["Date"])
        
        # Apply transformation based on viz_type
        if viz_type == "normalized":
            # Normalize to 0-1 range
            min_val = pd_data[series].min()
            max_val = pd_data[series].max()
            if max_val > min_val:  # Prevent division by zero
                pd_data["value"] = (pd_data[series] - min_val) / (max_val - min_val)
                y_title = f"{series} (Normalized)"
            else:
                pd_data["value"] = pd_data[series]
                y_title = series
        elif viz_type == "pct_change":
            # Calculate percent change
            pd_data["value"] = pd_data[series].pct_change() * 100
            pd_data = pd_data.dropna()  # Remove NaN from pct_change
            y_title = f"{series} (% Change)"
        else:
            # Raw values
            pd_data["value"] = pd_data[series]
            y_title = series
        
        # Create a subplot with secondary y-axis for additional metrics if needed
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add main time series line
        line_fig = px.line(
            pd_data,
            x="Date",
            y="value",
            title=f"Time Evolution of {series}",
            labels={"Date": "Date", "value": y_title},
            template="plotly_white",
            render_mode="webgl"  # Faster rendering for large datasets
        )
        
        # Add the trace to our main figure
        fig.add_trace(line_fig.data[0])
        
        # Add trendline if requested
        if show_trendline and len(pd_data) > 1:
            # Add trendline using OLS regression
            from scipy import stats
            
            # Convert dates to numeric for regression
            pd_data['date_numeric'] = pd_data['Date'].astype(int) // 10**9  # Convert to Unix timestamp
            
            # Perform linear regression
            mask = ~np.isnan(pd_data['value'])
            if sum(mask) > 1:  # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    pd_data.loc[mask, 'date_numeric'],
                    pd_data.loc[mask, 'value']
                )
                
                # Create line using regression parameters
                pd_data['trendline'] = intercept + slope * pd_data['date_numeric']
                
                # Calculate trend strength and direction
                trend_strength = abs(r_value)
                trend_direction = "Rising" if slope > 0 else "Falling"
                
                # Add trend information to title
                trend_info = f"{trend_direction} Trend (R²={r_value**2:.2f})"
                fig.update_layout(
                    title=f"Time Evolution of {series} - {trend_info}", 
                    height=600,  # Adjust height
                    width=1000,  # Adjust width
                    title_x=0.5,
                    title_y=0.95,
                    title_yanchor="top",  # 确保标题的顶部对齐
                    margin=dict(l=40, r=40, t=150, b=250), # 增加底部边距
                    legend=dict(
                        orientation="h",  # 设置图例为水平布局
                        yanchor="bottom",  # 图例的y坐标对齐
                        y=1.05,  # 调整图例的位置，避免和图表重叠
                        xanchor="right",  # 图例的x坐标对齐
                        x=1  # 图例置于右边
                    ),
                    dragmode="pan",
                )
                
                # Add trendline to figure
                fig.add_trace(
                    go.Scatter(
                        x=pd_data['Date'],
                        y=pd_data['trendline'],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=1),
                        name=f"Trend ({trend_direction})"
                    )
                )
        
        # Add statistical markers if requested
        if show_stats_markers and len(pd_data) > 0:
            # Drop NaN values for statistical calculations
            clean_data = pd_data.dropna(subset=["value"])
            
            if len(clean_data) > 0:
                # Find max value
                max_idx = clean_data["value"].idxmax()
                max_point = clean_data.loc[max_idx]
                fig.add_trace(
                    go.Scatter(
                        x=[max_point["Date"]],
                        y=[max_point["value"]],
                        mode="markers+text",
                        marker=dict(size=10, color="red", symbol="star"),
                        text=["Max"],
                        textposition="top center",
                        name=f"Maximum ({max_point['value']:.2f})",
                        hoverinfo="x+y+name"
                    )
                )
                
                # Find min value
                min_idx = clean_data["value"].idxmin()
                min_point = clean_data.loc[min_idx]
                fig.add_trace(
                    go.Scatter(
                        x=[min_point["Date"]],
                        y=[min_point["value"]],
                        mode="markers+text",
                        marker=dict(size=10, color="blue", symbol="star"),
                        text=["Min"],
                        textposition="bottom center",
                        name=f"Minimum ({min_point['value']:.2f})",
                        hoverinfo="x+y+name"
                    )
                )
                
                # Add mean line
                mean_val = clean_data["value"].mean()
                fig.add_hline(
                    y=mean_val,
                    line_dash="dot",
                    line_width=1,
                    line_color="green",
                    annotation=dict(
                        text=f"Mean: {mean_val:.2f}",
                        xref="paper",
                        x=1.0,
                        showarrow=False,
                        yanchor="bottom"
                    )
                )
                
                # Add standard deviation bands
                std_val = clean_data["value"].std()
                fig.add_trace(
                    go.Scatter(
                        x=clean_data["Date"],
                        y=[mean_val + std_val] * len(clean_data),
                        mode="lines",
                        line=dict(color="rgba(0,128,0,0.2)", dash="dash", width=1),
                        name=f"+1σ ({mean_val + std_val:.2f})",
                        hoverinfo="name+y"
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=clean_data["Date"],
                        y=[mean_val - std_val] * len(clean_data),
                        mode="lines",
                        line=dict(color="rgba(0,128,0,0.2)", dash="dash", width=1),
                        name=f"-1σ ({mean_val - std_val:.2f})",
                        hoverinfo="name+y",
                        fill="tonexty",
                        fillcolor="rgba(0,128,0,0.1)"
                    )
                )
        
        # Add range selector and slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor="rgba(150, 200, 250, 0.4)",
                activecolor="rgba(100, 150, 200, 0.8)"
            )
        )
        
        # Update axis titles
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=y_title,
        )
        
        # Update layout for better appearance
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            hovermode="x unified",
            plot_bgcolor="white",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            ),
            height=600
        )
        
        # Add download button configuration
        config = {
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'time_series_{series}',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        
        # Return the figure directly with the render_widget decorator
        return fig
    
    # Generate time evolution statistics table
    @output
    @render.ui
    def output_ID_time_evolution_stats():
        """
        Generate a statistics table for the time evolution tab.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing time series statistics
        """
        filtered_data = get_time_evo_data()
        series = input.ID_time_evo_series()
        
        if filtered_data is None or filtered_data.is_empty():
            return pd.DataFrame({"Message": ["No data available"]})
        
        # Extract series data as numpy array for statistical calculations
        series_data = filtered_data.select(series).to_numpy().flatten()
        
        # Calculate statistics
        stats = {
            "Statistic": [
                "Count", "Mean", "Median", "Std Dev", "Min", "25%", 
                "50%", "75%", "Max", "Range", "Skewness", "Kurtosis"
            ],
            "Value": [
                len(series_data),
                np.mean(series_data),
                np.median(series_data),
                np.std(series_data),
                np.min(series_data),
                np.percentile(series_data, 25),
                np.percentile(series_data, 50),
                np.percentile(series_data, 75),
                np.max(series_data),
                np.max(series_data) - np.min(series_data),
                scipy_stats.skew(series_data),
                scipy_stats.kurtosis(series_data)
            ]
        }
        
        # Format values to 4 decimal places
        stats["Value"] = [round(v, 4) if isinstance(v, (float, np.float64)) else v for v in stats["Value"]]

        stats_df = pd.DataFrame(stats)
        transposed_df = stats_df.set_index("Statistic").T

        # Use gt to format the table
        gt_table = (GT(transposed_df)
            .tab_header(title="Series Statistics")
        )
        
        return ui.HTML(gt_table._repr_html_())  # Display the formatted table
    
    # Get filtered data for Statistics tab
    @reactive.calc
    def get_stats_data():
        """
        Filter data for the Statistics tab based on user selections.
        
        Returns
        -------
        pl.DataFrame
            Filtered dataframe for statistics calculations
        """
        new_data = _load_new_data()
        date_range = input.ID_stats_date_range()
        series = input.ID_stats_series()
        
        if not series:
            return None
        
        # Filter by date range
        if date_range[0] and date_range[1]:
            filtered = new_data.filter(
                (pl.col("Date") >= date_range[0]) & 
                (pl.col("Date") <= date_range[1])
            )
        else:
            filtered = new_data
        
        # Select only relevant columns
        filtered = filtered.select(["Date", series])
        
        return filtered
    
    # Generate statistics tables
    @output
    @render.ui
    def output_ID_stats_tables():
        """
        Generate statistical tables using great_tables.
        
        Returns
        -------
        shiny.ui.TagList
            UI elements containing statistical tables
        """
        filtered_data = get_stats_data()
        series = input.ID_stats_series()
        selected_stats = input.ID_selected_stats()
        
        if filtered_data is None or filtered_data.is_empty() or not selected_stats:
            return ui.p("No data available or no statistics selected")
        
        # Extract series data as numpy array
        series_data = filtered_data.select(series).to_numpy().flatten()
        
        # Create UI elements for each selected statistic group
        tables = []
        
        if "basic" in selected_stats:
            # Create basic statistics table
            basic_stats = pd.DataFrame({
                "Statistic": [
                    "Count", "Mean", "Median", "Standard Deviation", 
                    "Minimum", "Maximum", "Range"
                ],
                "Value": [
                    len(series_data),
                    np.mean(series_data),
                    np.median(series_data),
                    np.std(series_data),
                    np.min(series_data),
                    np.max(series_data),
                    np.max(series_data) - np.min(series_data)
                ]
            })
            
            # Create GT table
            gt_basic = (GT(basic_stats)
                .tab_header(title=f"Basic Statistics for {series}")
                .fmt_number(columns=["Value"], decimals=4)
                .tab_options(container_width="100%"))
            
            tables.append(ui.div(
                ui.h4("Basic Statistics"),
                ui.HTML(gt_basic._repr_html_()),  # Change to _repr_html_() method
                class_="mb-4"
            ))
        
        if "moments" in selected_stats:
            # Create moments table
            moments_stats = pd.DataFrame({
                "Moment": [
                    "Mean (1st Moment)",
                    "Variance (2nd Moment)",
                    "Skewness (3rd Moment)",
                    "Kurtosis (4th Moment)",
                    "Excess Kurtosis"
                ],
                "Value": [
                    np.mean(series_data),
                    np.var(series_data),
                    scipy_stats.skew(series_data),
                    scipy_stats.kurtosis(series_data, fisher=True),
                    scipy_stats.kurtosis(series_data, fisher=False)
                ],
                "Description": [
                    "Central tendency",
                    "Spread of data",
                    "Asymmetry of distribution",
                    "Tailedness relative to normal distribution",
                    "Tailedness (raw kurtosis - 3)"
                ]
            })
            
            # Create GT table
            gt_moments = (GT(moments_stats)
                .tab_header(title=f"Statistical Moments for {series}")
                .fmt_number(columns=["Value"], decimals=4)
                .tab_options(container_width="100%"))
            
            tables.append(ui.div(
                ui.h4("Statistical Moments"),
                ui.HTML(gt_moments._repr_html_()),  # Change to _repr_html_() method
                class_="mb-4"
            ))
        
        if "quantiles" in selected_stats:
            # Generate quantiles
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            quantile_values = np.quantile(series_data, quantiles)
            
            quantile_stats = pd.DataFrame({
                "Percentile": [f"{q*100}%" for q in quantiles],
                "Value": quantile_values
            })
            
            # Create GT table
            gt_quantiles = (GT(quantile_stats)
                .tab_header(title=f"Distribution Quantiles for {series}")
                .fmt_number(columns=["Value"], decimals=4)
                .tab_options(container_width="100%"))
            
            tables.append(ui.div(
                ui.h4("Distribution Quantiles"),
                ui.HTML(gt_quantiles._repr_html_()),  # Change to _repr_html_() method
                class_="mb-4"
            ))
        
        if "tests" in selected_stats:
            # Statistical tests
            test_results = []
            
            # Normality test
            shapiro_test = scipy_stats.shapiro(series_data)
            test_results.append({
                "Test": "Shapiro-Wilk Normality",
                "Statistic": shapiro_test[0],
                "p-value": shapiro_test[1],
                "Interpretation": "Normal distribution" if shapiro_test[1] > 0.05 else "Not normally distributed"
            })
            
            # Runs test for randomness
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_test = acorr_ljungbox(series_data, lags=[10])
                test_results.append({
                    "Test": "Ljung-Box (lag=10)",
                    "Statistic": lb_test["lb_stat"].iloc[0],
                    "p-value": lb_test["lb_pvalue"].iloc[0],
                    "Interpretation": "No autocorrelation" if lb_test["lb_pvalue"].iloc[0] > 0.05 else "Autocorrelation present"
                })
            except:
                # Fallback if statsmodels not available
                test_results.append({
                    "Test": "Ljung-Box (lag=10)",
                    "Statistic": np.nan,
                    "p-value": np.nan,
                    "Interpretation": "Test could not be computed"
                })
            
            # Stationarity test
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_test = adfuller(series_data)
                test_results.append({
                    "Test": "Augmented Dickey-Fuller",
                    "Statistic": adf_test[0],
                    "p-value": adf_test[1],
                    "Interpretation": "Stationary" if adf_test[1] < 0.05 else "Non-stationary"
                })
            except:
                # Fallback if statsmodels not available
                test_results.append({
                    "Test": "Augmented Dickey-Fuller",
                    "Statistic": np.nan,
                    "p-value": np.nan,
                    "Interpretation": "Test could not be computed"
                })
            
            # Create GT table for tests
            gt_tests = (GT(pd.DataFrame(test_results))
                .tab_header(title=f"Statistical Tests for {series}")
                .fmt_number(columns=["Statistic", "p-value"], decimals=4)
                .tab_options(container_width="100%"))
            
            tables.append(ui.div(
                ui.h4("Statistical Tests"),
                ui.HTML(gt_tests._repr_html_()),  # Change to _repr_html_() method
                class_="mb-4"
            ))
        
        # Return all tables
        return ui.div(*tables)
    
    # Generate distribution visualization
    @output
    @render_widget
    def output_ID_stats_dist_plot():
        """
        Generate a distribution plot for the selected series using plotly express.
        
        Creates an interactive distribution visualization that combines histogram
        and density plots, with statistical markers. The visualization is rendered
        through shinywidgets for enhanced interactivity.
        
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure for rendering with render_widget
        """
        import plotly.express as px
        import plotly.graph_objects as go
        
        filtered_data = get_stats_data()
        series = input.ID_stats_series()
        
        if filtered_data is None or filtered_data.is_empty():
            # Return an empty plot if no data
            fig = px.scatter(title="No data available")
            fig.add_annotation(
                text="Select a series and date range to view data",
                showarrow=False,
                font=dict(size=14),
            )
            return fig
        
        # Convert to pandas for easier plotting
        pd_data = filtered_data.select([series]).to_pandas()
        
        # Create distribution plot using plotly express
        fig = px.histogram(
            pd_data,
            x=series,
            title=f"Distribution of {series}",
            nbins=30,
            opacity=0.7,
            color_discrete_sequence=["royalblue"],
            marginal="box",  # Add a box plot to the top
            histnorm="probability density",  # Normalize histogram for density comparison
            template="plotly_white",
        )
        
        # Store the maximum y-value of the histogram for reference
        hist_max = None
        if len(fig.data) > 0 and hasattr(fig.data[0], 'y') and fig.data[0].y is not None and len(fig.data[0].y) > 0:
            hist_max = fig.data[0].y.max()
        
        # Add KDE (kernel density estimate) if there are enough points
        clean_data = pd_data[series].dropna()
        if len(clean_data) > 5:
            try:
                # Use px to create a KDE
                kde_data = pd.DataFrame({
                    series: np.linspace(clean_data.min(), clean_data.max(), 1000)
                })
                
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(clean_data)
                kde_data["density"] = kde(kde_data[series])
                
                # Scale density to match histogram height - with robust error handling
                if hist_max is not None and kde_data["density"].max() > 0:
                    kde_data["density"] = kde_data["density"] * (hist_max / kde_data["density"].max())
                
                # Add density curve
                fig.add_scatter(
                    x=kde_data[series],
                    y=kde_data["density"],
                    mode="lines",
                    line=dict(color="crimson", width=2),
                    name="Density"
                )
            except Exception as e:
                # Skip KDE if there's an error
                print(f"Error calculating KDE: {e}")
        
        # Add statistical markers
        if len(clean_data) > 0:
            mean_val = clean_data.mean()
            std_val = clean_data.std()
            median_val = clean_data.median()
            
            # Add mean line
            fig.add_vline(
                x=mean_val,
                line_width=2,
                line_dash="solid",
                line_color="green",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top"
            )
            
            # Add median line
            fig.add_vline(
                x=median_val,
                line_width=2,
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Median: {median_val:.2f}",
                annotation_position="bottom"
            )
            
            # Add standard deviation lines
            if std_val > 0:
                fig.add_vline(
                    x=mean_val + std_val,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="+1σ",
                    annotation_position="top"
                )
                
                fig.add_vline(
                    x=mean_val - std_val,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="-1σ",
                    annotation_position="top"
                )
                
                # Add shaded area for 1 standard deviation - with robust error handling
                max_y_value = None
                if hist_max is not None:
                    max_y_value = hist_max
                elif len(fig.data) > 0 and hasattr(fig.data[0], 'y') and fig.data[0].y is not None and len(fig.data[0].y) > 0:
                    max_y_value = fig.data[0].y.max()
                else:
                    # Default value if we can't determine the maximum
                    max_y_value = 1.0
                    
                fig.add_shape(
                    type="rect",
                    x0=mean_val - std_val,
                    x1=mean_val + std_val,
                    y0=0,
                    y1=max_y_value,
                    fillcolor="rgba(0,128,0,0.1)",
                    line=dict(width=0),
                    layer="below"
                )
        
        # Update layout
        fig.update_layout(
            xaxis_title=series,
            yaxis_title="Density",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            bargap=0.1,
            height=400, 
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="white",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            ),
            hovermode="closest"
        )
        
        return fig
    
    # Get filtered data for Heatmap tab
    @reactive.calc
    def get_heatmap_data():
        """
        Filter data for the Heatmap tab based on user selections.
        
        Returns
        -------
        pl.DataFrame
            Filtered dataframe for heatmap visualization
        """
        new_data = _load_new_data()
        date_range = input.ID_heatmap_date_range()
        
        # Filter by date range
        if date_range[0] and date_range[1]:
            filtered = new_data.filter(
                (pl.col("Date") >= date_range[0]) & 
                (pl.col("Date") <= date_range[1])
            )
        else:
            filtered = new_data
        
        return filtered
    
    # Generate heatmap
    @output
    @render_widget
    def output_ID_d3_heatmap():
        """
        Generate an interactive heatmap using plotly express and shinywidgets.
        
        Creates a heatmap visualization that can display correlation matrices,
        distance matrices, or time series values based on user selection.
        The visualization supports different ordering methods and color scales.
        
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly heatmap figure rendered via shinywidgets
        """
        import plotly.express as px
        import plotly.graph_objects as go
        
        filtered_data = get_heatmap_data()
        ordering = input.ID_heatmap_ordering()
        color_scale = input.ID_heatmap_color_scale()
        metric = input.ID_heatmap_metric()
        
        if filtered_data is None or filtered_data.is_empty():
            # Return empty plot if no data
            fig = px.scatter(title="No data available for heatmap visualization")
            fig.add_annotation(
                text="Select a date range with available data",
                showarrow=False,
                font=dict(size=14),
            )
            return fig
        
        # Get all series names except date
        series_names = [col for col in filtered_data.columns if col != "Date"]
        
        # If too many series, limit to a manageable number for visualization
        if (len(series_names) > 50):
            # Provide a warning annotation
            warning_msg = f"Showing only 50 of {len(series_names)} series for better visualization"
            series_names = series_names[:50]
        else:
            warning_msg = None
        
        # Convert to pandas for easier manipulation
        pd_data = filtered_data.select(["Date"] + series_names).to_pandas()
        pd_data["Date"] = pd.to_datetime(pd_data["Date"])
        pd_data.set_index("Date", inplace=True)
        
        # Handle missing data to avoid visualization issues - FIXED HERE
        pd_data = pd_data.ffill().bfill().fillna(0)
        
        # Create heatmap based on selected metric
        if metric == "correlation":
            # Calculate correlation matrix
            corr_matrix = pd_data.corr(method='pearson')
            
            # Apply ordering if requested
            if ordering == "clustering":
                try:
                    from scipy.cluster.hierarchy import linkage, dendrogram
                    from scipy.spatial.distance import squareform
                    
                    # Use 1-|correlation| as distance measure for clustering
                    dist_array = squareform(1 - corr_matrix.abs())
                    link = linkage(dist_array, method='ward')
                    
                    # Get ordering from dendrogram
                    order = dendrogram(link, no_plot=True)['leaves']
                    ordered_series = [corr_matrix.index[i] for i in order if i < len(corr_matrix.index)]
                    
                    # Reindex matrix with new order
                    corr_matrix = corr_matrix.reindex(ordered_series)[ordered_series]
                    
                    # Add cluster boundaries
                    cluster_boundaries = None  # Will be defined if we can detect distinct clusters
                    
                    # Try to determine clusters by distance threshold
                    try:
                        from scipy.cluster.hierarchy import fcluster
                        num_clusters = min(8, max(2, int(len(series_names)/6)))  # Reasonable number of clusters
                        clusters = fcluster(link, num_clusters, criterion='maxclust')
                        
                        # Find boundaries between clusters in the ordered list
                        ordered_clusters = [clusters[order.index(i)] if i < len(order) else 0 
                                           for i in range(len(corr_matrix))]
                        
                        boundaries = []
                        for i in range(1, len(ordered_clusters)):
                            if ordered_clusters[i] != ordered_clusters[i-1]:
                                boundaries.append(i - 0.5)
                        
                        if boundaries:
                            cluster_boundaries = boundaries
                    except Exception:
                        cluster_boundaries = None
                
                except Exception:
                    # If clustering fails, fall back to default ordering
                    pass
                    
            elif ordering == "name":
                # Sort alphabetically
                corr_matrix = corr_matrix.reindex(sorted(corr_matrix.columns), axis=1)
                corr_matrix = corr_matrix.reindex(sorted(corr_matrix.index), axis=0)
                
            # Create correlation heatmap with plotly express
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Series", y="Series", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale=color_scale,
                zmin=-1,
                zmax=1,
                title=f"Correlation Heatmap ({len(series_names)} series)",
                aspect="auto",  # Maintain aspect ratio based on screen size
                text_auto='.2f' if len(corr_matrix) <= 20 else False,  # Show text values if not too many
            )
            
            # Add cluster boundary lines if available
            if ordering == "clustering" and cluster_boundaries:
                for boundary in cluster_boundaries:
                    # Add vertical line
                    fig.add_vline(x=boundary, line_width=2, line_dash="dash", line_color="black")
                    # Add horizontal line
                    fig.add_hline(y=boundary, line_width=2, line_dash="dash", line_color="black")
                    
        elif metric == "distance":
            # Calculate distance matrix using correlation
            from scipy.spatial.distance import pdist, squareform
            
            # Fill NAs to avoid distance calculation errors
            clean_data = pd_data.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate distance matrix
            dist_array = pdist(clean_data.T, metric='correlation')
            dist_matrix = pd.DataFrame(
                squareform(dist_array),
                index=series_names,
                columns=series_names
            )
            
            # Apply ordering if needed (similar to correlation case)
            if ordering == "clustering":
                try:
                    from scipy.cluster.hierarchy import linkage, dendrogram
                    link = linkage(dist_array, method='ward')
                    order = dendrogram(link, no_plot=True)['leaves']
                    ordered_series = [dist_matrix.index[i] for i in order if i < len(dist_matrix.index)]
                    dist_matrix = dist_matrix.reindex(ordered_series)[ordered_series]
                except Exception:
                    pass
            
            elif ordering == "name":
                dist_matrix = dist_matrix.reindex(sorted(dist_matrix.columns), axis=1)
                dist_matrix = dist_matrix.reindex(sorted(dist_matrix.index), axis=0)
            
            # Create heatmap using plotly express
            fig = px.imshow(
                dist_matrix,
                labels=dict(x="Series", y="Series", color="Distance"),
                x=dist_matrix.columns,
                y=dist_matrix.index,
                color_continuous_scale=color_scale,
                title=f"Distance Heatmap ({len(series_names)} series)",
                aspect="auto",
                text_auto='.2f' if len(dist_matrix) <= 20 else False,
                template="plotly_white",
            )
                
        else:  # values - time series values
            # Resample to reduce data points if needed
            if len(pd_data) > 100:
                # Resample to weekly or monthly based on data size
                freq = 'M' if len(pd_data) > 500 else 'W'
                pd_data = pd_data.resample(freq).mean()
            
            # Prepare for heatmap - transpose so series are rows
            values_df = pd_data.T
            
            # Apply ordering
            if ordering == "name":
                values_df = values_df.reindex(sorted(values_df.index))
            elif ordering == "frequency":
                # Sort by mean value
                means = values_df.mean(axis=1)
                values_df = values_df.reindex(means.sort_values(ascending=False).index)
            
            # Create heatmap using plotly express
            fig = px.imshow(
                values_df,
                labels=dict(x="Date", y="Series", color="Value"),
                x=[d.strftime('%Y-%m-%d') for d in pd_data.index],
                y=values_df.index,
                color_continuous_scale=color_scale,
                title=f"Time Series Values Heatmap ({len(pd_data.index)} time points × {len(series_names)} series)",
                aspect="auto",
                template="plotly_white",
            )
            
            # Customize x-axis for dates
            fig.update_xaxes(
                tickangle=45,
                tickmode='array',
                tickvals=[d.strftime('%Y-%m-%d') for d in pd_data.index[::max(1, len(pd_data.index)//10)]],  # Show only some dates
                ticktext=[d.strftime('%Y-%m-%d') for d in pd_data.index[::max(1, len(pd_data.index)//10)]]
            )
        
        # Common layout updates
        fig.update_layout(
            height=700,
            xaxis=dict(side="bottom"),
            yaxis=dict(side="left"),
            coloraxis_colorbar=dict(
                title=metric.capitalize(),
                thicknessmode="pixels", 
                thickness=20,
                lenmode="pixels", 
                len=600,
                yanchor="top", 
                y=1,
                ticks="outside",
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            ),
            margin=dict(l=40, r=40, t=80, b=40),
        )
        
        # Add warning if we limited the number of series
        if warning_msg:
            fig.add_annotation(
                text=warning_msg,
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=12, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                borderpad=4
            )
        
        # Config for better user interaction
        config = {
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'heatmap_{metric}',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        
        return fig
    
    @output
    @render.ui
    def output_ID_heatmap_interpretation():
        """
        Generate interpretation text for the heatmap.
        
        Returns
        -------
        shiny.ui.HTML
            HTML element containing the interpretation
        """
        metric = input.ID_heatmap_metric()
        
        if metric == "correlation":
            interpretation = """
            <div class="p-3 bg-light border rounded">
                <h5>How to interpret the correlation heatmap:</h5>
                <ul>
                    <li><strong>Red cells (value close to 1)</strong>: Strong positive correlation - series move in the same direction</li>
                    <li><strong>Blue cells (value close to -1)</strong>: Strong negative correlation - series move in opposite directions</li>
                    <li><strong>White/light colored cells (value close to 0)</strong>: Little to no correlation</li>
                    <li>The diagonal always shows perfect correlation (value = 1) as each series correlates perfectly with itself</li>
                    <li>This visualization helps identify patterns, relationships, and potential redundancies in your data</li>
                </ul>
                <p>When analyzing correlations, consider that:</p>
                <ul>
                    <li>Correlation does not imply causation</li>
                    <li>High correlation might indicate redundant information in your data</li>
                    <li>Clustering of correlated variables often reveals underlying factors or domains</li>
                </ul>
            </div>
            """
        elif metric == "distance":
            interpretation = """
            <div class="p-3 bg-light border rounded">
                <h5>How to interpret the distance heatmap:</h5>
                <ul>
                    <li><strong>Darker cells (lower values)</strong>: Series are more similar to each other</li>
                    <li><strong>Lighter cells (higher values)</strong>: Series are more different from each other</li>
                    <li>The diagonal always shows zero distance as each series is identical to itself</li>
                    <li>This visualization helps identify clusters and outliers in your data</li>
                </ul>
                <p>Distance matrices are useful for:</p>
                <ul>
                    <li>Identifying groups of similar time series</li>
                    <li>Finding outlier series that behave differently from others</li>
                    <li>Understanding the overall structure and relationships in your data</li>
                </ul>
            </div>
            """
        else:  # values
            interpretation = """
            <div class="p-3 bg-light border rounded">
                <h5>How to interpret the time series heatmap:</h5>
                <ul>
                    <li><strong>Darker/warmer colors</strong>: Higher values in the time series</li>
                    <li><strong>Lighter/cooler colors</strong>: Lower values in the time series</li>
                    <li>Each row represents one time series, and each column represents a time point</li>
                    <li>This visualization helps identify patterns over time and similarities between series</li>
                </ul>
                <p>Look for these patterns:</p>
                <ul>
                    <li>Horizontal patterns: Consistent behavior of specific series</li>
                    <li>Vertical patterns: Time periods with similar behavior across multiple series</li>
                    <li>Diagonal patterns: Progressive changes or delays in effects across series</li>
                </ul>
            </div>
            """
        
        return ui.HTML(interpretation)
    
