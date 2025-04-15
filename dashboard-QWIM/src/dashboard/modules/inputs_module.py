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

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        ui.h2("ETF Asset Visualization"),
        ui.navset_card_tab(
            ui.nav_panel(
                "Investment Pool",
                    ui.h4("ETFs Basic Information"),
                    ui.h6("We have selected the following 15 representative ETFs covering multiple asset classes such as stocks, bonds, commodities and real estate:"),
                    ui.output_data_frame("ETF_pool_table"),  # 改为使用widget输出
                    ui.tags.style("""
                        .datagrid table { 
                            font-size: 0.85em;
                            line-height: 1.2;
                        }
                        .datagrid th {
                            padding: 4px 8px;
                        }
                        .datagrid td {
                            padding: 3px 6px;
                        }
                    """), 
                    ui.hr(),  # Divider between tables
                    ui.h4("ETFs Daily Close Price Overview"),  # New Table Title
                    ui.h6("Choose the following time range to explore more!"),
                    ui.input_date_range(  # Time range selection for the new table
                        "new_data_date_range",
                        "Select Time Range:",
                        start="2025-01-01",
                        end="2025-03-01",
                    ),
                    ui.output_data_frame("new_data_table"),  # New Table to display the selected data
                    ui.tags.style("""
                        .datagrid table { 
                            font-size: 0.85em;
                            line-height: 1.2;
                        }
                        .datagrid th {
                            padding: 4px 8px;
                        }
                        .datagrid td {
                            padding: 3px 6px;
                        }
                    """)
            ),
            ui.nav_panel(
                "ETF Close Price Trends",  # Second sub-tab
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_selectize(
                            "selected_etfs", 
                            "Select ETFs to visualize:", 
                            choices=["SPY", "IWM", "EFA", "EEM", "AGG", "LQD", "HYG", "TLT", "GLD", "VNQ", "DBC", "VT", "XLE", "XLK", "UUP"], 
                            selected=["EFA", "IWM", "HYG", "GLD", "LQD", "VNQ"],
                            multiple=True,  # Enable multi-selection
                            width="100%",
                        ),
                        ui.input_date_range(
                            "date_range", 
                            "Select Date Range:", 
                            start="2008-07-01", 
                            end="2025-03-01",
                        ),
                        # apply_filters
                        ui.input_action_button(
                            "ID_apply_filters",  # 按钮 ID
                            "Apply Filters",  # 按钮标签
                            class_="btn-primary"  # 按钮样式
                        ), 
                    ),
                    ui.h4("ETFs Close Price Overview"),
                    ui.h6("Choose the following time range and select ETFs to explore more!"),
                    output_widget("output_ID_overview_plot"),   # 在这里渲染图表
                ),
            )

            # Other sub tab...


        )
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

    # 普通的函数，用来更新ETF表格数据
    def _update_etf_table():
        """Return ETF data to be rendered in a table"""
        etf_data = {
            "Ticker": ["SPY", "IWM", "EFA", "EEM", "AGG", "LQD", "HYG", "TLT", "GLD", "VNQ", "DBC", "VT", "XLE", "XLK", "UUP"],
            "ETF Full Name": ["SPDR S&P 500 ETF", "iShares Russell 2000 ETF", "iShares MSCI EAFE ETF", "iShares MSCI Emerging Markets ETF", "iShares Core US Aggregate Bond ETF", "iShares iBoxx $ Investment Grade Corporate Bond ETF", "iShares iBoxx $ High Yield Corporate Bond ETF", "iShares 20+ Year Treasury Bond ETF", "SPDR Gold Shares", "Vanguard Real Estate ETF", "Invesco DB Commodity Index Tracking Fund", "Vanguard Total World Stock ETF", "Energy Select Sector SPDR Fund", "Technology Select Sector SPDR Fund", "Invesco DB US Dollar Index Bullish Fund"],
            "Type": ["Large-Cap Equity", "Small-Cap Equity", "International Equity", "Emerging Markets", "Aggregate Bonds", "Investment Grade Corporate Bonds", "High Yield Bonds", "Long-Term Bonds", "Commodities", "REITs", "Commodities", "Global Stock", "Energy Sector", "Technology Sector", "USD Index"]
        }
        return pd.DataFrame(etf_data)
    
    # New data section
    def _load_new_data():
        """Load the new dataset from the raw/etf_data directory."""
        data_path = Path("data/raw/etf_data.csv")
        if not data_path.exists():
            print(f"Data file not found at {data_path}")  # Print data file error
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        # Load the data using pandas (or polars if necessary)
        df = pd.read_csv(data_path)
        print(f"Loaded ETF data: \n{df.head()}")  # Print the first few rows of the loaded data

        # Ensure 'Date' column is in datetime format, handle errors if any
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors='raise')  # Ensure 'Date' column is datetime
            print(f"Date column converted successfully. Data types:\n{df.dtypes}")  # Check data types
        except Exception as e:
            print(f"Error while converting Date column: {e}")
            raise
        
        # Check if 'Date' column is correctly converted
        print(f"Date column type: {df['Date'].dtype}")
        
        # Print first few rows to confirm
        print(df.head())
        
        return df
    
    # 在服务器端渲染表格
    @output
    @render.data_frame
    def ETF_pool_table():
        """Render the interactive ETF data table"""
        df = _update_etf_table()
        print(f"ETF Pool Table Data: \n{df.head()}")  # Print ETF table data
        
        return render.DataGrid(
            df,
            row_selection_mode="none",
            filters=False,  # 启用列过滤
            summary=True,  # 显示统计摘要
            height="400px",
            width="100%"
        )
    
    @output
    @render.data_frame
    def new_data_table():
        """Render the new data table based on selected time range."""
        try:
            # Load the new data from the file
            df = _load_new_data()
            # Apply time range filter
            start_date = pd.to_datetime(input.new_data_date_range()[0])  # 将 start_date 转换为 datetime64
            end_date = pd.to_datetime(input.new_data_date_range()[1])  # 将 end_date 转换为 datetime64
            print(f"Selected date range: {start_date} to {end_date}")
            df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
            print(f"Filtered Data:\n{df_filtered.head()}")  # Print filtered data

            # 格式化日期列，只保留日期部分
            df_filtered["Date"] = df_filtered["Date"].dt.strftime('%Y-%m-%d')

            # 格式化每列数据的小数位数
            df_filtered = df_filtered.round(2)  # 例如保留4位小数
            
            # Return the filtered data as a DataFrame
            return render.DataGrid(
                df_filtered,
                row_selection_mode="none",
                filters=False,  # Enable column filtering
                summary=True,  # Display summary statistics
                height="400px",
                width="100%"
            )
        except Exception as e:
            # Log the error and return an empty DataFrame
            logger.error(f"Error loading data: {str(e)}")
            
            # Return an empty DataFrame with the expected structure
            return pd.DataFrame({"Date": [], "Ticker": [], "Value": []})
    
    # Continue with other effects...
    
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
        print("Apply filters triggered!")
        data = _load_new_data()
        selected_etfs = list(input.selected_etfs())
        date_range = input.date_range()

        # 列名检查
        print("数据列名:", data.columns.tolist())
        print("用户选择的ETF:", selected_etfs)
        missing_columns = [etf for etf in selected_etfs if etf not in data.columns]
        if missing_columns:
            print(f"错误：列 {missing_columns} 不存在！")
            return {"filtered_data": pd.DataFrame(), "selected_etfs": []}

        # 数据处理
        selected_columns = ["Date"] + selected_etfs
        filtered_data = data[selected_columns].copy()
        filtered_data["Date"] = pd.to_datetime(filtered_data["Date"], errors="coerce")
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_data = filtered_data[(filtered_data["Date"] >= start_date) & (filtered_data["Date"] <= end_date)]

        print("筛选后日期范围:", filtered_data["Date"].min(), "至", filtered_data["Date"].max())
        return {"filtered_data": filtered_data, "selected_etfs": selected_etfs}

    @output
    @reactive.effect
    @render_widget
    def output_ID_overview_plot():
        filters_result = apply_filters()
        filtered = filters_result["filtered_data"]
        selected = filters_result["selected_etfs"]

        if not selected or filtered is None or filtered.empty:
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
                text="Select one or more ETFs and apply filters to view data",
                showarrow=False,
                font=dict(size=14),
            )
            return fig

        # 处理数据
        pd_data = filtered.copy()
        pd_data["Date"] = pd.to_datetime(pd_data["Date"], errors="coerce").dt.tz_localize(None)
        pd_data["Date_str"] = pd_data["Date"].dt.strftime("%Y-%m-%d")  # 格式化日期为字符串
        pd_data = pd_data.dropna(subset=["Date", "Date_str"] + selected).reset_index(drop=True)

        # 创建图表
        fig = go.Figure()
        for etf in selected:
            if etf in pd_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pd_data["Date_str"],  # 使用字符串日期
                        y=pd_data[etf],
                        mode="lines+markers",
                        name=etf,
                        line=dict(width=2),
                    )
                )
            else:
                print(f"列 {etf} 不存在！")

        # 配置布局
        fig.update_layout(
            title="ETF Close Price Trends",
            xaxis_title="Date",
            yaxis_title="Close Price",
            xaxis=dict(
                tickformat="%Y-%m-%d",  # 指定日期格式
                tickangle=45,
            ),
            yaxis=dict(autorange=True),
            template="plotly_white",
            height=600,
            dragmode="pan",      # 拖动时平移图表
        )
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