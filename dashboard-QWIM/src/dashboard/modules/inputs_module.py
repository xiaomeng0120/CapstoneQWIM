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
        