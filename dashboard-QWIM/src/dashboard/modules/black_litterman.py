"""
Black-Litterman Module for QWIM Dashboard
=======================================
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

def load_data():
    ff5_path = Path("data/processed/ff5_mom_data.csv")
    ff_factors = pd.read_csv(ff5_path)
    print(ff_factors.head())

    etf_path = Path("data/raw/etf_data.csv")
    etf_data = pd.read_csv(etf_path)
    print(etf_data.head())

    return ff_factors, etf_data

@module.ui
def model1_ui():
    """Create the UI for Model 1 (Factor-based Black-Litterman Model) tab.
    
    This function generates the complete user interface for Model 1 tab, organizing
    content into multiple sub-tabs for analysis and visualization.
    
    Returns
    -------
    shiny.ui.page_fluid
        A fluid page layout containing all UI elements for Model 1 tab
    """
    return ui.page_fluid(
        ui.h2("Factor-based Black-Litterman Model Analysis"),
        ui.navset_tab(
            ui.nav_panel(
                "Try Model",  # Main Panel Tab
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Risk Aversion and Tau"),
                        ui.input_slider("risk_aversion", "Select Risk Aversion:", min=0, max=5, value=2),
                        ui.input_slider("tau", "Select Tau:", min=0.01, max=1, value=0.05),
                        
                        ui.h4("Custom Investor View (DIY)"),
                        
                        # MKT Factor Selection
                        ui.input_checkbox("factor_mkt", "Include MKT"),
                        ui.input_radio_buttons("factor_mkt_direction", "Direction for MKT:", choices={"↑": "Up", "↓": "Down"}, selected="↑"),
                        ui.input_select("factor_mkt_value", "Select Factor Value for MKT:", 
                                        choices=[0.0002, 0.0003, -0.0001],
                                        selected=0.0002
                        ),
                        
                        # SMB Factor Selection
                        ui.input_checkbox("factor_smb", "Include SMB"),
                        ui.input_radio_buttons("factor_smb_direction", "Direction for SMB:", choices={"↑": "Up", "↓": "Down"}, selected="↑"),
                        ui.input_select("factor_smb_value", "Select Factor Value for SMB:", 
                                        choices=[0.0002, 0.0003, -0.0001],
                                        selected=0.0002
                        ),
                        
                        # HML Factor Selection
                        ui.input_checkbox("factor_hml", "Include HML"),
                        ui.input_radio_buttons("factor_hml_direction", "Direction for HML:", choices={"↑": "Up", "↓": "Down"}, selected="↑"),
                        ui.input_select("factor_hml_value", "Select Factor Value for HML:", 
                                        choices=[0.0002, 0.0003, -0.0001],
                                        selected=0.0002
                        ),
                        
                        # RMW Factor Selection
                        ui.input_checkbox("factor_rmw", "Include RMW"),
                        ui.input_radio_buttons("factor_rmw_direction", "Direction for RMW:", choices={"↑": "Up", "↓": "Down"}, selected="↑"),
                        ui.input_select("factor_rmw_value", "Select Factor Value for RMW:", 
                                        choices=[0.0002, 0.0003, -0.0001],
                                        selected=0.0002
                        ),
                        
                        # CMA Factor Selection
                        ui.input_checkbox("factor_cma", "Include CMA"),
                        ui.input_radio_buttons("factor_cma_direction", "Direction for CMA:", choices={"↑": "Up", "↓": "Down"}, selected="↑"),
                        ui.input_select("factor_cma_value", "Select Factor Value for CMA:", 
                                        choices=[0.0002, 0.0003, -0.0001],
                                        selected=0.0002
                        ),
                        
                        # MOM Factor Selection
                        ui.input_checkbox("factor_mom", "Include MOM"),
                        ui.input_radio_buttons("factor_mom_direction", "Direction for MOM:", choices={"↑": "Up", "↓": "Down"}, selected="↑"),
                        ui.input_select("factor_mom_value", "Select Factor Value for MOM:", 
                                        choices=[0.0002, 0.0003, -0.0001],
                                        selected=0.0002
                        ),
                        
                        ui.input_text("custom_name", "Name Your Custom View:", value="Custom View 1"),
                        ui.input_action_button("apply_investor_view", "Apply Selection", class_="btn-primary"),
                    ),
                    ui.h3("Model Overview"),
                    ui.output_table("output_investor_views_table")
                )
            ),
            ui.nav_panel(
                "Dynamic vs Static",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Strategy Selection"),
                        ui.input_select("ID_strategy_type", "Select Strategy Type:", choices=["Static", "Dynamic"], selected="Dynamic"),
                    ),
                    ui.h3("Strategy Performance Comparison"),
                    output_widget("output_ID_strategy_comparison"),
                ),
            ),
            ui.nav_panel(
                "Rolling Window",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Rolling Window Length"),
                        ui.input_slider("ID_window_length", "Select Rolling Window Length:", min=20, max=250, value=60, step=10),
                    ),
                    ui.h3("Performance Metrics"),
                    ui.output_ui("output_ID_rolling_window_performance"),
                ),
            ),
            ui.nav_panel(
                "Stress Testing",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Stress Test Parameters"),
                        ui.input_select("ID_stress_test_type", "Select Stress Test Type:", choices=["Historical", "Monte Carlo"], selected="Historical"),
                    ),
                    ui.h3("Stress Test Results"),
                    output_widget("output_ID_stress_test_results"),
                ),
            ),
            ui.nav_panel(
                "Weights",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Portfolio Weights Options"),
                        ui.input_select("ID_portfolio_weights_type", "Select Weights Type:", choices=["Equal", "Risk-based", "Market-cap"], selected="Equal"),
                    ),
                    ui.h3("Portfolio Weights Visualization"),
                    output_widget("output_ID_portfolio_weights"),
                ),
            ),
        ),
    )




@module.server
def model1_server(input, output, session, data_r, series_names_r):
    """Server logic for Model 1 (Factor-based Black-Litterman Model) tab.
    
    This function implements the server logic for the Model 1 tab, processing user inputs
    and generating corresponding visualizations and performance metrics. Each sub-tab is
    handled individually with its own reactive calculations.
    
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
        Reactive value containing all available series names
    
    Notes
    -----
    This function contains several reactive components for handling each sub-tab's logic:
    - Risk aversion, tau, investor view selection
    - Strategy comparison
    - Rolling window analysis
    - Stress testing
    - Portfolio weights visualization
    """
    
    return 0

if __name__ == "__main__":
    # 如果直接运行该脚本，会执行以下代码：
    etf_data, factor_data = load_data()
        