"""
Black-Litterman Module for QWIM Dashboard
=======================================
"""
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO

import yfinance as yf
from pandas import to_datetime
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
import statsmodels.api as sm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data():
    """   拿到两个数据集:ETF和ff5+mom   """
    ff5_path = Path("data/processed/ff5_mom_data.csv")
    ff_factors = pd.read_csv(
        ff5_path,
        parse_dates=['Date'],  # 解析日期列
        index_col='Date'       # 设为索引避免参与计算
    )
    #print(ff_factors)

    etf_path = Path("data/raw/etf_data.csv")
    etf_data = pd.read_csv(
        etf_path,
        parse_dates=['Date'],
        index_col='Date'
    )
    #print(etf_data)
    print("**************End for def: load_data()*************************")

    return ff_factors, etf_data

def get_etfReturn(etf_data):
    """   计算资产协方差矩阵(Covariance Matrix)   --但cov好像没怎么用到。。。"""
    etf_returns = etf_data.pct_change().dropna()
    #print(etf_returns)
    print("**************End for def get_etfReturn(etf_data)**************")
    return etf_returns

def split_trainTest():
    """   划分开test和train   """
    ff_factors, etf_data = load_data()
    etf_returns = get_etfReturn(etf_data)

    train_start_date = "2008-07-01"
    train_end_date = "2019-12-31"
    test_start_date = "2020-01-01"
    test_end_date = "2024-12-31"

    # 切分 Fama-French 因子数据和 ETF 收益数据
    train_ff_factors = ff_factors.loc[train_start_date:train_end_date]
    test_ff_factors = ff_factors.loc[test_start_date:test_end_date]

    train_etf_data = etf_data.loc[train_start_date:train_end_date]
    test_etf_data = etf_data.loc[test_start_date:test_end_date]

    # 检查划分是否成功
    #print(f"Train period ETF data: {train_etf_data.shape}")
    #print(f"Test period ETF data: {test_etf_data.shape}")
    #print(f"Train period ff5 data: {train_ff_factors.shape}")
    #print(f"Test period ff5 data: {test_ff_factors.shape}")

    # 切分 etf_returns
    train_etf_returns = etf_returns.loc[train_start_date:train_end_date]
    test_etf_returns = etf_returns.loc[test_start_date:test_end_date]
    #print(f"Train period etf_returns: {train_etf_returns.shape}")
    #print(f"Test period etf_returns: {test_etf_returns.shape}")

    print("**************End for def split_trainTest()********************")
    return train_ff_factors, test_ff_factors, train_etf_data, test_etf_data, train_etf_returns, test_etf_returns

def calMarketWeights():
    # 定义ETF列表
    tickers = ['SPY', 'IWM', 'EFA', 'EEM', 'AGG', 'LQD', 'HYG', 'TLT', 'GLD', 'VNQ', 'DBC', 'VT', 'XLE', 'XLK', 'UUP']

    # 初始化AUM字典
    aum_dict = {}

    # 获取每个ETF的总资产
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        aum = info.get("totalAssets", None)
        #print(f"{ticker} AUM: {aum}")
        aum_dict[ticker] = aum

    # 构建权重Series
    aum_series = pd.Series(aum_dict)
    market_weights = aum_series / aum_series.sum()

    print("**************End for def calMarketWeights()*******************")
    return market_weights

def prepStep():
    train_ff_factors, test_ff_factors, train_etf_data, test_etf_data, train_etf_returns, test_etf_returns = split_trainTest()

    # ✅ Step 1.1：计算 ETF 超额收益率
    #  定义无风险收益率序列
    train_rf_series = train_ff_factors['RF']
    train_etf_excess_returns = train_etf_returns.sub(train_rf_series, axis=0)
    test_rf_series = test_ff_factors['RF']
    test_etf_excess_returns = test_etf_returns.sub(test_rf_series, axis=0)

    # ✅ Step 1.2：时间对齐（ETF 收益 和 因子收益）
    # 对齐 ETF 和因子数据的时间范围
    common_train_dates = train_etf_excess_returns.index.intersection(train_ff_factors.index)
    common_train_dates = common_train_dates[common_train_dates >= '2008-07-02']

    # 截取重叠时间段
    train_etf_excess_returns = train_etf_excess_returns.loc[common_train_dates]
    train_factor_returns = train_ff_factors.loc[common_train_dates].drop(columns='RF')
    
    # 对齐 ETF 和因子数据的时间范围
    common_test_dates = test_etf_excess_returns.index.intersection(test_ff_factors.index)
    common_test_dates = common_test_dates[common_test_dates >= '2008-07-02']

    # 截取重叠时间段
    test_etf_excess_returns = test_etf_excess_returns.loc[common_test_dates]
    test_factor_returns = test_ff_factors.loc[common_test_dates].drop(columns='RF')

    # ✅ Step 1.3：对每个 ETF 回归其超额收益 vs 各因子
    betas = {}
    residual_vars = {}

    for etf in train_etf_excess_returns.columns:
        y = train_etf_excess_returns[etf]
        X = train_factor_returns

        # 对齐非 NaN 时间点
        data = pd.concat([y, X], axis=1).dropna()
        y_valid = data[etf]
        X_valid = data.drop(columns=[etf])

        # 添加常数项用于回归截距
        X_valid = sm.add_constant(X_valid)

        # 回归
        model = sm.OLS(y_valid.astype(float), X_valid.astype(float)).fit()

        betas[etf] = model.params[1:]          # 提取因子暴露系数（去除截距）
        residual_vars[etf] = model.mse_resid   # 提取残差方差
    
    # ✅ Step 1.4：构建 β（B 矩阵）和残差方差矩阵（D）
    # 构建因子暴露矩阵 B（ETF × 因子）
    B = pd.DataFrame(betas).T

    # 构建残差方差矩阵 D（对角阵）
    D = np.diag(list(residual_vars.values()))

    # ✅ Step 1.5：估计因子协方差矩阵 Ω（Omega）
    # 转换成 float 类型（很重要）
    train_factor_returns = train_factor_returns.astype(float)

    # 输入数据为 (n_periods x n_factors)，需转置为 (n_factors x n_periods)
    Omega = np.cov(train_factor_returns.values.T)

    # 保留列名，构建为 DataFrame
    Omega_df = pd.DataFrame(Omega, index=train_factor_returns.columns, columns=train_factor_returns.columns)

    # 查看结果
    #print(Omega_df)

    # ✅ Step 1.6：构建资产协方差矩阵 Σ（Sigma）
    # 资产协方差矩阵 Σ = B @ Ω @ B.T + D
    Sigma = B @ Omega_df @ B.T + D
    #print(Sigma)

    print("**************End for def prepStep()*******************")
    return train_factor_returns, train_factor_returns, B, Omega_df, Sigma

def metricGenerate(risk_aversion, tau, P_f, Q_f):
    market_weights = calMarketWeights()
    #print(market_weights)
    w_m = market_weights

    # ✅ Step 2.1: 计算隐含均值（π）
    pi = risk_aversion * Sigma.dot(w_m)

    # ✅ Step 2.2: 构造投资者视角 P 和 Q、引入risk_aversion、tau

    # ✅ Step 2.3: 计算implied factor return（𝜓~）
    from numpy.linalg import pinv
    # implied asset-level return
    pi = risk_aversion * Sigma.values @ market_weights.values.reshape(-1, 1)

    # B matrix: (num_assets x num_factors)
    B_mat = B.values  # already ETF × factor
    B_pinv = pinv(B_mat)  # pseudo-inverse

    psi_tilde = B_pinv @ pi  # implied factor return (K x 1)

    # ✅Step 2.4: 在 factor space 做 Bayesian 更新
    # factor covariance matrix: Omega_df
    Omega_f = Omega_df.values
    tau = 0.025
    Omega_prior = tau * Omega_f

    # 构建观点协方差 Φ（factor-level）
    # 计算 view 协方差（只保留对角）
    Phi_f = np.diag(np.diag(P_f @ Omega_prior @ P_f.T))  # shape: (V x V）

    # 可选：转为 DataFrame 方便 debug 和查看
    Phi_f_df = pd.DataFrame(Phi_f, index=[f'View{i+1}' for i in range(P_f.shape[0])],
                            columns=[f'View{i+1}' for i in range(P_f.shape[0])])

    #print("✅ Factor-level view uncertainty matrix Φ_f:")
    #print(Phi_f_df)
    # Black-Litterman posterior in factor space
    middle_term = np.linalg.inv(P_f @ Omega_prior @ P_f.T + Phi_f)
    psi_bl = psi_tilde + Omega_prior @ P_f.T @ middle_term @ (Q_f - P_f @ psi_tilde)

    # ✅ Step 2.5: 计算后验期望收益 𝜇_𝐵𝐿（映射回 asset space）
    mu_bl = B_mat @ psi_bl  # (N_assets x 1)
    bl_mean_returns = pd.Series(mu_bl.flatten(), index=B.index)
    print(bl_mean_returns)
    print("Sum of weights:", market_weights.sum())  # 应该约等于 1
    print()

    print("Reach the End")

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
                        ui.input_slider("risk_aversion", "Select Risk Aversion:", min=0, max=5, value=1, step=0.5),
                        ui.input_slider("tau", "Select Tau:", min=0, max=5, value=1, step=0.25),
                        
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
    @reactive.Effect
    @reactive.event(input.apply_investor_view)
    def apply_investor_view():
        # Print "hi" to confirm the action is triggered
        print("hi")
        
        # Print Risk Aversion and Tau
        print(f"Risk Aversion: {input.risk_aversion.get()}")
        print(f"Tau: {input.tau.get()}")
        print()
        
        # Print MKT Factor
        print(f"Include MKT: {input.factor_mkt.get()}")
        print(f"Direction for MKT: {input.factor_mkt_direction.get()}")
        print(f"Factor Value for MKT: {input.factor_mkt_value.get()}")
        print()
        
        # Print SMB Factor
        print(f"Include SMB: {input.factor_smb.get()}")
        print(f"Direction for SMB: {input.factor_smb_direction.get()}")
        print(f"Factor Value for SMB: {input.factor_smb_value.get()}")
        print()
        
        # Print HML Factor
        print(f"Include HML: {input.factor_hml.get()}")
        print(f"Direction for HML: {input.factor_hml_direction.get()}")
        print(f"Factor Value for HML: {input.factor_hml_value.get()}")
        print()
        
        # Print RMW Factor
        print(f"Include RMW: {input.factor_rmw.get()}")
        print(f"Direction for RMW: {input.factor_rmw_direction.get()}")
        print(f"Factor Value for RMW: {input.factor_rmw_value.get()}")
        print()
        
        # Print CMA Factor
        print(f"Include CMA: {input.factor_cma.get()}")
        print(f"Direction for CMA: {input.factor_cma_direction.get()}")
        print(f"Factor Value for CMA: {input.factor_cma_value.get()}")
        print()
        
        # Print MOM Factor
        print(f"Include MOM: {input.factor_mom.get()}")
        print(f"Direction for MOM: {input.factor_mom_direction.get()}")
        print(f"Factor Value for MOM: {input.factor_mom_value.get()}")
        print()
        
        # Print custom view name
        print(f"Custom View Name: {input.custom_name.get()}")
        print()
        
        # Optionally, return the selected values for further processing or visualization
        return {
            "risk_aversion": input.risk_aversion.get(),
            "tau": input.tau.get(),
            "mkt_factor": {
                "include": input.factor_mkt.get(),
                "direction": input.factor_mkt_direction.get(),
                "value": input.factor_mkt_value.get(),
            },
            "smb_factor": {
                "include": input.factor_smb.get(),
                "direction": input.factor_smb_direction.get(),
                "value": input.factor_smb_value.get(),
            },
            "hml_factor": {
                "include": input.factor_hml.get(),
                "direction": input.factor_hml_direction.get(),
                "value": input.factor_hml_value.get(),
            },
            "rmw_factor": {
                "include": input.factor_rmw.get(),
                "direction": input.factor_rmw_direction.get(),
                "value": input.factor_rmw_value.get(),
            },
            "cma_factor": {
                "include": input.factor_cma.get(),
                "direction": input.factor_cma_direction.get(),
                "value": input.factor_cma_value.get(),
            },
            "mom_factor": {
                "include": input.factor_mom.get(),
                "direction": input.factor_mom_direction.get(),
                "value": input.factor_mom_value.get(),
            },
            "custom_view_name": input.custom_name.get()
        }


if __name__ == "__main__":
    # 如果直接运行该脚本，会执行以下代码：
    pd.options.display.float_format = '{:.16f}'.format
    np.set_printoptions(precision=16, suppress=False)

    risk_aversion = 2.5
    tau = 0.025
    P_f = np.array([
        [1, 0, 0, 0, 0, 0],  # Mkt-RF
        [0, 0, 0, 0, 0, 1]   # MOM
    ])
    Q_f = np.array([
        [0.07/252],
        [0.03/252]
    ])
    

    train_factor_returns, train_factor_returns, B, Omega_df, Sigma = prepStep()
    metricGenerate(risk_aversion, tau, P_f, Q_f)
    print("Main END")
        