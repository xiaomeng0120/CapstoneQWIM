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
import cvxpy as cp
from numpy.linalg import pinv

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

def calculate_max_drawdown(portfolio_value):
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return max_drawdown, drawdown

def calculate_max_10_day_drawdown(portfolio_value, window=10):
    max_dd_10day = 0
    for i in range(len(portfolio_value) - window):
        window_slice = portfolio_value[i:i+window]
        peak = window_slice.max()
        trough = window_slice.min()
        drawdown = (trough - peak) / peak
        max_dd_10day = min(max_dd_10day, drawdown)
    return max_dd_10day

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

    # 检查划分是否成功
    #print(f"Train period ff5 data: {train_ff_factors.shape}")
    #print(f"Test period ff5 data: {test_ff_factors.shape}")

    # 切分 etf_returns
    train_etf_returns = etf_returns.loc[train_start_date:train_end_date]
    test_etf_returns = etf_returns.loc[test_start_date:test_end_date]
    #print(f"Train period etf_returns: {train_etf_returns.shape}")
    #print(f"Test period etf_returns: {test_etf_returns.shape}")

    print("**************End for def split_trainTest()********************")
    return train_ff_factors, test_ff_factors, train_etf_returns, test_etf_returns

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

def prepStep(train_ff_factors, test_ff_factors, train_etf_returns, test_etf_returns):
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

    print("**************End for def prepStep()***************************")
    return train_factor_returns, train_factor_returns, B, Omega_df, Sigma, test_etf_excess_returns

def metricGenerate(risk_aversion, tau, P_f, Q_f, Sigma, B, Omega_df, test_etf_excess_returns, test_factor_returns):
    market_weights = calMarketWeights()
    w_m = market_weights

    # ✅ Step 2.1: 计算隐含均值（π）
    pi = risk_aversion * Sigma.dot(w_m)
    
    # ✅ Step 2.2: 构造投资者视角 P 和 Q、引入risk_aversion和tau

    # ✅ Step 2.3: 计算implied factor return（𝜓~）
    # implied asset-level return
    pi = risk_aversion * Sigma.values @ market_weights.values.reshape(-1, 1)

    # B matrix: (num_assets x num_factors)
    B_mat = B.values  # already ETF × factor
    B_pinv = pinv(B_mat)  # pseudo-inverse

    psi_tilde = B_pinv @ pi  # implied factor return (K x 1)
    
    # ✅Step 2.4: 在 factor space 做 Bayesian 更新
    # factor covariance matrix: Omega_df
    Omega_f = Omega_df.values
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
    #print(bl_mean_returns)
    #print("Sum of weights:", market_weights.sum())  # 应该约等于 1

    # ✅ Step 3: 组合权重优化（Optimal portfolio weights）
    # 使用 factor-space 后验期望收益
    mu_bl_vec = bl_mean_returns.values.reshape(-1, 1)   # 转为列向量
    n_assets = mu_bl_vec.shape[0]

    # 定义优化变量：资产权重向量
    w = cp.Variable((n_assets, 1))

    # 目标函数：最大化收益 - 风险惩罚
    objective = cp.Maximize(w.T @ mu_bl_vec - (risk_aversion / 2) * cp.quad_form(w, Sigma.values))

    # 约束条件：权重和为1，且不允许做空（可以改）
    constraints = [cp.sum(w) == 1, w >= 0]

    # 建立并求解优化问题
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # 提取最优权重
    w_opt = pd.Series(w.value.flatten(), index=Sigma.columns)
    pd.options.display.float_format = '{:.16f}'.format
    #print("✅ Optimal Portfolio Weights (Factor View):")
    #print(w_opt)

    # 计算每日组合收益（假设你已有组合权重 w_opt 和资产日收益 etf_excess_returns）
    daily_portfolio_return = test_etf_excess_returns @ w_opt

    # 🟢 以下是各种指标计算
    mean_return = daily_portfolio_return.mean()
    annual_return = (1 + mean_return)**252 - 1
    geometric_return = (np.prod(1 + daily_portfolio_return))**(252 / len(daily_portfolio_return)) - 1
    min_return = daily_portfolio_return.min()

    # 假设 portfolio_returns 是每日组合收益率（按日）
    initial_value = 100
    portfolio_value = (1 + daily_portfolio_return).cumprod() * initial_value
    max_dd, drawdown_series = calculate_max_drawdown(portfolio_value)
    max_10_day_dd = calculate_max_10_day_drawdown(portfolio_value)

    volatility = daily_portfolio_return.std()
    sharpe_ratio = mean_return / volatility * np.sqrt(252)
    # 年化波动率 = 日波动率 * sqrt(252)
    annualized_volatility = volatility * np.sqrt(252)

    skewness = daily_portfolio_return.skew()
    kurtosis = daily_portfolio_return.kurt()
    confidence_level = 0.95
    VaR_95 = -np.percentile(daily_portfolio_return, (1 - confidence_level) * 100)
    CVaR_95 = -daily_portfolio_return[daily_portfolio_return <= -VaR_95].mean()

     # ✅ Step 5：收益归因分析（Return Attribution）
     # ✅ Step 5.1: 组合隐含溢价 π̃(x)、因子贡献 ψ̃(x)、残差 ν̆(x)
    # 投资组合的总隐含溢价
    pi_tilde_x = w_opt @ bl_mean_returns

    # 组合因子暴露 beta(x)
    beta_x = B.T @ w_opt  # shape: (K因子,)

    # 从文献结构：组合系统性溢价 ψ̃(x) = β(x)^T × ψ̃
    # ψ̃ = 隐含因子溢价 ≈ mean of factor returns
    psi_tilde = test_factor_returns.mean()  # shape: (K因子,)
    psi_tilde_x = beta_x @ psi_tilde

    # 残差部分
    residual_premium = pi_tilde_x - psi_tilde_x

    # ✅ Step 5.2: 组合因子解释度  𝑅𝑐^2(𝑥)
    numerator = w_opt.T @ B @ Omega_df @ B.T @ w_opt
    denominator = w_opt.T @ Sigma @ w_opt
    R_squared_c = numerator / denominator

    '''
    print(f"""
        📊 Portfolio Performance Summary:
        ----------------------------------------
        Mean Daily Return      : {mean_return:.4%}
        Annualized Return      : {annual_return:.4%}
        Geometric Return       : {geometric_return:.4%}
        Minimum Daily Return   : {min_return:.4%}
        Volatility (daily)     : {volatility:.4%}
        Volatility (annual)    : {annualized_volatility:.4%}
        Sharpe Ratio (annual)  : {sharpe_ratio:.4f}
        Skewness               : {skewness:.4f}
        Kurtosis (excess)      : {kurtosis:.4f}
        Max Drawdown           : {max_dd:.4%}
        Max 10-Day Drawdown    : {max_10_day_dd:.4%}
        VaR 95% (1-day)        : {VaR_95:.4%}
        CVaR 95% (1-day)       : {CVaR_95:.4%}
        Hidden Alpha π̃(x)     : {pi_tilde_x:.4%}
        Factor Return ψ̃(x)   : {psi_tilde_x:.4%}
        Residual     ν̆(x)     : {residual_premium:.4%}
        Factor   R²_c(x)   : {R_squared_c:.4%}
        """)
    '''

    results = {
        "Mean Daily Return": f"{mean_return:.4%}",
        "Annualized Return": f"{annual_return:.4%}",
        "Geometric Return": f"{geometric_return:.4%}",
        "Minimum Daily Return": f"{min_return:.4%}",
        "Volatility (daily)": f"{volatility:.4%}",
        "Volatility (annual)": f"{annualized_volatility:.4%}",
        "Sharpe Ratio (annual)": f"{sharpe_ratio:.4f}",
        "Skewness": f"{skewness:.4f}",
        "Kurtosis (excess)": f"{kurtosis:.4f}",
        "Max Drawdown": f"{max_dd:.4%}",
        "Max 10-Day Drawdown": f"{max_10_day_dd:.4%}",
        "VaR 95% (1-day)": f"{VaR_95:.4%}",
        "CVaR 95% (1-day)": f"{CVaR_95:.4%}",
        "Hidden Alpha": f"{pi_tilde_x:.4%}",
        "Factor Return": f"{psi_tilde_x:.4%}",
        "Residual": f"{residual_premium:.4%}",
        "Factor R sqaured": f"{R_squared_c:.4%}"
    }

    print("Reach the End")
    return results


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
                        ui.input_slider("risk_aversion", "Select Risk Aversion:", min=0, max=5, value=2.5, step=0.5),
                        ui.input_slider("tau", "Select Tau:", min=0.01, max=0.1, value=0.025, step=0.005),
                        
                        ui.h4("Custom Investor View (DIY)"),
                        
                        # MKT Factor Selection
                        ui.input_checkbox("factor_mkt", "Include MKT"),
                        ui.input_radio_buttons("factor_mkt_direction", "Direction for MKT:", choices={"Up": "Up", "Down": "Down"}, selected="Up"),
                        ui.input_select("factor_mkt_value", "Select Factor Value for MKT (Annual Effect):", 
                                        choices=[0.02, 0.04, 0.06, 0.08, 0.10],
                                        selected=0.06
                        ),
                        
                        # SMB Factor Selection
                        ui.input_checkbox("factor_smb", "Include SMB"),
                        ui.input_radio_buttons("factor_smb_direction", "Direction for SMB:", choices={"Up": "Up", "Down": "Down"}, selected="Up"),
                        ui.input_select("factor_smb_value", "Select Factor Value for SMB (Annual Effect):", 
                                        choices=[0.02, 0.04, 0.06, 0.08, 0.10],
                                        selected=0.06
                        ),
                        
                        # HML Factor Selection
                        ui.input_checkbox("factor_hml", "Include HML"),
                        ui.input_radio_buttons("factor_hml_direction", "Direction for HML:", choices={"Up": "Up", "Down": "Down"}, selected="Up"),
                        ui.input_select("factor_hml_value", "Select Factor Value for HML (Annual Effect):", 
                                        choices=[0.02, 0.04, 0.06, 0.08, 0.10],
                                        selected=0.06
                        ),
                        
                        # RMW Factor Selection
                        ui.input_checkbox("factor_rmw", "Include RMW"),
                        ui.input_radio_buttons("factor_rmw_direction", "Direction for RMW:", choices={"Up": "Up", "Down": "Down"}, selected="Up"),
                        ui.input_select("factor_rmw_value", "Select Factor Value for RMW (Annual Effect):", 
                                        choices=[0.02, 0.04, 0.06, 0.08, 0.10],
                                        selected=0.06
                        ),
                        
                        # CMA Factor Selection
                        ui.input_checkbox("factor_cma", "Include CMA"),
                        ui.input_radio_buttons("factor_cma_direction", "Direction for CMA:", choices={"Up": "Up", "Down": "Down"}, selected="Up"),
                        ui.input_select("factor_cma_value", "Select Factor Value for CMA (Annual Effect):", 
                                        choices=[0.02, 0.04, 0.06, 0.08, 0.10],
                                        selected=0.06
                        ),
                        
                        # MOM Factor Selection
                        ui.input_checkbox("factor_mom", "Include MOM"),
                        ui.input_radio_buttons("factor_mom_direction", "Direction for MOM:", choices={"Up": "Up", "Down": "Down"}, selected="Up"),
                        ui.input_select("factor_mom_value", "Select Factor Value for MOM (Annual Effect):", 
                                        choices=[0.02, 0.04, 0.06, 0.08, 0.10],
                                        selected=0.06
                        ),
                        
                        ui.input_text("custom_name", "Name Your Custom View:", value="Custom View 1"),
                        ui.input_action_button("apply_investor_view", "Apply Selection", class_="btn-primary"),
                    ),
                    ui.h3("Model Overview", style="margin-top: 0px; margin-bottom: 5px;"),
                    ui.markdown("""
                        **Welcome to the Custom Portfolio Builder!**

                        You can create your own investment view by selecting factors and specifying the expected annual effect. Here's a quick guide:

                        **Factors Explained**:
                        - **Risk Aversion**: Determines how much risk you are willing to take. Higher means less risk.
                        - **Tau**: Confidence in your custom views. Higher means more confidence.
                        - **MKT**: Market risk (overall equity premium)
                        - **SMB**: Size (small minus big)
                        - **HML**: Value (high book-to-market minus low)
                        - **RMW**: Profitability (robust minus weak)
                        - **CMA**: Investment (conservative minus aggressive)
                        - **MOM**: Momentum (winners minus losers)  
                        
                        *Value picked for factor is annual increse or decrese, e.g. 0.02 stands for 2%.*  

                        Select the factors you believe in, set their direction and strength, and click **Apply** to see your view compared to baseline strategies!  
                        📝 Note that: Running Custom Portfolio takes some time ⏳ and your choice **will be saved** if each choice have a **different name**!
                        """),
                    ui.output_text("status"),
                    ui.div(
                        output_widget("output_investor_views_table"),
                        style="height: 500px; overflow-y: auto;"
                    ), 
                    ui.hr(style="margin-top: 5px; margin-bottom: 5px;"),
                    ui.h3("View Performance (Scatter)", style="margin-top: 10px; margin-bottom: 5px;"),
                    ui.div(
                        output_widget("output_investor_views_scatter"),
                        style="height: 400px; overflow-y: auto;"
                    ),
                    ui.hr(style="margin-top: 5px; margin-bottom: 5px;"),
                    ui.h3("Basic 7 Views Info", style="margin-top: 10px; margin-bottom: 5px;"),
                    ui.h6("All these 7 views based on risk aversion=2.5 and tau=0.025. Welcome to try more with risk aversion and tau using the sidebar!", style="margin-top: 10px; margin-bottom: 5px;"),
                    ui.output_data_frame("output_viewsInfo_table")
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

    # ✅ 初始化状态文本
    status_text = reactive.Value("Ready")

    @output
    @render.text
    def status():
        return status_text.get()

    view_update_trigger = reactive.Value(0)
    # 在此初始化一个存储表格的地方
    all_results = pd.read_csv("data/processed/7_views.csv", index_col=0)

    @output
    @render_plotly
    def output_investor_views_table():
        _ = view_update_trigger.get()
        CSV_PATH = Path("data/processed/7_views.csv")
        try:
            # 读取CSV并确保列名和索引为字符串
            all_results = pd.read_csv(CSV_PATH, index_col=0)
            all_results.columns = all_results.columns.astype(str)  # 列名转为字符串（兼容空格）
            all_results.index = all_results.index.astype(str)       # 索引转为字符串

            # 替换列名中的空格为下划线
            all_results.columns = all_results.columns.str.replace(" ", "_")
            all_results.index = all_results.index.str.replace(" ", "_")

            # 构建表头和单元格数据
            header_values = ["Metrics"] + list(all_results.columns)
            cell_values = [
                all_results.index.tolist()  # 第一列：指标名称（如"Mean Daily Return"）
            ] + [
                all_results[col].astype(str).tolist() for col in all_results.columns  # 后续列数据（如"View A"）
            ]

            # 生成 Plotly 表格
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=header_values,
                    align="center",
                    fill_color="lightgrey",
                    font=dict(size=12)
                ),
                cells=dict(
                    values=cell_values,  # 直接传入列式数据
                    align="center",
                    height=24
                ),
                columnwidth=[300] + [120] * (len(all_results.columns))  # 设置第一列宽度为300，其余列宽度为100
            )])
            
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=700
            )
            return fig

        except Exception as e:
            print("Error:", e)
            return go.Figure()



    def build_P_and_Q(input):
        # 设定因子顺序 ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        factor_names = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        
        # 构建 P_f 矩阵和 Q_f 向量
        P_f = []
        Q_f = []
        
        # 根据用户输入的选择逐个构建 P_f 和 Q_f
        if input.factor_mkt.get():
            Q_f.append(float(input.factor_mkt_value.get()) / 252)  # 假设用户输入的值是年化的，将其转为日收益
            if input.factor_mkt_direction.get() == "Up":
                P_f.append([1, 0, 0, 0, 0, 0])  # 假设 MKT 在第 1 个位置
            else:
                P_f.append([-1, 0, 0, 0, 0, 0])  # 假设 MKT 在第 1 个位置
        
        if input.factor_smb.get():
            Q_f.append(float(input.factor_smb_value.get()) / 252)  # 同样转化为日收益
            if input.factor_smb_direction.get() == "Up":
                P_f.append([0, 1, 0, 0, 0, 0])  # 假设 SMB 在第 2 个位置
            else:
                P_f.append([0, -1, 0, 0, 0, 0])  # 假设 SMB 在第 2 个位置
            
        if input.factor_hml.get():
            Q_f.append(float(input.factor_hml_value.get()) / 252)  # 转化为日收益
            if input.factor_hml_direction.get() == "Up":
                P_f.append([0, 0, 1, 0, 0, 0])  # 假设 HML 在第 3 个位置
            else:
                P_f.append([0, 0, -1, 0, 0, 0])  # 假设 HML 在第 3 个位置
            
        if input.factor_rmw.get():
            Q_f.append(float(input.factor_hml_value.get()) / 252)  # 转化为日收益
            if input.factor_rmw_direction.get() == "Up":
                P_f.append([0, 0, 0, 1, 0, 0])  # 假设 RMW 在第 4 个位置
            else: 
                P_f.append([0, 0, 0, -1, 0, 0])  # 假设 RMW 在第 4 个位置
            
        if input.factor_cma.get():
            Q_f.append(float(input.factor_hml_value.get()) / 252)  # 转化为日收益
            if input.factor_cma_direction.get() == "Up":
                P_f.append([0, 0, 0, 0, 1, 0])  # 假设 CMA 在第 5 个位置
            else:
                P_f.append([0, 0, 0, 0, -1, 0])  # 假设 CMA 在第 5 个位置
            
        if input.factor_mom.get():
            Q_f.append(float(input.factor_hml_value.get()) / 252)  # 转化为日收益
            if input.factor_mom_direction.get() == "Up":
                P_f.append([0, 0, 0, 0, 0, 1])  # 假设 MOM 在第 6 个位置
            else:
                P_f.append([0, 0, 0, 0, 0, -1])  # 假设 MOM 在第 6 个位置


        # 将 P_f 和 Q_f 转化为 NumPy 数组
        P_f = np.array(P_f)
        Q_f = np.array(Q_f).reshape(-1, 1)

        return P_f, Q_f


    @reactive.Effect
    @reactive.event(input.apply_investor_view)
    def apply_investor_view():
        # Print "hi" to confirm the action is triggered
        print("hi")
        # 检查用户是否选择了至少一个因子
        if not any([
            input.factor_mkt.get(),
            input.factor_smb.get(),
            input.factor_hml.get(),
            input.factor_rmw.get(),
            input.factor_cma.get(),
            input.factor_mom.get()
        ]):
            status_text.set("❌ No factors selected. Please select at least one factor.")
            return  # 不执行后续操作
    
        status_text.set("Working...")
        custom_view_name = input.custom_name.get()
        print(custom_view_name)
        
        P_f, Q_f = build_P_and_Q(input)
        
        risk_aversion = float(input.risk_aversion.get())
        tau = float(input.tau.get())

        train_ff_factors, test_ff_factors, train_etf_returns, test_etf_returns = split_trainTest()
        train_factor_returns, test_factor_returns, B, Omega_df, Sigma, test_etf_excess_returns = prepStep(train_ff_factors, test_ff_factors, train_etf_returns, test_etf_returns)

        custom_combination = metricGenerate(
            risk_aversion,
            tau,
            P_f,
            Q_f,
            Sigma,
            B,
            Omega_df,
            test_etf_excess_returns,
            test_factor_returns
        )

        # 将新的一列添加到表格中
        all_results[custom_view_name] = custom_combination
        status_text.set("✅ View added successfully!")

        #print(all_results[custom_view_name])

        # 更新 CSV 文件（以便表格内容持久化）
        all_results.to_csv("data/processed/7_views.csv")
        view_update_trigger.set(view_update_trigger.get() + 1)

    @output
    @render_plotly
    def output_investor_views_scatter():
        _ = view_update_trigger.get()  # reactive trigger to refresh chart on update

        # 读取 CSV 文件
        try:
            results_path = Path("data/processed/7_views.csv")
            df = pd.read_csv(results_path, index_col=0)
            df.columns = df.columns.astype(str)

            # 选择需要的指标列（先做存在性校验）
            if "Annualized Return" not in df.index or "Sharpe Ratio (annual)" not in df.index:
                print("Missing required rows in results file.")
                return go.Figure()

            # 提取 Annualized Return 和 Sharpe Ratio (annual)
            annual_return = df.loc["Annualized Return"].astype(str).str.replace('%', '').astype(float)
            sharpe_ratio = df.loc["Sharpe Ratio (annual)"].astype(float)
            views = df.columns.tolist()

            # 每个点颜色不同
            colors = px.colors.qualitative.Set1  # 可选: Set1, Set2, Pastel, Bold, etc.
            color_map = {view: colors[i % len(colors)] for i, view in enumerate(views)}
            marker_colors = [color_map[view] for view in views]

            # 创建交互式图表
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=annual_return,
                y=sharpe_ratio,
                mode="markers+text",
                text=views,
                textposition="top center",
                marker=dict(
                    size=12,
                    color=marker_colors,
                    line=dict(width=1, color='black')  # 加黑边提升视觉
                ),
                hovertemplate="View: %{text}<br>Annual Return: %{x}%<br>Sharpe Ratio: %{y}<extra></extra>",
            ))

            fig.update_layout(
                title="Sharpe Ratio vs Annualized Return",
                title_font=dict(size=18),
                xaxis=dict(
                    title="Annual Return (%)",
                    gridcolor="lightgrey",
                    zeroline=False
                ),
                yaxis=dict(
                    title="Sharpe Ratio",
                    gridcolor="lightgrey",
                    zeroline=False
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=20, r=20, t=40, b=40),
                height=500,
                showlegend=False
            )
            
            return fig

        except Exception as e:
            print("Error in scatter plot:", e)
            return go.Figure()

    @output
    @render.data_frame
    def output_viewsInfo_table():
        # 定义表格数据
        data = {
            "View Name": [
                "A: Growth", "B: Defensive", "C: Small-cap", "D: Inflation-style",
                "E: Balanced (Recommended)", "F: Value-style", "G: Growth + MOM Tilt"
            ],
            "Factor Exposure (P_f)": [
                "MKT↑, MOM↑, HML↓", "MKT↓, MOM↓, RMW↑", "SMB↑, MOM↓", "HML↑, CMA↑, MOM↓",
                "MKT↑, MOM↑, SMB↑, RMW↑", "MKT↑, HML↑, MOM↓", "MKT↑, MOM↑"
            ],
            "Expected Return (Q_f)": [
                "[0.0002, 0.0003, -0.0001]", "[-0.0001, -0.0002, 0.0003]", "[0.0003, -0.0001]",
                "[0.0004, 0.0003, -0.0002]", "[0.0001, 0.00015, 0.0002, 0.00025]",
                "[0.0001, 0.0003, -0.0002]", "[0.07/252, 0.03/252]"
            ]
        }
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        return df




if __name__ == "__main__":
    # 如果直接运行该脚本，会执行以下代码：
    # Recalculate View A-G when run in main
    pd.options.display.float_format = '{:.16f}'.format
    np.set_printoptions(precision=16, suppress=False)

    parameter_sets = [
        {   
            # View A: Growth
            "name": "View A",
            "risk_aversion": 2.5,
            "tau": 0.025,
            "P_f": np.array([
                [1, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 1],
                [0, 0, -1, 0, 0, 0]
            ]),
            "Q_f": np.array([
                [0.0002],
                [0.0003],
                [0.0001]
            ])
        },
        {
            "name": "View B",
            "risk_aversion": 2.5,
            "tau": 0.025,
            "P_f": np.array([
                [-1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1],
                [0, 0, 0, 1, 0, 0]
            ]),
            "Q_f": np.array([
                [0.0001],
                [0.0002],
                [0.0003]
            ])
        },
        {
            "name": "View C",
            "risk_aversion": 2.5,
            "tau": 0.025,
            "P_f": np.array([
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1]
            ]),
            "Q_f": np.array([
                [0.0003],
                [0.0001]
            ])
        },
        {
            "name": "View D",
            "risk_aversion": 2.5,
            "tau": 0.025,
            "P_f": np.array([
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, -1]
            ]),
            "Q_f": np.array([
                [0.0004],
                [0.0003],
                [0.0002]
            ])
        },
        {
            "name": "View E",
            "risk_aversion": 2.5,
            "tau": 0.025,
            "P_f": np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]
            ]),
            "Q_f": np.array([
                [0.0001],
                [0.00015],
                [0.0002],
                [0.00025]
            ])
        },
        {
            "name": "View F",
            "risk_aversion": 2.5,
            "tau": 0.025,
            "P_f": np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, -1]
            ]),
            "Q_f": np.array([
                [0.0001],
                [0.0003],
                [0.0002]
            ])
        },
        {
            "name": "View G",
            "risk_aversion": 2.5,
            "tau": 0.025,
            "P_f": np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1]
            ]),
            "Q_f": np.array([
                [0.07/252],
                [0.03/252]
            ])
        }
        # 你可以继续添加更多组合
    ]
    
    # 加载和划分数据
    train_ff_factors, test_ff_factors, train_etf_returns, test_etf_returns = split_trainTest()
    # 计算step1 -- 不需要反复计算的部分
    train_factor_returns, test_factor_returns, B, Omega_df, Sigma, test_etf_excess_returns = prepStep(train_ff_factors, test_ff_factors, train_etf_returns, test_etf_returns)

    all_results = {}

    for params in parameter_sets:
        print(f"🔍 Running: {params['name']}")
        
        result = metricGenerate(
            params["risk_aversion"],
            params["tau"],
            params["P_f"],
            params["Q_f"],
            Sigma,
            B,
            Omega_df,
            test_etf_excess_returns,
            test_factor_returns
        )
    
        # 存储结果
        all_results[params["name"]] = result

    # 转成DataFrame展示
    df_results = pd.DataFrame(all_results)
    print(df_results)

    save_path = Path("data/processed/7_views.csv")
    df_results.to_csv(save_path)
    copy_path = Path("data/raw/7_views.csv")
    df_results.to_csv(save_path)


    # 真正的反复计算部分
    #perf_metrics = metricGenerate(risk_aversion, tau, P_f, Q_f, Sigma, B, Omega_df, test_etf_excess_returns, test_factor_returns)
    

    '''
    results = {
        "Mean Daily Return": mean_return,
        "Annualized Return": annual_return,
        "Geometric Return": geometric_return,
        "Minimum Daily Return": min_return,
        "Volatility (daily)": volatility,
        "Volatility (annual)": annualized_volatility,
        "Sharpe Ratio (annual)": sharpe_ratio,
        "Skewness": skewness,
        "Kurtosis (excess)": kurtosis,
        "Max Drawdown": max_dd,
        "Max 10-Day Drawdown": max_10_day_dd,
        "VaR 95% (1-day)": VaR_95,
        "CVaR 95% (1-day)": CVaR_95,
        "Hidden Alpha π̃(x)": pi_tilde_x,
        "Factor Return ψ̃(x)": psi_tilde_x,
        "Residual ν̆(x)": residual_premium,
        "Factor R²_c(x)": R_squared_c
    }
    '''

    print("Main END")
        