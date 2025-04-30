"""
CVaR Portfolio Optimization Module for QWIM Dashboard
===============================================

This module implements portfolio optimization using Conditional Value at Risk (CVaR)
as the risk measure instead of the traditional Black-Litterman approach.
"""
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from great_tables import GT
from plotly.subplots import make_subplots
from plotnine import (aes, element_text, facet_wrap, geom_line, geom_point,
                      ggplot, labs, scale_color_brewer, theme, theme_minimal)
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from shiny import module, reactive, render, ui, render_ui
from shinywidgets import render_plotly, output_widget, render_widget
import cvxpy as cp
from sklearn.linear_model import LinearRegression

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        logger.info(f"[PERFORMANCE] {self.name} took {self.duration:.2f} seconds")


def load_data():
    """Load ETF and factor data efficiently"""
    # Use Path for cross-platform compatibility and relative paths
    base_path = Path(__file__).parent.parent.parent.parent
    ff5_path = base_path / "data/processed/ff5_mom_data.csv"
    etf_path = base_path / "data/raw/etf_data.csv"

    logger.info(f"Loading data from:\nFF5: {ff5_path}\nETF: {etf_path}")

    try:
        # Read data with modern datetime parsing
        ff_factors = pd.read_csv(
            ff5_path,
            parse_dates=['Date'],
            date_format='%Y-%m-%d',
            index_col='Date'
        )
        logger.info(f"Successfully loaded FF5 data with shape: {ff_factors.shape}")

        etf_data = pd.read_csv(
            etf_path,
            parse_dates=['Date'],
            date_format='%Y-%m-%d',
            index_col='Date'
        )
        logger.info(f"Successfully loaded ETF data with shape: {etf_data.shape}")

        # Pre-sort data by date for better performance in time-series operations
        ff_factors = ff_factors.sort_index()
        etf_data = etf_data.sort_index()

        return ff_factors, etf_data

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def get_etf_returns(etf_data):
    """Calculate asset returns efficiently"""
    # Use numpy operations for better performance
    prices = etf_data.values
    returns = np.diff(prices, axis=0) / prices[:-1]

    # Create DataFrame with proper dates
    returns_df = pd.DataFrame(
        returns,
        index=etf_data.index[1:],
        columns=etf_data.columns
    )

    return returns_df


def calculate_cvar(returns, weights, alpha=0.05):
    """
    Calculate the Conditional Value at Risk (CVaR) for a portfolio efficiently

    Parameters:
    -----------
    returns : numpy.ndarray
        Matrix of asset returns (n_samples, n_assets)
    weights : numpy.ndarray
        Portfolio weights
    alpha : float
        Confidence level (default: 0.05 for 95% CVaR)

    Returns:
    --------
    float
        CVaR value
    """
    # Calculate portfolio returns using vectorized operations
    portfolio_returns = returns @ weights

    # Calculate VaR efficiently
    var = np.percentile(portfolio_returns, alpha * 100)

    # Calculate CVaR using vectorized operations
    tail_returns = portfolio_returns[portfolio_returns <= var]
    cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var

    return cvar


def calculate_portfolio_metrics(returns, weights, risk_free_rate=0.02):
    """
    Calculate portfolio performance metrics.

    Args:
        returns (np.ndarray): Array of returns with shape (n_periods, n_assets) or (n_periods,)
        weights (np.ndarray): Array of weights with shape (n_assets,)
        risk_free_rate (float): Annual risk-free rate

    Returns:
        dict: Dictionary containing portfolio metrics
    """
    try:
        # Convert to numpy arrays if pandas
        if isinstance(returns, pd.Series) or isinstance(returns, pd.DataFrame):
            returns = returns.values
        if isinstance(weights, pd.Series):
            weights = weights.values

        # Ensure returns is 2D
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)

        # Ensure weights is 1D
        if len(weights.shape) > 1:
            weights = weights.flatten()

        # Calculate portfolio returns
        if returns.shape[1] == len(weights):
            portfolio_returns = returns @ weights
        else:
            # If dimensions don't match, we might have pre-calculated portfolio returns
            portfolio_returns = returns.flatten()

        # Calculate metrics
        annual_return = (1 + np.mean(portfolio_returns)) ** 252 - 1
        annual_vol = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Calculate CVaR
        sorted_returns = np.sort(portfolio_returns)
        var_cutoff = int(len(sorted_returns) * 0.05)
        cvar = np.mean(sorted_returns[:var_cutoff])

        # Calculate maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns / running_max - 1
        max_drawdown = np.min(drawdowns)

        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'cvar': cvar,
            'max_drawdown': max_drawdown
        }

    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        return {
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'cvar': 0.0,
            'max_drawdown': 0.0
        }


def calculate_max_drawdown(returns):
    """
    Calculate the maximum drawdown from a series of returns.

    Parameters:
        returns (pd.Series or np.ndarray): Array of returns

    Returns:
        float: Maximum drawdown
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_dd = drawdowns.min()
    return max_dd


def split_train_test(ff_factors, etf_returns):
    """Split data into training and testing periods"""
    train_start_date = "2008-07-01"
    train_end_date = "2019-12-31"
    test_start_date = "2020-01-01"
    test_end_date = "2024-12-31"

    train_ff_factors = ff_factors.loc[train_start_date:train_end_date]
    test_ff_factors = ff_factors.loc[test_start_date:test_end_date]

    train_etf_returns = etf_returns.loc[train_start_date:train_end_date]
    test_etf_returns = etf_returns.loc[test_start_date:test_end_date]

    return train_ff_factors, test_ff_factors, train_etf_returns, test_etf_returns


def optimize_cvar_portfolio(returns, alpha=0.05, target_return=None, constraints=None):
    """
    Optimize portfolio weights to minimize CVaR.

    Args:
        returns (pd.DataFrame): Asset returns
        alpha (float): Confidence level for CVaR (default: 0.05 for 95% CVaR)
        target_return (float): Target portfolio return (optional)
        constraints (list): Additional constraints (optional)

    Returns:
        np.array: Optimal portfolio weights
    """
    try:
        # Input validation
        if returns.shape[1] < 2:
            raise ValueError("Need at least 2 assets for optimization")
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            raise ValueError("Returns contain NaN or Inf values")
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")

        n_assets = returns.shape[1]
        n_samples = returns.shape[0]

        # Convert returns to numpy array if needed
        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        # Scale returns to improve numerical stability
        returns_std = np.std(returns, axis=0)
        returns_std[returns_std < 1e-8] = 1  # Prevent division by zero
        scaled_returns = returns / returns_std

        # Define variables
        w = cp.Variable(n_assets)
        aux = cp.Variable(n_samples)
        v = cp.Variable()

        # Basic constraints with numerical stability
        constraints = [
            cp.sum(w) == 1,  # Budget constraint
            w >= 0.001,  # Long only with minimum weight
            w <= 0.999  # Maximum weight constraint
        ]

        # CVaR constraint
        cvar_constraints = [
            aux >= 0,
            aux >= -scaled_returns @ w - v
        ]
        constraints.extend(cvar_constraints)

        # Target return constraint if specified
        if target_return is not None:
            if not isinstance(target_return, (int, float)):
                raise ValueError("Target return must be a number")
            mean_returns = np.mean(returns, axis=0)
            constraints.append(mean_returns @ w >= target_return)

        # Objective: Minimize CVaR
        objective = cp.Minimize(v + (1 / (alpha * n_samples)) * cp.sum(aux))

        # Solve with robust settings
        prob = cp.Problem(objective, constraints)

        # Try ECOS first with more iterations and tighter tolerances
        try:
            logger.info("Attempting optimization with ECOS solver...")
            result = prob.solve(
                solver=cp.ECOS,
                verbose=False,
                max_iters=5000,
                abstol=1e-10,
                reltol=1e-10,
                feastol=1e-10
            )
            if prob.status in ["optimal", "optimal_inaccurate"]:
                logger.info("ECOS solver succeeded")
                weights = np.array(w.value)
            else:
                raise cp.error.SolverError("ECOS failed to find optimal solution")

        except cp.error.SolverError:
            # Try SCS with more iterations and relaxed tolerance
            logger.warning("ECOS failed, trying SCS solver...")
            try:
                result = prob.solve(
                    solver=cp.SCS,
                    verbose=False,
                    max_iters=10000,
                    eps=1e-8,
                    alpha=1.8,
                    scale=0.1,
                    normalize=True
                )
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    logger.info("SCS solver succeeded")
                    weights = np.array(w.value)
                else:
                    raise cp.error.SolverError("SCS failed to find optimal solution")

            except cp.error.SolverError:
                logger.warning("Both solvers failed, trying OSQP...")
                try:
                    result = prob.solve(
                        solver=cp.OSQP,
                        verbose=False,
                        max_iter=10000,
                        eps_abs=1e-8,
                        eps_rel=1e-8,
                        polish=True,
                        adaptive_rho=True
                    )
                    if prob.status in ["optimal", "optimal_inaccurate"]:
                        logger.info("OSQP solver succeeded")
                        weights = np.array(w.value)
                    else:
                        raise cp.error.SolverError("OSQP failed to find optimal solution")
                except cp.error.SolverError:
                    logger.warning("All solvers failed, falling back to minimum variance")
                    return calculate_min_variance_portfolio(returns)

        # Clean up small weights and renormalize
        if weights is not None:
            weights[weights < 1e-4] = 0
            weights = weights / np.sum(weights)

            # Verify constraints
            weight_sum = np.sum(weights)
            min_weight = np.min(weights)
            max_weight = np.max(weights)

            logger.info(f"Optimization results - Sum: {weight_sum:.6f}, Min: {min_weight:.6f}, Max: {max_weight:.6f}")

            if abs(weight_sum - 1) > 1e-4 or min_weight < 0 or max_weight > 1:
                logger.warning("Weight constraints violated, using minimum variance portfolio")
                return calculate_min_variance_portfolio(returns)

            return weights
        else:
            logger.error("Optimization failed to produce valid weights")
            return calculate_min_variance_portfolio(returns)

    except Exception as e:
        logger.error(f"Error in optimize_cvar_portfolio: {str(e)}")
        return calculate_min_variance_portfolio(returns)


def calculate_min_variance_portfolio(returns):
    """Calculate the minimum variance portfolio when optimization fails."""
    # Convert returns to numpy array if it's a DataFrame
    if isinstance(returns, pd.DataFrame):
        returns = returns.values

    # Ensure returns is a 2D array
    if len(returns.shape) == 1:
        returns = returns.reshape(-1, 1)

    n_assets = returns.shape[1]

    try:
        cov_matrix = np.cov(returns.T)

        # Define the optimization problem
        weights = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0.001,  # Small minimum weight to prevent numerical issues
            weights <= 0.999  # Upper bound to prevent concentration
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=True)

        if weights.value is None:
            logging.warning("Minimum variance optimization failed. Returning equal weights...")
            return np.ones(n_assets) / n_assets

        # Ensure the weights sum to 1 and are non-negative (handle numerical errors)
        final_weights = np.array(weights.value)
        final_weights = np.maximum(final_weights, 0)  # Ensure non-negativity
        final_weights = final_weights / np.sum(final_weights)  # Normalize to sum to 1

        return final_weights

    except Exception as e:
        logging.error(f"Error in calculate_min_variance_portfolio: {str(e)}")
        # Return equal weights as last resort
        return np.ones(n_assets) / n_assets


def dynamic_rebalancing(returns, window_length=252, rebalance_freq=21, target_return=0.10):
    """
    Perform dynamic portfolio rebalancing using CVaR optimization.

    Args:
        returns (pd.DataFrame): Asset returns
        window_length (int): Length of rolling window in days
        rebalance_freq (int): Rebalancing frequency in days
        target_return (float): Target portfolio return

    Returns:
        tuple: (portfolio_returns, weights_df) where portfolio_returns is a pandas Series
              of daily returns and weights_df is a DataFrame of portfolio weights
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame")

    n_assets = returns.shape[1]
    dates = returns.index
    n_periods = len(returns) - window_length

    # Initialize arrays
    portfolio_returns = []
    weights_history = []
    return_dates = []

    # Initial weights (equal weight)
    current_weights = np.ones(n_assets) / n_assets

    for t in range(n_periods):
        if t % rebalance_freq == 0:  # Rebalance every rebalance_freq days
            # Get rolling window of returns
            window_returns = returns.iloc[t:t + window_length]

            try:
                # Optimize portfolio
                optimal_weights = optimize_cvar_portfolio(
                    returns=window_returns,
                    alpha=0.95,
                    target_return=target_return
                )
                # Validate weights
                if optimal_weights is not None and not np.any(np.isnan(optimal_weights)) and not np.any(
                        np.isinf(optimal_weights)):
                    current_weights = optimal_weights
                else:
                    logger.warning(f"Invalid weights at period {t}, using previous weights")
            except Exception as e:
                logger.warning(f"Optimization failed at period {t}, using previous weights: {str(e)}")
                # Keep using current weights
                pass

        # Store weights
        weights_history.append(current_weights)

        # Calculate portfolio return for this period
        next_returns = returns.iloc[t + window_length]
        port_return = np.dot(current_weights, next_returns)

        # Validate portfolio return
        if np.isnan(port_return) or np.isinf(port_return):
            logger.warning(f"Invalid portfolio return at period {t}, using 0")
            port_return = 0.0

        # Store return and date
        portfolio_returns.append(port_return)
        return_dates.append(dates[t + window_length])

    # Create weights DataFrame with proper index
    weights_df = pd.DataFrame(
        weights_history,
        index=dates[window_length:window_length + len(weights_history)],
        columns=returns.columns
    )

    # Create returns Series with proper index
    portfolio_returns = pd.Series(portfolio_returns, index=return_dates)

    # Final validation of returns series
    portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], 0)
    portfolio_returns = portfolio_returns.fillna(0)

    return portfolio_returns, weights_df


def plot_portfolio_performance(returns):
    """Plot portfolio performance over time."""
    with Timer("Plot Portfolio Performance"):
        try:
            # Convert returns to Series if numpy array
            if isinstance(returns, np.ndarray):
                returns = pd.Series(returns)
            elif isinstance(returns, pd.DataFrame):
                # If DataFrame, calculate portfolio returns using equal weights
                weights = np.ones(len(returns.columns)) / len(returns.columns)
                returns = returns.dot(weights)

            # # Ensure index is datetime
            if not isinstance(returns.index, pd.DatetimeIndex):
                returns.index = pd.to_datetime(returns.index)

            # Filter data to start from test_start_date
            # test_start_date = "2020-01-01"
            # returns = returns[returns.index >= test_start_date]

            # Calculate cumulative returns
            cum_rets = (1 + returns).cumprod()

            # Create figure
            fig = go.Figure()

            # Add trace with explicit dates
            fig.add_trace(go.Scatter(
                x=[str(item.date()) for item in cum_rets.index],
                y=cum_rets.values,
                name='Portfolio',
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>'
            ))

            # Update layout with better formatting
            fig.update_layout(
                title={
                    'text': 'Portfolio Performance',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': dict(size=18)
                },
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                template="plotly_white",
                showlegend=True,
                height=500,
                width=1000,
                margin=dict(l=50, r=50, t=50, b=50),
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis=dict(
                    gridcolor="lightgrey",
                    zeroline=False,
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    type='date',
                    tickformat='%Y-%m-%d',
                    range=[cum_rets.index[0], cum_rets.index[-1]]  # Keep the correct date range
                ),
                yaxis=dict(
                    gridcolor="lightgrey",
                    zeroline=False,
                    tickformat='.1%',
                    hoverformat='.2%'
                ),
                hovermode='x unified'
            )

            logger.info(f"Successfully created performance plot with {len(returns)} data points")
            return fig

        except Exception as e:
            logger.error(f"Error in plot_portfolio_performance: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig


def plot_portfolio_analysis(returns, metric):
    """Plot portfolio analysis based on selected metric."""
    try:
        # Convert returns to Series if numpy array
        if isinstance(returns, np.ndarray):
            if returns.ndim > 1:
                # If 2D array, calculate portfolio returns using equal weights
                weights = np.ones(returns.shape[1]) / returns.shape[1]
                returns = np.dot(returns, weights)
            returns = pd.Series(returns)
        elif isinstance(returns, pd.DataFrame):
            # If DataFrame, calculate portfolio returns using equal weights
            weights = np.ones(len(returns.columns)) / len(returns.columns)
            returns = returns.dot(weights)

        fig = go.Figure()

        if metric == "Returns":
            # Calculate kernel density estimation for smoother distribution
            kde = stats.gaussian_kde(returns)
            x_range = np.linspace(returns.min(), returns.max(), 100)
            y_range = kde(x_range)

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                fill='tozeroy',
                name='Returns Distribution',
                line=dict(color='#1f77b4', width=2)
            ))

            # Add histogram as overlay
            fig.add_trace(go.Histogram(
                x=returns,
                name='Returns Histogram',
                opacity=0.5,
                nbinsx=50,
                marker_color='#1f77b4',
                showlegend=False
            ))

            fig.update_layout(
                title='Portfolio Returns Distribution',
                xaxis_title='Return',
                yaxis_title='Density',
                bargap=0.1
            )

        elif metric == "Volatility":
            vol = returns.rolling(21).std() * np.sqrt(252)
            fig.add_trace(go.Scatter(
                x=vol.index,
                y=vol,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{x}<br>Volatility: %{y:.2%}<extra></extra>'
            ))

            # Add mean line
            mean_vol = vol.mean()
            fig.add_hline(
                y=mean_vol,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_vol:.1%}",
                annotation_position="bottom right"
            )

            fig.update_layout(
                title='Rolling Annualized Volatility (21-day)',
                xaxis_title='Date',
                yaxis_title='Volatility',
                yaxis_tickformat='.0%'
            )

        elif metric == "CVaR":
            # Calculate rolling CVaR
            rolling_cvar = returns.rolling(63).apply(
                lambda x: calculate_cvar(x.values.reshape(-1, 1), np.array([1]))
            )

            fig.add_trace(go.Scatter(
                x=rolling_cvar.index,
                y=rolling_cvar,
                mode='lines',
                name='Rolling CVaR',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{x}<br>CVaR: %{y:.2%}<extra></extra>'
            ))

            # Add mean line
            mean_cvar = rolling_cvar.mean()
            fig.add_hline(
                y=mean_cvar,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_cvar:.1%}",
                annotation_position="bottom right"
            )

            fig.update_layout(
                title='Rolling CVaR (63-day)',
                xaxis_title='Date',
                yaxis_title='CVaR',
                yaxis_tickformat='.0%'
            )

        elif metric == "Drawdown":
            # Calculate drawdown series
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max

            fig.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns,
                mode='lines',
                name='Drawdown',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                hovertemplate='%{x}<br>Drawdown: %{y:.2%}<extra></extra>'
            ))

            # Add maximum drawdown line
            max_dd = drawdowns.min()
            fig.add_hline(
                y=max_dd,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Max Drawdown: {max_dd:.1%}",
                annotation_position="bottom right"
            )

            fig.update_layout(
                title='Drawdown Analysis',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                yaxis_tickformat='.0%'
            )

        # Common layout settings
        fig.update_layout(
            template='plotly_white',
            showlegend=True,
            height=500,
            width=1000,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                gridcolor="lightgrey",
                zeroline=False,
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
            ),
            yaxis=dict(
                gridcolor="lightgrey",
                zeroline=False
            ),
            hovermode='x unified'
        )

        return fig

    except Exception as e:
        logger.error(f"Error in plot_portfolio_analysis: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def plot_portfolio_weights(weights_df, asset_names=None):
    """Plot portfolio weights over time."""
    with Timer("Plot Portfolio Weights"):
        try:
            # Convert weights to DataFrame if numpy array
            if isinstance(weights_df, np.ndarray):
                if asset_names is None:
                    asset_names = [f'Asset {i + 1}' for i in range(weights_df.shape[1])]
                weights_df = pd.DataFrame(weights_df, columns=asset_names)

            # Ensure weights are in decimal form
            if weights_df.max().max() > 1:
                weights_df = weights_df / 100

            # Create figure
            fig = go.Figure()

            # Add trace for each asset with custom colors
            colors = px.colors.qualitative.Set1
            for i, column in enumerate(weights_df.columns):
                fig.add_trace(go.Scatter(
                    x=[str(item.date()) for item in weights_df.index],
                    y=weights_df[column],
                    name=column,
                    mode='lines',
                    stackgroup='one',  # Enable stacking
                    line=dict(width=0.5),
                    fillcolor=colors[i % len(colors)],
                    hovertemplate=f"{column}: %{y:.1%}<extra></extra>"
                ))

            # Update layout
            fig.update_layout(
                title={
                    'text': 'Portfolio Weights Over Time',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': dict(size=18)
                },
                xaxis_title="Date",
                yaxis_title="Weight",
                template="plotly_white",
                showlegend=True,
                height=500,
                width=1000,
                margin=dict(l=50, r=50, t=50, b=50),
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis=dict(
                    gridcolor="lightgrey",
                    zeroline=False,
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
                ),
                yaxis=dict(
                    gridcolor="lightgrey",
                    zeroline=False,
                    tickformat='.0%',
                    hoverformat='.1%',
                    range=[0, 1]  # Force y-axis from 0 to 1 (0% to 100%)
                ),
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
            )

            logger.info(
                f"Successfully created weights plot with {len(weights_df)} time periods and {len(weights_df.columns)} assets")
            return fig

        except Exception as e:
            logger.error(f"Error in plot_portfolio_weights: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig


def plot_stress_test(returns_dict, benchmark_returns, crisis_periods):
    """Plot stress test analysis comparing portfolio performance during crisis periods."""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(crisis_periods.keys()),
            vertical_spacing=0.1
        )

        colors = px.colors.qualitative.Set1

        for idx, (name, (start, end)) in enumerate(crisis_periods.items()):
            row = (idx // 2) + 1
            col = (idx % 2) + 1

            # Extract crisis period data
            for i, (strategy_name, returns) in enumerate(returns_dict.items()):
                crisis_returns = returns.loc[start:end]
                cum_returns = (1 + crisis_returns).cumprod()

                fig.add_trace(
                    go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns,
                        name=strategy_name,
                        mode='lines',
                        line=dict(color=colors[i]),
                        showlegend=idx == 0  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )

            # Add benchmark
            benchmark_crisis = benchmark_returns.loc[start:end]
            cum_benchmark = (1 + benchmark_crisis).cumprod()

            fig.add_trace(
                go.Scatter(
                    x=cum_benchmark.index,
                    y=cum_benchmark,
                    name='Benchmark',
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    showlegend=idx == 0
                ),
                row=row, col=col
            )

        fig.update_layout(
            height=800,
            width=1000,
            title='Stress Test Analysis',
            template='plotly_white',
            showlegend=True,
            margin=dict(l=50, r=50, t=100, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        # Update all subplot axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    title_text="Date",
                    gridcolor="lightgrey",
                    zeroline=False,
                    row=i,
                    col=j
                )
                fig.update_yaxes(
                    title_text="Cumulative Return",
                    gridcolor="lightgrey",
                    zeroline=False,
                    row=i,
                    col=j
                )

        return fig

    except Exception as e:
        logger.error(f"Error in plot_stress_test: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


@module.ui
def model4_ui():
    """Create the UI for CVaR Portfolio Optimization tab."""
    return ui.page_fluid(
        ui.h2("Portfolio Optimization using Conditional Value at Risk (CVaR)"),
        ui.div(
            ui.markdown('''*This section is handled by Ge Meng.*'''),
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Portfolio Optimization",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Optimization Parameters"),
                        ui.input_slider("confidence_level", "Confidence Level (%):", min=90, max=99, value=95, step=1),
                        ui.input_action_button("optimize_portfolio", "Optimize Portfolio", class_="btn-primary")
                    ),
                    ui.h3("Portfolio Performance"),
                    output_widget("portfolio_performance_plot"),
                    ui.h4("Portfolio Metrics"),
                    ui.output_table("portfolio_metrics_table")
                )
            ),
            ui.nav_panel(
                "Dynamic Rebalancing",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Rebalancing Parameters"),
                        ui.input_slider("window_length", "Rolling Window Length (days):", min=30, max=90, value=40,
                                        step=10),
                        ui.input_slider("rebalance_freq", "Rebalancing Frequency (days):", min=5, max=15, value=10,
                                        step=5),
                        ui.input_action_button("run_dynamic", "Run Dynamic Strategy", class_="btn-primary")
                    ),
                    ui.h3("Strategy Performance"),
                    output_widget("dynamic_performance_plot")
                )
            ),
            ui.nav_panel(
                "Portfolio Analysis",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Analysis Parameters"),
                        ui.input_select(
                            "analysis_metric",
                            "Select Metric:",
                            choices=["Returns", "Volatility", "CVaR", "Drawdown"]
                        ),
                        ui.input_action_button("update_analysis", "Update Analysis", class_="btn-primary")
                    ),
                    ui.h3("Portfolio Analysis"),
                    output_widget("portfolio_analysis_plot")
                )
            )
        )
    )


@module.server
def model4_server(input, output, session, data_r, series_names_r):
    """Server logic for CVaR Portfolio Optimization tab."""
    # Initialize reactive values for caching results
    cached_weights = reactive.Value(None)
    cached_returns = reactive.Value(None)
    cached_metrics = reactive.Value(None)

    # Initialize data
    try:
        ff_factors, etf_data = load_data()
        etf_returns = get_etf_returns(etf_data)

        # Split data into train and test sets
        train_start_date = "2008-07-01"
        train_end_date = "2019-12-31"
        test_start_date = "2020-01-01"
        test_end_date = "2024-12-31"

        train_returns = etf_returns.loc[train_start_date:train_end_date]
        test_returns = etf_returns.loc[test_start_date:test_end_date]

        logger.info(f"Train data shape: {train_returns.shape}")
        logger.info(f"Test data shape: {test_returns.shape}")

        # Initialize with equal weights portfolio
        n_assets = train_returns.shape[1]
        initial_weights = np.ones(n_assets) / n_assets
        initial_returns = test_returns @ initial_weights
        initial_metrics = calculate_portfolio_metrics(test_returns, initial_weights)

        # Set initial values
        cached_weights.set(initial_weights)
        cached_returns.set(initial_returns)
        cached_metrics.set(initial_metrics)

    except Exception as e:
        logger.error(f"Error loading data in CVaR module: {str(e)}")
        return

    @reactive.Effect
    @reactive.event(input.optimize_portfolio)
    def optimize_static_portfolio():
        """Handle static portfolio optimization"""
        with Timer("Static Portfolio Optimization"):
            try:
                # Get parameters
                alpha = 1 - (input.confidence_level() / 100)
                logger.info(f"Optimizing portfolio with alpha={alpha}")

                # Verify data dimensions
                logger.info(f"Training returns shape before optimization: {train_returns.shape}")

                # Get optimization results without target return constraint
                weights = optimize_cvar_portfolio(
                    returns=train_returns,
                    alpha=alpha,
                    target_return=None  # Remove target return constraint
                )

                # Calculate out-of-sample performance
                if isinstance(weights, pd.Series):
                    weights = weights.values

                logger.info(f"Weights shape: {weights.shape}")
                logger.info(f"Test returns shape: {test_returns.shape}")

                # Ensure dimensions match
                if len(weights) != test_returns.shape[1]:
                    raise ValueError(
                        f"Dimension mismatch: weights ({len(weights)}) != returns columns ({test_returns.shape[1]})")

                # Calculate portfolio returns and metrics
                portfolio_returns = test_returns @ weights
                metrics = calculate_portfolio_metrics(test_returns, weights)

                # Update cached values
                cached_weights.set(weights)
                cached_returns.set(portfolio_returns)
                cached_metrics.set(metrics)

                logger.info("Portfolio optimization completed successfully")

            except Exception as e:
                logger.error(f"Error in portfolio optimization: {str(e)}")
                ui.notification_show(
                    "Error in portfolio optimization. Please check the parameters and try again.",
                    type="error",
                    duration=5
                )

    @reactive.Effect
    @reactive.event(input.run_dynamic)
    def run_dynamic_strategy():
        """Handle dynamic portfolio rebalancing"""
        try:
            with Timer("Dynamic Strategy Execution"):
                # Get parameters
                window_len = input.window_length()
                rebalance_freq = input.rebalance_freq()
                alpha = 1 - (input.confidence_level() / 100)

                logger.info(f"Running dynamic strategy with window={window_len}, freq={rebalance_freq}")
                logger.info(f"Training data shape: {train_returns.shape}")

                # Run dynamic strategy without target return
                returns_series, weights_df = dynamic_rebalancing(
                    returns=train_returns,
                    window_length=window_len,
                    rebalance_freq=rebalance_freq,
                    target_return=None  # Remove target return constraint
                )

                # Calculate metrics
                metrics = calculate_portfolio_metrics(returns_series, weights_df.iloc[-1].values)

                # Update cached values
                cached_weights.set(weights_df)
                cached_returns.set(returns_series)
                cached_metrics.set(metrics)

                logger.info("Dynamic strategy completed successfully")

        except Exception as e:
            logger.error(f"Error in dynamic strategy: {str(e)}")
            ui.notification_show(
                "Error in dynamic strategy. Please check the parameters and try again.",
                type="error",
                duration=5
            )

    @output
    @render_widget
    def portfolio_performance_plot():
        """Render portfolio performance plot"""
        if cached_returns.get() is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No portfolio data available. Please run optimization first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        try:
            returns_series = pd.Series(cached_returns.get())
            fig = plot_portfolio_performance(returns_series)
            if fig is None:
                raise ValueError("Failed to generate portfolio performance plot")
            return fig

        except Exception as e:
            logger.error(f"Error rendering performance plot: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

    @output
    @render.table
    def portfolio_metrics_table():
        """Render portfolio metrics table"""
        if cached_metrics.get() is None:
            return None

        try:
            metrics_df = pd.DataFrame(
                cached_metrics.get().items(),
                columns=['Metric', 'Value']
            )
            # Format the values for better display
            metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2%}" if abs(x) < 1 else f"{x:.4f}")
            return metrics_df
        except Exception as e:
            logger.error(f"Error rendering metrics table: {str(e)}")
            return None

    @output
    @render_widget
    def portfolio_analysis_plot():
        """Render portfolio analysis plot"""
        if cached_returns.get() is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No portfolio data available. Please run optimization first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        try:
            metric = input.analysis_metric()
            returns_series = pd.Series(cached_returns.get(), index=test_returns.index)
            return plot_portfolio_analysis(returns_series, metric)
        except Exception as e:
            logger.error(f"Error rendering analysis plot: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

    @output
    @render_widget
    def portfolio_weights_plot():
        """Render portfolio weights plot"""
        if cached_weights.get() is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No portfolio data available. Please run optimization first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        try:
            weights = cached_weights.get()
            is_dynamic = isinstance(weights, pd.DataFrame)
            if is_dynamic:
                return plot_portfolio_weights(weights)
            else:
                weights_series = pd.Series(weights, index=train_returns.columns)
                return plot_portfolio_weights(pd.DataFrame(weights_series).T)
        except Exception as e:
            logger.error(f"Error rendering weights plot: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

    @output
    @render_widget
    def dynamic_performance_plot():
        """Render dynamic strategy performance plot"""
        if cached_returns.get() is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No dynamic strategy data available. Please run the strategy first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        try:
            returns_series = pd.Series(cached_returns.get())
            fig = plot_portfolio_performance(returns_series)
            if fig is None:
                raise ValueError("Failed to generate dynamic performance plot")

            fig.update_layout(
                title="Dynamic Strategy Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                template="plotly_white"
            )
            return fig

        except Exception as e:
            logger.error(f"Error rendering dynamic performance plot: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig


def test_portfolio_optimization():
    """Test the portfolio optimization process."""
    try:
        logger.info("Starting portfolio optimization test...")

        # 1. Load and prepare data
        logger.info("Step 1: Loading data...")
        ff_factors, etf_data = load_data()
        etf_returns = get_etf_returns(etf_data)

        # Split data
        train_start_date = "2008-07-01"
        train_end_date = "2019-12-31"
        test_start_date = "2020-01-01"
        test_end_date = "2024-12-31"

        train_returns = etf_returns.loc[train_start_date:train_end_date]
        test_returns = etf_returns.loc[test_start_date:test_end_date]

        logger.info(f"Train data shape: {train_returns.shape}")
        logger.info(f"Test data shape: {test_returns.shape}")
        logger.info(f"Asset names: {train_returns.columns.tolist()}")

        # 2. Test different optimization scenarios
        scenarios = [
            {"name": "Conservative", "alpha": 0.05, "target_return": 0.05},
            {"name": "Moderate", "alpha": 0.05, "target_return": 0.10},
            {"name": "Aggressive", "alpha": 0.05, "target_return": 0.15}
        ]

        for scenario in scenarios:
            logger.info(f"\nTesting {scenario['name']} scenario:")
            logger.info(f"Parameters: alpha={scenario['alpha']}, target_return={scenario['target_return']}")

            try:
                # 3. Run optimization
                weights = optimize_cvar_portfolio(
                    returns=train_returns,
                    alpha=scenario['alpha'],
                    target_return=scenario['target_return']
                )

                # 4. Verify optimization results
                if isinstance(weights, pd.Series):
                    weights = weights.values

                # Check weight constraints
                sum_weights = np.sum(weights)
                min_weight = np.min(weights)
                max_weight = np.max(weights)

                logger.info("Weight constraints:")
                logger.info(f"Sum of weights: {sum_weights:.6f} (should be close to 1.0)")
                logger.info(f"Min weight: {min_weight:.6f} (should be >= 0)")
                logger.info(f"Max weight: {max_weight:.6f}")

                # 5. Calculate and verify portfolio metrics
                portfolio_returns = test_returns @ weights
                metrics = calculate_portfolio_metrics(test_returns, weights)

                logger.info("\nPortfolio metrics:")
                for metric, value in metrics.items():
                    logger.info(f"{metric}: {value:.4f}")

            except Exception as e:
                logger.error(f"Error in {scenario['name']} scenario: {str(e)}")
                continue

        logger.info("\nPortfolio optimization test completed.")
        return True

    except Exception as e:
        logger.error(f"Portfolio optimization test failed: {str(e)}")
        return False


def calculate_annualized_return(returns):
    """Calculate the annualized return from a series of returns."""
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252  # Assuming 252 trading days per year
    return (1 + total_return) ** (1 / n_years) - 1


def calculate_annualized_volatility(returns):
    """Calculate the annualized volatility from a series of returns."""
    return returns.std() * np.sqrt(252)  # Assuming 252 trading days per year


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate the Sharpe ratio from a series of returns."""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def test_dynamic_rebalancing():
    """Test the dynamic rebalancing strategy."""
    logging.info("Loading data...")
    ff5_data, etf_data = load_data()

    # Set parameters for dynamic rebalancing
    window_length = 252  # One year of trading days
    rebalance_freq = 21  # Monthly rebalancing
    confidence_level = 0.95
    target_return = 0.10  # 10% target annual return

    with Timer(name="dynamic_strategy") as t:
        try:
            # Run dynamic rebalancing strategy
            portfolio_returns, weights_df = dynamic_rebalancing(
                returns=etf_data,
                window_length=window_length,
                rebalance_freq=rebalance_freq,
                target_return=target_return
            )
            print("=================")
            print(portfolio_returns)
            print("=================")
            # Calculate performance metrics
            ann_return = calculate_annualized_return(portfolio_returns)
            ann_vol = calculate_annualized_volatility(portfolio_returns)
            sharpe = calculate_sharpe_ratio(portfolio_returns)
            max_dd = calculate_max_drawdown(portfolio_returns)

            # Log results
            logger.info("Dynamic Strategy Results:")
            logger.info(f"Annualized Return: {ann_return:.4f}")
            logger.info(f"Annualized Volatility: {ann_vol:.4f}")
            logger.info(f"Sharpe Ratio: {sharpe:.4f}")
            logger.info(f"Maximum Drawdown: {max_dd:.4f}")
            logger.info(f"Number of rebalancing periods: {len(weights_df)}")
            logger.info(f"Portfolio returns shape: {len(portfolio_returns)}")
            logger.info(f"Weights matrix shape: {weights_df.shape}")

            # Create and test plots
            logger.info("Creating performance plot...")
            fig_perf = plot_portfolio_performance(portfolio_returns)
            logger.info("Performance plot created successfully")

            logger.info("Creating weights plot...")
            fig_weights = plot_portfolio_weights(weights_df, etf_data.columns)
            logger.info("Weights plot created successfully")

            # Optionally save plots
            # fig_perf.write_html("portfolio_performance.html")
            # fig_weights.write_html("portfolio_weights.html")

            return True

        except Exception as e:
            logger.error(f"Error in test_dynamic_rebalancing: {str(e)}")
            raise


def test_portfolio_analysis():
    """Test the portfolio analysis functionality."""
    try:
        logger.info("\nStarting portfolio analysis test...")

        # 1. Load and prepare data
        logger.info("Step 1: Loading data...")
        ff_factors, etf_data = load_data()
        etf_returns = get_etf_returns(etf_data)

        # Generate a sample portfolio for testing
        n_assets = len(etf_returns.columns)
        weights = np.ones(n_assets) / n_assets  # Equal-weight portfolio

        # 2. Test different analysis metrics
        metrics = ["Returns", "Volatility", "CVaR", "Drawdown"]

        for metric in metrics:
            logger.info(f"\nTesting {metric} analysis:")

            try:
                # 3. Generate analysis plot
                fig = plot_portfolio_analysis(etf_returns, metric)

                logger.info(f"Successfully generated {metric} analysis plot")

                # 4. Calculate relevant metrics
                if metric == "Returns":
                    portfolio_returns = etf_returns @ weights
                    mean_return = np.mean(portfolio_returns)
                    std_return = np.std(portfolio_returns)
                    logger.info(f"Mean return: {mean_return:.4f}")
                    logger.info(f"Return std: {std_return:.4f}")

                elif metric == "Volatility":
                    vol = pd.Series(etf_returns @ weights).rolling(21).std() * np.sqrt(252)
                    logger.info(f"Average annualized volatility: {vol.mean():.4f}")

                elif metric == "CVaR":
                    cvar = calculate_cvar(etf_returns, weights)
                    logger.info(f"Portfolio CVaR: {cvar:.4f}")

                elif metric == "Drawdown":
                    max_dd, dd_series = calculate_max_drawdown(etf_returns @ weights)
                    logger.info(f"Maximum drawdown: {max_dd:.4f}")

            except Exception as e:
                logger.error(f"Error in {metric} analysis: {str(e)}")
                continue

        # 5. Test portfolio weights visualization
        try:
            weights_fig = plot_portfolio_weights(pd.Series(weights, index=etf_returns.columns))
            logger.info("\nSuccessfully generated portfolio weights plot")
        except Exception as e:
            logger.error(f"Error in portfolio weights visualization: {str(e)}")

        logger.info("\nPortfolio analysis test completed.")
        return True

    except Exception as e:
        logger.error(f"Portfolio analysis test failed: {str(e)}")
        return False


# Add test execution to main
if __name__ == "__main__":
    test_portfolio_optimization()
    test_dynamic_rebalancing()
    test_portfolio_analysis()